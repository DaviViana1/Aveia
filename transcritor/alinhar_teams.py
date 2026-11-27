"""Nomeia os "Locutor N" de uma gravação pelo VTT (legenda) do Teams.

    .venv\\Scripts\\python -m transcritor.alinhar_teams <pasta_saida> <arquivo.vtt>

1. Lê o VTT (`<v Nome>texto</v>` com tempos).
2. Estima o deslocamento entre o VTT e a gravação pelo texto parecido das primeiras falas (mediana das diferenças).
3. Cada "Locutor N" recebe o nome do VTT com a maior sobreposição de tempo (um nome só vai para um locutor).
4. Reescreve `locutores.json`, `meta.json` (origem_nomes = "teams") e `transcricao.md`.

Imprime só contagens (nenhum nome vai para a tela).
"""
from __future__ import annotations

import difflib
import json
import re
import statistics
import sys
import unicodedata
from pathlib import Path

from . import saida

PRIMEIRAS_FALAS = 15       # falas da gravação usadas para estimar o deslocamento
SIMILARIDADE_MIN = 0.55    # parecença de texto para valer como par
MIN_PALAVRAS = 4           # fala menor que isto não serve de âncora
_TEMPO = re.compile(r"(?:(\d+):)?(\d{1,2}):(\d{2})[.,](\d{1,3})")
_LINHA_TEMPO = re.compile(r"^\s*(\S+)\s+-->\s+(\S+)")
_VOZ = re.compile(r"<v(?:\.[^\s>]*)?\s+([^>]+)>(.*?)(?:</v>|$)", re.S)


def _seg(t: str) -> float:
    m = _TEMPO.fullmatch(t.strip())
    if not m:
        raise ValueError(f"tempo inválido: {t}")
    h, mi, s, ms = m.groups()
    return int(h or 0) * 3600 + int(mi) * 60 + int(s) + int(ms.ljust(3, "0")) / 1000


def ler_vtt(texto: str) -> list[dict]:
    """Cortes do VTT: [{inicio, fim, nome, texto}]. Corte sem `<v Nome>` é ignorado."""
    cortes: list[dict] = []
    blocos = re.split(r"\r?\n\s*\r?\n", texto.replace("﻿", ""))
    for bloco in blocos:
        linhas = bloco.splitlines()
        for i, ln in enumerate(linhas):
            m = _LINHA_TEMPO.match(ln)
            if not m:
                continue
            try:
                ini, fim = _seg(m.group(1)), _seg(m.group(2))
            except ValueError:
                break
            for nome, fala in _VOZ.findall("\n".join(linhas[i + 1:])):
                t = " ".join(re.sub(r"<[^>]+>", "", fala).split())
                if nome.strip() and t:
                    cortes.append({"inicio": ini, "fim": fim, "nome": nome.strip(), "texto": t})
            break
    return cortes


def _norm(t: str) -> str:
    t = unicodedata.normalize("NFKD", t.lower()).encode("ascii", "ignore").decode("ascii")
    return " ".join(re.sub(r"[^a-z0-9 ]+", " ", t).split())


def estimar_deslocamento(falas: list[dict], cortes: list[dict]) -> tuple[float, int]:
    """(deslocamento em s, pares usados). `tempo_vtt = tempo_gravacao + deslocamento`. Sem pares suficientes: (0, n)."""
    cn = [(c, _norm(c["texto"])) for c in cortes]
    difs = []
    for f in falas[:PRIMEIRAS_FALAS]:
        tf = _norm(f["texto"])
        if len(tf.split()) < MIN_PALAVRAS:
            continue
        melhor, melhor_r = None, 0.0
        for c, tc in cn:
            r = difflib.SequenceMatcher(None, tf, tc).ratio()
            if r > melhor_r:
                melhor, melhor_r = c, r
        if melhor is not None and melhor_r >= SIMILARIDADE_MIN:
            difs.append(melhor["inicio"] - f["inicio"])
    if len(difs) < 2:
        return 0.0, len(difs)
    return float(statistics.median(difs)), len(difs)


def _sobreposicao(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def nomear(falas: list[dict], cortes: list[dict], deslocamento: float) -> dict[str, str | None]:
    """Para cada rótulo, o nome do VTT com a maior sobreposição de tempo; cada nome serve a um rótulo só (vence o maior)."""
    pontos: dict[tuple[str, str], float] = {}
    rotulos: list[str] = []
    for f in falas:
        if f["locutor"] not in rotulos:
            rotulos.append(f["locutor"])
        a0, a1 = f["inicio"] + deslocamento, (f.get("fim") or f["inicio"]) + deslocamento
        for c in cortes:
            o = _sobreposicao(a0, a1, c["inicio"], c["fim"])
            if o > 0:
                chave = (f["locutor"], c["nome"])
                pontos[chave] = pontos.get(chave, 0.0) + o
    resultado: dict[str, str | None] = {r: None for r in rotulos}
    usados: set[str] = set()
    for (rot, nome), _ in sorted(pontos.items(), key=lambda kv: -kv[1]):
        if resultado[rot] is None and nome not in usados:
            resultado[rot] = nome
            usados.add(nome)
    return resultado


def alinhar(pasta: Path, vtt: Path) -> dict:
    pasta = Path(pasta)
    meta, falas = saida.ler(pasta)
    cortes = ler_vtt(Path(vtt).read_text(encoding="utf-8", errors="replace"))
    desloc, pares = estimar_deslocamento(falas, cortes)
    nomes = nomear(falas, cortes, desloc)
    meta["origem_nomes"] = "teams"
    meta.pop("locutores", None)
    saida.gravar(pasta, meta, falas, nomes, (pasta / "resumo.md").read_text(encoding="utf-8"))
    return {"cortes": len(cortes), "falas": len(falas), "locutores": len(nomes),
            "nomeados": sum(1 for n in nomes.values() if n), "pares": pares, "deslocamento_s": round(desloc, 1)}


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("uso: python -m transcritor.alinhar_teams <pasta_saida> <arquivo.vtt>")
        return 2
    r = alinhar(Path(argv[0]), Path(argv[1]))
    print(json.dumps(r))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
