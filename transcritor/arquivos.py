"""Guarda em disco o áudio, a transcrição e a análise de cada nota, com limite por pasta.

    gravacoes/
        Áudio/          <base>.webm, <base>-2.webm ...   (uma por gravação da nota)
        Transcrições/   <base>.txt
        Análises/       <base>.md

`<base>` (AAAAMMDD-HHMMSS-titulo) é o mesmo nas três pastas e é escolhido pela interface, uma vez por nota.

Cada pasta guarda no máximo LIMITE arquivos. Passou disso, nada é apagado nem trocado: a nota segue só
no navegador (localStorage) e o áudio é descartado, como era antes. Reescrever um arquivo que já existe
(ex.: a transcrição com os nomes dos locutores atualizados) não ocupa vaga nova.
Excluir a nota na interface apaga os arquivos dela (`apagar`), liberando as vagas.
"""
from __future__ import annotations

import os
import re
import threading
from pathlib import Path

from .pipeline import Fala, _tempo, para_texto

LIMITE = 5
PASTAS = {"audio": "Áudio", "transcricao": "Transcrições", "analise": "Análises"}

_trava = threading.Lock()
_BASE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$")


_RESERVADOS = {"con", "prn", "aux", "nul", *(f"com{i}" for i in range(1, 10)), *(f"lpt{i}" for i in range(1, 10))}


def base_valida(base: str | None) -> bool:
    """A base vira nome de arquivo: só letras, números, ponto, hífen e sublinhado (nada de caminhos nem nomes reservados do Windows)."""
    return bool(base) and bool(_BASE.match(base)) and ".." not in base and base.split(".")[0].lower() not in _RESERVADOS


def criar_pastas(raiz: Path) -> None:
    for nome in PASTAS.values():
        (raiz / nome).mkdir(parents=True, exist_ok=True)


def _ha_vaga(pasta: Path, destino: Path) -> bool:
    if destino.exists():   # reescrever o mesmo arquivo não ocupa vaga
        return True
    return sum(1 for f in pasta.iterdir() if f.is_file()) < LIMITE


def guardar_audio(raiz: Path, origem: Path, base: str, parte: int) -> bool:
    """Move o áudio pendente para Áudio/. False se a pasta está cheia (o chamador apaga o áudio)."""
    nome = base if parte <= 1 else f"{base}-{parte}"
    pasta = raiz / PASTAS["audio"]
    destino = pasta / (nome + origem.suffix)
    with _trava:
        pasta.mkdir(parents=True, exist_ok=True)
        if not origem.exists() or not _ha_vaga(pasta, destino):
            return False
        os.replace(origem, destino)
    return True


def _gravar(raiz: Path, chave: str, base: str, extensao: str, conteudo: str) -> bool:
    pasta = raiz / PASTAS[chave]
    destino = pasta / (base + extensao)
    with _trava:
        pasta.mkdir(parents=True, exist_ok=True)
        if not _ha_vaga(pasta, destino):
            return False
        temporario = destino.with_name(destino.name + ".tmp")
        temporario.write_text(conteudo, encoding="utf-8")
        os.replace(temporario, destino)
    return True


def _hms(segundos) -> str:
    try:
        return _tempo(float(segundos))[:8]
    except (TypeError, ValueError):
        return str(segundos)


def texto_transcricao(titulo: str, base: str, duracao: float, falas: list[dict], destaques: list[dict] | None = None) -> str:
    """Transcrição em texto. Os destaques do usuário entram como `[HH:MM:SS] >>> DESTAQUE: texto` antes da
    primeira fala que começa depois deles (e depois da última fala, se não houver nenhuma)."""
    cabecalho = f"# {titulo.strip() or 'Sem título'}\nArquivo: {base} · Duração: {_hms(duracao)}\n\n"
    marcas = sorted(((float(d["t"]), " ".join(str(d["texto"]).split())) for d in destaques or [] if str(d.get("texto", "")).strip()),
                    key=lambda m: m[0])
    blocos: list[str] = []
    for f in falas:
        while marcas and marcas[0][0] < f["inicio"]:
            t, texto = marcas.pop(0)
            blocos.append(f"[{_hms(t)}] >>> DESTAQUE: {texto}")
        blocos.append(para_texto([Fala(f["inicio"], f.get("fim") or f["inicio"], f["locutor"], f["texto"])]))
    blocos += [f"[{_hms(t)}] >>> DESTAQUE: {texto}" for t, texto in marcas]
    return cabecalho + "\n\n".join(blocos) + "\n"


def texto_analise(titulo: str, base: str, r: dict) -> str:
    linhas = [f"# {titulo.strip() or 'Sem título'}", f"Arquivo: {base}" + (f" · Modelo: {r['modelo']}" if r.get("modelo") else ""), ""]
    perfil = r.get("perfil") or {}
    for aviso in perfil.get("avisos") or []:
        linhas += [f"> {aviso}", ""]
    if r.get("resumo"):
        linhas += ["## Resumo", str(r["resumo"]), ""]
    if r.get("destaques"):
        linhas.append("## Destaques")
        for d in r["destaques"]:
            texto = f" — {d['texto']}" if d.get("texto") else ""
            linhas.append(f"- **[{_hms(d.get('t', 0))}] {d.get('marca', '')}**{texto}")
        linhas.append("")
    if r.get("topicos"):
        linhas += ["## Pontos-chave"] + [f"- {t}" for t in r["topicos"]] + [""]
    if r.get("capitulos"):
        linhas.append("## Linha do tempo")
        for c in r["capitulos"]:
            resumo = f" — {c['resumo']}" if c.get("resumo") else ""
            linhas.append(f"- **[{_hms(c.get('inicio', 0))}] {c.get('titulo', '')}**{resumo}")
        linhas.append("")
    if r.get("decisoes"):
        linhas += ["## Decisões"] + [f"- {d}" for d in r["decisoes"]] + [""]
    if r.get("acoes"):
        linhas.append("## Próximos passos")
        for a in r["acoes"]:
            extra = "".join(f" ({a[k]})" for k in ("responsavel", "prazo") if a.get(k))
            linhas.append(f"- [ ] {a.get('tarefa', '')}{extra}")
        linhas.append("")
    return "\n".join(linhas)


def guardar_transcricao(raiz: Path, base: str, titulo: str, duracao: float, falas: list[dict],
                      destaques: list[dict] | None = None) -> bool:
    return _gravar(raiz, "transcricao", base, ".txt", texto_transcricao(titulo, base, duracao, falas, destaques))


def guardar_analise(raiz: Path, base: str, titulo: str, resultado: dict) -> bool:
    return _gravar(raiz, "analise", base, ".md", texto_analise(titulo, base, resultado))


def apagar(raiz: Path, base: str) -> list[str]:
    """Apaga os arquivos de uma nota nas três pastas (só os que têm exatamente este nome-base).

    No Áudio entram `<base>.ext` e `<base>-2.ext`, `<base>-3.ext`... (as gravações seguintes da nota).
    Devolve os caminhos removidos, relativos a `raiz`.
    """
    if not base_valida(base):
        return []
    partes = re.compile(re.escape(base) + r"(-\d+)?")
    removidos: list[str] = []
    with _trava:
        for chave, ext in (("audio", None), ("transcricao", ".txt"), ("analise", ".md")):
            pasta = raiz / PASTAS[chave]
            if not pasta.is_dir():
                continue
            for f in pasta.iterdir():
                if not f.is_file():
                    continue
                if chave == "audio":
                    casa = partes.fullmatch(f.stem) is not None
                else:
                    casa = f.name == base + ext
                if casa:
                    f.unlink(missing_ok=True)
                    removidos.append(f"{PASTAS[chave]}/{f.name}")
    return removidos


def _arquivos_da_raiz(raiz: Path) -> list[Path]:
    """Tudo o que `gravacoes/` guarda: as três pastas e o que está pendente na raiz (áudio de trabalho e o .json dele)."""
    achados: list[Path] = []
    if raiz.is_dir():
        achados += [f for f in raiz.iterdir() if f.is_file()]
        for nome in PASTAS.values():
            pasta = raiz / nome
            if pasta.is_dir():
                achados += [f for f in pasta.iterdir() if f.is_file()]
    return achados


def situacao(raiz: Path, dias: int, agora: float | None = None) -> dict:
    """Quantos arquivos há, a idade do mais antigo e quantos passaram de `dias` (só números; nada é apagado)."""
    import time
    agora = time.time() if agora is None else agora
    idades = []
    for f in _arquivos_da_raiz(raiz):
        try:
            idades.append((agora - f.stat().st_mtime) / 86400)
        except OSError:
            continue
    return {"arquivos": len(idades), "mais_antigo_dias": round(max(idades), 1) if idades else 0,
            "passaram_do_prazo": sum(1 for i in idades if i > dias), "prazo_dias": dias}


def apagar_tudo(raiz: Path) -> int:
    """Apaga o áudio, as transcrições e as análises guardados, mais o pendente na raiz. Mantém as pastas. Devolve quantos."""
    n = 0
    with _trava:
        for f in _arquivos_da_raiz(raiz):
            try:
                f.unlink(missing_ok=True)
                n += 1
            except OSError as e:
                print(f"Não foi possível apagar {f.name}: {e}")
    return n
