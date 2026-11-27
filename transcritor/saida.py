"""Saída de uma reunião gravada por controle: a pasta `saida/<AAAA-MM-DD>_<HHMM>_<titulo>/` com os arquivos do contrato.

Arquivos: `meta.json`, `transcricao.md`, `resumo.md`, `locutores.json`, `falas.json` (as falas com os rótulos originais
"Locutor N", para poder renomear depois) e, por último e de forma atômica, `concluido.flag`. Quem lê a pasta só pode
confiar nela quando o `concluido.flag` existe. Nada aqui vai para o Git (`saida/` está no `.gitignore`).
"""
from __future__ import annotations

import json
import os
import re
import time
import unicodedata
from datetime import datetime
from pathlib import Path

SECOES = ("Resumo", "Pontos-chave", "Decisões", "Próximos passos")


def slug(titulo: str, limite: int = 50) -> str:
    t = unicodedata.normalize("NFKD", titulo or "").encode("ascii", "ignore").decode("ascii").lower()
    t = re.sub(r"[^a-z0-9]+", "-", t).strip("-")[:limite].strip("-")
    return t or "reuniao"


def pasta_para(raiz: Path, inicio: datetime, titulo: str) -> Path:
    """Caminho único da pasta (não cria). Repetindo data, hora e título, acrescenta -2, -3..."""
    base = f"{inicio:%Y-%m-%d_%H%M}_{slug(titulo)}"
    p = Path(raiz) / base
    n = 2
    while p.exists():
        p = Path(raiz) / f"{base}-{n}"
        n += 1
    return p


def _hms(s: float) -> str:
    s = max(0, int(s))
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"


def montar_transcricao(falas: list[dict], nomes: dict[str, str | None] | None = None) -> str:
    """Uma linha por fala: `[HH:MM:SS] Nome ou Locutor N: texto`."""
    nomes = nomes or {}
    linhas = []
    for f in falas:
        quem = nomes.get(f["locutor"]) or f["locutor"]
        texto = " ".join(str(f["texto"]).split())
        linhas.append(f"[{_hms(f['inicio'])}] {quem}: {texto}")
    return "\n".join(linhas) + ("\n" if linhas else "")


def montar_resumo(r: dict | None) -> str:
    """As quatro seções do contrato; com IA desligada ou falha (`r` None) as seções ficam vazias."""
    r = r or {}
    acoes = []
    for a in r.get("acoes") or []:
        extra = "".join(f" ({a[k]})" for k in ("responsavel", "prazo") if a.get(k))
        acoes.append(f"- {a.get('tarefa', '')}{extra}")
    corpo = {
        "Resumo": str(r.get("resumo") or "").strip(),
        "Pontos-chave": "\n".join(f"- {t}" for t in r.get("topicos") or []),
        "Decisões": "\n".join(f"- {d}" for d in r.get("decisoes") or []),
        "Próximos passos": "\n".join(acoes),
    }
    return "\n".join(f"## {s}\n{corpo[s]}\n" for s in SECOES)


def _atomico(destino: Path, conteudo: str) -> None:
    tmp = destino.with_name(destino.name + ".tmp")
    tmp.write_text(conteudo, encoding="utf-8")
    for tentativa in range(10):   # no Windows o rename falha se um leitor está com o arquivo aberto: tenta de novo
        try:
            os.replace(tmp, destino)
            return
        except PermissionError:
            if tentativa == 9:
                raise
            time.sleep(0.2)


def _json(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)


def gravar(pasta: Path, meta: dict, falas: list[dict], locutores: dict[str, str | None], resumo_md: str) -> None:
    """Grava tudo e, por último, o `concluido.flag` (atômico: aparece inteiro ou não aparece)."""
    pasta = Path(pasta)
    pasta.mkdir(parents=True, exist_ok=True)
    (pasta / "concluido.flag").unlink(missing_ok=True)   # regravação: a pasta volta a "incompleta" até terminar
    _atomico(pasta / "falas.json", _json(falas))
    _atomico(pasta / "transcricao.md", montar_transcricao(falas, locutores))
    _atomico(pasta / "resumo.md", resumo_md)
    _atomico(pasta / "locutores.json", _json(locutores))
    _atomico(pasta / "meta.json", _json({**meta, "locutores": locutores}))
    _atomico(pasta / "concluido.flag", datetime.now().isoformat(timespec="seconds") + "\n")


def ler(pasta: Path) -> tuple[dict, list[dict]]:
    """(meta, falas com rótulos originais) de uma pasta já gravada."""
    pasta = Path(pasta)
    meta = json.loads((pasta / "meta.json").read_text(encoding="utf-8"))
    falas = json.loads((pasta / "falas.json").read_text(encoding="utf-8"))
    return meta, falas
