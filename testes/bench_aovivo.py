"""Mede o modo ao vivo sem esperar o tempo real: entrega o arquivo inteiro a uma `Sessao` de uma vez (o pior caso, a fila
atrasada) e compara o corte de blocos antigo (primeira pausa depois de 15 s) com o adaptativo (última pausa da janela).

Uso (da raiz do projeto):
  .venv\\Scripts\\python testes\\bench_aovivo.py ref58-webm,fleurs --politica 0,1 --reps 1

Saída: uma linha por execução e `testes/resultados-local/bruto/aovivo_blocos__<politica>__<clipe>.json` (número de blocos,
chamadas de janela de 30 s, tempo total, parar até pronto, WER contra a referência humana). Sem rede e sem IA externa.
Os tempos valem só como comparação lado a lado na mesma máquina (CPU compartilhada varia 17 a 60 %).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ["AVEIA_SEM_AQUECER"] = "1"

from testes.wer import wer  # noqa: E402

AR = RAIZ / "testes" / "audio-referencia"
CL = RAIZ / "testes" / "resultados-local" / "clipes"
BRUTO = RAIZ / "testes" / "resultados-local" / "bruto"


def clipe(nome: str) -> tuple[Path, str]:
    """(arquivo de áudio, texto de referência humano)."""
    if nome == "ref58-webm":
        ref = " ".join(f["texto"] for f in json.load(open(AR / "referencia.json", encoding="utf-8")))
        return AR / "audio.webm", ref
    tabela = {"fleurs": ("fleurs-leitura.wav", "fleurs-leitura.txt"), "mls-livro": ("mls-audiolivro.wav", "mls-audiolivro.txt"),
              "mls-ruido-webm": ("mls-ruido.webm", "mls-ruido.txt"), "cv-webm": ("cv-microfones.webm", "cv-microfones.txt")}
    a, t = tabela[nome]
    return CL / a, (CL / t).read_text(encoding="utf-8").strip()


def rodar(nome: str, adaptativo: bool) -> dict:
    from transcritor import aovivo, pipeline
    aovivo.BLOCO_ADAPTATIVO = adaptativo
    origem, ref = clipe(nome)
    dados = origem.read_bytes()
    pasta = Path(tempfile.mkdtemp(prefix="aveia-bench-"))
    fila = ThreadPoolExecutor(max_workers=1)
    s = aovivo.Sessao("bench000", pasta / ("bench" + origem.suffix), "pt", None, lambda fn, *a: fila.submit(fn, *a))
    t0 = time.monotonic()
    for seq, i in enumerate(range(0, len(dados), 32768)):
        s.receber(seq, dados[i:i + 32768])
    s.marcar_fim(0.0)
    t_fim = time.monotonic()
    meta: dict = {}
    falas = fila.submit(s.concluir, None, None, meta).result()
    t1 = time.monotonic()
    texto = " ".join(f.texto for f in falas)
    blocos = list(s.blocos)
    janelas = sum(max(1, math.ceil((b - a) / 30.0)) for a, b in blocos)
    r = dict(clipe=nome, politica="adaptativo" if adaptativo else "antigo", blocos=len(blocos),
             duracoes=[round(b - a, 1) for a, b in blocos], janelas_30s=janelas, total_s=round(t1 - t0, 1),
             parar_ao_pronto_s=round(t1 - t_fim, 1), wer=round(wer(ref, texto) * 100, 2), palavras=len(texto.split()),
             locutores=len({f.locutor for f in falas}), lacunas=len(meta.get("lacunas") or []))
    s.liberar()
    fila.shutdown(wait=False)
    return r


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("clipes", help="lista separada por vírgula: ref58-webm, fleurs, mls-livro, mls-ruido-webm, cv-webm")
    ap.add_argument("--politica", default="0,1", help="0 = corte antigo, 1 = adaptativo (padrão: os dois, alternados)")
    ap.add_argument("--reps", type=int, default=1)
    a = ap.parse_args()
    BRUTO.mkdir(parents=True, exist_ok=True)
    from transcritor import pipeline
    pipeline.aquecer(em_segundo_plano=False)
    for rep in range(1, a.reps + 1):
        for nome in a.clipes.split(","):
            for pol in a.politica.split(","):
                r = rodar(nome, pol == "1")
                r["rep"] = rep
                arq = BRUTO / f"aovivo_blocos__{r['politica']}__{nome}__r{rep}.json"
                arq.write_text(json.dumps(r, ensure_ascii=False, indent=1), encoding="utf-8")
                print({k: v for k, v in r.items() if k != "duracoes"}, r["duracoes"], flush=True)


if __name__ == "__main__":
    main()
