"""Roda pipeline.transcrever (caminho classico real) no ditado, com ou sem glossario. Uso: rodar_gloss.py sem|com"""
import json
import os
import sys
import time
from pathlib import Path
from datetime import datetime

os.environ["HF_HUB_OFFLINE"] = "1"
RAIZ = str(Path(__file__).resolve().parents[3])  # raiz do projeto (este arquivo fica em testes/resultados-local/scripts/)
sys.path.insert(0, RAIZ)
os.chdir(RAIZ)

from transcritor import motores, pipeline  # noqa: E402
from testes.wer import wer  # noqa: E402

GLOSSARIO = "throughput, cluster, deploy, pipeline, nodes, latency, compliance, SLA, parsing, APIs RESTful, microserviço"
modo = sys.argv[1]
glossario = GLOSSARIO if modo == "com" else None

capturadas = []
original = motores.palavras


def espiao(*a, **k):
    for p in original(*a, **k):
        capturadas.append([round(p.start, 2), round(p.end, 2), p.word])
        yield p


motores.palavras = espiao

audio = os.path.join(RAIZ, "testes", "audio-ditado", "ditado.m4a")
meta = {}
inicio = datetime.now()
t0 = time.monotonic()
falas = pipeline.transcrever(audio, idioma="pt", motor="local", meta=meta, glossario=glossario)
dt = time.monotonic() - t0
fim = datetime.now()

texto = " ".join(f.texto for f in falas)
ref = open(os.path.join(RAIZ, "testes", "audio-ditado", "transcricao-referencia.txt"), encoding="utf-8").read()
saida = dict(
    configuracao=modo, glossario=glossario, inicio=inicio.isoformat(timespec="seconds"), fim=fim.isoformat(timespec="seconds"),
    tempo_s=round(dt, 1), texto=texto, n_palavras=len(capturadas), wer=round(wer(ref, texto) * 100, 2),
    falas=[[round(f.inicio, 2), round(f.fim, 2), f.locutor, f.texto] for f in falas],
    palavras=capturadas, meta=meta, env_hf_offline=os.environ.get("HF_HUB_OFFLINE"),
)
destino = os.path.join(RAIZ, "testes", "resultados-local", "bruto", f"gloss__{modo}__ditado.json")
with open(destino, "w", encoding="utf-8") as f:
    json.dump(saida, f, ensure_ascii=False, indent=1, default=str)
print("inicio", saida["inicio"], "fim", saida["fim"], "tempo", saida["tempo_s"], "palavras", saida["n_palavras"], "wer", saida["wer"])
print(texto)
print(json.dumps(meta, ensure_ascii=False, default=str)[:600])
