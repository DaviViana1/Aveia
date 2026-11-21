r"""Compara embeddings (mesma segmentação, agrupamento Ward com k real): DER final por modelo de embedding.
    python testes/diarizacao/emb_eval.py wespeaker_r34lm,titanet_s,...
"""
import sys
from pathlib import Path
AQUI = Path(__file__).resolve().parent
sys.path[:0] = [str(AQUI.parent.parent), str(AQUI)]
import dataclasses
import numpy as np
from faster_whisper.audio import decode_audio
import candidatos as C
from metricas import avaliar, ler_rttm
from benchmark import conjuntos, refinar_ref
from transcritor import diarizacao_onnx as D

modelos = sys.argv[1].split(",")
metodo = int(sys.argv[2]) if len(sys.argv) > 2 else 1
p = dataclasses.replace(D.Params(), metodo=metodo)
sets = conjuntos([])
print(f"{'embedding':20}" + "".join(f"{n[:14]:>15}" for n, _, _ in sets) + "   média")
for m in modelos:
    linha, ders = f"{m:20}", []
    for nome, wav, rp in sets:
        a = decode_audio(str(wav), sampling_rate=16000)
        ref = ler_rttm(rp)
        if nome.startswith("ref-"):
            ref = refinar_ref(ref, wav)
        an, t = C.carregar_analise(C.SEG_PY3, C.EMB[m], a, p)
        k = len({s for _, _, s in ref})
        tr, _ = D.diarizar_analise(an, k, p)
        r = avaliar(ref, tr, len(a) / 16000)
        ders.append(r["der"])
        linha += f"{100*r['der']:15.1f}"
    print(linha + f"{100*np.mean(ders):8.1f}", flush=True)
