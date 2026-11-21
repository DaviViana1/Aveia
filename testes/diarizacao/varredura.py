r"""Varredura de agrupamento sobre análises em cache (etapas 1-2 não são recalculadas).

    .venv\Scripts\python testes\diarizacao\varredura.py -e campp_zhen,titanet_s -l 0.04,0.05,0.06 [-m 3] [--k]

Imprime DER por grupo e nº de locutores estimados (estimado/real) por arquivo, para cada (embedding, limiar).
No método 3 (padrão) `-l` varre `limiar_var` (fração da variância); `--extra fusao=0.7,min_fala_locutor=10` ajusta o resto.
"""
from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

import numpy as np

AQUI = Path(__file__).resolve().parent
sys.path[:0] = [str(AQUI.parent.parent), str(AQUI)]

from faster_whisper.audio import decode_audio  # noqa: E402

import candidatos as C  # noqa: E402
from benchmark import conjuntos, refinar_ref  # noqa: E402
from metricas import avaliar, ler_rttm  # noqa: E402
from transcritor import diarizacao_onnx as D  # noqa: E402

EMB_ALIAS = {"campp_zhen": "3dspk_campp_zhen"}


def carregar(emb: str, p):
    """[(nome, ref, dur, Analise)] para todos os conjuntos."""
    out = []
    for nome, wav, rp in conjuntos([]):
        a = decode_audio(str(wav), sampling_rate=16000)
        ref = ler_rttm(rp)
        if nome.startswith("ref-"):
            ref = refinar_ref(ref, wav)
        an, _ = C.carregar_analise(C.SEG_PY3, C.EMB[emb], a, p)
        out.append((nome, ref, len(a) / 16000, an))
    return out


def avaliar_conjunto(dados, p, usar_k: bool):
    res = []
    for nome, ref, dur, an in dados:
        k = len({s for _, _, s in ref}) if usar_k else None
        tr, _ = D.diarizar_analise(an, k, p)
        m = avaliar(ref, [(a, b, f"s{g}") for a, b, g in tr], dur)
        res.append((nome, m["der"], m["n_hip"], m["n_ref"]))
    return res


def resumo(res):
    g = lambda pref: [r for r in res if r[0].startswith(pref)]
    s = ""
    for pref in ("mls", "ami", "ref"):
        sub = g(pref)
        s += f" {pref}:{100*np.mean([r[1] for r in sub]):5.1f}"
    s += f" | todos:{100*np.mean([r[1] for r in res]):5.1f} | "
    s += " ".join(f"{r[2]}/{r[3]}" for r in res)
    return s


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-e", default="campp_zhen,titanet_s")
    ap.add_argument("-l", default="1.0")
    ap.add_argument("-m", type=int, default=3)
    ap.add_argument("--k", action="store_true")
    ap.add_argument("--extra", default="", help="chave=valor,... de Params")
    a = ap.parse_args()
    extra = {}
    for kv in filter(None, a.extra.split(",")):
        k, v = kv.split("=")
        extra[k] = type(getattr(D.Params(), k))(float(v))
    for emb in a.e.split(","):
        p0 = dataclasses.replace(D.Params(), metodo=a.m, **extra)
        dados = carregar(emb, p0)
        print(f"-- {emb} metodo={a.m}  (nomes: {[d[0] for d in dados]})")
        for lim in [float(x) for x in a.l.split(",")]:
            p = dataclasses.replace(p0, limiar_var=lim) if a.m == 3 else dataclasses.replace(p0, limiar=lim)
            print(f" lim={lim:5.2f} auto  ", resumo(avaliar_conjunto(dados, p, False)), flush=True)
        if a.k:
            print(f" k dado        ", resumo(avaliar_conjunto(dados, p0, True)), flush=True)
