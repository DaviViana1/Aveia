"""DER e JER (próprios, só numpy), em quadros de 10 ms.

DER = (fala perdida + falso alarme + confusão de locutor) / fala de referência, com sobreposição
contada (várias vozes no mesmo quadro). A correspondência referência↔hipótese é a que maximiza o tempo
em comum. `collar` é a tolerância ±s ao redor de cada fronteira da referência (NIST: 0,25 s de cada lado;
no pyannote.metrics isso equivale a collar=0.5).
JER = média, por locutor de referência, de 1 - interseção/união (locutor sem par = 100 %).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

QUADRO = 0.01


def ler_rttm(caminho: str | Path) -> list[tuple[float, float, str]]:
    out = []
    for linha in Path(caminho).read_text(encoding="utf-8").splitlines():
        p = linha.split()
        if len(p) >= 8 and p[0] == "SPEAKER":
            out.append((float(p[3]), float(p[3]) + float(p[4]), p[7]))
    return out


def escrever_rttm(caminho: str | Path, nome: str, trechos) -> None:
    linhas = [f"SPEAKER {nome} 1 {a:.3f} {b - a:.3f} <NA> <NA> {s} <NA> <NA>" for a, b, s in sorted(trechos, key=lambda t: t[0])]
    Path(caminho).write_text("\n".join(linhas) + "\n", encoding="utf-8")


def _matriz(trechos, nomes: list[str], n: int) -> np.ndarray:
    m = np.zeros((len(nomes), n), dtype=bool)
    idx = {k: i for i, k in enumerate(nomes)}
    for a, b, s in trechos:
        i, j = int(round(a / QUADRO)), int(round(b / QUADRO))
        m[idx[s], max(i, 0): min(j, n)] = True
    return m


def _melhor_pareamento(ov: np.ndarray) -> dict[int, int]:
    """Pareamento (ref i -> hip j) que maximiza a soma de `ov`; DP sobre subconjuntos do lado menor."""
    nr, nh = ov.shape
    if nr == 0 or nh == 0:
        return {}
    transposto = nr > nh
    if transposto:
        ov, nr, nh = ov.T, nh, nr
    # linhas = lado menor (máscara), colunas = lado maior
    dp = {0: (0.0, ())}
    for j in range(nh):
        novo = dict(dp)
        for mask, (v, par) in dp.items():
            for i in range(nr):
                if not mask >> i & 1 and ov[i, j] > 0:
                    m2, v2 = mask | 1 << i, v + ov[i, j]
                    if m2 not in novo or novo[m2][0] < v2:
                        novo[m2] = (v2, par + ((i, j),))
        dp = novo
    _, pares = max(dp.values(), key=lambda x: x[0])
    return {j: i for i, j in pares} if transposto else {i: j for i, j in pares}


def avaliar(ref, hip, duracao: float | None = None, collar: float = 0.25, pular_sobreposicao: bool = False) -> dict:
    """Retorna dict com der, jer, miss, fa, conf (frações do tempo de fala de ref) e n_ref/n_hip."""
    fim = max([b for _, b, _ in ref] + [b for _, b, _ in hip] + [duracao or 0.0])
    n = int(np.ceil(fim / QUADRO)) + 1
    nomes_r = sorted({s for _, _, s in ref})
    nomes_h = sorted({s for _, _, s in hip})
    R, H = _matriz(ref, nomes_r, n), _matriz(hip, nomes_h, n)

    valido = np.ones(n, dtype=bool)
    if collar > 0:
        c = int(round(collar / QUADRO))
        for a, b, _ in ref:
            for t in (a, b):
                k = int(round(t / QUADRO))
                valido[max(k - c, 0): k + c] = False
    if pular_sobreposicao:
        valido &= R.sum(0) <= 1

    ov = (R[:, None, valido] & H[None, :, valido]).sum(-1).astype(float) if len(nomes_r) and len(nomes_h) else np.zeros((len(nomes_r), len(nomes_h)))
    par = _melhor_pareamento(ov)

    nr, nh = R[:, valido].sum(0), H[:, valido].sum(0)
    certos = np.zeros(valido.sum())
    Rv, Hv = R[:, valido], H[:, valido]
    for i, j in par.items():
        certos += Rv[i] & Hv[j]
    total = nr.sum()
    miss = np.maximum(nr - nh, 0).sum()
    fa = np.maximum(nh - nr, 0).sum()
    conf = np.minimum(nr, nh).sum() - certos.sum()

    # JER (sem collar)
    Rj, Hj = R, H
    ovj = (Rj[:, None] & Hj[None]).sum(-1).astype(float) if len(nomes_r) and len(nomes_h) else np.zeros((len(nomes_r), len(nomes_h)))
    parj = _melhor_pareamento(ovj)
    jers = []
    for i in range(len(nomes_r)):
        if i in parj:
            inter = (Rj[i] & Hj[parj[i]]).sum()
            uniao = (Rj[i] | Hj[parj[i]]).sum()
            jers.append(1 - inter / uniao if uniao else 0.0)
        else:
            jers.append(1.0)
    return {
        "der": float((miss + fa + conf) / total) if total else 0.0,
        "miss": float(miss / total), "fa": float(fa / total), "conf": float(conf / total),
        "jer": float(np.mean(jers)) if jers else 0.0,
        "n_ref": len(nomes_r), "n_hip": len(nomes_h), "fala_ref": float(total * QUADRO),
    }
