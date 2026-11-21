"""Benchmark de diarização: roda cada candidato nos áudios de testes/diarizacao/dados/ (e
testes/audio-referencia/ se existir), mede DER/JER/nº de locutores/RTF e imprime a tabela.

    .venv\\Scripts\\python testes\\diarizacao\\benchmark.py                       # baseline + método atual do app
    .venv\\Scripts\\python testes\\diarizacao\\benchmark.py -c baseline,atual -d mls,ami,ref
    .venv\\Scripts\\python testes\\diarizacao\\benchmark.py --oracle                # também com o nº real de locutores
    .venv\\Scripts\\python testes\\diarizacao\\benchmark.py --recalcular            # ignora o cache de resultados

Gera os dados antes com `preparar_dados.py`. Resultados ficam em testes/diarizacao/resultados/ (cache por candidato e áudio).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parent.parent
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(AQUI))

from faster_whisper.audio import decode_audio  # noqa: E402

import candidatos  # noqa: E402
from metricas import avaliar, escrever_rttm, ler_rttm  # noqa: E402

DADOS = AQUI / "dados"
REF_APP = RAIZ / "testes" / "audio-referencia"
RESULTADOS = AQUI / "resultados"


def conjuntos(filtro: list[str]) -> list[tuple[str, Path, Path]]:
    """(nome, wav, rttm)."""
    lista = [(w.stem, w, w.with_suffix(".rttm")) for w in sorted(DADOS.glob("*.wav"))]
    if (REF_APP / "audio.wav").exists() and (REF_APP / "referencia.rttm").exists():
        lista.append(("ref-audio", REF_APP / "audio.wav", REF_APP / "referencia.rttm"))
    if filtro:
        lista = [x for x in lista if any(x[0].startswith(f) for f in filtro)]
    return lista


def refinar_ref(ref, wav: Path):
    """A referência do app vem de timestamps de palavras do Whisper (inclui pausas dentro do turno): corta-se o que o
    VAD por energia considera silêncio (pausa >= 0,3 s), para que `miss` não seja inflado por pausas que nenhum sistema deve rotular."""
    from preparar_dados import _regioes_fala
    x = decode_audio(str(wav), sampling_rate=16000)
    voz = _regioes_fala(x, pausa_min=0.3)
    out = []
    for a, b, s in ref:
        for va, vb in voz:
            ia, ib = max(a, va), min(b, vb)
            if ib - ia > 0.05:
                out.append((ia, ib, s))
    return out


def rodar(nome_cand: str, nome: str, wav: Path, k: int | None, recalcular: bool) -> tuple[list, float, float]:
    tag = f"{nome_cand}__{nome}__k{k or 0}"
    rttm, meta = RESULTADOS / f"{tag}.rttm", RESULTADOS / f"{tag}.json"
    if not recalcular and rttm.exists() and meta.exists():
        m = json.loads(meta.read_text())
        return ler_rttm(rttm), m["dur"], m["tempo"]
    audio = decode_audio(str(wav), sampling_rate=16000)
    dur = len(audio) / 16000
    t0 = time.perf_counter()
    fn = candidatos.obter(nome_cand)
    trechos = fn(audio, k)
    tempo = getattr(fn, 'tempo', None) or (time.perf_counter() - t0)
    RESULTADOS.mkdir(exist_ok=True)
    escrever_rttm(rttm, nome, [(t[0], t[1], f"spk{t[2]}") for t in trechos])
    meta.write_text(json.dumps({"dur": dur, "tempo": tempo}))
    return ler_rttm(rttm), dur, tempo


def principal() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--candidatos", default="baseline,atual")
    ap.add_argument("-d", "--dados", default="")
    ap.add_argument("--oracle", action="store_true", help="roda também informando o nº real de locutores")
    ap.add_argument("--recalcular", action="store_true")
    ap.add_argument("--detalhe", action="store_true", help="tabela por áudio, não só a média")
    a = ap.parse_args()

    cands = [c for c in a.candidatos.split(",") if c]
    for c in cands:
        try:
            candidatos.obter(c)
        except (KeyError, ValueError, IndexError):
            sys.exit(f"candidato desconhecido: {c} (veja candidatos.obter)")
    sets = conjuntos([d for d in a.dados.split(",") if d])
    if not sets:
        sys.exit("sem áudios: rode preparar_dados.py")
    modos = [("auto", False)] + ([("oracle", True)] if a.oracle else [])

    linhas = []
    for c in cands:
        for modo, usar_k in modos:
            res = []
            for nome, wav, ref_path in sets:
                ref = ler_rttm(ref_path)
                if nome.startswith("ref-"):
                    ref = refinar_ref(ref, wav)
                k = len({s for _, _, s in ref}) if usar_k else None
                hip, dur, tempo = rodar(c, nome, wav, k, a.recalcular)
                m = avaliar(ref, hip, dur)
                m0 = avaliar(ref, hip, dur, collar=0.0)
                m.update(der0=m0["der"], dur=dur, tempo=tempo, nome=nome)
                res.append(m)
            linhas.append((c, modo, res))

    if a.detalhe:
        print("\n== por áudio ==")
        print(f"{'candidato':26} {'modo':6} {'áudio':14} {'DER%':>6} {'miss':>5} {'fa':>5} {'conf':>5} {'JER%':>6} {'locut.':>7} {'RTF':>5}")
        for c, modo, res in linhas:
            for m in res:
                print(f"{c:26} {modo:6} {m['nome']:14} {100*m['der']:6.1f} {100*m['miss']:5.1f} {100*m['fa']:5.1f} {100*m['conf']:5.1f} "
                      f"{100*m['jer']:6.1f} {m['n_hip']:>3}/{m['n_ref']:<3} {m['tempo']/m['dur']:5.2f}")
    print("\n== média por grupo (DER com collar ±0,25 s; DER0 = sem collar) ==")
    grupos = {"mls-pt (sintético)": "mls", "AMI (real, en)": "ami", "referência do app (real, pt)": "ref"}
    cab = f"{'candidato':26} {'modo':6}"
    for g in grupos:
        cab += f" | {g[:22]:22}"
    print(cab + " | RTF")
    print(f"{'':33}" + " | DER%  DER0%  JER%  loc.  " * len(grupos))
    for c, modo, res in linhas:
        s = f"{c:26} {modo:6}"
        for g, pref in grupos.items():
            sub = [m for m in res if m["nome"].startswith(pref)]
            if not sub:
                s += " | " + " " * 22
                continue
            media = lambda k: sum(m[k] for m in sub) / len(sub)
            loc = "/".join(f"{m['n_hip']}" for m in sub) + f"({'/'.join(str(m['n_ref']) for m in sub)})"
            s += f" | {100*media('der'):5.1f} {100*media('der0'):5.1f} {100*media('jer'):5.1f} {loc:>10}"
        rtf = sum(m["tempo"] for m in res) / sum(m["dur"] for m in res)
        print(s + f" | {rtf:.2f}")


if __name__ == "__main__":
    principal()
