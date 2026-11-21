"""Mede o WER do Whisper (e a saída da diarização) com e sem pré-processamento, por variante degradada.

Uso (da raiz do projeto):
  .venv\\Scripts\\python testes\\microfone\\medir.py <pasta_variantes> <referencia.txt> <saida.json> [config ...]
configs = valores de AVEIA_PREPROC (`off`, `hpf`, `hpf,norm`, `hpf,norm,denoise`...). Padrão: off hpf,norm.
Transcreve com faster-whisper via transcritor.pipeline.transcrever (o mesmo caminho do app), sem diarização no WER;
com --diar mede também nº de locutores/fatias de turnos.  Resultados acumulam em <saida.json> (retomável).
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
import unicodedata
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
os.cpu_count = lambda: int(os.environ.get("AVEIA_THREADS", "4"))   # CPU compartilhada: não ocupa todos os núcleos
sys.path.insert(0, str(RAIZ))


def normalizar(t: str) -> list[str]:
    t = unicodedata.normalize("NFKD", t.lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    t = re.sub(r"[^a-z0-9 ]+", " ", t)
    return t.split()


def wer(ref: list[str], hip: list[str]) -> tuple[float, int, int, int]:
    n, m = len(ref), len(hip)
    d = list(range(m + 1))
    for i in range(1, n + 1):
        prev, d[0] = d[0], i
        for j in range(1, m + 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (ref[i - 1] != hip[j - 1]))
            prev, d[j] = d[j], cur
    return d[m] / max(n, 1), n, m, d[m]


def rodar(wav: Path, config: str, diar: bool):
    os.environ["AVEIA_PREPROC"] = config
    from transcritor.pipeline import transcrever
    t0 = time.time()
    falas = transcrever(str(wav), idioma="pt", diarizacao=diar)
    return falas, time.time() - t0


if __name__ == "__main__":
    pasta, ref_txt, saida = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    resto = [a for a in sys.argv[4:] if not a.startswith("--")]
    diar = "--diar" in sys.argv
    configs = resto or ["off", "hpf,norm"]
    ref = normalizar(ref_txt.read_text(encoding="utf-8"))
    res = json.loads(saida.read_text(encoding="utf-8")) if saida.exists() else {}
    so = [v for v in os.environ.get("AVEIA_VARIANTES", "").split(",") if v]
    for wav in sorted(pasta.glob("*.wav")):
        if so and wav.stem not in so:
            continue
        for cfg in configs:
            chave = f"{wav.stem}|{cfg}" + ("|diar" if diar else "")
            if chave in res:
                continue
            falas, seg = rodar(wav, cfg, diar)
            hip = normalizar(" ".join(f.texto for f in falas))
            w, n, m, e = wer(ref, hip)
            res[chave] = {"wer": round(w * 100, 2), "ref": n, "hip": m, "erros": e, "seg": round(seg, 1),
                          "locutores": len({f.locutor for f in falas}), "falas": len(falas)}
            print(chave, res[chave], flush=True)
            saida.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
