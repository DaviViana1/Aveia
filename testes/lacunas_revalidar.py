"""Revalidação da defesa contra truncamento (retranscrição de lacunas em `pipeline._palavras`). Roda o Whisper real.

Uso (da raiz do projeto; defina HF_HUB_OFFLINE=1; os resultados ficam em testes/resultados-local/bruto/lac__*.json e
NÃO são refeitos se o arquivo existir: apague para repetir):
  .venv\\Scripts\\python testes\\lacunas_revalidar.py lacunas <clipe>                    # só lê o cache e o VAD (sem Whisper)
  .venv\\Scripts\\python testes\\lacunas_revalidar.py pedacos <clipe> <n_lacuna> <max_s> [rodar]   # sem 'rodar' só lista os pedaços
  .venv\\Scripts\\python testes\\lacunas_revalidar.py arquivo <clipe>                    # pipeline.transcrever com o código novo
  .venv\\Scripts\\python testes\\lacunas_revalidar.py aovivo                             # ref58-webm pela sessão ao vivo, em blocos
Clipes: ref58-webm, cv, cv-webm, conv-4loc, conv-2loc, mls-A, mls-B, fleurs, ...  (nomes de testes/local_decodificacao.py).
Linha do tempo de cada chamada ao Whisper em testes/resultados-local/log_lacunas.txt.
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ["AVEIA_PREPROC"] = ""          # como no app por padrão: sem pré-processamento (os caches foram feitos assim)
os.environ.pop("AVEIA_LACUNAS", None)

import numpy as np  # noqa: E402
from faster_whisper.audio import decode_audio  # noqa: E402

from testes.wer import normalizar, wer  # noqa: E402
from transcritor import pipeline  # noqa: E402

SR = 16000
OUT = RAIZ / "testes" / "resultados-local"
BRUTO = OUT / "bruto"
LOG = OUT / "log_lacunas.txt"
AR = RAIZ / "testes" / "audio-referencia"
CLIPES = OUT / "clipes"
ARQUIVO = {   # clipe -> (arquivo de áudio, texto de referência ou None, cache da linha de base "atual")
    "ref58-webm": (AR / "audio.webm", "ref58", "q2__atual__ref58-webm"),
    "cv": (CLIPES / "cv-microfones.wav", CLIPES / "cv-microfones.txt", "q2__atual__cv"),
    "cv-webm": (CLIPES / "cv-microfones.webm", CLIPES / "cv-microfones.txt", "q2__atual__cv-webm"),
    "conv-4loc": (RAIZ / "testes/diarizacao/dados/mls-pt-4loc.wav", None, "q2__atual__conv-4loc"),
    "conv-2loc": (RAIZ / "testes/diarizacao/dados/mls-pt-2loc.wav", None, "q2__atual__conv-2loc"),
    "mls-A": (CLIPES / "mls-A.wav", None, "q2__atual__mls-A"),
    "mls-B": (CLIPES / "mls-B.wav", None, "q2__atual__mls-B"),
    "fleurs": (CLIPES / "fleurs-leitura.wav", CLIPES / "fleurs-leitura.txt", "q2__atual__fleurs"),
}
_chamadas = [0]


def registrar(msg: str) -> None:
    linha = f"{datetime.now():%H:%M:%S} {msg}"
    print(linha, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(linha + "\n")


def referencia(clipe: str) -> str | None:
    r = ARQUIVO[clipe][1]
    if r is None:
        return None
    if r == "ref58":
        return " ".join(f["texto"] for f in json.load(open(AR / "referencia.json", encoding="utf-8")))
    return Path(r).read_text(encoding="utf-8").strip()


def baseline(clipe: str) -> dict:
    return json.loads((BRUTO / f"{ARQUIVO[clipe][2]}.json").read_text(encoding="utf-8"))


def modelo_medido():
    """O modelo do app, com cada `transcribe` registrado (conta as chamadas = execuções do Whisper)."""
    m = pipeline._carregar_modelo(pipeline.MODELO)
    if not getattr(m, "_medido", False):
        original = m.transcribe

        def medido(audio, *a, **k):
            _chamadas[0] += 1
            n, t0 = _chamadas[0], time.time()
            seg, info = original(audio, *a, **k)
            seg = list(seg)       # força a decodificação aqui para medir o tempo
            registrar(f"  transcribe #{n}: {len(audio) / SR:.1f} s de áudio, {time.time() - t0:.1f} s")
            return iter(seg), info

        m.transcribe = medido
        m._medido = True
    return m


def contagem(texto: str) -> int:
    return len(normalizar(texto).split())


def cmd_lacunas(clipe: str):
    audio = decode_audio(str(ARQUIVO[clipe][0]), sampling_rate=SR)
    base = baseline(clipe)
    tempos = [(a, b) for a, b, _ in base["palavras_tempo"]]
    fala = pipeline._trechos_fala(audio)
    for i, (a, b) in enumerate(pipeline._buracos_com_fala(tempos, 0.0, len(audio) / SR, fala)):
        print(f"lacuna {i}: {a:.1f}-{b:.1f} s ({b - a:.1f} s), fala do VAD {pipeline._fala_em(fala, a, b):.1f} s")


def cmd_pedacos(clipe: str, n: int, maximo: float, rodar: bool):
    audio = decode_audio(str(ARQUIVO[clipe][0]), sampling_rate=SR)
    base = baseline(clipe)
    fala = pipeline._trechos_fala(audio)
    a, b = pipeline._buracos_com_fala([(x, y) for x, y, _ in base["palavras_tempo"]], 0.0, len(audio) / SR, fala)[n]
    cortes = pipeline._cortes_em_pedacos(audio, a, b, fala, maximo)
    pedacos = [(i, f) for i, f in zip(cortes, cortes[1:]) if pipeline._fala_em(fala, i, f) >= pipeline.PEDACO_MIN_FALA_S]
    print(f"{clipe} lacuna {n}: {a:.1f}-{b:.1f} s, fala {pipeline._fala_em(fala, a, b):.1f} s; "
          f"{len(pedacos)} pedaços (máx {maximo} s): " + ", ".join(f"{i:.1f}-{f:.1f}" for i, f in pedacos))
    if not rodar:
        return
    arq = BRUTO / f"lac__pedacos{maximo:g}__{clipe}__{n}.json"
    if arq.exists():
        print("já feito:", arq.name)
        return
    m = modelo_medido()
    opcoes = pipeline._opcoes_decodificacao("pt")

    def decodificar(fatia):
        seg, _ = m.transcribe(fatia, **opcoes)
        return [w for s in seg for w in (s.words or [])]

    inicio = time.time()
    registrar(f"INÍCIO pedaços {clipe} lacuna {n} (máx {maximo} s)")
    novas = pipeline._recuperar_em_pedacos(decodificar, audio, a, b, fala, maximo)
    dt = time.time() - inicio
    fala_s = pipeline._fala_em(fala, a, b)
    texto = "".join(p.word for p in novas).strip()
    r = dict(clipe=clipe, lacuna=[round(a, 2), round(b, 2)], fala_s=round(fala_s, 1), maximo=maximo, pedacos=len(pedacos),
             palavras=sum(len(pipeline._norm(p.word)) for p in novas), por_s=round(
                 sum(len(pipeline._norm(p.word)) for p in novas) / fala_s, 2),
             aceitavel=pipeline._aceitavel(novas, fala_s), texto=texto, tempo=round(dt, 1),
             palavras_tempo=[(round(p.start, 2), round(p.end, 2), p.word) for p in novas])
    arq.write_text(json.dumps(r, ensure_ascii=False), encoding="utf-8")
    registrar(f"FIM pedaços {clipe} lacuna {n}: {r['palavras']} palavras, {r['por_s']} por s, aceitável={r['aceitavel']}, {dt:.1f} s")
    print(texto[:600])


def cmd_arquivo(clipe: str):
    arq = BRUTO / f"lac__arquivo__{clipe}.json"
    if arq.exists():
        print("já feito:", arq.name)
        print({k: v for k, v in json.loads(arq.read_text(encoding="utf-8")).items() if k not in ("texto", "falas")})
        return
    modelo_medido()
    base, ref = baseline(clipe), referencia(clipe)
    meta: dict = {}
    n0 = _chamadas[0]
    registrar(f"INÍCIO arquivo {clipe}")
    t0 = time.time()
    falas = pipeline.transcrever(str(ARQUIVO[clipe][0]), diarizacao=False, motor="local", meta=meta)
    dt = time.time() - t0
    texto = " ".join(f.texto for f in falas)
    r = dict(clipe=clipe, palavras_antes=base["palavras"], palavras_depois=contagem(texto), wer_antes=base.get("wer"),
             wer_depois=round(wer(ref, texto) * 100, 2) if ref else None, lacunas=meta.get("lacunas"), avisos=meta.get("avisos"),
             tempo=round(dt, 1), tempo_antes=base.get("tempo"), chamadas_whisper=_chamadas[0] - n0,
             inicios_em_ordem=all(a.inicio <= b.inicio for a, b in zip(falas, falas[1:])),
             texto=texto, falas=[(f.inicio, f.fim, f.texto) for f in falas])
    arq.write_text(json.dumps(r, ensure_ascii=False), encoding="utf-8")
    registrar(f"FIM arquivo {clipe}: {r['palavras_antes']} -> {r['palavras_depois']} palavras, WER {r['wer_antes']} -> {r['wer_depois']}, "
              f"{dt:.1f} s, {r['chamadas_whisper']} chamadas ao Whisper")
    print({k: v for k, v in r.items() if k not in ("texto", "falas")})


def cmd_aovivo():
    """ref58-webm pela sessão ao vivo (aovivo.Sessao) em blocos de 15 a 30 s, com Whisper real e diarização desligada."""
    from unittest import mock

    from transcritor import aovivo
    from transcritor import diarizacao as diarizacao_mod
    clipe = "ref58-webm"
    arq = BRUTO / f"lac__aovivo__{clipe}.json"
    if arq.exists():
        print("já feito:", arq.name)
        print({k: v for k, v in json.loads(arq.read_text(encoding="utf-8")).items() if k not in ("texto", "falas")})
        return
    modelo_medido()
    base, ref = baseline(clipe), referencia(clipe)
    dados = (AR / "audio.webm").read_bytes()
    fila = ThreadPoolExecutor(max_workers=1)
    tmp = OUT / "_aovivo_tmp"                     # fora de gravacoes/: o arquivo da sessão
    tmp.mkdir(exist_ok=True)
    n0 = _chamadas[0]
    registrar("INÍCIO aovivo ref58-webm")
    t0 = time.time()
    with mock.patch.object(pipeline, "diarizar", lambda a, num_locutores=None: []), \
            mock.patch.object(diarizacao_mod, "analise_incremental", lambda *a, **k: None):
        s = aovivo.Sessao("lac-aovivo", tmp / "sessao.webm", "pt", None, lambda fn, *a: fila.submit(fn, *a))
        try:
            for seq, i in enumerate(range(0, len(dados), 16384)):
                s.receber(seq, dados[i:i + 16384])
            s.marcar_fim(57.7)
            meta: dict = {}
            fila.submit(lambda: None).result()    # a fila é de 1 worker: espera os blocos já agendados
            falas = s.concluir(meta=meta)
            blocos = list(s.blocos)
        finally:
            s.liberar()
            try:
                (tmp / "sessao.webm").unlink(missing_ok=True)
                tmp.rmdir()
            except OSError:
                pass
    dt = time.time() - t0
    texto = " ".join(f.texto for f in falas)
    r = dict(clipe=clipe, via="aovivo", blocos=blocos, palavras_antes=base["palavras"], palavras_depois=contagem(texto),
             wer_antes=base.get("wer"), wer_depois=round(wer(ref, texto) * 100, 2), lacunas=meta.get("lacunas"),
             avisos=meta.get("avisos"), tempo=round(dt, 1), chamadas_whisper=_chamadas[0] - n0, degradada=s.degradada,
             inicios_em_ordem=all(a.inicio <= b.inicio for a, b in zip(falas, falas[1:])), texto=texto)
    arq.write_text(json.dumps(r, ensure_ascii=False), encoding="utf-8")
    registrar(f"FIM aovivo: {r['palavras_antes']} -> {r['palavras_depois']} palavras, WER {r['wer_antes']} -> {r['wer_depois']}, "
              f"blocos {blocos}, {dt:.1f} s, {r['chamadas_whisper']} chamadas")
    print({k: v for k, v in r.items() if k != "texto"})


if __name__ == "__main__":
    a = sys.argv[1:]
    if not a:
        print(__doc__)
    elif a[0] == "lacunas":
        cmd_lacunas(a[1])
    elif a[0] == "pedacos":
        cmd_pedacos(a[1], int(a[2]), float(a[3]), len(a) > 4 and a[4] == "rodar")
    elif a[0] == "arquivo":
        cmd_arquivo(a[1])
    elif a[0] == "aovivo":
        cmd_aovivo()
