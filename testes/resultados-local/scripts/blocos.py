"""Etapa B: Sessao ao vivo (blocos <=30 s), alimentada sem esperar tempo real, Whisper real, config atual, glossario vazio."""
import json, os, sys, time, tempfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from unittest import mock
RAIZ = Path(__file__).resolve().parents[3]  # raiz do projeto
sys.path.insert(0, str(RAIZ)); sys.path.insert(0, str(RAIZ / "testes"))
os.environ["HF_HUB_OFFLINE"] = "1"; os.environ["AI_GATEWAY_API_KEY"] = ""; os.environ["GROQ_API_KEY"] = ""
import lacunas_revalidar as L
from testes.wer import wer, normalizar
from transcritor import pipeline, aovivo
from transcritor import diarizacao as diarizacao_mod
from faster_whisper.audio import decode_audio
CL = L.CLIPES
MAPA = {
    "real-ganho-30db": (RAIZ / "testes/microfone/ref-real/variantes/ganho-30db.wav", "ref58", "q1b__atual__real-ganho-30db"),
    "cv": L.ARQUIVO["cv"], "cv-webm": L.ARQUIVO["cv-webm"], "conv-2loc": L.ARQUIVO["conv-2loc"], "conv-4loc": L.ARQUIVO["conv-4loc"],
}
LOGB = L.OUT / "log_blocos.txt"
def reg(m):
    l = f"{datetime.now():%H:%M:%S} {m}"; print(l, flush=True); open(LOGB, "a", encoding="utf-8").write(l + "\n")
L.registrar = reg
def ref_de(c):
    r = MAPA[c][1]
    if r is None: return None
    if r == "ref58": return L.referencia("ref58-webm")
    return Path(r).read_text(encoding="utf-8").strip()
def base_de(c): return json.loads((L.BRUTO / f"{MAPA[c][2]}.json").read_text(encoding="utf-8"))
def rodar(c):
    arq = L.BRUTO / f"blocos_atual__{c}.json"
    if arq.exists(): print("ja feito", arq.name); return
    L.modelo_medido()
    path = MAPA[c][0]; dados = Path(path).read_bytes(); ext = Path(path).suffix.lstrip(".")
    audio = decode_audio(str(path), sampling_rate=16000); dur = len(audio) / 16000
    fala = pipeline._trechos_fala(audio)
    base = base_de(c); ref = ref_de(c)
    fila = ThreadPoolExecutor(max_workers=1)
    tmp = Path(os.environ["TEMP_BLOCOS"]); tmp.mkdir(exist_ok=True)
    n0 = L._chamadas[0]
    reg(f"INICIO blocos {c} ({dur:.1f} s, .{ext})")
    t0 = time.time()
    with mock.patch.object(pipeline, "diarizar", lambda a, num_locutores=None: []), \
            mock.patch.object(diarizacao_mod, "analise_incremental", lambda *a, **k: None):
        s = aovivo.Sessao("blocos-" + c, tmp / f"s.{ext}", "pt", None, lambda fn, *a: fila.submit(fn, *a))
        try:
            for seq, i in enumerate(range(0, len(dados), 16384)):
                s.receber(seq, dados[i:i + 16384])
            s.marcar_fim(dur)
            fila.submit(lambda: None).result()
            meta = {}
            falas = s.concluir(meta=meta)
            blocos = list(s.blocos); palavras = list(s._palavras); degradada = s.degradada
        finally:
            s.liberar()
            try: (tmp / f"s.{ext}").unlink(missing_ok=True)
            except OSError: pass
    dt = time.time() - t0
    texto = " ".join(f.texto for f in falas)
    tempos = [(p.start, p.end) for p in palavras]
    resta = [(round(a, 1), round(b, 1), round(pipeline._fala_em(fala, a, b), 1)) for a, b in pipeline._buracos_com_fala(tempos, 0.0, dur, fala)]
    lac = meta.get("lacunas") or []
    r = dict(clipe=c, via="aovivo-blocos", arquivo=str(path), dur=round(dur, 1), blocos=[(round(a, 1), round(b, 1)) for a, b in blocos],
             palavras=len(normalizar(texto).split()), wer=round(wer(ref, texto) * 100, 2) if ref else None,
             wer_sem_acentos=round(wer(ref, texto, True) * 100, 2) if ref else None,
             base_palavras=base["palavras"], base_wer=base.get("wer"),
             lacunas_meta=lac, avisos=meta.get("avisos"), lacunas_restantes_com_fala=resta, degradada=degradada,
             tempo=round(dt, 1), chamadas_whisper=L._chamadas[0] - n0, texto=texto)
    arq.write_text(json.dumps(r, ensure_ascii=False), encoding="utf-8")
    reg(f"FIM blocos {c}: {r['palavras']} palavras (base {r['base_palavras']}), WER {r['wer']} (base {r['base_wer']}), blocos {r['blocos']}, "
        f"lacunas_meta {len(lac)}, restantes {resta}, {dt:.1f} s, {r['chamadas_whisper']} chamadas")
if __name__ == "__main__":
    os.environ["TEMP_BLOCOS"] = str(L.OUT / "_blocos_tmp")
    for c in sys.argv[1:]: rodar(c)
