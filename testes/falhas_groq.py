r"""Simula falhas da Groq (servidor falso local, testes/groq_falso.py) chamando o pipeline real, sem usar a chave verdadeira.
Uso (raiz do projeto): .venv\Scripts\python testes\falhas_groq.py [cenario ...]
O fallback local carrega o Whisper (lento com a CPU ocupada); o áudio é recortado de testes/audio-referencia/audio.wav."""
import os, subprocess, sys, time, json
from pathlib import Path
import tempfile, wave
RAIZ = Path(__file__).resolve().parent.parent
SC = Path(__file__).resolve().parent
TMP = Path(tempfile.mkdtemp(prefix="aveia-falhas-"))


def recorte(segundos: int) -> str:
    with wave.open(str(RAIZ / "testes" / "audio-referencia" / "audio.wav")) as w:
        n = min(w.getnframes(), segundos * w.getframerate()); dados = w.readframes(n); p = (w.getnchannels(), w.getsampwidth(), w.getframerate())
    destino = TMP / f"c{segundos}.wav"
    with wave.open(str(destino), "wb") as o:
        o.setnchannels(p[0]); o.setsampwidth(p[1]); o.setframerate(p[2]); o.writeframes(dados)
    return str(destino)
sys.path.insert(0, str(RAIZ))
os.environ["GROQ_API_KEY"] = "chave-de-teste"   # chave falsa: nunca usa a real nestes testes
from transcritor import motores, pipeline

CENARIOS = {  # nome: (sequência do servidor falso, env extra, áudio)
    "429-depois-ok":   ("429,429,ok", {}, 25),
    "429-cota-diaria": ("429long", {}, 25),
    "chave-invalida":  ("401", {}, 25),
    "servidor-503":    ("500", {"AVEIA_GROQ_TIMEOUT": "5"}, 25),
    "timeout":         ("hang", {"AVEIA_GROQ_TIMEOUT": "3"}, 25),
    "parcial":         ("ok,429long", {"AVEIA_GROQ_BLOCO": "20"}, 50),
    "413-divide":      ("413,ok,ok", {}, 57),
    "arquivo-grande":  ("ok", {"AVEIA_GROQ_MAX_BYTES": "300000"}, 57),
    "sem-chave":       (None, {"GROQ_API_KEY": ""}, 25),
    "sem-servidor":    (None, {"AVEIA_GROQ_URL": "http://127.0.0.1:9/x", "AVEIA_GROQ_TIMEOUT": "3"}, 25),
}
alvo = sys.argv[1:] or list(CENARIOS)
porta = 8791
for nome in alvo:
    seq, extra, audio = CENARIOS[nome]
    proc = None
    antes = {k: os.environ.get(k) for k in ("AVEIA_GROQ_URL", "AVEIA_GROQ_TIMEOUT", "AVEIA_GROQ_BLOCO", "AVEIA_GROQ_MAX_BYTES", "GROQ_API_KEY")}
    if seq:
        porta += 1
        proc = subprocess.Popen([sys.executable, str(SC / "groq_falso.py"), str(porta), seq], stdout=subprocess.PIPE, text=True)
        os.environ["AVEIA_GROQ_URL"] = f"http://127.0.0.1:{porta}/v1/audio/transcriptions"
        time.sleep(1.5)
    os.environ.update(extra)
    meta, msgs = {}, []
    t0 = time.time()
    try:
        falas = pipeline.transcrever(recorte(audio), diarizacao=False, motor="groq", meta=meta,
                                     progresso=lambda p, m: msgs.append(m) if (not msgs or msgs[-1] != m) else None)
        res = {"erro": None, "falas": len(falas), "texto": " ".join(f.texto for f in falas)[:70]}
    except Exception as e:
        res = {"erro": f"{type(e).__name__}: {e}"}
    res.update(cenario=nome, tempo=round(time.time() - t0, 1), motor=meta.get("motor"), avisos=meta.get("avisos"),
               esperas=[m for m in msgs if "aguardando" in m or "tentando" in m][:3])
    if proc:
        proc.terminate()
        out = proc.communicate(timeout=10)[0]
        res["reqs_ao_servidor"] = len([l for l in out.splitlines() if l.startswith("req")])
    for k, v in antes.items():
        if v is None: os.environ.pop(k, None)
        else: os.environ[k] = v
    print(json.dumps(res, ensure_ascii=False), flush=True)
