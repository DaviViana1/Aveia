"""Benchmark dos motores de transcrição (WER, números, nomes próprios, silêncio, RTF).

Uso (da raiz do projeto):
  .venv\\Scripts\\python testes\\bench_motores.py run  <motor> <variante> <pasta_clipes> [clipe ...]
  .venv\\Scripts\\python testes\\bench_motores.py report <pasta_resultados> [<pasta_clipes>]

Cada clipe é `nome.(wav|webm|...)` + `nome.txt` (transcrição humana). Resultados em cache JSON
(<pasta_resultados>/<motor>__<variante>__<clipe>.json) para reavaliar sem rodar de novo.
Dependências só do benchmark: jiwer e num2words (pip install jiwer num2words).
Chaves (GROQ_API_KEY) são lidas só do ambiente/.env, nunca impressas.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
import unicodedata
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
RESULT_DIR = Path(os.environ.get("AVEIA_BENCH_OUT", RAIZ / "testes" / "resultados"))


def _carregar_env():
    p = RAIZ / ".env"
    if p.exists():
        for l in p.read_text(encoding="utf-8").splitlines():
            if "=" in l and not l.lstrip().startswith("#"):
                k, v = l.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip("\"'"))


# ---------------- normalização / métricas ----------------
def _num(m: re.Match) -> str:
    from num2words import num2words
    s = m.group(0)
    try:
        if "," in s:
            return " " + num2words(float(s.replace(".", "").replace(",", ".")), lang="pt_BR") + " "
        return " " + num2words(int(s.replace(".", "")), lang="pt_BR") + " "
    except Exception:
        return " " + s + " "


def normalizar(t: str) -> str:
    t = unicodedata.normalize("NFC", t).lower()
    t = t.replace("%", " por cento ").replace("r$", " reais ").replace("º", " ").replace("ª", " ")
    t = re.sub(r"\d+(?:\.\d{3})*(?:,\d+)?", _num, t)
    t = re.sub(r"[-–—/]", " ", t)
    t = re.sub(r"[^\w\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def tokens_estritos(t: str) -> list[str]:
    """Palavras com caixa original + sinais de pontuação como tokens (mede legibilidade)."""
    t = unicodedata.normalize("NFC", t)
    t = re.sub(r"\s+", " ", t)
    return re.findall(r"[\w'’-]+|[.,;:?!…]", t)


def wer(ref: list[str] | str, hip: list[str] | str) -> float:
    import jiwer
    r = ref if isinstance(ref, str) else " ".join(ref)
    h = hip if isinstance(hip, str) else " ".join(hip)
    if not r.strip():
        return 0.0
    return jiwer.wer(r, h if h.strip() else "x")


def wer_tokens(ref: list[str], hip: list[str]) -> float:
    import jiwer
    # jiwer trabalha com strings: mapeia cada token a um símbolo único
    voc: dict[str, str] = {}
    cod = lambda ts: " ".join(voc.setdefault(t, f"w{len(voc)}") for t in ts)  # noqa: E731
    return jiwer.wer(cod(ref), cod(hip) if hip else "x")


def numeros_e_nomes(ref_raw: str, hip_raw: str) -> dict:
    hip_n = set(normalizar(hip_raw).split())
    hip_txt = normalizar(hip_raw)
    nums = re.findall(r"\d+(?:\.\d{3})*(?:,\d+)?", ref_raw)
    n_ok = 0
    for n in nums:
        alvo = normalizar(n)
        if alvo and alvo in hip_txt:
            n_ok += 1
    # nomes próprios: palavra capitalizada que não abre frase
    toks = re.findall(r"[\w'’-]+|[.!?]", unicodedata.normalize("NFC", ref_raw))
    nomes = []
    for i, t in enumerate(toks):
        if t[:1].isupper() and not t.isupper() and i > 0 and toks[i - 1] not in ".!?":
            nomes.append(normalizar(t))
    nomes = [n for n in nomes if n]
    nm_ok = sum(1 for n in nomes if n in hip_n)
    return {"num_total": len(nums), "num_ok": n_ok, "nome_total": len(nomes), "nome_ok": nm_ok}


# ---------------- motores ----------------
_modelos: dict = {}


def _whisper(nome: str):
    from faster_whisper import WhisperModel
    if nome not in _modelos:
        _modelos[nome] = WhisperModel(nome, device="auto", compute_type="int8", cpu_threads=os.cpu_count() or 0)
    return _modelos[nome]


BASE_LOCAL = dict(
    modelo="large-v3-turbo", language="pt", word_timestamps=True, beam_size=5, vad_filter=True,
    vad_parameters=dict(min_silence_duration_ms=500, speech_pad_ms=300),
    condition_on_previous_text=False, hallucination_silence_threshold=2.0,
    no_speech_threshold=0.5, compression_ratio_threshold=2.2,
    initial_prompt="Reunião de trabalho em português do Brasil, com números, orçamento e faturamento.",
)
# nome -> alterações sobre BASE_LOCAL (None remove a chave)
VARIANTES_LOCAL = {
    "base": {},
    "beam1": {"beam_size": 1},
    "beam10": {"beam_size": 10},
    "sem-prompt": {"initial_prompt": None},
    "sem-vad": {"vad_filter": False, "vad_parameters": None},
    "vad-sensivel": {"vad_parameters": dict(min_silence_duration_ms=300, speech_pad_ms=400, threshold=0.4)},
    "temp0": {"temperature": 0.0},
    "cond-true": {"condition_on_previous_text": True},
    "auto-idioma": {"language": None},
    "v3": {"modelo": "large-v3"},
    "medium": {"modelo": "medium"},
    "base-rep2": {}, "base-rep3": {},   # repetições da base (mede a variação de uma execução para outra)
    "sem-halluc": {"hallucination_silence_threshold": None},
    "temp0-sem-halluc": {"temperature": 0.0, "hallucination_silence_threshold": None},
    "repeticao": {"repetition_penalty": 1.1, "no_repeat_ngram_size": 4},
}


def rodar_local(audio, variante: str, extras: dict | None = None):
    cfg = dict(BASE_LOCAL)
    cfg.update(VARIANTES_LOCAL[variante])
    if extras:
        cfg.update(extras)
    modelo = cfg.pop("modelo")
    cfg = {k: v for k, v in cfg.items() if v is not None}
    seg, _ = _whisper(modelo).transcribe(audio, **cfg)
    palavras = [w for s in seg for w in (s.words or [])]
    return [(w.start, w.end, w.word) for w in palavras]


def rodar_groq(audio, variante: str):
    """Pelo mesmo caminho da integração (motores.palavras) para medir exatamente o que o app faz."""
    from transcritor import motores
    # variante: "base" | "v3" (whisper-large-v3) | "bloco120" (blocos de 120 s), combináveis com "+"
    for k in ("AVEIA_GROQ_BLOCO", "AVEIA_GROQ_MODELO", "AVEIA_GROQ_PROMPT"): os.environ.pop(k, None)
    for v in variante.split("+"):
        if v == "v3":
            os.environ["AVEIA_GROQ_MODELO"] = "whisper-large-v3"
        elif v.startswith("bloco"):
            os.environ["AVEIA_GROQ_BLOCO"] = v[5:]
        elif v == "prompt":
            os.environ["AVEIA_GROQ_PROMPT"] = BASE_LOCAL["initial_prompt"]
    meta: dict = {}
    pal = list(motores.palavras(audio, len(audio) / 16000, "pt", lambda p, m: None, motor="groq", meta=meta))
    print("   meta:", {k: v for k, v in meta.items() if k != "avisos"}, meta.get("avisos"))
    if meta.get("motor") != "groq":
        raise RuntimeError(f"Groq não foi usada: {meta}")
    return [(p.start, p.end, p.word) for p in pal]


_sherpa = {}


def rodar_parakeet(audio, variante: str):
    import sherpa_onnx as so
    import numpy as np
    pasta = Path(os.environ.get("AVEIA_PARAKEET_DIR", RAIZ / "models" / "sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8"))
    if "r" not in _sherpa:
        _sherpa["r"] = so.OfflineRecognizer.from_transducer(
            encoder=str(pasta / "encoder.int8.onnx"), decoder=str(pasta / "decoder.int8.onnx"),
            joiner=str(pasta / "joiner.int8.onnx"), tokens=str(pasta / "tokens.txt"),
            num_threads=os.cpu_count() or 4, sample_rate=16000, feature_dim=128, decoding_method="greedy_search",
            model_type="nemo_transducer")
    r = _sherpa["r"]
    saida = []
    # blocos de ~30 s (o modelo é offline; áudio longo precisa ser fatiado)
    passo = 30 * 16000
    for i in range(0, len(audio), passo):
        s = r.create_stream()
        s.accept_waveform(16000, audio[i:i + passo].astype(np.float32))
        r.decode_stream(s)
        res = s.result
        ts = list(res.timestamps) if res.timestamps else []
        toks = list(res.tokens)
        off = i / 16000
        # tokens SentencePiece: "▁" marca início de palavra
        palavra, ini = "", None
        for j, t in enumerate(toks):
            tt = ts[j] + off if j < len(ts) else off
            if t.startswith("▁") and palavra:
                saida.append((ini, tt, " " + palavra)); palavra, ini = "", None
            if ini is None:
                ini = tt
            palavra += t.replace("▁", "")
        if palavra:
            saida.append((ini, ini + 0.3, " " + palavra))
    return saida


def carregar_audio(caminho: Path):
    from faster_whisper.audio import decode_audio
    return decode_audio(str(caminho), sampling_rate=16000)


def achar_clipes(pasta: Path, nomes: list[str]):
    out = []
    for txt in sorted(pasta.glob("*.txt")):
        if nomes and txt.stem not in nomes:
            continue
        for ext in (".webm", ".wav", ".mp3", ".flac", ".ogg", ".m4a"):
            a = txt.with_suffix(ext)
            if a.exists():
                out.append((txt.stem, a, txt))
                break
    return out


def cmd_run(motor: str, variante: str, pasta: Path, nomes: list[str]):
    _carregar_env()
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    for nome, a, txt in achar_clipes(pasta, nomes):
        saida = RESULT_DIR / f"{motor}__{variante}__{nome}.json"
        if saida.exists():
            print("já existe", saida.name); continue
        audio = carregar_audio(a)
        dur = len(audio) / 16000
        partes_v = variante.split("+")
        if "pre" in partes_v:   # mesmo pré-processamento do app (transcritor/audio.py: passa-altas + volume)
            from transcritor.audio import preparar
            audio = preparar(audio)
        variante_motor = "+".join(v for v in partes_v if v != "pre") or "base"
        t0 = time.time()
        try:
            pal = {"local": lambda: rodar_local(audio, variante_motor), "groq": lambda: rodar_groq(audio, variante_motor),
                   "parakeet": lambda: rodar_parakeet(audio, variante_motor)}[motor]()
        except Exception as e:
            print(nome, "ERRO", type(e).__name__, str(e)[:200]); continue
        dt = time.time() - t0
        hip = "".join(w for _, _, w in pal).strip()
        saida.write_text(json.dumps({"motor": motor, "variante": variante, "clipe": nome, "duracao": dur,
                                     "tempo": dt, "palavras": pal, "texto": hip}, ensure_ascii=False), encoding="utf-8")
        ref = txt.read_text(encoding="utf-8")
        print(f"{motor}/{variante} {nome}: dur {dur:.0f}s tempo {dt:.1f}s RTF {dt / dur:.2f} "
              f"WER_norm {wer(normalizar(ref), normalizar(hip)):.3f}")


def cmd_report(pasta_res: Path, pasta_clipes: Path | None):
    import statistics as st
    pasta_clipes = pasta_clipes or RAIZ / "testes" / "audio-referencia"
    refs = {}
    for p in pasta_res.glob("*.json"):
        d = json.loads(p.read_text(encoding="utf-8"))
        if d["clipe"] not in refs:
            t = pasta_clipes / f"{d['clipe']}.txt"
            if t.exists():
                refs[d["clipe"]] = t.read_text(encoding="utf-8")
    linhas: dict[tuple, list] = {}
    for p in sorted(pasta_res.glob("*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        if d["clipe"] not in refs:
            continue
        ref, hip = refs[d["clipe"]], d["texto"]
        com_pont = bool(re.search(r"[.,?!]", ref)) and any(c.isupper() for c in ref[1:])
        r = {"clipe": d["clipe"], "wer": wer(normalizar(ref), normalizar(hip)), "rtf": d["tempo"] / d["duracao"],
             "dur": d["duracao"], "pal": len(normalizar(ref).split()),
             "estrito": wer_tokens(tokens_estritos(ref), tokens_estritos(hip)) if com_pont else None}
        r.update(numeros_e_nomes(ref, hip))
        motor_f, variante_f = p.stem.split("__")[:2]   # o nome do arquivo manda (permite renomear resultados)
        linhas.setdefault((motor_f, variante_f), []).append(r)
    print(f"{'motor/variante':26s} {'clipes':>6s} {'WER%':>6s} {'WERtot%':>8s} {'estrito%':>9s} {'núm':>7s} {'nomes':>7s} {'RTF':>5s}  por clipe (WER%)")
    for (m, v), rs in sorted(linhas.items()):
        tot_err = sum(r["wer"] * r["pal"] for r in rs); tot_pal = sum(r["pal"] for r in rs)
        est = [r["estrito"] for r in rs if r["estrito"] is not None]
        nt = sum(r["num_total"] for r in rs); no = sum(r["num_ok"] for r in rs)
        mt = sum(r["nome_total"] for r in rs); mo = sum(r["nome_ok"] for r in rs)
        por = " ".join(f"{r['clipe'][:8]}={r['wer'] * 100:.1f}" for r in rs)
        print(f"{m + '/' + v:26s} {len(rs):6d} {st.mean(r['wer'] for r in rs) * 100:6.2f} {tot_err / tot_pal * 100:8.2f} "
              f"{(st.mean(est) * 100 if est else float('nan')):9.2f} {no:3d}/{nt:<3d} {mo:3d}/{mt:<3d} "
              f"{st.mean(r['rtf'] for r in rs):5.2f}  {por}")


if __name__ == "__main__":
    a = sys.argv[1:]
    if a and a[0] == "run":
        cmd_run(a[1], a[2], Path(a[3]), a[4:])
    elif a and a[0] == "report":
        cmd_report(Path(a[1]), Path(a[2]) if len(a) > 2 else None)
    else:
        print(__doc__)
