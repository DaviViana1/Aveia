"""Medições do motor local (faster-whisper large-v3-turbo, int8, CPU): truncamento, config, blocos, hotwords, velocidade.

Uso (da raiz do projeto, uma linha por tabela do RESULTADOS.md):
  .venv\\Scripts\\python testes\\local_decodificacao.py clipes      # monta clipes MLS (sem referência) em testes/resultados-local/clipes/
  .venv\\Scripts\\python testes\\local_decodificacao.py q1          # Q1 truncamento (variantes de parâmetros / entrada)
  .venv\\Scripts\\python testes\\local_decodificacao.py q1b         # Q1 complemento: combinações e demais clipes
  .venv\\Scripts\\python testes\\local_decodificacao.py pipeline    # Q1: transcritor.pipeline.transcrever(webm, diarizacao=False)
  .venv\\Scripts\\python testes\\local_decodificacao.py q2          # Q2: configs candidatas em todos os clipes
  .venv\\Scripts\\python testes\\local_decodificacao.py q3          # Q3: modo bloco (VAD, <=30 s) x arquivo inteiro
  .venv\\Scripts\\python testes\\local_decodificacao.py q4          # Q4: hotwords
  .venv\\Scripts\\python testes\\local_decodificacao.py q5          # Q5: velocidade (rodar por último; usa a CPU toda)
  .venv\\Scripts\\python testes\\local_decodificacao.py tabelas     # imprime as tabelas a partir do cache bruto

Resultados brutos em testes/resultados-local/bruto/<questao>__<variante>__<clipe>.json (cache: apague para refazer).
Sem rede, sem IA externa. WER = testes/wer.py (minúsculas, hífen vira espaço, sem pontuação; números ficam como
o Whisper escreveu, então "2,3 milhões" x "dois milhões e trezentos mil" conta como erro igualmente em todas as variantes).
"""
from __future__ import annotations

import json
import os
import sys
import time
import wave
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
from testes.wer import distancia, normalizar, wer  # noqa: E402

SR = 16000
OUT = RAIZ / "testes" / "resultados-local"
BRUTO = OUT / "bruto"
CLIPES = OUT / "clipes"
AR = RAIZ / "testes" / "audio-referencia"
TTS = RAIZ / "testes" / "microfone" / "ref-tts"
MLS = RAIZ / "testes" / "diarizacao" / "cache" / "mls"
THREADS_PADRAO = int(os.environ.get("AVEIA_THREADS", os.cpu_count() or 12))

DICA_PT = "Reunião de trabalho em português do Brasil, com números, orçamento e faturamento."
# Config atual de transcritor/pipeline.py::_palavras (cópia fiel)
ATUAL = dict(
    language="pt", word_timestamps=True, beam_size=5,
    vad_filter=True, vad_parameters=dict(min_silence_duration_ms=500, speech_pad_ms=300),
    condition_on_previous_text=False, hallucination_silence_threshold=2.0,
    no_speech_threshold=0.5, compression_ratio_threshold=2.2, initial_prompt=DICA_PT,
)


def cfg(**alt) -> dict:
    c = dict(ATUAL)
    c.update(alt)
    return {k: v for k, v in c.items() if v is not None}


# ------------------------------------------------------------------ clipes
def ler_wav(p: Path) -> np.ndarray:
    with wave.open(str(p)) as w:
        assert w.getframerate() == SR and w.getnchannels() == 1
        return np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768


def gravar_wav(p: Path, a: np.ndarray):
    p.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(p), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes((np.clip(a, -1, 1) * 32767).astype(np.int16).tobytes())


def montar_mls():
    """Clipes de ~2-2,5 min: enunciados MLS consecutivos (índice crescente) com 0,3-0,6 s de silêncio. NÃO há transcrição
    no cache (só os .flac): os clipes ficam sem referência humana."""
    from faster_whisper.audio import decode_audio
    rng = np.random.default_rng(7)
    man = {}
    for nome, pasta in (("mls-A", "3050_2941_000"), ("mls-B", "4405_3604_000"), ("mls-C", "5739_4739_000")):
        fs = sorted((MLS / pasta).glob("*.flac"))
        partes, itens, t = [], [], 0.0
        for f in fs:
            a = decode_audio(str(f), sampling_rate=SR)
            if t + len(a) / SR > 150:
                break
            gap = rng.uniform(0.3, 0.6)
            partes += [a, np.zeros(int(gap * SR), np.float32)]
            itens.append({"arquivo": f.name, "inicio": round(t, 2), "dur": round(len(a) / SR, 2)})
            t += len(a) / SR + gap
        gravar_wav(CLIPES / f"{nome}.wav", np.concatenate(partes))
        man[nome] = {"pasta": pasta, "dur": round(t, 1), "enunciados": itens}
        print(nome, pasta, f"{t:.1f}s", len(itens), "enunciados")
    (CLIPES / "manifesto.json").write_text(json.dumps(man, ensure_ascii=False, indent=1), encoding="utf-8")


def carregar_clipes() -> dict[str, dict]:
    """nome -> {audio, ref (str|None)}. ref58 (wav), ref58-webm (decodifica o webm como o app), tts148, mls-*."""
    from faster_whisper.audio import decode_audio
    ref58 = " ".join(f["texto"] for f in json.load(open(AR / "referencia.json", encoding="utf-8")))
    c = {
        "ref58": dict(audio=decode_audio(str(AR / "audio.wav"), sampling_rate=SR), ref=ref58),
        "ref58-webm": dict(audio=decode_audio(str(AR / "audio.webm"), sampling_rate=SR), ref=ref58),
        "tts148": dict(audio=decode_audio(str(TTS / "referencia.wav"), sampling_rate=SR),
                       ref=(TTS / "referencia.txt").read_text(encoding="utf-8")),
    }
    for n in ("mls-A", "mls-B", "mls-C"):
        p = CLIPES / f"{n}.wav"
        if p.exists():
            c[n] = dict(audio=ler_wav(p), ref=None)       # sem transcrição humana: só contagem de palavras/divergência
    for n in ("2loc", "3loc", "4loc"):                     # conversas sintéticas com troca de locutor (só leitura, sem referência de texto)
        p = RAIZ / "testes" / "diarizacao" / "dados" / f"mls-pt-{n}.wav"
        if p.exists():
            c[f"conv-{n}"] = dict(audio=decode_audio(str(p), sampling_rate=SR), ref=None)
    for n, (a, t) in HUMANOS_ARQ.items():                  # corpus com referência humana (clipes/<áudio> + <txt>)
        if (CLIPES / a).exists() and (CLIPES / t).exists():
            c[n] = dict(audio=decode_audio(str(CLIPES / a), sampling_rate=SR), ref=(CLIPES / t).read_text(encoding="utf-8").strip())
    return c


def fw(r: dict) -> str:
    """WER formatado; 'sem ref' quando o clipe não tem transcrição humana (mls-A/B/C)."""
    return f"{r['wer']:.2f}" if r.get("wer") is not None else "sem ref"


# nome do clipe -> (áudio, referência humana) em testes/resultados-local/clipes/
HUMANOS_ARQ = {
    "fleurs": ("fleurs-leitura.wav", "fleurs-leitura.txt"),
    "mls-livro": ("mls-audiolivro.wav", "mls-audiolivro.txt"),
    "mls-ruido": ("mls-ruido.wav", "mls-ruido.txt"),
    "mls-ruido-webm": ("mls-ruido.webm", "mls-ruido.txt"),
    "cv": ("cv-microfones.wav", "cv-microfones.txt"),
    "cv-webm": ("cv-microfones.webm", "cv-microfones.txt"),
    "mls-longo-webm": ("mls-longo.webm", "mls-longo.txt"),
}


# ------------------------------------------------------------------ motor
_modelos: dict = {}


def modelo(threads: int = THREADS_PADRAO):
    from faster_whisper import WhisperModel
    if threads not in _modelos:
        t0 = time.time()
        _modelos[threads] = WhisperModel("large-v3-turbo", device="auto", compute_type="int8", cpu_threads=threads)
        _modelos[threads].t_carga = time.time() - t0
    return _modelos[threads]


def transcrever(audio, c: dict, threads: int = THREADS_PADRAO, lote: int = 0):
    """Devolve (palavras [(ini,fim,txt)], segundos). `lote`>0 usa BatchedInferencePipeline com batch_size=lote."""
    m = modelo(threads)
    c = dict(c)
    t0 = time.time()
    if lote:
        from faster_whisper import BatchedInferencePipeline
        seg, _ = BatchedInferencePipeline(model=m).transcribe(audio, batch_size=lote, **c)
    else:
        seg, _ = m.transcribe(audio, **c)
    pal = [(w.start, w.end, w.word) for s in seg for w in (s.words or [])]
    return pal, time.time() - t0


def texto(pal) -> str:
    return "".join(w for _, _, w in pal).strip()


def resumo(pal, ref, dur) -> dict:
    t = texto(pal)
    r = dict(palavras=len(normalizar(t).split()), fim_ultima=round(pal[-1][1], 1) if pal else 0.0, dur=round(dur, 1), texto=t)
    if ref:
        r["wer"] = round(wer(ref, t) * 100, 2)
        r["wer_sem_acentos"] = round(wer(ref, t, True) * 100, 2)
        r["ref_palavras"] = len(normalizar(ref).split())
    return r


def rodar(questao: str, variante: str, clipe: str, clipes: dict, c: dict, **kw) -> dict:
    BRUTO.mkdir(parents=True, exist_ok=True)
    fil = os.environ.get("AVEIA_CLIPES")      # execução em paralelo: cada processo só toca os clipes da sua lista (listas disjuntas)
    if fil and clipe not in fil.split(","):
        return {}
    arq = BRUTO / f"{questao}__{variante}__{clipe}.json"
    if arq.exists():
        return json.loads(arq.read_text(encoding="utf-8"))
    if questao == "q4" and variante == "rec" and CONFIG_REC == {"condition_on_previous_text": True}:   # mesma config do q2 cond-true
        ant = BRUTO / f"q2__cond-true__{clipe}.json"
        if ant.exists():
            d = json.loads(ant.read_text(encoding="utf-8")); d.update(questao="q4", variante="rec", reaproveitado_de="q2")
            arq.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
            return d
    if questao == "q2":   # mesma config já medida em Q1 (a decodificação é determinística: ver atual-rep2/3): reaproveita
        for q in ("q1", "q1b", "q1c"):
            ant = BRUTO / f"{q}__{variante}__{clipe}.json"
            if ant.exists():
                d = json.loads(ant.read_text(encoding="utf-8")); d["questao"] = "q2"; d["reaproveitado_de"] = q
                arq.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
                return d
    audio = clipes[clipe]["audio"]
    pal, dt = transcrever(audio, c, **kw)
    r = resumo(pal, clipes[clipe]["ref"], len(audio) / SR)
    r.update(questao=questao, variante=variante, clipe=clipe, tempo=round(dt, 1), hotwords=c.get("hotwords"),
             palavras_tempo=[(round(a, 2), round(b, 2), w) for a, b, w in pal])
    arq.write_text(json.dumps(r, ensure_ascii=False), encoding="utf-8")
    print(f"{questao}/{variante:28s} {clipe:11s} pal {r['palavras']:4d} fim {r['fim_ultima']:6.1f}/{r['dur']:.1f} "
          f"WER {fw(r):>7s} ({dt:.0f}s)", flush=True)
    return r


# ------------------------------------------------------------------ variantes
VAR = {
    "atual": {},
    "beam1": dict(beam_size=1),
    "sem-halluc": dict(hallucination_silence_threshold=None),
    "temp0": dict(temperature=0.0),
    "sem-prompt": dict(initial_prompt=None),
    "temp0+sem-halluc": dict(temperature=0.0, hallucination_silence_threshold=None),
    "beam1+sem-halluc": dict(beam_size=1, hallucination_silence_threshold=None),
    "sem-halluc+sem-prompt": dict(hallucination_silence_threshold=None, initial_prompt=None),
    "beam1+sem-prompt": dict(beam_size=1, initial_prompt=None),
    "sem-vad": dict(vad_filter=False, vad_parameters=None),
    "sem-wordts": dict(word_timestamps=False),
    "cond-true": dict(condition_on_previous_text=True),
    "no-speech0.6": dict(no_speech_threshold=0.6),
    "beam3": dict(beam_size=3),
    "beam3+cond-true": dict(beam_size=3, condition_on_previous_text=True),
}


def preparado(audio):
    from transcritor.audio import Config, preparar
    return preparar(audio, Config(hpf=True, norm=True))


def q1():
    cl = carregar_clipes()
    for v, alt in VAR.items():
        rodar("q1", v, "ref58", cl, cfg(**alt))
    rodar("q1", "atual", "ref58-webm", cl, cfg())          # entrada .webm
    rodar("q1", "beam1", "ref58-webm", cl, cfg(beam_size=1))
    cl["ref58-pre"] = dict(audio=preparado(cl["ref58"]["audio"]), ref=cl["ref58"]["ref"])    # AVEIA_PREPROC=hpf,norm
    cl["ref58-webm-pre"] = dict(audio=preparado(cl["ref58-webm"]["audio"]), ref=cl["ref58"]["ref"])
    rodar("q1", "atual", "ref58-pre", cl, cfg())
    rodar("q1", "atual", "ref58-webm-pre", cl, cfg())
    rodar("q1", "beam1", "ref58-pre", cl, cfg(beam_size=1))
    # repetição da config atual (determinismo)
    for i in (2, 3):
        rodar("q1", f"atual-rep{i}", "ref58", cl, cfg())


VAR_WEBM = {
    "sem-halluc": dict(hallucination_silence_threshold=None),
    "temp0": dict(temperature=0.0),
    "sem-prompt": dict(initial_prompt=None),
    "sem-vad": dict(vad_filter=False, vad_parameters=None),
    "cond-true": dict(condition_on_previous_text=True),
    "no-speech-off": dict(no_speech_threshold=None),
    "logprob-off": dict(log_prob_threshold=None),
    "no-speech-off+logprob-off": dict(no_speech_threshold=None, log_prob_threshold=None),
    "beam2": dict(beam_size=2),
    "beam3": dict(beam_size=3),
    "beam5-sem-compress": dict(compression_ratio_threshold=None),
    "vad-pad500": dict(vad_parameters=dict(min_silence_duration_ms=500, speech_pad_ms=500)),
    "vad-thr0.35": dict(vad_parameters=dict(min_silence_duration_ms=500, speech_pad_ms=300, threshold=0.35)),
}


def q1c():
    """Diagnóstico no .webm (onde o truncamento aparece): qual parâmetro isolado o remove?"""
    cl = carregar_clipes()
    for v, alt in VAR_WEBM.items():
        rodar("q1c", v, "ref58-webm", cl, cfg(**alt))


def diag():
    """Log DEBUG do faster-whisper (config atual) no clipe dado em argv[2] (padrão: ref58-webm, a janela que perde a carta)."""
    import logging
    logging.basicConfig(level=logging.DEBUG, format="%(name)s: %(message)s")
    logging.getLogger("faster_whisper").setLevel(logging.DEBUG)
    cl = carregar_clipes()
    pal, dt = transcrever(cl[sys.argv[2] if len(sys.argv) > 2 else "ref58-webm"]["audio"], cfg())
    print("palavras", len(pal), "fim", pal[-1][1])


def q1b():
    """Variantes degradadas do clipe de referência (mesmo texto) e demais clipes: o truncamento aparece também fora do ref58?"""
    cl = carregar_clipes()
    ref58 = cl["ref58"]["ref"]
    from faster_whisper.audio import decode_audio
    for n in ("limpo", "ruido", "eco", "ganho-30db", "banda8k", "clipping"):
        cl[f"real-{n}"] = dict(audio=ler_wav(RAIZ / "testes/microfone/ref-real/variantes" / f"{n}.wav"), ref=ref58)
    for nome in [k for k in cl if k.startswith("real-")] + ["tts148"]:
        for v in ("atual", "beam1", "sem-halluc", "temp0"):
            rodar("q1b", v, nome, cl, cfg(**VAR[v]))


def q1d():
    """Q1 (reforço): candidatas com condition_on_previous_text=True nas variantes degradadas do clipe de referência (q1b mediu atual/beam1)."""
    cl = carregar_clipes()
    for n in ("limpo", "ruido", "eco", "ganho-30db", "banda8k", "clipping"):
        cl[f"real-{n}"] = dict(audio=ler_wav(RAIZ / "testes/microfone/ref-real/variantes" / f"{n}.wav"), ref=cl["ref58"]["ref"])
    for nome in [k for k in cl if k.startswith("real-")]:
        for v, alt in (("cond-true", VAR["cond-true"]), ("beam3+cond-true", dict(beam_size=3, condition_on_previous_text=True))):
            rodar("q1d", v, nome, cl, cfg(**alt))


_vad_cache: dict = {}


def _ts_vad(audio):
    k = (id(audio), len(audio))
    if k not in _vad_cache:
        from faster_whisper.vad import VadOptions, get_speech_timestamps
        _vad_cache[k] = [(t["start"] / SR, t["end"] / SR) for t in get_speech_timestamps(audio, VadOptions(min_silence_duration_ms=500, speech_pad_ms=300))]
    return _vad_cache[k]


def lacunas_com_fala(audio, pal, min_lacuna=3.0, min_fala=2.0):
    """Lacunas sem palavras (> min_lacuna s) onde o VAD detecta >= min_fala s de fala: sintoma do 'pulo de janela'."""
    ts = _ts_vad(audio)
    out = []
    pontos = [(0.0, 0.0)] + [(a, b) for a, b, _ in pal] + [(len(audio) / SR, len(audio) / SR)]
    for (a0, b0), (a1, b1) in zip(pontos, pontos[1:]):
        if a1 - b0 > min_lacuna:
            fala = sum(max(0.0, min(f, a1) - max(i, b0)) for i, f in ts)
            if fala >= min_fala:
                out.append((b0, a1))
    return out


def recuperar(audio, pal, c, **kw):
    """Re-transcreve cada lacuna com fala como bloco independente e funde (lista de palavras com tempos absolutos)."""
    novas = list(pal)
    for a, b in lacunas_com_fala(audio, pal):
        a0, b0 = max(0.0, a - 0.3), min(len(audio) / SR, b + 0.3)
        p, _ = transcrever(audio[int(a0 * SR):int(b0 * SR)], c, **kw)
        novas += [(x + a0, y + a0, w) for x, y, w in p if x + a0 >= a - 0.05]
    return sorted(novas, key=lambda t: t[0])


def q2b():
    """Rede de segurança: detectar lacuna com fala (>3 s sem palavras, VAD com >=2 s de fala) e re-transcrever só ela. Aplicada aos casos que truncaram."""
    cl = carregar_clipes()
    ref58 = cl["ref58"]["ref"]
    for n in ("ganho-30db", "banda8k"):
        cl[f"real-{n}"] = dict(audio=ler_wav(RAIZ / "testes/microfone/ref-real/variantes" / f"{n}.wav"), ref=ref58)
    casos = [("atual", "real-banda8k", "q1b"), ("cond-true", "real-banda8k", "q1d"), ("beam3+cond-true", "real-banda8k", "q1d"), ("beam1", "real-banda8k", "q1b"),("atual", "ref58-webm", "q1"), ("beam3", "ref58", "q2"), ("beam1+sem-prompt", "mls-B", "q2"), ("atual", "real-ganho-30db", "q1b")]
    for var, clipe, q in casos:
        arq = BRUTO / f"q2b__recuperar-{var}__{clipe}.json"
        if arq.exists():
            continue
        d = json.loads((BRUTO / f"{q}__{var}__{clipe}.json").read_text(encoding="utf-8"))
        pal = [tuple(x) for x in d["palavras_tempo"]]
        audio = cl[clipe]["audio"]
        lac = lacunas_com_fala(audio, pal)
        novas = recuperar(audio, pal, cfg(**(VAR.get(var, {}))))
        r = resumo(novas, cl[clipe]["ref"], len(audio) / SR)
        r.update(questao="q2b", variante=f"recuperar-{var}", clipe=clipe, tempo=0, lacunas=[(round(a, 1), round(b, 1)) for a, b in lac],
                 antes=dict(palavras=d["palavras"], wer=d.get("wer")), palavras_tempo=[(round(a, 2), round(b, 2), w) for a, b, w in novas])
        arq.write_text(json.dumps(r, ensure_ascii=False), encoding="utf-8")
        print(f"q2b {var}/{clipe}: lacunas {r['lacunas']}  antes {d['palavras']} pal WER {d.get('wer')} -> depois {r['palavras']} pal WER {r.get('wer')}", flush=True)


def pipeline():
    from transcritor.pipeline import transcrever as t
    cl = carregar_clipes()
    arq = BRUTO / "pipeline__transcrever-webm__ref58-webm.json"
    if arq.exists():
        print(arq.read_text(encoding="utf-8")[:300]); return
    t0 = time.time()
    falas = t(str(AR / "audio.webm"), diarizacao=False)
    dt = time.time() - t0
    txt = " ".join(f.texto for f in falas)
    r = dict(palavras=len(normalizar(txt).split()), fim_ultima=round(falas[-1].fim, 1), wer=round(wer(cl["ref58"]["ref"], txt) * 100, 2),
             tempo=round(dt, 1), texto=txt, falas=[(f.inicio, f.fim, f.texto) for f in falas], env_preproc=os.environ.get("AVEIA_PREPROC", ""))
    arq.write_text(json.dumps(r, ensure_ascii=False), encoding="utf-8")
    print({k: v for k, v in r.items() if k not in ("texto", "falas")})


# ------------------------------------------------------------------ Q2
def candidatas() -> dict[str, dict]:
    d = {k: VAR[k] for k in ("atual", "beam1", "temp0", "sem-prompt", "beam1+sem-prompt", "cond-true")}
    d["beam3"] = dict(beam_size=3)
    d["beam3+cond-true"] = dict(beam_size=3, condition_on_previous_text=True)
    return d


def q2():
    cl = carregar_clipes()
    for nome in ("ref58", "ref58-webm", "tts148", "mls-A", "mls-B"):
        for v, alt in candidatas().items():
            if nome in cl:
                rodar("q2", v, nome, cl, cfg(**alt))


# ------------------------------------------------------------------ Q3: blocos
def blocos_vad(audio, min_bloco=15.0, max_bloco=30.0, pausa=0.5):
    """Fecha o bloco na primeira pausa >= `pausa` s depois de `min_bloco` s; corte duro em `max_bloco`. Devolve [(ini, fim)] em s."""
    from faster_whisper.vad import VadOptions, get_speech_timestamps
    ts = get_speech_timestamps(audio, VadOptions(min_silence_duration_ms=int(pausa * 1000), speech_pad_ms=0))
    seg = [(t["start"] / SR, t["end"] / SR) for t in ts]
    if not seg:
        return [(0.0, len(audio) / SR)]
    blocos, ini, fim_ant = [], seg[0][0], seg[0][1]
    for s, e in seg[1:]:
        # há pausa >= `pausa` entre fim_ant e s (garantido pelo VAD); fecha se o bloco já passou de min_bloco
        if fim_ant - ini >= min_bloco:
            blocos.append((ini, fim_ant)); ini = s
        elif e - ini > max_bloco:     # corte duro: o bloco não pode passar de max_bloco
            blocos.append((ini, fim_ant)); ini = s
        fim_ant = e
        while fim_ant - ini > max_bloco:   # fala contínua longa: corta em max_bloco
            blocos.append((ini, ini + max_bloco)); ini += max_bloco
    blocos.append((ini, fim_ant))
    total = len(audio) / SR
    return [(max(0.0, a - 0.25), min(total, b + 0.25)) for a, b in blocos]   # folga de 0,25 s para não comer a borda


def transcrever_blocos(audio, c: dict, blocos=None, **kw):
    blocos = blocos_vad(audio) if blocos is None else blocos
    pal, dt, por = [], 0.0, []
    for a, b in blocos:
        p, t = transcrever(audio[int(a * SR):int(b * SR)], c, **kw)
        dt += t
        pal += [(x + a, y + a, w) for x, y, w in p]
        por.append((round(a, 1), round(b, 1), len(p)))
    return pal, dt, por


def q3(config_nome="rec"):
    cl = carregar_clipes()
    c = cfg(**CONFIG_REC)
    BRUTO.mkdir(parents=True, exist_ok=True)
    for nome in ("ref58", "ref58-webm", "tts148", "mls-A", "mls-B"):
        if nome not in cl:
            continue
        arq = BRUTO / f"q3__blocos-{config_nome}__{nome}.json"
        if not arq.exists():
            pal, dt, por = transcrever_blocos(cl[nome]["audio"], c)
            r = resumo(pal, cl[nome]["ref"], len(cl[nome]["audio"]) / SR)
            r.update(questao="q3", variante=f"blocos-{config_nome}", clipe=nome, tempo=round(dt, 1), blocos=por,
                     palavras_tempo=[(round(a, 2), round(b, 2), w) for a, b, w in pal])
            arq.write_text(json.dumps(r, ensure_ascii=False), encoding="utf-8")
            print(f"q3/blocos {nome}: {len(por)} blocos, pal {r['palavras']}, WER {fw(r)}", flush=True)
        rodar("q3", f"inteiro-{config_nome}", nome, cl, c)


# ------------------------------------------------------------------ Q4: hotwords
GLOSSARIO_FORA = "Kubernetes, Salesforce, PostgreSQL, Itaú, Embraer, OKR, SLA, Brasília, Camargo, Tiradentes"
GLOSSARIO_TTS = "Marina, Ricardo, Fernanda, Aveia"


def q4():
    cl = carregar_clipes()
    for nome in ("ref58", "tts148", "mls-A", "mls-B"):
        if nome not in cl:
            continue
        rodar("q4", "rec", nome, cl, cfg(**CONFIG_REC))
        rodar("q4", "hotwords-fora", nome, cl, cfg(**CONFIG_REC, hotwords=GLOSSARIO_FORA))
    rodar("q4", "hotwords-nomes", "tts148", cl, cfg(**CONFIG_REC, hotwords=GLOSSARIO_TTS))
    rodar("q4", "hotwords-nomes+fora", "tts148", cl, cfg(**CONFIG_REC, hotwords=GLOSSARIO_TTS + ", " + GLOSSARIO_FORA))
    rodar("q4", "hotwords-nomes", "ref58", cl, cfg(**CONFIG_REC, hotwords="Inês, Filipe, Carteiro"))
    # a config atual (beam 5, com prompt) + hotwords, para ver se o prompt genérico + hotwords trunca
    rodar("q4", "atual+hotwords-fora", "ref58", cl, cfg(hotwords=GLOSSARIO_FORA))


# CONFIG_REC é definido depois das medições de Q1/Q2 (ver CONFIG_REC_JSON); padrão provisório = atual
_REC_ARQ = OUT / "config_recomendada.json"
CONFIG_REC = json.loads(_REC_ARQ.read_text(encoding="utf-8")) if _REC_ARQ.exists() else {}


# ------------------------------------------------------------------ Q5: velocidade
def q5():
    cl = carregar_clipes()
    nome = os.environ.get("AVEIA_Q5_CLIPE", "tts148")
    audio, ref = cl[nome]["audio"], cl[nome]["ref"]
    dur = len(audio) / SR
    cn = os.environ.get("AVEIA_Q5_CONFIG", "atual")      # config por nome (VAR); a medida final usa 'atual' (a de pipeline._palavras)
    c = cfg(**VAR[cn])
    configs = [(f"threads{t}", dict(threads=t), c) for t in (4, 6, 8, 12)]          # threads12 = beam 5, é a comparação do beam1
    configs += [("beam1-threads12", dict(threads=12), {**c, "beam_size": 1})]
    configs += [(f"batched{b}-threads12", dict(threads=12, lote=b), c) for b in (4, 8)]
    reps = int(os.environ.get("AVEIA_Q5_REPS", "2"))
    arq = BRUTO / f"q5__tempos__{nome}.json" if cn == "atual" else BRUTO / f"q5__tempos_{cn}__{nome}.json"
    res = json.loads(arq.read_text(encoding="utf-8")) if arq.exists() else []
    feitos = {(r["variante"], r["rep"]) for r in res}
    for rep in range(1, reps + 1):          # ordem intercalada: a cada rodada percorre todas as configs
        for nome_c, kw, cc in configs:
            if (nome_c, rep) in feitos or (os.environ.get("AVEIA_Q5_SO") and nome_c not in os.environ["AVEIA_Q5_SO"].split(",")):
                continue
            t0 = time.time(); modelo(kw["threads"]); carga = modelo(kw["threads"]).t_carga
            pal, dt = transcrever(audio, cc, **kw)
            r = dict(variante=nome_c, rep=rep, clipe=nome, tempo=round(dt, 1), rtf=round(dt / dur, 3), carga=round(carga, 1),
                     palavras=len(normalizar(texto(pal)).split()), wer=round(wer(ref, texto(pal)) * 100, 2) if ref else None,
                     fim_ultima=round(pal[-1][1], 1))
            res.append(r)
            arq.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
            print("q5", r, flush=True)


def q5par():
    """Dois processos ao mesmo tempo, 6 threads cada, cada um transcreve o clipe (config 'atual') uma vez. O modelo é carregado
    ANTES; os dois esperam um instante de largada comum (epoch) e o relógio vai da largada até o último terminar.
    Pai: `q5par` (reps em AVEIA_Q5_REPS, padrão 2). Filho: `q5par_filho <largada_epoch> <saida.json>`."""
    import subprocess
    nome = os.environ.get("AVEIA_Q5_CLIPE", "tts148")
    arq = BRUTO / f"q5par__{nome}.json"
    res = json.loads(arq.read_text(encoding="utf-8")) if arq.exists() else []
    for rep in range(len(res) + 1, int(os.environ.get("AVEIA_Q5_REPS", "2")) + 1):
        largada = time.time() + 45
        outs = [BRUTO / f"q5par__tmp{i}.json" for i in (1, 2)]
        env = {**os.environ, "AVEIA_THREADS": "6", "HF_HUB_OFFLINE": "1"}
        ps = [subprocess.Popen([sys.executable, str(Path(__file__)), "q5par_filho", repr(largada), str(o)], env=env) for o in outs]
        for p in ps:
            p.wait()
        fil = [json.loads(o.read_text(encoding="utf-8")) for o in outs]
        for o in outs:
            o.unlink()
        r = dict(rep=rep, clipe=nome, relogio=round(max(f["fim_epoch"] for f in fil) - largada, 1),
                 tempos=[f["tempo"] for f in fil], palavras=[f["palavras"] for f in fil], wer=[f["wer"] for f in fil])
        res.append(r)
        arq.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
        print("q5par", r, flush=True)


def q5par_filho():
    nome = os.environ.get("AVEIA_Q5_CLIPE", "tts148")
    cl = carregar_clipes()
    audio, ref = cl[nome]["audio"], cl[nome]["ref"]
    modelo(6)
    largada = float(sys.argv[2])
    while time.time() < largada:
        time.sleep(0.05)
    pal, dt = transcrever(audio, cfg(), threads=6)
    Path(sys.argv[3]).write_text(json.dumps(dict(tempo=round(dt, 1), fim_epoch=time.time(), palavras=len(normalizar(texto(pal)).split()),
                                                 wer=round(wer(ref, texto(pal)) * 100, 2) if ref else None)), encoding="utf-8")


# ------------------------------------------------------------------ precisão: corpus com referência humana
HUMANOS = ("fleurs", "mls-livro", "mls-ruido", "mls-ruido-webm", "cv", "cv-webm")
TRES = ("atual", "cond-true", "beam1")


# ------------------------------------------------------------------ etapa 2: dica em toda janela (hotwords = DICA_PT)
Q6_VAR = {
    "atual+dica-hw": dict(initial_prompt=None, hotwords=DICA_PT),      # dica só como hotwords (vale em toda janela)
    "atual+dica-hw+prompt": dict(hotwords=DICA_PT),                    # mantém initial_prompt (1ª janela) e acrescenta hotwords
}
Q6_TRUNCAM = ("ref58-webm", "real-ganho-30db", "cv", "cv-webm", "conv-2loc", "conv-4loc")
Q6_CONTROLE = ("fleurs", "conv-3loc")


def q6(var=None):
    """Etapa 2. `q6 atual+dica-hw` / `q6 atual+dica-hw+prompt` (clipes que truncam + controle; filtre com AVEIA_CLIPES) ou
    `q6 lote8` / `q6 lote4` (BatchedInferencePipeline com config atual nos clipes AVEIA_CLIPES). Saída: bruto/q6__<var>__<clipe>.json."""
    var = var or sys.argv[2]
    cl = clipes_todos()
    if var.startswith("lote"):
        for n in (os.environ.get("AVEIA_CLIPES") or "ref58-webm,cv-webm,conv-4loc").split(","):
            rodar("q6", var, n, cl, cfg(), lote=int(var[4:]))
        return
    for n in Q6_TRUNCAM + Q6_CONTROLE:
        rodar("q6", var, n, cl, cfg(**Q6_VAR[var]))


def _base(q_prefixos, v, n):
    for q in q_prefixos:
        p = BRUTO / f"{q}__{v}__{n}.json"
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))


def eco_dica(txt: str) -> int:
    """Trigramas consecutivos de DICA_PT (normalizada) encontrados na saída normalizada: 0 = sem eco."""
    d, t = normalizar(DICA_PT).split(), " " + normalizar(txt) + " "
    return sum(t.count(" " + " ".join(d[i:i + 3]) + " ") for i in range(len(d) - 2))


def q6tab():
    """Por clipe: atual / cond-true (já medidos) e as execuções q6: palavras, WER, lacunas com fala (n, s), trigramas da dica na saída."""
    cl = clipes_todos()
    for n in Q6_TRUNCAM + Q6_CONTROLE:
        a = cl[n]["audio"]
        print(f"\n## {n} ({len(a) / SR:.1f} s)")
        linhas = [("atual", _base(("q2", "q1", "q1b", "q1c", "q1d"), "atual", n)), ("cond-true", _base(("q2", "q1c", "q1d"), "cond-true", n))]
        linhas += [(v, _base(("q6",), v, n)) for v in list(Q6_VAR) + ["lote4", "lote8"]]
        for v, r in linhas:
            if not r:
                continue
            pal = [tuple(x) for x in r["palavras_tempo"]]
            lac = lacunas_com_fala(a, pal)
            ts = _ts_vad(a)
            fala = sum(sum(max(0.0, min(f, y) - max(i, x)) for i, f in ts) for x, y in lac)
            print(f"{v:22s} pal {r['palavras']:4d} WER {fw(r):>7s} lacunas_com_fala {len(lac)} ({fala:4.1f} s) eco_dica_trigramas {eco_dica(r['texto'])} fim {r['fim_ultima']:.1f}/{r['dur']:.1f} tempo {r['tempo']:.0f}s")


def q2h(etapa="curtos"):
    """Q2 nos clipes com referência humana. etapa: sanidade | curtos | longo (mls-longo-webm: só atual e cond-true)."""
    cl = carregar_clipes()
    if etapa == "sanidade":
        for n in ("mls-livro", "fleurs"):
            rodar("q2", "atual", n, cl, cfg())
    elif etapa == "curtos":
        for n in HUMANOS:
            for v in TRES:
                rodar("q2", v, n, cl, cfg(**VAR[v]))
    elif etapa == "longo":
        for v in ("atual", "cond-true"):
            rodar("q2", v, "mls-longo-webm", cl, cfg(**VAR[v]))


# ------------------------------------------------------------------ utilidades de alinhamento / repetição
def _palavras_norm(pal):
    """[(tempo, palavra normalizada)]: uma entrada por palavra contada pelo WER (um token do Whisper pode virar 0 ou 2+)."""
    return [(a, x) for a, _b, w in pal for x in normalizar(w).split()]


def alinhar(ref: list[str], hip: list[str]):
    """Alinhamento de Levenshtein palavra a palavra: [(op, i, j)], op em ok|sub|del|ins (i = índice na ref, j = na hipótese)."""
    n, m = len(ref), len(hip)
    D = [list(range(m + 1))]
    for i in range(1, n + 1):
        ri, prev, cur = ref[i - 1], D[-1], [i]
        for j in range(1, m + 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ri != hip[j - 1])))
        D.append(cur)
    i, j, ops = n, m, []
    while i > 0 or j > 0:
        if i > 0 and j > 0 and D[i][j] == D[i - 1][j - 1] + (ref[i - 1] != hip[j - 1]):
            ops.append(("ok" if ref[i - 1] == hip[j - 1] else "sub", i - 1, j - 1)); i -= 1; j -= 1
        elif i > 0 and D[i][j] == D[i - 1][j] + 1:
            ops.append(("del", i - 1, None)); i -= 1
        else:
            ops.append(("ins", None, j - 1)); j -= 1
    ops.reverse()
    return ops


def repeticoes(palavras: list[str], nmin=4, nmax=40, k=3):
    """Trechos de >= nmin palavras repetidos >= k vezes em sequência: [(índice, n, repetições, texto)]."""
    L, out, i = len(palavras), [], 0
    while i < L:
        achou = None
        for n in range(nmin, nmax + 1):
            if i + k * n > L:
                break
            base, reps = palavras[i:i + n], 1
            while palavras[i + reps * n:i + (reps + 1) * n] == base:
                reps += 1
            if reps >= k:
                achou = (i, n, reps, " ".join(base)); break
        if achou:
            out.append(achou); i += achou[1] * achou[2]
        else:
            i += 1
    return out


def _trigramas(ws):
    return {tuple(ws[i:i + 3]) for i in range(len(ws) - 2)}


# ------------------------------------------------------------------ recuperação de lacunas aplicada a todos os resultados
VAR_TODAS = {**VAR, **VAR_WEBM}


def clipes_todos():
    cl = carregar_clipes()
    for n in ("limpo", "ruido", "eco", "ganho-30db", "banda8k", "clipping"):
        cl[f"real-{n}"] = dict(audio=ler_wav(RAIZ / "testes/microfone/ref-real/variantes" / f"{n}.wav"), ref=cl["ref58"]["ref"])
    return cl


def recuperar2(audio, pal, c, **kw):
    """Como `recuperar`, mas devolve também o detalhe por lacuna (palavras recuperadas, se veio vazia) e o tempo gasto."""
    t0 = time.time()
    lac = lacunas_com_fala(audio, pal)
    t_det = time.time() - t0
    novas, det, t_rec = list(pal), [], 0.0
    for a, b in lac:
        a0, b0 = max(0.0, a - 0.3), min(len(audio) / SR, b + 0.3)
        p, dt = transcrever(audio[int(a0 * SR):int(b0 * SR)], c, **kw)
        t_rec += dt
        itens = [(x + a0, y + a0, w) for x, y, w in p if x + a0 >= a - 0.05]
        novas += itens
        fala_s = sum(max(0.0, min(f, b) - max(i, a)) for i, f in _ts_vad(audio))
        det.append(dict(ini=round(a, 1), fim=round(b, 1), fala_s=round(fala_s, 1), palavras=len(itens), vazio=len(itens) == 0, texto=texto(itens)))
    return sorted(novas, key=lambda t: t[0]), dict(lacunas=det, t_detectar=round(t_det, 1), t_retranscrever=round(t_rec, 1))


def rec():
    """Aplica detecção + recuperação de lacunas a CADA resultado em bruto/ (q2, q1, q1b, q1c, q1d; uma vez por variante x clipe).
    Grava bruto/rec__<variante>__<clipe>.json com antes/depois, lacunas, texto inserido e o tempo extra."""
    cl = clipes_todos()
    vistos = set()
    for q in ("q2", "q1", "q1b", "q1c", "q1d"):
        for p in sorted(BRUTO.glob(f"{q}__*.json")):
            d = json.loads(p.read_text(encoding="utf-8"))
            v, c = d.get("variante"), d.get("clipe")
            if (v, c) in vistos or c not in cl or v not in VAR_TODAS or not d.get("palavras_tempo"):
                continue
            vistos.add((v, c))
            arq = BRUTO / f"rec__{v}__{c}.json"
            if arq.exists():
                continue
            audio, pal = cl[c]["audio"], [tuple(x) for x in d["palavras_tempo"]]
            novas, info = recuperar2(audio, pal, cfg(**VAR_TODAS[v]))
            r = resumo(novas, cl[c]["ref"], len(audio) / SR)
            antes_ws = [w for _, w in _palavras_norm(pal)]
            ins_ws = [w for _, w in _palavras_norm([x for x in novas if x not in set(pal)])]
            dup3 = len(_trigramas(ins_ws) & _trigramas(antes_ws))
            r.update(questao="rec", variante=v, clipe=c, origem=d["questao"], tempo=round(info["t_detectar"] + info["t_retranscrever"], 1),
                     tempo_extra=round(info["t_detectar"] + info["t_retranscrever"], 1), t_detectar=info["t_detectar"],
                     t_retranscrever=info["t_retranscrever"], antes=dict(palavras=d["palavras"], wer=d.get("wer"), fim_ultima=d["fim_ultima"]),
                     lacunas=info["lacunas"], inseridas=len(ins_ws), inseridas_texto=" ".join(ins_ws), trigramas_inseridos_ja_no_texto=dup3,
                     texto_igual=normalizar(texto(novas)) == normalizar(d["texto"]),
                     repeticoes=repeticoes([w for _, w in _palavras_norm(novas)]),
                     palavras_tempo=[(round(a, 2), round(b, 2), w) for a, b, w in novas])
            arq.write_text(json.dumps(r, ensure_ascii=False), encoding="utf-8")
            print(f"rec {v:20s} {c:15s} lacunas {[(x['ini'], x['fim'], x['palavras']) for x in info['lacunas']]} "
                  f"pal {d['palavras']}->{r['palavras']} WER {d.get('wer')}->{r.get('wer')} +{r['tempo_extra']}s inseridas={len(ins_ws)} dup3={dup3}", flush=True)


def tabela_rec():
    """variante x clipe: 'WER% (palavras)' sem -> com recuperação de lacunas. Sem ref: só palavras. Também lista lacunas e vazios."""
    d = {}
    for p in BRUTO.glob("rec__*.json"):
        r = json.loads(p.read_text(encoding="utf-8")); d[(r["variante"], r["clipe"])] = r
    clipes = ["ref58", "ref58-webm", "tts148", "fleurs", "mls-livro", "mls-ruido", "mls-ruido-webm", "cv", "cv-webm", "mls-longo-webm", "mls-A", "mls-B", "conv-2loc", "conv-3loc", "conv-4loc"]
    for v in TRES:
        print(f"\n### {v}")
        for c in clipes:
            r = d.get((v, c))
            if not r:
                continue
            a = r["antes"]
            wa = f"{a['wer']:.1f}%" if a.get("wer") is not None else "sem ref"
            wd = f"{r['wer']:.1f}%" if r.get("wer") is not None else "sem ref"
            lac = "; ".join(f"{x['ini']}-{x['fim']}s ({x['palavras']} pal{', VAZIO' if x['vazio'] else ''})" for x in r["lacunas"]) or "nenhuma"
            print(f"{c:15s} sem: {wa:>8s} ({a['palavras']:4d})  com: {wd:>8s} ({r['palavras']:4d})  extra {r['tempo_extra']:5.1f}s  igual={r['texto_igual']}  inseridas={r['inseridas']} dup3={r['trigramas_inseridos_ja_no_texto']}  lacunas: {lac}")
    print("\n### demais variantes/degradados com lacuna detectada")
    for (v, c), r in sorted(d.items()):
        if v in TRES and c in clipes:
            continue
        if r["lacunas"]:
            a = r["antes"]
            print(f"{v:22s} {c:16s} pal {a['palavras']}->{r['palavras']} WER {a.get('wer')}->{r.get('wer')} extra {r['tempo_extra']}s  "
                  + "; ".join(f"{x['ini']}-{x['fim']}s ({x['palavras']} pal{', VAZIO' if x['vazio'] else ''})" for x in r["lacunas"]))
    print("\n### texto inserido nos casos sem truncamento (falsos positivos), variante TRES")
    for (v, c), r in sorted(d.items()):
        if r["lacunas"] and (r["antes"].get("wer") is None or r["antes"]["wer"] < 25):
            for x in r["lacunas"]:
                print(f"{v:12s} {c:15s} lacuna {x['ini']}-{x['fim']}s -> {x['palavras']} pal: {x['texto'][:200]!r}")


# ------------------------------------------------------------------ áudio longo
def faixas(ref: str, pal, passo=120.0):
    """WER por faixa de `passo` s: alinha (Levenshtein) a referência inteira com a hipótese e atribui cada operação ao tempo da
    palavra da hipótese correspondente (deleções herdam o tempo da última palavra da hipótese vista). Total = WER normal."""
    r = normalizar(ref).split()
    h = _palavras_norm(pal)
    ops = alinhar(r, [w for _, w in h])
    nf = int(max(t for t, _ in h) // passo) + 1
    f = [dict(ref=0, hip=0, sub=0, dele=0, ins=0) for _ in range(nf)]
    ult = 0.0
    for op, i, j in ops:
        t = h[j][0] if j is not None else ult
        ult = t
        k = min(int(t // passo), nf - 1)
        if op in ("ok", "sub", "del"):
            f[k]["ref"] += 1
        if op in ("ok", "sub", "ins"):
            f[k]["hip"] += 1
        if op == "sub": f[k]["sub"] += 1
        if op == "del": f[k]["dele"] += 1
        if op == "ins": f[k]["ins"] += 1
    for k, x in enumerate(f):
        x["faixa"] = f"{int(k * passo)}-{int((k + 1) * passo)}s"
        x["wer"] = round((x["sub"] + x["dele"] + x["ins"]) / x["ref"] * 100, 1) if x["ref"] else None
    return f


def longo():
    """mls-longo-webm: WER total e por faixa de 2 min (alinhamento), repetições e lacunas, sem e com recuperação."""
    cl = carregar_clipes()
    ref = cl["mls-longo-webm"]["ref"]
    base = normalizar(ref).split()
    print("repetições (4+ palavras, 3+ vezes seguidas) na REFERÊNCIA:", repeticoes(base))
    saida = {}
    for v in ("atual", "cond-true"):
        for pref, rotulo in (("q2", "sem recuperação"), ("rec", "com recuperação")):
            arq = BRUTO / f"{pref}__{v}__mls-longo-webm.json"
            if not arq.exists():
                continue
            d = json.loads(arq.read_text(encoding="utf-8"))
            pal = [tuple(x) for x in d["palavras_tempo"]]
            f = faixas(ref, pal)
            ws = [w for _, w in _palavras_norm(pal)]
            rep = repeticoes(ws)
            lac = lacunas_com_fala(cl["mls-longo-webm"]["audio"], pal)
            print(f"\n== {v} {rotulo}: palavras {d['palavras']} (ref {d['ref_palavras']}) WER {d['wer']} fim {d['fim_ultima']}/{d['dur']}s "
                  f"| repetições={rep} | lacunas com fala={[(round(a, 1), round(b, 1)) for a, b in lac]}")
            for x in f:
                print(f"   {x['faixa']:>10s} ref {x['ref']:4d} hip {x['hip']:4d} sub {x['sub']:3d} del {x['dele']:3d} ins {x['ins']:3d} WER {x['wer']}")
            saida[f"{v}|{rotulo}"] = dict(wer=d["wer"], palavras=d["palavras"], faixas=f, repeticoes=rep, lacunas=lac)
    (OUT / "analise_longo.json").write_text(json.dumps(saida, ensure_ascii=False, indent=1), encoding="utf-8")


# ------------------------------------------------------------------ Q3 (continuação): corte idêntico ao do aovivo.py
def blocos_aovivo(audio):
    """Reproduz o planejamento de transcritor/aovivo.py::Sessao._planejar sobre o áudio inteiro (usa as próprias funções _fala_vad,
    escolher_pausa e _ponto_quieto do módulo): cortes SEM folga, no meio da pausa; bloco >= 15 s contado desde o cursor; corte duro ~29 s."""
    from transcritor import aovivo as av
    minimo, maximo = int(av.BLOCO_MIN_S * SR), int(av.BLOCO_MAX_S * SR)
    n, cur, out = len(audio), 0, []
    while cur < n:
        fim_j = min(n, cur + maximo)
        janela = audio[cur:fim_j]
        tam = len(janela)
        fala = av._fala_vad(janela)
        if sum(b - a for a, b in fala) < av.FALA_MIN_S * SR:
            cur = fim_j; continue                 # sem fala: não gasta Whisper
        corte = av.escolher_pausa(fala, tam, minimo)
        if corte is None and tam >= maximo:
            alvo = av._ponto_quieto(janela, av.CORTE_DURO_ALVO_S, av.CORTE_DURO_RAIO_S)
            corte = min(max(int(alvo * SR), minimo), maximo)
        if corte is None:                         # cauda (finalização): o resto vai num bloco só
            corte = tam
        out.append((cur / SR, (cur + corte) / SR)); cur += corte
    return out


def inteiro(conf, nome, cl, c):
    """Arquivo inteiro com a mesma config: reaproveita o q2 (decodificação determinística) quando a config coincide."""
    arq = BRUTO / f"q3__inteiro-{conf}__{nome}.json"
    if arq.exists():
        return
    var = {"rec": "cond-true", "atual": "atual"}[conf]
    ant = BRUTO / f"q2__{var}__{nome}.json"
    if ant.exists() and (conf != "rec" or CONFIG_REC == {"condition_on_previous_text": True}):
        d = json.loads(ant.read_text(encoding="utf-8"))
        d.update(questao="q3", variante=f"inteiro-{conf}", reaproveitado_de="q2")
        arq.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
        return
    rodar("q3", f"inteiro-{conf}", nome, cl, c)


def q3h():
    """q3h <corte: vad|aovivo> <config: rec|atual> <clipe1,clipe2,...>  (corte vad = blocos_vad do script; aovivo = blocos_aovivo)"""
    corte = sys.argv[2] if len(sys.argv) > 2 else "vad"
    conf = sys.argv[3] if len(sys.argv) > 3 else "rec"
    nomes = sys.argv[4].split(",") if len(sys.argv) > 4 else ["fleurs", "mls-livro", "mls-ruido-webm"]
    cl = carregar_clipes()
    c = cfg(**CONFIG_REC) if conf == "rec" else cfg()
    pref = "blocos" if corte == "vad" else "aovivo"
    for nome in nomes:
        arq = BRUTO / f"q3__{pref}-{conf}__{nome}.json"
        if not arq.exists():
            audio = cl[nome]["audio"]
            bl = None if corte == "vad" else blocos_aovivo(audio)
            pal, dt, por = transcrever_blocos(audio, c, blocos=bl)
            r = resumo(pal, cl[nome]["ref"], len(audio) / SR)
            r.update(questao="q3", variante=f"{pref}-{conf}", clipe=nome, tempo=round(dt, 1), blocos=por,
                     palavras_tempo=[(round(a, 2), round(b, 2), w) for a, b, w in pal])
            arq.write_text(json.dumps(r, ensure_ascii=False), encoding="utf-8")
            print(f"q3/{pref}-{conf} {nome}: {len(por)} blocos, pal {r['palavras']}, WER {fw(r)} ({dt:.0f}s)", flush=True)
        inteiro(conf, nome, cl, c)


def q3tab():
    """Blocos x inteiro: palavras, WER, e ONDE diferem (distância ao corte mais próximo; borda = <= 2 s) e, com referência,
    quais erros são novos nos blocos / corrigidos pelos blocos."""
    import difflib
    for p in sorted(BRUTO.glob("q3__*.json")):
        _, var, clipe = p.stem.split("__")
        if var.startswith("inteiro"):
            continue
        conf = var.split("-", 1)[1]
        b = json.loads(p.read_text(encoding="utf-8"))
        i = json.loads((BRUTO / f"q3__inteiro-{conf}__{clipe}.json").read_text(encoding="utf-8"))
        pb, pi = _palavras_norm(b["palavras_tempo"]), _palavras_norm(i["palavras_tempo"])
        wb, wi = [w for _, w in pb], [w for _, w in pi]
        cortes = [(b["blocos"][k][1] + b["blocos"][k + 1][0]) / 2 for k in range(len(b["blocos"]) - 1)]

        def dist(t):
            return min((abs(t - x) for x in cortes), default=float("inf"))

        diffs = []
        for op, a0, a1, b0, b1 in difflib.SequenceMatcher(None, wi, wb, autojunk=False).get_opcodes():
            if op == "equal":
                continue
            t = pb[b0][0] if b1 > b0 and b0 < len(pb) else (pi[a0][0] if a0 < len(pi) else 0.0)
            diffs.append((round(t, 1), round(dist(t), 1), " ".join(wi[a0:a1]), " ".join(wb[b0:b1])))
        borda = sum(1 for d in diffs if d[1] <= 2.0)
        print(f"\n## {var} / {clipe}: {len(b['blocos'])} blocos | pal bloco={b['palavras']} inteiro={i['palavras']} | WER bloco={fw(b)} inteiro={fw(i)} "
              f"| div bloco x inteiro={wer(i['texto'], b['texto']) * 100:.1f}% | tempo bloco={b['tempo']}s inteiro={i['tempo']}s")
        print(f"   diferenças bloco x inteiro: {len(diffs)} (na borda <=2 s do corte: {borda}; no meio: {len(diffs) - borda})")
        for d in diffs[:40]:
            print(f"     t={d[0]:7.1f}s dist_corte={d[1]:5.1f}s  inteiro={d[2]!r}  bloco={d[3]!r}")
        if b.get("ref_palavras"):
            r = normalizar(open_ref(clipe)).split()
            st = {}
            for nome_h, pw in (("b", wb), ("i", wi)):
                st[nome_h] = {ii: (op, jj) for op, ii, jj in alinhar(r, pw) if ii is not None}
            novos = [(r[k], pb[st["b"][k][1]][1] if st["b"][k][1] is not None else "-", pb[st["b"][k][1]][0] if st["b"][k][1] is not None else None)
                     for k in range(len(r)) if st["b"][k][0] != "ok" and st["i"][k][0] == "ok"]
            corrig = [r[k] for k in range(len(r)) if st["b"][k][0] == "ok" and st["i"][k][0] != "ok"]
            print(f"   erros que só os blocos têm: {len(novos)}; erros que só o inteiro tem: {len(corrig)}")
            for w, h, t in novos[:40]:
                print(f"     bloco errou: ref={w!r} hip={h!r}" + (f" t={t:.1f}s dist_corte={dist(t):.1f}s" if t is not None else " (palavra apagada)"))
            print(f"     só o inteiro errou: {corrig[:40]}")
        print(f"   blocos={b['blocos']}")


def open_ref(clipe: str) -> str:
    return carregar_clipes_ref()[clipe]


_refs: dict = {}


def carregar_clipes_ref() -> dict:
    """Referências humanas sem decodificar áudio."""
    if not _refs:
        _refs["ref58"] = _refs["ref58-webm"] = " ".join(f["texto"] for f in json.load(open(AR / "referencia.json", encoding="utf-8")))
        _refs["tts148"] = (TTS / "referencia.txt").read_text(encoding="utf-8")
        for n, (_a, t) in HUMANOS_ARQ.items():
            if (CLIPES / t).exists():
                _refs[n] = (CLIPES / t).read_text(encoding="utf-8").strip()
    return _refs


# ------------------------------------------------------------------ Q4 (continuação): glossário / hotwords
FORA = ["Kubernetes", "Salesforce", "PostgreSQL", "Itaú", "Embraer", "OKR", "SLA", "Brasília", "Camargo", "Tiradentes",
        "Bradesco", "Petrobras", "Magazine Luiza", "Natura", "Ambev", "Suzano", "Localiza", "Totvs", "Stefanini", "Sicredi",
        "Cielo", "Gerdau", "Marfrig", "Nubank", "iFood", "Mercado Livre", "CNPJ", "LGPD", "ERP", "CRM",
        "KPI", "EBITDA", "Scrum", "Jira", "Confluence", "Datadog", "Terraform", "Odoo", "Aveia", "Ruan Cavalcanti",
        "Thiago Nascimento", "Bianca Vasconcelos", "Rogério Albuquerque", "Débora Siqueira", "Gustavo Tavares", "Wellington Prado",
        "Aracaju", "Uberlândia", "Zeferino", "Quintanilha", "Pernambuco Telecom", "Ferreira Lacerda"]
DENTRO = {
    "cv": ["Otacílio Costa", "Itamari", "Tangará da Serra", "Machu Picchu", "Taj Mahal", "Tanabi", "Caldeirão Grande",
           "Alfredo Barbosa dos Santos", "Gabriel Landi", "Alexandre Albuquerque de Freitas"],
    "tts148": ["Marina", "Ricardo", "Fernanda"],
    "ref58": ["Inês", "Filipe"],
    "ref58-webm": ["Inês", "Filipe"],
}
# termos contados na saída (os do glossário + controles que NÃO entram no glossário)
CONTAR = {
    "cv": DENTRO["cv"] + ["Petra", "Coliseu", "Gramado", "Contagem", "Sicília", "Carlos de Moura Ramos", "Xavier", "Patricia",
                          "Frederico", "Adriano", "Juliana", "Letícia"],
    "tts148": DENTRO["tts148"] + ["São Paulo", "Curitiba"],
    "ref58": DENTRO["ref58"], "ref58-webm": DENTRO["ref58"],
}


def contar_termo(termo: str, txt: str, sem_acentos=False) -> int:
    return (" " + normalizar(txt, sem_acentos) + " ").count(" " + normalizar(termo, sem_acentos) + " ")


def fora_para(nome: str) -> list[str]:
    """Termos fora de contexto: tira qualquer um que compartilhe palavra (> 3 letras) com a referência do clipe."""
    ws = set(normalizar(carregar_clipes_ref()[nome]).split())
    return [t for t in FORA if not any(w in ws for w in normalizar(t).split() if len(w) > 3)]


def glossarios(nome: str) -> dict[str, str]:
    d, f = DENTRO[nome], fora_para(nome)
    g = {"dentro": d, "g10": d + f[:max(0, 10 - len(d))], "fora10": f[:10]}   # (g50/fora50 cortados do escopo)
    return {k: ", ".join(v) for k, v in g.items()}


def q4b():
    """Hotwords x config (atual / cond-true) x clipe; linha de base sem hotwords vem do Q2. Glossários: dentro (só termos do áudio),
    g10/g50 (dentro + fora de contexto até 10/50 termos), fora10/fora50 (nenhum termo do áudio)."""
    cl = carregar_clipes()
    for nome in ("ref58", "cv", "tts148", "ref58-webm"):
        for cn, alt in (("atual", {}), ("cond-true", dict(condition_on_previous_text=True))):
            rodar("q2", cn, nome, cl, cfg(**alt))
            vistos = set()
            for hn, hw in glossarios(nome).items():
                if hw in vistos:
                    continue                      # (cv: dentro == g10)
                vistos.add(hw)
                rodar("q4", f"{cn}+hw-{hn}", nome, cl, cfg(**alt, hotwords=hw))


def q4tab():
    """Para cada clipe e execução (q2 sem hotwords + q4): palavras, WER, lacunas com fala, termos certos e termos fora de contexto inseridos."""
    cl = carregar_clipes()
    todos_fora = set(FORA) | {t.strip() for t in GLOSSARIO_FORA.split(",")}
    for nome in ("ref58", "ref58-webm", "cv", "tts148", "mls-A", "mls-B"):
        ref = cl[nome]["ref"]
        arqs = [BRUTO / f"q2__{v}__{nome}.json" for v in ("atual", "cond-true")] + sorted(BRUTO.glob(f"q4__*__{nome}.json"))
        print(f"\n## {nome}" + (f" | termos contados: {CONTAR[nome]}" if nome in CONTAR else ""))
        for p in arqs:
            if not p.exists():
                continue
            r = json.loads(p.read_text(encoding="utf-8"))
            pal = [tuple(x) for x in r["palavras_tempo"]]
            lac = lacunas_com_fala(cl[nome]["audio"], pal) if pal else []
            ins = [t for t in sorted(todos_fora) if contar_termo(t, r["texto"]) > (contar_termo(t, ref) if ref else 0)]
            linha = f"{r['variante']:30s} pal {r['palavras']:4d} WER {fw(r):>7s} lacunas_com_fala {len(lac)} fim {r['fim_ultima']:.1f}/{r['dur']:.1f}"
            if nome in CONTAR:
                ts = CONTAR[nome]
                ok = [t for t in ts if contar_termo(t, r["texto"]) >= max(1, contar_termo(t, ref))]
                ok2 = [t for t in ts if contar_termo(t, r["texto"], True) >= max(1, contar_termo(t, ref, True))]
                linha += f" | termos certos {len(ok)}/{len(ts)} (sem acento {len(ok2)}/{len(ts)}) | faltam: {[t for t in ts if t not in ok]}"
            linha += f" | termos de fora inseridos: {ins}"
            print(linha)


# ------------------------------------------------------------------ truncamento em conversa (sem referência de texto)
CONV = ("conv-2loc", "conv-3loc", "conv-4loc")


def conv():
    """atual / cond-true / beam1 nas conversas sintéticas com troca de locutor (sem referência de texto)."""
    cl = carregar_clipes()
    for n in CONV:
        for v in TRES:
            rodar("q2", v, n, cl, cfg(**VAR[v]))


def _trocas(nome: str):
    """Instantes (s) de troca de locutor segundo o RTTM da conversa sintética."""
    seg = []
    for ln in (RAIZ / "testes" / "diarizacao" / "dados" / f"mls-pt-{nome[5:]}.rttm").read_text(encoding="utf-8").splitlines():
        f = ln.split()
        if len(f) > 7 and f[0] == "SPEAKER":
            seg.append((float(f[3]), float(f[3]) + float(f[4]), f[7]))
    seg.sort()
    return [b[0] for a, b in zip(seg, seg[1:]) if a[2] != b[2]]


def convtab():
    import difflib
    cl = carregar_clipes()
    pal = {}
    for n in CONV:
        dur = len(cl[n]["audio"]) / SR
        trocas = _trocas(n)
        ts = _ts_vad(cl[n]["audio"])
        print(f"\n## {n}: {dur:.1f}s ({dur / 60:.2f} min), {len(trocas)} trocas de locutor (RTTM), fala VAD total {sum(f - i for i, f in ts):.1f}s")
        for v in TRES:
            d = json.loads((BRUTO / f"q2__{v}__{n}.json").read_text(encoding="utf-8"))
            p = [tuple(x) for x in d["palavras_tempo"]]
            pal[(n, v)] = d
            lac = lacunas_com_fala(cl[n]["audio"], p)
            fala = [sum(max(0.0, min(f, b) - max(i, a)) for i, f in ts) for a, b in lac]
            linha = f"{v:10s} pal {d['palavras']:4d} fim {d['fim_ultima']}/{d['dur']} | lacunas com fala: {len(lac)} ({len(lac) / (dur / 60):.2f}/min), {sum(fala):.1f}s de fala nelas"
            print(linha)
            for (a, b), fs in zip(lac, fala):
                # distância ao início da lacuna até a troca de locutor mais próxima e pausa do VAD logo antes
                prox = min((abs(a - t) for t in trocas), default=99)
                ant = [f for i, f in ts if f <= a + 0.5]
                pausa_antes = a - ant[-1] if ant else a
                print(f"    lacuna {a:6.1f}-{b:6.1f}s ({b - a:4.1f}s, fala {fs:4.1f}s) troca de locutor mais próxima do início: {prox:4.1f}s; pausa VAD antes: {pausa_antes:4.1f}s")
            r = BRUTO / f"rec__{v}__{n}.json"
            if r.exists():
                rr = json.loads(r.read_text(encoding="utf-8"))
                lac2 = lacunas_com_fala(cl[n]["audio"], [tuple(x) for x in rr["palavras_tempo"]])
                print(f"    depois de recuperar: pal {rr['palavras']} (+{rr['palavras'] - d['palavras']}), lacunas com fala restantes: {len(lac2)} {[(round(a, 1), round(b, 1)) for a, b in lac2]}, "
                      f"vazias: {sum(1 for x in rr['lacunas'] if x['vazio'])}, extra {rr['tempo_extra']}s")
        for v1, v2 in (("atual", "cond-true"), ("atual", "beam1"), ("cond-true", "beam1")):
            a, b = pal[(n, v1)], pal[(n, v2)]
            wa, wb = normalizar(a["texto"]).split(), normalizar(b["texto"]).split()
            so1, so2 = [], []
            for op, a0, a1, b0, b1 in difflib.SequenceMatcher(None, wa, wb, autojunk=False).get_opcodes():
                if op != "equal":
                    so1.append(" ".join(wa[a0:a1])); so2.append(" ".join(wb[b0:b1]))
            print(f"  {v1} x {v2}: divergência {wer(a['texto'], b['texto']) * 100:.1f}% | {len(so1)} trechos diferentes; palavras só em {v2}: {sum(len(x.split()) for x in so2)}, só em {v1}: {sum(len(x.split()) for x in so1)}")
            if (v1, v2) == ("atual", "cond-true"):
                for x, y in list(zip(so1, so2))[:25]:
                    print(f"      {v1}={x[:90]!r}  {v2}={y[:90]!r}")


def thrchk():
    """O texto muda com o nº de threads? Repete com 6 threads duas execuções já medidas com 12 (ref58 cond-true; ref58-webm atual, que trunca)."""
    cl = carregar_clipes()
    for var, clipe, q in (("cond-true", "ref58", "q2"), ("atual", "ref58-webm", "q2")):
        ant = json.loads((BRUTO / f"{q}__{var}__{clipe}.json").read_text(encoding="utf-8"))
        pal, dt = transcrever(cl[clipe]["audio"], cfg(**VAR[var]), threads=6)
        igual = texto(pal) == ant["texto"]
        iguais_tempos = [(round(a, 2), round(b, 2), w) for a, b, w in pal] == [tuple(x) for x in ant["palavras_tempo"]]
        print(f"thrchk {var}/{clipe}: 6 threads {len(normalizar(texto(pal)).split())} pal vs 12 threads {ant['palavras']} pal | texto idêntico={igual} | palavras+tempos idênticos={iguais_tempos} ({dt:.0f}s)", flush=True)


# ------------------------------------------------------------------ tabelas
def matriz(questao: str, ref_var: str = "beam1"):
    """variante x clipe: 'WER% (palavras)'. Clipes sem referência humana: 'div%' = divergência de palavras contra a variante `ref_var`
    do mesmo clipe (NÃO é WER: é só uma medida de quanto o texto muda)."""
    d = {}
    for p in BRUTO.glob(f"{questao}__*.json"):
        r = json.loads(p.read_text(encoding="utf-8")); d[(r["variante"], r["clipe"])] = r
    clipes = sorted({c for _, c in d}); vars_ = sorted({v for v, _ in d}, key=lambda v: (v != "atual", v))
    print(f"{'variante':24s}" + "".join(f"{c:>20s}" for c in clipes))
    for v in vars_:
        linha = f"{v:24s}"
        for c in clipes:
            r = d.get((v, c))
            if not r:
                linha += f"{'-':>20s}"; continue
            if "wer" in r:
                cel = f"{r['wer']:.1f}% ({r['palavras']})"
            else:
                rf = d.get((ref_var, c))
                cel = f"div {wer(rf['texto'], r['texto']) * 100:.1f}% ({r['palavras']})" if rf else f"({r['palavras']})"
            linha += f"{cel:>20s}"
        print(linha)


def tabelas():
    import collections
    por = collections.defaultdict(list)
    for p in sorted(BRUTO.glob("*.json")):
        if p.name.startswith("q5__"):
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        if "variante" in d:
            por[d["questao"]].append(d)
    for q, rs in por.items():
        print(f"\n== {q}")
        print(f"{'variante':28s} {'clipe':15s} {'pal':>4s} {'ref':>4s} {'fim/dur':>11s} {'WER%':>6s} {'semAc%':>7s} {'tempo':>6s}")
        for r in rs:
            sa = f"{r['wer_sem_acentos']:.2f}" if r.get("wer_sem_acentos") is not None else "sem ref"
            print(f"{r['variante']:28s} {r['clipe']:15s} {r['palavras']:4d} {r.get('ref_palavras', 0):4d} "
                  f"{r['fim_ultima']:5.1f}/{r['dur']:<5.1f} {fw(r):>7s} {sa:>7s} {r['tempo']:6.1f}")
    f = BRUTO / "q5__tempos__tts148.json"
    if f.exists():
        print("\n== q5")
        for r in json.loads(f.read_text(encoding="utf-8")):
            print(r)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    fn = {"clipes": montar_mls, "q1": q1, "q1b": q1b, "pipeline": pipeline, "q1c": q1c, "q1d": q1d, "q2b": q2b, "diag": diag, "q2": q2, "q3": q3, "q4": q4, "q5": q5, "q6": q6, "q6tab": q6tab, "q5par": q5par, "q5par_filho": q5par_filho, "tabelas": tabelas, "q3tab": q3tab, "q2h": lambda: q2h(*sys.argv[2:3]), "rec": rec, "thrchk": thrchk, "conv": conv, "convtab": convtab, "tabela_rec": tabela_rec, "longo": longo, "q3h": q3h, "q4b": q4b, "q4tab": q4tab, "matriz": lambda: matriz(sys.argv[2], *(sys.argv[3:4]))}.get(cmd)
    if not fn:
        sys.exit(__doc__)
    fn()
