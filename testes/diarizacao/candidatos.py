"""Candidatos de diarização do benchmark. Cada um é fn(audio_float32_16k, num_locutores|None) -> [(início, fim, id)]."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import sherpa_onnx

MODELOS = Path(__file__).resolve().parents[2] / "models"
PESQ = MODELOS / "pesquisa"
THREADS = 4


def _baseline(audio, k):
    import baseline_antigo
    return [(t.inicio, t.fim, t.locutor) for t in baseline_antigo.diarizar(audio, num_locutores=k)]


def sherpa(seg: Path, emb: Path, limiar: float, threads: int = THREADS):
    """sherpa-onnx puro (sem consolidação): segmentação pyannote + embeddings + clustering aglomerativo."""
    def fn(audio, k):
        cfg = sherpa_onnx.OfflineSpeakerDiarizationConfig(
            segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
                pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(model=str(seg)), num_threads=threads),
            embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(emb), num_threads=threads),
            clustering=sherpa_onnx.FastClusteringConfig(num_clusters=k or -1, threshold=limiar),
            min_duration_on=0.3, min_duration_off=0.5)
        sd = sherpa_onnx.OfflineSpeakerDiarization(cfg)
        return [(r.start, r.end, r.speaker) for r in sd.process(audio).sort_by_start_time()]
    return fn


SEG_PY3 = MODELOS / "sherpa-onnx-pyannote-segmentation-3-0" / "model.onnx"
SEG_REV1 = PESQ / "sherpa-onnx-reverb-diarization-v1" / "model.onnx"
EMB = {
    "titanet_s": MODELOS / "nemo_en_titanet_small.onnx",
    "eres2net_zh": MODELOS / "3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx",
    "wespeaker_r34lm": PESQ / "wespeaker_en_voxceleb_resnet34_LM.onnx",
    "wespeaker_campp": PESQ / "wespeaker_en_voxceleb_CAM++_LM.onnx",
    "3dspk_campp_zhen": PESQ / "3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx",
    "eres2net_en": PESQ / "3dspeaker_speech_eres2net_sv_en_voxceleb_16k.onnx",
    "titanet_l": PESQ / "nemo_en_titanet_large.onnx",
}

def _atual(audio, k):
    """O que o app faz hoje: `transcritor.diarizacao.diarizar` (motor conforme AVEIA_DIARIZACAO; padrão novo, com fallback)."""
    from transcritor import diarizacao
    return [(t.inicio, t.fim, t.locutor) for t in diarizacao.diarizar(audio, num_locutores=k)]


def _antigo(audio, k):
    """O método antigo pelo código atual do app (extrator/diarizador reutilizados): deve reproduzir o `baseline`."""
    from transcritor import diarizacao
    return [(t.inicio, t.fim, t.locutor) for t in diarizacao._diarizar_antigo(audio, k, 0.65, "titanet_small", 4)]


CANDIDATOS = {"baseline": _baseline, "atual": _atual, "antigo": _antigo}


# ---- motor próprio (transcritor/diarizacao_onnx.py), com cache da análise (etapas 1-2) para varrer o clustering
import hashlib  # noqa: E402
import time  # noqa: E402
import dataclasses  # noqa: E402

from transcritor import diarizacao_onnx as D  # noqa: E402

CACHE_AN = Path(__file__).resolve().parent / "cache" / "analise"


def carregar_analise(seg: Path, emb: Path, audio, p) -> tuple["D.Analise", float]:
    """Etapas 1-2 do motor próprio, com cache em disco (segmentação e embeddings separados). Retorna (análise, segundos de cálculo)."""
    h = hashlib.md5(audio[::97].tobytes() + str(len(audio)).encode()).hexdigest()[:10]
    CACHE_AN.mkdir(parents=True, exist_ok=True)
    f_seg = CACHE_AN / f"seg-{seg.parent.name}-{h}-{p.passo_janela}.npz"
    if f_seg.exists():
        z = np.load(f_seg)
        ativ, t_seg = z["ativ"], float(z["t"])
    else:
        t0 = time.perf_counter()
        ativ = D.segmentar(audio, seg, p, THREADS)
        t_seg = time.perf_counter() - t0
        np.savez(f_seg, ativ=ativ, t=t_seg)
    f = CACHE_AN / f"emb-{seg.parent.name}-{emb.stem}-{h}-{p.passo_janela}-{p.min_fala_emb}-{p.min_fala_emb_ov}.npz"
    if f.exists():
        z = np.load(f, allow_pickle=True)
        e, idx, dur, t_emb = z["emb"], [tuple(x) for x in z["idx"]], z["dur"], float(z["t"])
    else:
        t0 = time.perf_counter()
        e, idx, dur = D.extrair_embeddings(audio, ativ, emb, p, THREADS)
        t_emb = time.perf_counter() - t0
        np.savez(f, emb=e, idx=np.array(idx), dur=dur, t=t_emb)
    return D.Analise(len(audio), p.passo_janela, ativ, e, idx, dur), t_seg + t_emb


def onnx_motor(seg: Path, emb: Path, **kw):
    """Candidato do motor próprio. `kw` sobrescreve campos de Params. O tempo reportado inclui a análise (mesmo em cache)."""
    p = dataclasses.replace(D.Params(), **kw)

    def fn(audio, k):
        an, t_an = carregar_analise(seg, emb, audio, p)
        t1 = time.perf_counter()
        trechos, _ = D.diarizar_analise(an, k, p)
        fn.tempo = t_an + (time.perf_counter() - t1)
        return trechos
    return fn


SEGS = {"py3": SEG_PY3, "rev1": SEG_REV1}


def obter(nome: str):
    """Resolve nomes dinâmicos:  baseline | sh-<seg>-<emb>-<limiar> (sherpa puro) | mo-<seg>-<emb>-<limiar>[-chave=valor...] (motor próprio)."""
    if nome in CANDIDATOS:
        return CANDIDATOS[nome]
    partes = nome.split("-")
    tipo, seg, emb, lim, extras = partes[0], partes[1], partes[2], float(partes[3]), partes[4:]
    if tipo == "sh":
        CANDIDATOS[nome] = sherpa(SEGS[seg], EMB[emb], lim)
    else:
        kw = {"limiar": lim}
        for e in extras:
            k, v = e.split("=")
            kw[k] = type(getattr(D.Params(), k))(float(v)) if not isinstance(getattr(D.Params(), k), bool) else bool(int(v))
        CANDIDATOS[nome] = onnx_motor(SEGS[seg], EMB[emb], **kw)
    return CANDIDATOS[nome]
