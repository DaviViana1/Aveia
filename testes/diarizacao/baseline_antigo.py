"""CÓPIA CONGELADA da diarização original do Aveia (sherpa-onnx + titanet_small + _consolidar), baseline do benchmark."""
from __future__ import annotations

import os
import tarfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import sherpa_onnx

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
_BASE = "https://github.com/k2-fsa/sherpa-onnx/releases/download"
SEGMENTATION_URL = f"{_BASE}/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2"
EMBEDDINGS = {
    "titanet_small": f"{_BASE}/speaker-recongition-models/nemo_en_titanet_small.onnx",
    "eres2net": f"{_BASE}/speaker-recongition-models/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx",
}


@dataclass
class Trecho:
    inicio: float
    fim: float
    locutor: int


def _baixar(url: str, destino: Path) -> None:
    destino.parent.mkdir(parents=True, exist_ok=True)
    print(f"Baixando {url} ...")
    urllib.request.urlretrieve(url, destino)


def _modelo_segmentacao() -> Path:
    onnx = MODELS_DIR / "sherpa-onnx-pyannote-segmentation-3-0" / "model.onnx"
    if not onnx.exists():
        tar = MODELS_DIR / "segmentation.tar.bz2"
        _baixar(SEGMENTATION_URL, tar)
        with tarfile.open(tar) as t:
            t.extractall(MODELS_DIR, filter="data")
        tar.unlink()
    return onnx


def _modelo_embedding(nome: str) -> Path:
    arquivo = MODELS_DIR / Path(EMBEDDINGS[nome]).name
    if not arquivo.exists():
        _baixar(EMBEDDINGS[nome], arquivo)
    return arquivo


SAMPLE_RATE = 16000
# Ruído, sobreposição e falas curtas fazem o agrupamento automático dividir uma pessoa em vários
# "locutores". Na estimativa automática, `_consolidar` reagrupa pelo timbre depois da diarização.
MIN_FALA_LOCUTOR = 10.0   # s: abaixo disso (e de 4% da fala total) o locutor é ruído e é absorvido
MIN_FRACAO_LOCUTOR = 0.04
SIMILARIDADE_FUSAO = float(os.environ.get("AVEIA_FUSAO_LOCUTORES") or 0.6)   # cosseno entre timbres


def _vetor_timbre(extrator, audio: np.ndarray, trechos: list[Trecho]) -> np.ndarray | None:
    """Timbre médio de um locutor, a partir dos trechos mais longos dele."""
    longos = sorted((t for t in trechos if t.fim - t.inicio >= 0.8), key=lambda t: t.inicio - t.fim)[:8]
    if not longos:
        longos = sorted(trechos, key=lambda t: t.inicio - t.fim)[:1]
    vetores = []
    for t in longos:
        a = int(t.inicio * SAMPLE_RATE)
        amostras = audio[a : a + int(min(t.fim - t.inicio, 15.0) * SAMPLE_RATE)]
        if len(amostras) < 0.4 * SAMPLE_RATE:
            continue
        s = extrator.create_stream()
        s.accept_waveform(SAMPLE_RATE, amostras)
        s.input_finished()
        v = np.array(extrator.compute(s), dtype=np.float64)
        vetores.append(v / (np.linalg.norm(v) or 1.0))
    if not vetores:
        return None
    m = np.mean(vetores, axis=0)
    return m / (np.linalg.norm(m) or 1.0)


def _consolidar(audio: np.ndarray, trechos: list[Trecho], embedding: str, threads: int) -> list[Trecho]:
    """Absorve locutores minúsculos e funde os que têm o mesmo timbre."""
    por_loc: dict[int, list[Trecho]] = {}
    for t in trechos:
        por_loc.setdefault(t.locutor, []).append(t)
    if len(por_loc) < 2:
        return trechos

    extrator = sherpa_onnx.SpeakerEmbeddingExtractor(
        sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(_modelo_embedding(embedding)), num_threads=threads)
    )
    tempo = {k: sum(t.fim - t.inicio for t in v) for k, v in por_loc.items()}
    vetor = {k: _vetor_timbre(extrator, audio, v) for k, v in por_loc.items()}
    total = sum(tempo.values())
    minimo = max(MIN_FALA_LOCUTOR, MIN_FRACAO_LOCUTOR * total)

    def sim(a: int, b: int) -> float:
        return -1.0 if vetor[a] is None or vetor[b] is None else float(vetor[a] @ vetor[b])

    destino = {k: k for k in por_loc}   # rótulo original -> rótulo final

    def fundir(origem: int, alvo: int) -> None:
        for k, d in destino.items():
            if d == origem:
                destino[k] = alvo
        peso_alvo, peso_origem = tempo[alvo], tempo.pop(origem)
        if vetor[alvo] is not None and vetor[origem] is not None:
            m = vetor[alvo] * peso_alvo + vetor[origem] * peso_origem
            vetor[alvo] = m / (np.linalg.norm(m) or 1.0)
        elif vetor[alvo] is None:
            vetor[alvo] = vetor[origem]
        del vetor[origem]
        tempo[alvo] = peso_alvo + peso_origem

    # 1) locutores minúsculos vão para o mais parecido entre os que sobram
    for k in sorted(tempo, key=tempo.get):
        grandes = [j for j in tempo if j != k and tempo[j] >= minimo]
        if tempo[k] < minimo and grandes:
            fundir(k, max(grandes, key=lambda j: sim(k, j)))
    # 2) fusão por semelhança de timbre
    while len(tempo) > 1:
        a, b = max(((x, y) for x in tempo for y in tempo if x < y), key=lambda p: sim(*p))
        if sim(a, b) < SIMILARIDADE_FUSAO:
            break
        fundir(b, a)

    return [Trecho(t.inicio, t.fim, destino[t.locutor]) for t in trechos]


def diarizar(
    audio: np.ndarray,
    num_locutores: int | None = None,
    limiar: float = 0.65,
    embedding: str = "titanet_small",
    threads: int = 4,
) -> list[Trecho]:
    """Recebe áudio mono float32 a 16 kHz. Se `num_locutores` for None, estima sozinho
    (`limiar` maior => menos locutores) e consolida os locutores parecidos."""
    config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                model=str(_modelo_segmentacao())
            ),
            num_threads=threads,
        ),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=str(_modelo_embedding(embedding)), num_threads=threads
        ),
        clustering=sherpa_onnx.FastClusteringConfig(
            num_clusters=num_locutores or -1, threshold=limiar
        ),
        min_duration_on=0.3,
        min_duration_off=0.5,
    )
    if not config.validate():
        raise RuntimeError("Configuração de diarização inválida (modelos ausentes?).")
    sd = sherpa_onnx.OfflineSpeakerDiarization(config)
    resultado = sd.process(audio).sort_by_start_time()
    trechos = [Trecho(r.start, r.end, r.speaker) for r in resultado]
    return trechos if num_locutores else _consolidar(audio, trechos, embedding, threads)
