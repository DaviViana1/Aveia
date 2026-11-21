"""Diarização (quem falou quando). Roda em CPU, sem token e sem torch.

Dois motores, escolhidos por `AVEIA_DIARIZACAO=novo|antigo` (padrão: novo):
- novo (`diarizacao_onnx.py`): segmentação pyannote 3.0 em janelas de 10 s + embeddings 3D-Speaker CAM++ por
  (janela, locutor local) + agrupamento Ward com estimativa automática do nº de locutores. Se falhar ou não
  devolver nada, cai sozinho no antigo.
- antigo: `OfflineSpeakerDiarization` do sherpa-onnx (pyannote 3.0 + TitaNet small + FastClustering) + `_consolidar`.
"""
from __future__ import annotations

import os
import tarfile
import threading
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import sherpa_onnx

from . import diarizacao_onnx as _onnx

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
_BASE = "https://github.com/k2-fsa/sherpa-onnx/releases/download"
SEGMENTATION_URL = f"{_BASE}/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2"
EMBEDDINGS = {
    "titanet_small": f"{_BASE}/speaker-recongition-models/nemo_en_titanet_small.onnx",
    "eres2net": f"{_BASE}/speaker-recongition-models/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx",
    "campp_zhen": f"{_BASE}/speaker-recongition-models/3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx",
}
EMBEDDING_NOVO = "campp_zhen"   # embedding do motor novo (escolhido no benchmark: testes/diarizacao/ e README)


@dataclass
class Trecho:
    inicio: float
    fim: float
    locutor: int


def _baixar(url: str, destino: Path) -> None:
    destino.parent.mkdir(parents=True, exist_ok=True)
    print(f"Baixando {url} ...")
    parcial = destino.with_name(destino.name + ".part")   # só vira o arquivo final quando o download termina inteiro
    try:
        urllib.request.urlretrieve(url, parcial)
        os.replace(parcial, destino)
    finally:
        parcial.unlink(missing_ok=True)


def _caminho_segmentacao() -> Path:
    return MODELS_DIR / "sherpa-onnx-pyannote-segmentation-3-0" / "model.onnx"


def _caminho_embedding(nome: str) -> Path:
    return MODELS_DIR / Path(EMBEDDINGS[nome]).name


def _modelo_segmentacao() -> Path:
    onnx = _caminho_segmentacao()
    if not onnx.exists():
        tar = MODELS_DIR / "segmentation.tar.bz2"
        _baixar(SEGMENTATION_URL, tar)
        with tarfile.open(tar) as t:
            t.extractall(MODELS_DIR, filter="data")
        tar.unlink()
    return onnx


def _modelo_embedding(nome: str) -> Path:
    arquivo = _caminho_embedding(nome)
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


# Método antigo: extrator e diarizador são criados uma vez e reutilizados (antes eram refeitos a cada chamada).
_ANTIGO: dict = {}
_TRAVA_ANTIGO = threading.RLock()   # os objetos do sherpa não são documentados como thread-safe: uma chamada por vez


def _extrator_antigo(embedding: str, threads: int):
    chave = ("ext", embedding, threads)
    if chave not in _ANTIGO:
        _ANTIGO[chave] = sherpa_onnx.SpeakerEmbeddingExtractor(
            sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(_modelo_embedding(embedding)), num_threads=threads)
        )
    return _ANTIGO[chave]


def _consolidar(audio: np.ndarray, trechos: list[Trecho], embedding: str, threads: int) -> list[Trecho]:
    """Absorve locutores minúsculos e funde os que têm o mesmo timbre."""
    por_loc: dict[int, list[Trecho]] = {}
    for t in trechos:
        por_loc.setdefault(t.locutor, []).append(t)
    if len(por_loc) < 2:
        return trechos

    extrator = _extrator_antigo(embedding, threads)
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


def _diarizar_novo(audio: np.ndarray, num_locutores: int | None, threads: int) -> list[Trecho]:
    """Motor novo (diarizacao_onnx.py). Modelos resolvidos/baixados como os demais; sessões ONNX reutilizadas."""
    trechos = _onnx.diarizar_audio(
        audio, num_locutores, _modelo_segmentacao(), _modelo_embedding(EMBEDDING_NOVO), _onnx.Params(), threads
    )
    return [Trecho(a, b, g) for a, b, g in trechos]


def analise_incremental(threads: int = 4):
    """`AnaliseIncremental` pronta para o modo ao vivo, ou None. Mesmos modelos, `Params()` e `threads` de `_diarizar_novo`
    (o resultado é idêntico ao do lote). Devolve None se o motor não é o novo, se `AVEIA_DIAR_AOVIVO=0` (padrão: ligado) ou se
    falta algum modelo em disco: durante uma gravação nunca se baixa nada. Nunca levanta exceção."""
    try:
        if _motor() != "novo":
            return None
        if (os.environ.get("AVEIA_DIAR_AOVIVO") or "1").strip() == "0":
            return None
        seg, emb = _caminho_segmentacao(), _caminho_embedding(EMBEDDING_NOVO)
        faltam = [str(c) for c in (seg, emb) if not c.exists()]
        if faltam:
            print(f"Diarização ao vivo desligada: modelo ausente ({', '.join(faltam)}); a diarização roda no fim.")
            return None
        return _onnx.AnaliseIncremental(seg, emb, _onnx.Params(), threads)
    except Exception as e:  # noqa: BLE001
        print(f"Diarização ao vivo desligada ({type(e).__name__}: {e}); a diarização roda no fim.")
        return None


def _motor() -> str:
    """`AVEIA_DIARIZACAO=novo|antigo` (lido a cada chamada); qualquer outro valor vale como o padrão, `novo`."""
    valor = (os.environ.get("AVEIA_DIARIZACAO") or "novo").strip().lower()
    if valor not in ("novo", "antigo"):
        print(f"AVEIA_DIARIZACAO={valor!r} desconhecido (use novo ou antigo); usando novo.")
        return "novo"
    return valor


def diarizar(
    audio: np.ndarray,
    num_locutores: int | None = None,
    limiar: float = 0.65,
    embedding: str = "titanet_small",
    threads: int = 4,
) -> list[Trecho]:
    """Recebe áudio mono float32 a 16 kHz. Se `num_locutores` for None, estima sozinho.
    Motor novo por padrão (`AVEIA_DIARIZACAO=antigo` volta ao método anterior); se o novo falhar ou não devolver
    nada, usa o antigo. `limiar` e `embedding` valem só para o método antigo (`limiar` maior => menos locutores)."""
    global _ultimo
    if _motor() == "novo":
        _ultimo = ("novo", "")
        try:
            trechos = _diarizar_novo(audio, num_locutores, threads)
            if trechos:
                return trechos
            print("Diarização nova não devolveu trechos; usando o método antigo.")
            _ultimo = ("antigo", "o método novo não devolveu trechos")
        except Exception as e:  # noqa: BLE001 - qualquer falha (modelo ausente, rede, ONNX) cai no método antigo
            print(f"Diarização nova falhou ({type(e).__name__}: {e}); usando o método antigo.")
            _ultimo = ("antigo", f"{type(e).__name__}: {e}")
    else:
        _ultimo = ("antigo", "")   # escolha de AVEIA_DIARIZACAO=antigo, sem aviso
    return _diarizar_antigo(audio, num_locutores, limiar, embedding, threads)


_ultimo: tuple[str, str] = ("novo", "")


def ultimo_motor() -> tuple[str, str]:
    """(motor, motivo) da última chamada de `diarizar`: `("antigo", "ModuleNotFoundError: ...")` quando o método novo falhou e o
    antigo assumiu (a interface avisa: o antigo erra mais a contagem de pessoas), `("antigo", "")` quando foi escolha."""
    return _ultimo


def _diarizar_antigo(audio: np.ndarray, num_locutores: int | None, limiar: float, embedding: str, threads: int) -> list[Trecho]:
    """Método anterior: sherpa-onnx `OfflineSpeakerDiarization` + `_consolidar` (sem K)."""
    with _TRAVA_ANTIGO:
        return _diarizar_antigo_sem_trava(audio, num_locutores, limiar, embedding, threads)


def _diarizar_antigo_sem_trava(audio, num_locutores, limiar, embedding, threads) -> list[Trecho]:
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
    chave = ("sd", embedding, threads)
    sd = _ANTIGO.get(chave)
    if sd is None:
        sd = _ANTIGO[chave] = sherpa_onnx.OfflineSpeakerDiarization(config)
    else:
        sd.set_config(config)   # só o agrupamento muda entre chamadas (nº de locutores, limiar)
    resultado = sd.process(audio).sort_by_start_time()
    trechos = [Trecho(r.start, r.end, r.speaker) for r in resultado]
    return trechos if num_locutores else _consolidar(audio, trechos, embedding, threads)
