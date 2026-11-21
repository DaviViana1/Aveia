"""Pré-processamento de áudio antes do Whisper e da diarização.

Entrada: ondas mono float32 a 16 kHz (já decodificadas por `decode_audio`). Etapas (todas opcionais,
controladas por `AVEIA_PREPROC`, ver `configuracao()`):

* ``hpf``   — passa-altas Butterworth de 2ª ordem em ~80 Hz (zumbido de rede, vento, pancadas na mesa).
* ``norm``  — normalização por nível de fala: leva o RMS dos trechos com voz a ~-23 dBFS (ganho limitado a
              +30 dB / -12 dB) e protege o pico com um limitador suave. Só amplifica; não "levanta" ruído em
              áudio sem fala.
* ``denoise`` — redução de ruído espectral (`afftdn` do FFmpeg/PyAV). Desligado por padrão: ver README
              (a medição de WER não mostrou ganho com o Whisper large-v3-turbo).

Sem dependências novas: usa numpy e os filtros do PyAV (FFmpeg) que o faster-whisper já traz.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

SAMPLE_RATE = 16000
ALVO_RMS_DB = -23.0          # nível-alvo da fala (RMS dos quadros ativos)
GANHO_MAX_DB = 30.0
GANHO_MIN_DB = -12.0
PICO_MAX = 0.97


@dataclass(frozen=True)
class Config:
    hpf: bool = True
    norm: bool = True
    denoise: bool = False
    hpf_hz: float = 80.0


def configuracao() -> Config:
    """`AVEIA_PREPROC`: `0`/`off` desliga tudo; lista separada por vírgula (`hpf,norm,denoise`) escolhe etapas.
    Padrão: desligado (`AVEIA_PREPROC=hpf,norm` liga passa-altas + normalização)."""
    bruto = os.environ.get("AVEIA_PREPROC", "").strip().lower()
    if not bruto:
        return Config(hpf=False, norm=False, denoise=False)   # desligado por padrão até a medição em áudio limpo (ver README)
    if bruto in {"0", "off", "nao", "não", "false", "none"}:
        return Config(hpf=False, norm=False, denoise=False)
    etapas = {e.strip() for e in bruto.split(",")}
    return Config(hpf="hpf" in etapas or "1" in etapas, norm="norm" in etapas or "1" in etapas, denoise="denoise" in etapas)


# ---------------------------------------------------------------- medição

def _db(x: float) -> float:
    return 20.0 * np.log10(max(float(x), 1e-9))


def medir(audio: np.ndarray) -> dict:
    """Pico, RMS da fala (quadros de 30 ms acima de -50 dB do pico... ver abaixo), % de amostras saturadas."""
    if audio.size == 0:
        return {"pico_db": -120.0, "fala_rms_db": -120.0, "saturado_pct": 0.0, "duracao": 0.0}
    pico = float(np.max(np.abs(audio)))
    quadros = _rms_quadros(audio)
    fala = _rms_fala(quadros)
    return {
        "pico_db": float(round(_db(pico), 1)),
        "fala_rms_db": float(round(_db(fala), 1)),
        "saturado_pct": round(float(np.mean(np.abs(audio) >= 0.995)) * 100, 3),
        "duracao": round(audio.size / SAMPLE_RATE, 2),
    }


def _rms_quadros(audio: np.ndarray, ms: int = 30) -> np.ndarray:
    n = SAMPLE_RATE * ms // 1000
    k = audio.size // n
    if k == 0:
        return np.array([np.sqrt(np.mean(audio ** 2))]) if audio.size else np.zeros(1)
    q = audio[: k * n].reshape(k, n).astype(np.float64)
    return np.sqrt(np.mean(q ** 2, axis=1))


def _rms_fala(rms_quadros: np.ndarray) -> float:
    """RMS (energia média) dos quadros de fala. Limiar adaptativo: 12 dB abaixo do percentil 95 dos quadros,
    nunca abaixo do piso de ruído (percentil 10) + 6 dB, para não contar silêncio/ruído como fala."""
    if rms_quadros.size == 0:
        return 0.0
    p95 = np.percentile(rms_quadros, 95)
    piso = np.percentile(rms_quadros, 10)
    limiar = max(p95 * 10 ** (-12 / 20), piso * 10 ** (6 / 20))
    ativos = rms_quadros[rms_quadros >= limiar]
    if ativos.size == 0:
        ativos = rms_quadros
    return float(np.sqrt(np.mean(ativos ** 2)))


# ---------------------------------------------------------------- etapas

def _filtrar_av(audio: np.ndarray, filtros: list[str]) -> np.ndarray:
    """Passa o áudio (mono f32 16 kHz) por um grafo de filtros do FFmpeg e devolve o resultado, com o mesmo tamanho."""
    from fractions import Fraction  # noqa: PLC0415

    import av  # noqa: PLC0415  (import tardio: só quando há filtro)

    g = av.filter.Graph()
    src = g.add_abuffer(sample_rate=SAMPLE_RATE, format="flt", layout="mono", time_base=Fraction(1, SAMPLE_RATE))
    ultimo = src
    for f in filtros:
        nome, _, args = f.partition("=")
        no = g.add(nome, args or None)
        ultimo.link_to(no)
        ultimo = no
    sink = g.add("abuffersink")
    ultimo.link_to(sink)
    g.configure()

    saida: list[np.ndarray] = []
    passo = SAMPLE_RATE  # 1 s por quadro
    pts = 0

    def coletar():
        while True:
            try:
                fr = sink.pull()
            except (av.error.BlockingIOError, av.error.EOFError):
                return
            saida.append(fr.to_ndarray().reshape(-1))

    for i in range(0, audio.size, passo):
        bloco = np.ascontiguousarray(audio[i:i + passo], dtype=np.float32).reshape(1, -1)
        fr = av.AudioFrame.from_ndarray(bloco, format="flt", layout="mono")
        fr.sample_rate = SAMPLE_RATE
        fr.pts = pts
        fr.time_base = Fraction(1, SAMPLE_RATE)
        pts += bloco.shape[1]
        src.push(fr)
        coletar()
    src.push(None)
    coletar()
    out = np.concatenate(saida) if saida else np.zeros(0, np.float32)
    # alguns filtros acrescentam/atrasam amostras: normaliza o tamanho
    if out.size >= audio.size:
        return out[: audio.size].astype(np.float32, copy=False)
    return np.pad(out, (0, audio.size - out.size)).astype(np.float32, copy=False)


def passa_altas(audio: np.ndarray, hz: float = 80.0) -> np.ndarray:
    return _filtrar_av(audio, [f"highpass=f={hz}:poles=2"])


def reduzir_ruido(audio: np.ndarray, reducao_db: float = 12.0) -> np.ndarray:
    """Redução espectral (afftdn). `nr` = quanto abaixar o ruído estimado (dB); `nt=w` = ruído branco; adaptativo."""
    return _filtrar_av(audio, [f"afftdn=nr={reducao_db}:nf=-40:tn=1"])


def normalizar(audio: np.ndarray, alvo_db: float = ALVO_RMS_DB) -> np.ndarray:
    """Ganho único para a gravação toda, calculado no nível da fala; limitador suave no pico."""
    if audio.size == 0:
        return audio
    fala = _rms_fala(_rms_quadros(audio))
    if fala < 1e-6:           # silêncio digital: nada a fazer
        return audio
    ganho_db = float(np.clip(alvo_db - _db(fala), GANHO_MIN_DB, GANHO_MAX_DB))
    saida = audio * np.float32(10 ** (ganho_db / 20))
    pico = float(np.max(np.abs(saida)))
    if pico > PICO_MAX:
        # compressão suave só do que passa de 0,6 (tanh), sem alterar o restante
        joelho = 0.6
        acima = np.abs(saida) > joelho
        s = np.sign(saida[acima])
        saida[acima] = s * (joelho + (PICO_MAX - joelho) * np.tanh((np.abs(saida[acima]) - joelho) / (PICO_MAX - joelho)))
    return saida.astype(np.float32, copy=False)


def preparar(audio: np.ndarray, cfg: Config | None = None) -> np.ndarray:
    """Aplica as etapas ligadas, nesta ordem: passa-altas → redução de ruído → normalização."""
    cfg = cfg or configuracao()
    if audio.size == 0 or not (cfg.hpf or cfg.norm or cfg.denoise):
        return audio
    saida = audio
    try:
        if cfg.hpf:
            saida = passa_altas(saida, cfg.hpf_hz)
        if cfg.denoise:
            saida = reduzir_ruido(saida)
        if cfg.norm:
            saida = normalizar(saida)
    except Exception as e:   # pré-processamento nunca pode impedir a transcrição
        print(f"Pré-processamento de áudio ignorado: {e}")
        return audio
    return saida
