"""Gera variantes degradadas de um áudio de referência (simulam microfones/ambientes ruins).

Uso (da raiz do projeto):
  .venv\\Scripts\\python testes\\microfone\\degradar.py <entrada.wav> <pasta_saida>

Variantes (WAV 16 kHz mono, 16 bits — a quantização faz parte da degradação):
  limpo        original
  ganho-30db   -30 dB (microfone muito baixo)
  ruido        ruído de fundo (rosa + "murmúrio") com SNR de 10 dB
  banda8k      telefone/Bluetooth HFP: 300-3400 Hz (CVSD 8 kHz) e volta a 16 kHz
  clipping     +24 dB e corte duro em ±1 (microfone saturado)
  eco          reverberação leve (atrasos de 90/190/310 ms)
"""
from __future__ import annotations

import sys
import wave
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ))
SR = 16000


def ler(caminho: str) -> np.ndarray:
    from faster_whisper.audio import decode_audio
    return decode_audio(caminho, sampling_rate=SR)


def gravar(caminho: Path, x: np.ndarray) -> None:
    pcm = (np.clip(x, -1, 1) * 32767).round().astype("<i2")
    with wave.open(str(caminho), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes(pcm.tobytes())


def _fft_banda(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1 / SR)
    # bordas suaves (cosseno de 100 Hz) para não soar como "parede de tijolo"
    g = np.clip((f - (lo - 100)) / 100, 0, 1) * np.clip(((hi + 100) - f) / 100, 0, 1)
    return np.fft.irfft(X * g, n=len(x)).astype(np.float32)


def _ruido(n: int, rng: np.random.Generator) -> np.ndarray:
    branco = rng.standard_normal(n).astype(np.float32)
    f = np.fft.rfftfreq(n, 1 / SR); f[0] = f[1]
    rosa = np.fft.irfft(np.fft.rfft(branco) / np.sqrt(f), n=n).astype(np.float32)
    return rosa / (np.std(rosa) + 1e-9)


def _rms_fala(x: np.ndarray) -> float:
    q = x[: len(x) // 480 * 480].reshape(-1, 480)
    r = np.sqrt((q.astype(np.float64) ** 2).mean(axis=1))
    return float(np.sqrt((r[r >= np.percentile(r, 95) * 0.25] ** 2).mean()))


def variantes(x: np.ndarray) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(7)
    x = x / max(1e-9, float(np.max(np.abs(x)))) * 0.7       # referência a -3 dBFS de pico
    fala = _rms_fala(x)
    ruido = _ruido(len(x), rng)
    ruido *= fala / 10 ** (10 / 20)                            # SNR 10 dB sobre a fala
    eco = x.copy()
    for atraso_ms, g in ((90, 0.35), (190, 0.22), (310, 0.14)):
        a = int(SR * atraso_ms / 1000); eco[a:] += g * x[:-a]
    return {
        "limpo": x,
        "ganho-30db": x * 10 ** (-30 / 20),
        "ruido": x + ruido,
        "banda8k": _fft_banda(x, 300, 3400),
        "clipping": np.clip(x * 10 ** (24 / 20), -1, 1),
        "eco": eco / max(1.0, float(np.max(np.abs(eco))) / 0.9),
    }


if __name__ == "__main__":
    entrada, saida = sys.argv[1], Path(sys.argv[2])
    saida.mkdir(parents=True, exist_ok=True)
    for nome, v in variantes(ler(entrada)).items():
        gravar(saida / f"{nome}.wav", v)
        print(nome, f"pico {20 * np.log10(np.max(np.abs(v)) + 1e-9):.1f} dBFS")
