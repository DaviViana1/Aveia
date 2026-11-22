"""Transcrição: entrega palavras com tempo (`Palavra`), que o pipeline junta com a diarização no mesmo contrato
`Fala(inicio, fim, locutor, texto)`.

O único motor é o faster-whisper `large-v3-turbo` na CPU: o áudio não sai do computador. O módulo mantém o nome e o
parâmetro `motor` por compatibilidade com a API e com as notas já salvas no navegador, mas qualquer valor vira `local`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterator

import numpy as np

SAMPLE_RATE = 16000


@dataclass
class Palavra:
    start: float
    end: float
    word: str          # com o espaço inicial, como no faster-whisper


def normalizar_motor(pedido: str | None = None) -> str:
    """Só existe o motor local; qualquer pedido (inclusive de versões antigas) vira `local`."""
    return "local"


def _ponto_quieto(audio: np.ndarray, alvo_s: float, raio_s: float = 8.0) -> float:
    """Instante de menor energia perto de `alvo_s` (para cortar o bloco no meio de uma pausa, não de uma palavra)."""
    a, b = max(0, int((alvo_s - raio_s) * SAMPLE_RATE)), min(len(audio), int((alvo_s + raio_s) * SAMPLE_RATE))
    jan = int(0.1 * SAMPLE_RATE)
    n = (b - a) // jan
    if n < 2:
        return alvo_s
    e = (audio[a:a + n * jan].reshape(n, jan) ** 2).mean(axis=1)
    return (a + (int(np.argmin(e)) + 0.5) * jan) / SAMPLE_RATE


# ---- despacho --------------------------------------------------------------------------------------------
def palavras(audio: np.ndarray, duracao: float, idioma: str | None, avisar: Callable[[float, str], None],
             motor: str | None = None, meta: dict | None = None, glossario: str | None = None) -> Iterator[Palavra]:
    """Gera as palavras (com tempo) do áudio com o Whisper local; `meta` recebe o motor usado, avisos e lacunas.

    `glossario` ajuda o Whisper a acertar a grafia de nomes e termos."""
    from . import pipeline   # import tardio: o pipeline importa este módulo

    meta = meta if meta is not None else {}
    meta.update(motor_pedido="local", motor="local", avisos=meta.get("avisos", []), lacunas=meta.get("lacunas", []))

    extra = {"glossario": glossario} if glossario else {}   # sem glossário a chamada é idêntica à de antes dele
    yield from pipeline._palavras(audio, duracao, idioma, avisar, lacunas=meta["lacunas"], **extra)
