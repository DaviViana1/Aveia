"""Gravação em dois canais: microfone (esquerda) e áudio do computador (direita).

Em reunião online, quem fala pelo microfone é "eu" e quem sai pelo computador são "os outros". Com os dois em canais
separados dá para saber quem é quem pelo CANAL, sem depender do timbre. Este módulo é puro (numpy + Silero VAD do
faster-whisper, sem I/O): recebe as duas ondas mono (float32, 16 kHz) e devolve

* `mix`: a soma dos canais, protegida contra estouro, que é o que o Whisper transcreve;
* `trechos_locais`: quando o USUÁRIO (microfone) está falando, em segundos;
* `valido`: False quando os canais não servem (cópias um do outro, ou um deles sem fala): quem chamou trata o arquivo
  como mono.

Regra por quadro de `QUADRO_S` (50 ms):

* `sistema` fala  -> é gente remota (pode ser os dois ao mesmo tempo);
* só o microfone fala -> é o usuário;
* os dois falam (conversa sobreposta, ou eco do alto-falante voltando pelo microfone) -> decide pela energia: o ganho
  de eco `g` é a MEDIANA de E_mic / E_sistema nos quadros em que o sistema fala (com fone, g ~ 0; com alto-falante, g é
  o quanto do som do computador volta pelo microfone). O quadro só é "local" se E_mic > g * E_sistema * MARGEM_ECO.
  O eco também persiste um instante depois que o sistema cala (reverberação, atraso): essa cauda (`CAUDA_ECO_S`) é
  tratada como "sistema falando" para esta comparação, usando a maior energia do sistema na janela.

O resultado é suavizado: trechos locais mais curtos que `LOCAL_MIN_S` somem e pausas menores que `PAUSA_MIN_S` entre
dois trechos locais são fechadas.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

TAXA = 16000
QUADRO_S = 0.05            # s: resolução da análise
MARGEM_ECO_DB = 6.0        # o microfone precisa passar do eco esperado por isto para o quadro ser "local"
MARGEM_ECO = 10 ** (MARGEM_ECO_DB / 10.0)   # em potência
CAUDA_ECO_S = 0.25         # s depois do sistema calar em que ainda pode haver eco no microfone
LOCAL_MIN_S = 0.25         # trecho local mais curto que isto é descartado
PAUSA_MIN_S = 0.30         # pausa menor que isto entre trechos locais é fechada
CORRELACAO_COPIA = 0.95    # |correlação| entre os canais acima disto = um é cópia do outro
RMS_SILENCIO = 1e-4        # (-80 dBFS) canal com RMS abaixo disto é silêncio
FALA_MIN_S = 0.5           # canal com menos fala que isto (VAD) não serve para separar
_ENERGIA_MIN = 1e-10       # piso de energia por quadro (evita divisão por zero)
PICO_MIX = 0.99


@dataclass
class Canais:
    mix: np.ndarray                                   # float32 mono: soma dos canais, sem estourar
    trechos_locais: list[tuple[float, float]] = field(default_factory=list)   # (início, fim) em s: o usuário falando
    valido: bool = True
    motivo: str = ""                                  # por que não é válido (diagnóstico)
    eco: float = 0.0                                  # ganho de eco estimado (E_mic / E_sistema)
    fala_sistema: float = 0.0                         # s de fala no canal do sistema
    fala_mic: float = 0.0                             # s de fala no canal do microfone


def misturar(esq: np.ndarray, dir: np.ndarray) -> np.ndarray:
    """Soma dos canais; se passar de PICO_MIX reduz o ganho inteiro (sem cortar a onda)."""
    n = min(len(esq), len(dir))
    mix = esq[:n].astype(np.float32) + dir[:n].astype(np.float32)
    pico = float(np.max(np.abs(mix))) if n else 0.0
    if pico > PICO_MIX:
        mix *= PICO_MIX / pico
    return mix


def _fala_quadros(audio: np.ndarray, n_quadros: int) -> np.ndarray:
    """Máscara booleana por quadro: há fala (Silero VAD) neste canal? Sem preenchimento, pausas curtas preservadas."""
    from faster_whisper.vad import VadOptions, get_speech_timestamps
    mascara = np.zeros(n_quadros, dtype=bool)
    if audio.size < TAXA // 4:
        return mascara
    opc = VadOptions(min_silence_duration_ms=150, speech_pad_ms=0, min_speech_duration_ms=150)
    quadro = int(QUADRO_S * TAXA)
    for t in get_speech_timestamps(audio, opc, sampling_rate=TAXA):
        a = int(np.ceil(t["start"] / quadro))     # quadros cujo início cai dentro da fala
        b = int(np.ceil(t["end"] / quadro))
        mascara[a:min(b, n_quadros)] = True
    return mascara


def _energia_quadros(audio: np.ndarray, n_quadros: int) -> np.ndarray:
    quadro = int(QUADRO_S * TAXA)
    util = np.zeros(n_quadros * quadro, dtype=np.float64)
    util[:min(len(audio), len(util))] = audio[:len(util)]
    return np.maximum((util.reshape(n_quadros, quadro) ** 2).mean(axis=1), _ENERGIA_MIN)


def _janela_max(e: np.ndarray, largura: int) -> np.ndarray:
    """max(e[i-largura+1 .. i]) para cada i."""
    saida = e.copy()
    for k in range(1, largura):
        np.maximum(saida[k:], e[:-k], out=saida[k:])
    return saida


def _dilatar(m: np.ndarray, quadros: int) -> np.ndarray:
    """Estende cada trecho verdadeiro `quadros` à frente."""
    saida = m.copy()
    for k in range(1, quadros + 1):
        saida[k:] |= m[:-k]
    return saida


def _intervalos(m: np.ndarray) -> list[tuple[int, int]]:
    """Trechos [a, b) de quadros verdadeiros consecutivos."""
    if not m.any():
        return []
    d = np.diff(np.concatenate(([0], m.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(d == 1).tolist(), np.flatnonzero(d == -1).tolist()))


def _suavizar(m: np.ndarray) -> list[tuple[int, int]]:
    """Fecha pausas curtas e descarta trechos curtos."""
    trechos = _intervalos(m)
    pausa = int(round(PAUSA_MIN_S / QUADRO_S))
    juntos: list[list[int]] = []
    for a, b in trechos:
        if juntos and a - juntos[-1][1] < pausa:
            juntos[-1][1] = b
        else:
            juntos.append([a, b])
    minimo = int(round(LOCAL_MIN_S / QUADRO_S))
    return [(a, b) for a, b in juntos if b - a >= minimo]


def _correlacao(a: np.ndarray, b: np.ndarray) -> float:
    a64, b64 = a.astype(np.float64), b.astype(np.float64)
    den = np.sqrt(float(np.dot(a64, a64)) * float(np.dot(b64, b64)))
    return float(np.dot(a64, b64) / den) if den > 0 else 0.0


def separar(esq: np.ndarray, dir: np.ndarray) -> Canais:
    """`esq` = microfone, `dir` = áudio do computador (mono, float32, 16 kHz). Ver a docstring do módulo."""
    n = min(len(esq), len(dir))
    esq, dir = np.asarray(esq[:n], dtype=np.float32), np.asarray(dir[:n], dtype=np.float32)
    mix = misturar(esq, dir)
    if n < TAXA // 2:
        return Canais(mix, [], False, "áudio curto demais")
    rms_e, rms_d = float(np.sqrt(np.mean(esq.astype(np.float64) ** 2))), float(np.sqrt(np.mean(dir.astype(np.float64) ** 2)))
    if rms_e < RMS_SILENCIO:
        return Canais(mix, [], False, "o canal do microfone está em silêncio")
    if rms_d < RMS_SILENCIO:
        return Canais(mix, [], False, "o canal do computador está em silêncio")
    if abs(_correlacao(esq, dir)) >= CORRELACAO_COPIA:
        return Canais(mix, [], False, "os dois canais são cópias um do outro")

    nq = n // int(QUADRO_S * TAXA)
    if nq < 2:
        return Canais(mix, [], False, "áudio curto demais")
    fala_m, fala_s = _fala_quadros(esq, nq), _fala_quadros(dir, nq)
    seg_m, seg_s = float(fala_m.sum()) * QUADRO_S, float(fala_s.sum()) * QUADRO_S
    if seg_s < FALA_MIN_S:
        return Canais(mix, [], False, "sem fala no canal do computador", 0.0, seg_s, seg_m)
    if seg_m < FALA_MIN_S:
        return Canais(mix, [], False, "sem fala no canal do microfone", 0.0, seg_s, seg_m)

    e_m, e_s = _energia_quadros(esq, nq), _energia_quadros(dir, nq)
    razao = e_m[fala_s] / e_s[fala_s]
    eco = float(np.median(razao)) if razao.size else 0.0

    cauda = int(round(CAUDA_ECO_S / QUADRO_S))
    sistema_ativo = _dilatar(fala_s, cauda)                 # sistema falando ou acabou de falar (eco possível)
    e_ref = _janela_max(e_s, cauda + 1)                      # maior energia do sistema na janela recente
    local = fala_m & (~sistema_ativo | (e_m > eco * e_ref * MARGEM_ECO))

    trechos = [(a * QUADRO_S, b * QUADRO_S) for a, b in _suavizar(local)]
    return Canais(mix, trechos, True, "", eco, seg_s, seg_m)
