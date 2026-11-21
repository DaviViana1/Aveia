"""Motores de transcrição plugáveis. Todos entregam palavras com tempo (`Palavra`), que o pipeline
junta com a diarização no mesmo contrato `Fala(inicio, fim, locutor, texto)`.

- `local` (padrão): faster-whisper `large-v3-turbo` na CPU. O áudio não sai do computador.
- `groq`  (opt-in): API da Groq (Whisper na nuvem). O ÁUDIO É ENVIADO à Groq. Se falhar (sem chave, chave
  inválida, cota/429, rede, timeout), o trecho que faltou é transcrito pelo motor local e o aviso aparece na UI.

Escolha: parâmetro `motor` da API (Ajustes) ou variável de ambiente `AVEIA_MOTOR`; sem nenhum dos dois, `local`.
"""
from __future__ import annotations

import io
import json
import os
import socket
import time
import urllib.error
import urllib.request
import uuid
from collections import deque
from dataclasses import dataclass
from typing import Callable, Iterator

import numpy as np

SAMPLE_RATE = 16000

MOTORES = {
    "local": "Whisper local (privado, no computador)",
    "groq": "Groq Whisper (nuvem, envia o áudio)",
}

# ---- Groq ------------------------------------------------------------------------------------------------
GROQ_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
GROQ_MODELO = "whisper-large-v3-turbo"
BLOCO_S = 600.0            # duração nominal de cada requisição (FLAC 16 kHz mono ≈ 10–17 MB; limite free: 25 MB)
SOBRA_S = 2.5              # áudio extra de cada lado de um bloco (contexto); as palavras são atribuídas por tempo
MAX_BYTES = 24 * 1024 * 1024
TIMEOUT_S = 180
TENTATIVAS = 4
ESPERA_MAX_S = 75.0        # acima disso o 429 é tratado como cota esgotada (ex.: limite por hora/dia) → fallback
GROQ_ASH_PADRAO = 7200     # audio-seconds/hour do plano free (console.groq.com/docs/rate-limits, 2026-11-05)


@dataclass
class Palavra:
    start: float
    end: float
    word: str          # com o espaço inicial, como no faster-whisper


class MotorIndisponivel(Exception):
    """O motor não conseguiu transcrever; a mensagem (pt-BR) vai para o aviso mostrado ao usuário."""


class SemChave(MotorIndisponivel):
    pass


class ChaveInvalida(MotorIndisponivel):
    pass


class CotaEsgotada(MotorIndisponivel):
    pass


class FalhaDeRede(MotorIndisponivel):
    pass


class _BlocoGrande(Exception):
    pass


def _env(nome: str, padrao: str = "") -> str:
    return os.environ.get(nome, padrao).strip()


def motor_padrao() -> str:
    m = _env("AVEIA_MOTOR", "local").lower()
    return m if m in MOTORES else "local"


def normalizar_motor(pedido: str | None) -> str:
    m = (pedido or "").strip().lower()
    return m if m in MOTORES else motor_padrao()


def groq_configurado() -> bool:
    return bool(_env("GROQ_API_KEY"))


def groq_modelo() -> str:
    return _env("AVEIA_GROQ_MODELO", GROQ_MODELO)


# ---- controle de uso (aproximado, só enquanto o app está aberto) ------------------------------------------
_uso: deque[tuple[float, float]] = deque()   # (instante, segundos de áudio enviados)


def _registrar_uso(segundos: float) -> None:
    _uso.append((time.time(), segundos))


def uso_ultima_hora() -> float:
    corte = time.time() - 3600
    while _uso and _uso[0][0] < corte:
        _uso.popleft()
    return sum(s for _, s in _uso)


def limite_hora() -> float:
    try:
        return float(_env("AVEIA_GROQ_ASH", str(GROQ_ASH_PADRAO)))
    except ValueError:
        return float(GROQ_ASH_PADRAO)


def estado() -> dict:
    """Para a UI (Ajustes): motores disponíveis e uso estimado da cota gratuita da Groq."""
    return {
        "padrao": motor_padrao(),
        "motores": [
            {"id": "local", "nome": MOTORES["local"], "nuvem": False, "disponivel": True},
            {"id": "groq", "nome": MOTORES["groq"], "nuvem": True, "disponivel": groq_configurado(),
             "modelo": groq_modelo(),
             "motivo": "" if groq_configurado() else "Defina GROQ_API_KEY no arquivo .env."},
        ],
        "groq_uso_hora_s": round(uso_ultima_hora()),
        "groq_limite_hora_s": round(limite_hora()),
    }


# ---- áudio -----------------------------------------------------------------------------------------------
def _flac(x: np.ndarray) -> bytes:
    """FLAC 16 kHz mono (sem perdas: o que a Groq recomenda para reduzir o tamanho)."""
    import av

    buf = io.BytesIO()
    pcm = (np.clip(x, -1.0, 1.0) * 32767).astype(np.int16).reshape(1, -1)
    with av.open(buf, "w", format="flac") as c:
        st = c.add_stream("flac", rate=SAMPLE_RATE)
        st.layout = "mono"
        fr = av.AudioFrame.from_ndarray(pcm, format="s16", layout="mono")
        fr.sample_rate = SAMPLE_RATE
        for p in st.encode(fr):
            c.mux(p)
        for p in st.encode(None):
            c.mux(p)
    return buf.getvalue()


def _ponto_quieto(audio: np.ndarray, alvo_s: float, raio_s: float = 8.0) -> float:
    """Instante de menor energia perto de `alvo_s` (para cortar o bloco no meio de uma pausa, não de uma palavra)."""
    a, b = max(0, int((alvo_s - raio_s) * SAMPLE_RATE)), min(len(audio), int((alvo_s + raio_s) * SAMPLE_RATE))
    jan = int(0.1 * SAMPLE_RATE)
    n = (b - a) // jan
    if n < 2:
        return alvo_s
    e = (audio[a:a + n * jan].reshape(n, jan) ** 2).mean(axis=1)
    return (a + (int(np.argmin(e)) + 0.5) * jan) / SAMPLE_RATE


def _bloco_s() -> float:
    try:
        return max(20.0, float(_env("AVEIA_GROQ_BLOCO", str(BLOCO_S))))
    except ValueError:
        return BLOCO_S


def _cortes(audio: np.ndarray, duracao: float) -> list[tuple[float, float]]:
    """Divide em blocos [início, fim) de ~BLOCO_S, cortando em pausas. Áudio curto = um bloco só."""
    alvo = _bloco_s()
    if duracao <= alvo * 1.2:
        return [(0.0, duracao)]
    pontos = [0.0]
    while duracao - pontos[-1] > alvo * 1.2:
        pontos.append(_ponto_quieto(audio, pontos[-1] + alvo, raio_s=min(8.0, alvo / 4)))
    pontos.append(duracao)
    return list(zip(pontos[:-1], pontos[1:]))


def _tem_fala(audio: np.ndarray, ini: float, fim: float) -> bool:
    """Blocos sem fala não são enviados (a Groq não tem VAD e o Whisper alucina em silêncio; também poupa cota)."""
    try:
        from faster_whisper.vad import VadOptions, get_speech_timestamps
    except Exception:
        return True
    trecho = audio[int(ini * SAMPLE_RATE):int(fim * SAMPLE_RATE)]
    fala = get_speech_timestamps(trecho, VadOptions(min_silence_duration_ms=500, speech_pad_ms=200))
    return sum(t["end"] - t["start"] for t in fala) / SAMPLE_RATE >= 0.4


# ---- cliente Groq ----------------------------------------------------------------------------------------
def _multipart(campos: list[tuple[str, str]], arquivo: bytes) -> tuple[bytes, str]:
    b = uuid.uuid4().hex
    corpo = b"".join(
        f'--{b}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode() for k, v in campos)
    corpo += (f'--{b}\r\nContent-Disposition: form-data; name="file"; filename="audio.flac"\r\n'
              f'Content-Type: audio/flac\r\n\r\n').encode() + arquivo + f"\r\n--{b}--\r\n".encode()
    return corpo, f"multipart/form-data; boundary={b}"


def _erro_da_resposta(e: urllib.error.HTTPError) -> str:
    try:
        d = json.loads(e.read().decode("utf-8", "replace"))
        return str(d.get("error", {}).get("message", d))[:200]
    except Exception:
        return ""


def _groq_chamar(flac: bytes, idioma: str | None, prompt: str | None, avisar_espera: Callable[[str], None]) -> dict:
    """Uma requisição à Groq com retry/backoff. Levanta MotorIndisponivel quando não adianta insistir."""
    chave = _env("GROQ_API_KEY")
    if not chave:
        raise SemChave("Chave da Groq ausente (GROQ_API_KEY no .env).")
    campos = [("model", groq_modelo()), ("response_format", "verbose_json"), ("temperature", "0"),
              ("timestamp_granularities[]", "word"), ("timestamp_granularities[]", "segment")]
    if idioma:
        campos.append(("language", idioma))
    if prompt:
        campos.append(("prompt", prompt))
    corpo, tipo = _multipart(campos, flac)
    url = _env("AVEIA_GROQ_URL", GROQ_URL)   # configurável só para testes locais com servidor falso
    ultimo = ""
    for tentativa in range(TENTATIVAS):
        req = urllib.request.Request(url, data=corpo, method="POST", headers={
            "Authorization": f"Bearer {chave}", "Content-Type": tipo, "User-Agent": "aveia/1.0"})
        try:
            with urllib.request.urlopen(req, timeout=float(_env("AVEIA_GROQ_TIMEOUT", str(TIMEOUT_S)))) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            msg = _erro_da_resposta(e)
            if e.code in (401, 403):
                raise ChaveInvalida(f"A Groq recusou a chave (HTTP {e.code}). Confira GROQ_API_KEY no .env.")
            if e.code == 413:
                raise _BlocoGrande()
            if e.code == 429:
                try:
                    espera = float(e.headers.get("retry-after") or 0)
                except ValueError:
                    espera = 0.0
                if espera > ESPERA_MAX_S:
                    raise CotaEsgotada(f"Cota gratuita da Groq esgotada (liberada em ~{int(espera // 60) + 1} min).")
                espera = espera or min(8.0 * 2 ** tentativa, ESPERA_MAX_S)
                ultimo = "limite de uso da Groq (429)"
                if tentativa == TENTATIVAS - 1:
                    raise CotaEsgotada("A Groq continuou limitando o uso (429) após várias tentativas.")
                avisar_espera(f"Limite da Groq atingido; aguardando {int(espera) + 1} s...")
                time.sleep(espera + 0.5)
                continue
            if e.code >= 500:
                ultimo = f"erro {e.code} no servidor da Groq"
            else:
                raise MotorIndisponivel(f"A Groq recusou a requisição (HTTP {e.code}): {msg}")
        except (urllib.error.URLError, socket.timeout, TimeoutError, ConnectionError, OSError) as e:
            ultimo = "tempo esgotado" if isinstance(e, (socket.timeout, TimeoutError)) or "timed out" in str(e) \
                else f"sem conexão com a Groq ({getattr(e, 'reason', e)})"
        if tentativa < TENTATIVAS - 1:
            avisar_espera(f"Groq indisponível ({ultimo}); tentando de novo...")
            time.sleep(min(2.0 * 2 ** tentativa, 20.0))
    raise FalhaDeRede(f"Groq indisponível: {ultimo}.")


def _palavras_da_resposta(d: dict) -> tuple[list[Palavra], int]:
    """Palavras com tempo, sem os trechos que o Whisper marca como silêncio/alucinação (e quantas foram descartadas)."""
    segs = d.get("segments") or []
    ruins = [(s["start"], s["end"]) for s in segs
             if (s.get("no_speech_prob", 0) > 0.6 and s.get("avg_logprob", 0) < -1.0)
             or s.get("compression_ratio", 0) > 2.4]
    out, descartadas = [], 0
    for w in d.get("words") or []:
        meio = (w["start"] + w["end"]) / 2
        if any(a <= meio <= b for a, b in ruins):
            descartadas += 1
            continue
        txt = str(w["word"]).strip()
        if txt:
            out.append(Palavra(float(w["start"]), float(w["end"]), " " + txt))
    return out, descartadas


def _groq_palavras(audio: np.ndarray, duracao: float, idioma: str | None, prompt: str | None,
                   avisar: Callable[[float, str], None], meta: dict,
                   local: Callable[[float], Iterator[Palavra]]) -> Iterator[Palavra]:
    """Groq em blocos com costura por tempo. Se um bloco falhar, o restante sai pelo motor local."""
    blocos = _cortes(audio, duracao)
    uso = uso_ultima_hora()
    if uso + duracao > 0.8 * limite_hora():
        meta["avisos"].append(
            f"Cota gratuita da Groq: ~{round(uso / 60)} min enviados na última hora (+{round(duracao / 60)} min desta "
            f"gravação) de um limite de ~{round(limite_hora() / 60)} min/h. Se acabar, o resto é transcrito localmente.")
    fila = list(blocos)
    enviados = 0.0
    while fila:
        ini, fim = fila.pop(0)
        fracao = 0.25 + 0.7 * ini / max(duracao, 1.0)
        if not _tem_fala(audio, ini, fim):
            continue
        a, b = max(0.0, ini - SOBRA_S), min(duracao, fim + SOBRA_S)
        flac = _flac(audio[int(a * SAMPLE_RATE):int(b * SAMPLE_RATE)])
        if len(flac) > int(_env("AVEIA_GROQ_MAX_BYTES", str(MAX_BYTES))) and fim - ini > 30:      # arquivo grande demais: divide o bloco ao meio
            meio = _ponto_quieto(audio, (ini + fim) / 2, raio_s=min(8.0, (fim - ini) / 4))
            fila[:0] = [(ini, meio), (meio, fim)]
            continue
        avisar(fracao, f"Transcrevendo na Groq (bloco {len(blocos) - len(fila)}/{len(blocos)})...")
        try:
            try:
                d = _groq_chamar(flac, idioma, prompt, lambda m: avisar(fracao, m))
            except _BlocoGrande:
                if fim - ini <= 30:
                    raise MotorIndisponivel("A Groq recusou o arquivo por tamanho.")
                meio = _ponto_quieto(audio, (ini + fim) / 2, raio_s=min(8.0, (fim - ini) / 4))
                fila[:0] = [(ini, meio), (meio, fim)]
                continue
        except MotorIndisponivel as e:
            desde = "Usado o motor local." if ini <= 0 else \
                f"A partir de {int(ini // 60):02d}:{int(ini % 60):02d} foi usado o motor local."
            meta["avisos"].append(f"{e} {desde}")
            meta["motor"] = "groq+local" if enviados > 0 else "local"
            for p in local(ini):
                yield p
            return
        enviados += b - a
        _registrar_uso(b - a)
        palavras_bloco, descartadas = _palavras_da_resposta(d)
        meta["descartadas"] = meta.get("descartadas", 0) + descartadas
        for p in palavras_bloco:
            inicio, fim_p = p.start + a, p.end + a
            meio_p = (inicio + fim_p) / 2
            if ini <= meio_p < fim or (not fila and meio_p >= fim):
                yield Palavra(inicio, fim_p, p.word)
    meta["groq_segundos"] = round(enviados)


# ---- despacho --------------------------------------------------------------------------------------------
def palavras(audio: np.ndarray, duracao: float, idioma: str | None, avisar: Callable[[float, str], None],
             motor: str | None = None, meta: dict | None = None, glossario: str | None = None) -> Iterator[Palavra]:
    """Gera as palavras (com tempo) do áudio com o motor pedido; `meta` recebe motor usado e avisos.

    `glossario` só vale para o Whisper local (inclusive no fallback da Groq): a Groq não o recebe (usa AVEIA_GROQ_PROMPT)."""
    from . import pipeline   # import tardio: o pipeline importa este módulo

    meta = meta if meta is not None else {}
    motor = normalizar_motor(motor)
    meta.update(motor_pedido=motor, motor=motor, avisos=meta.get("avisos", []), lacunas=meta.get("lacunas", []))

    extra = {"glossario": glossario} if glossario else {}   # sem glossário a chamada é idêntica à de antes dele

    def local(desde: float = 0.0) -> Iterator[Palavra]:
        if desde <= 0:
            yield from pipeline._palavras(audio, duracao, idioma, avisar, lacunas=meta["lacunas"], **extra)
            return
        resto = audio[int(desde * SAMPLE_RATE):]
        base = 0.25 + 0.7 * desde / max(duracao, 1.0)
        av = lambda p, m: avisar(base + (p - 0.25) * (1 - desde / max(duracao, 1.0)), m)  # noqa: E731
        achadas: list[dict] = []   # lacunas com tempo relativo a `resto`: voltam para a linha do tempo inteira
        try:
            for p in pipeline._palavras(resto, len(resto) / SAMPLE_RATE, idioma, av, lacunas=achadas, **extra):
                yield Palavra(p.start + desde, p.end + desde, p.word)
        finally:
            meta["lacunas"].extend({**l, "inicio": round(l["inicio"] + desde, 2), "fim": round(l["fim"] + desde, 2)}
                                   for l in achadas)

    if motor == "groq":
        if not groq_configurado():
            meta["avisos"].append("Groq selecionada, mas falta GROQ_API_KEY no .env: usado o motor local.")
            meta["motor"] = "local"
        else:
            # Sem prompt por padrão: medido, o prompt genérico do local (DICA_PT) fez a Groq truncar a transcrição
            # (perdeu ~30 s no clipe de referência). AVEIA_GROQ_PROMPT=<texto> (≤224 tokens) liga um glossário próprio.
            prompt = _env("AVEIA_GROQ_PROMPT") or None
            meta["modelo"] = groq_modelo()
            yield from _groq_palavras(audio, duracao, idioma, prompt, avisar, meta, local)
            return
    yield from local()
