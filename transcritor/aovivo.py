"""Transcrição durante a gravação ("ao vivo", pré-renderização).

O navegador envia os pedaços do MediaRecorder em ordem; cada sessão grava o arquivo, decodifica o fluxo que cresce
(PyAV lendo de um arquivo virtual que bloqueia até chegar mais byte), corta blocos de 15 a 30 s em pausas de fala e
transcreve cada bloco na fila única do Whisper. Ao parar a gravação só sobra a cauda: o texto completo aparece em
segundos. Em gravação de um canal, a parte cara da diarização (segmentação + embeddings, `AnaliseIncremental`) também é
paga durante a gravação, logo depois de cada bloco; ao parar resta só a cauda e o agrupamento. Sem isso (motor antigo,
`AVEIA_DIAR_AOVIVO=0`, modelo ausente, qualquer falha) a diarização roda no áudio inteiro ao parar, como antes.

O modo ao vivo é oportunista: qualquer dúvida (decodificação falhou, contagem de bytes não bate, sessão ociosa...)
marca a sessão como `degradada` e a finalização cai na transcrição clássica do mesmo arquivo. Nada aqui apaga o que o
navegador guarda.
"""
from __future__ import annotations

import math
import os
import threading
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Callable

import numpy as np

from . import canais as canais_mod
from . import diarizacao as diarizacao_mod
from . import pipeline
from .motores import Palavra, _ponto_quieto

TAXA = 16000                  # Hz: áudio decodificado (o mesmo do Whisper)
BLOCO_MIN_S = 15.0            # só fecha um bloco com pelo menos isto de áudio pendente...
BLOCO_MAX_S = 30.0            # ...e nunca deixa passar disto
# Cada chamada do Whisper custa uma janela inteira de 30 s (~17 s neste CPU) mesmo com 8 s de áudio (medido em 2026-11-20),
# então blocos curtos desperdiçam janela. Com áudio acumulado (fila atrasada, ou ao parar), em vez de cortar na PRIMEIRA pausa
# depois do mínimo, o corte adaptativo usa a ÚLTIMA pausa da janela: blocos mais cheios, menos chamadas. Medido em 5 clipes com
# texto de referência (testes/bench_aovivo.py, tempo total de CPU / WER): ref58 -14% / 8,3 -> 5,5; fleurs -38% / 4,0 igual;
# mls-livro -11% / 5,9 igual; mls-ruido -8% / 8,8 -> 9,4; cv-webm -33% / 8,2 -> 11,8. WER agregado +0,4 ponto, mas cv-webm passa
# do limite de 3 pontos por clipe: por isso o padrão fica DESLIGADO. `AVEIA_BLOCO_ADAPTATIVO=1` liga.
BLOCO_ADAPTATIVO = os.environ.get("AVEIA_BLOCO_ADAPTATIVO", "0").strip() == "1"
CORTE_DURO_ALVO_S = 29.0      # sem pausa até BLOCO_MAX_S: corta no instante mais quieto perto daqui
CORTE_DURO_RAIO_S = 1.0       # (28–30 s)
PAUSA_MIN_MS = 500            # silêncio que conta como pausa para fechar o bloco
FALA_MIN_S = 0.4              # bloco com menos fala que isto é pulado (o Whisper alucina em silêncio)
RESTO_SEM_FALA_S = 1.0        # ao pular um trecho sem fala, mantém este final (a fala pode estar começando)
AVALIAR_A_CADA_S = 1.0        # reavalia o corte quando chegar este tanto de áudio novo
OCIOSA_S = 30 * 60            # sem pedaço por este tempo (e sem finalizar): libera a memória, mantém o arquivo
VARREDURA_S = 60.0            # intervalo da limpeza de sessões ociosas
ESPERA_DECODIFICADOR_S = 60.0  # na finalização: tempo máximo esperando o decodificador chegar ao fim do fluxo
MAX_SESSOES = 4               # sessões simultâneas (cada uma guarda o áudio decodificado na memória)
MAX_PEDACO = 8 * 1024 * 1024  # bytes por pedaço
MAX_ADIANTADOS = 200          # pedaços fora de ordem guardados à espera do que falta
TOLERANCIA_DURACAO_S = 4.0    # decodificado menor que o gravado por mais que isto (ou 10%) = suspeito: cai no clássico
_SEGMENTO = TAXA * 120        # amostras por segmento do buffer de áudio (2 min)
ATRASO_DIAR_MAX_S = 60.0      # áudio pendente (além do bloco recém-transcrito) acima disto: não diariza agora, deixa p/ o fim

EXTENSOES = {"webm", "ogg", "oga", "mp4", "m4a", "mp3", "wav", "aac", "flac"}


class AoVivoIndisponivel(Exception):
    """O caminho ao vivo não pode concluir; quem chamou deve usar a transcrição clássica do arquivo."""


def extensao_valida(ext: str | None) -> str:
    ext = (ext or "").strip().lower().lstrip(".")
    return ext if ext in EXTENSOES else "webm"


def motivo_recusa(motor: str, preproc) -> str | None:
    """Por que o modo ao vivo não vale para este pedido (ou None). `motor` já normalizado; `preproc` = audio.configuracao()."""
    if motor != "local":
        return "O modo ao vivo só funciona com o motor local."
    if preproc.hpf or preproc.norm or preproc.denoise:
        return "O pré-processamento de áudio está ligado (AVEIA_PREPROC) e precisa do arquivo inteiro."
    return None


# ---------------------------------------------------------------------------------------------------------- áudio
class _Fluxo:
    """Arquivo virtual para o PyAV: `read` bloqueia até haver bytes ou o fluxo terminar. Devolve o que houver (não
    espera encher `n`), então o decodificador anda a cada pedaço recebido."""

    def __init__(self) -> None:
        self._buf = bytearray()
        self._cond = threading.Condition()
        self.terminou = False
        self.cancelado = False

    def anexar(self, dados: bytes) -> None:
        with self._cond:
            self._buf += dados
            self._cond.notify_all()

    def encerrar(self) -> None:
        with self._cond:
            self.terminou = True
            self._cond.notify_all()

    def cancelar(self) -> None:
        with self._cond:
            self.cancelado = True
            self._buf.clear()
            self._cond.notify_all()

    def read(self, n: int = -1) -> bytes:
        with self._cond:
            while not self._buf and not self.terminou and not self.cancelado:
                self._cond.wait()
            if self.cancelado or not self._buf:
                return b""
            k = len(self._buf) if n is None or n < 0 else min(n, len(self._buf))
            dados = bytes(self._buf[:k])
            del self._buf[:k]
            return dados

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return False


class _Pcm:
    """Áudio decodificado (float32, 16 kHz, mono) em segmentos de 2 min: cresce sem copiar tudo de novo."""

    def __init__(self) -> None:
        self._segs: list[np.ndarray] = []
        self.n = 0
        self._livre = False
        self._trava = threading.Lock()

    def anexar(self, a: np.ndarray) -> None:
        with self._trava:
            if self._livre:
                return
            i, tam = 0, len(a)
            while i < tam:
                s, off = divmod(self.n, _SEGMENTO)
                if s == len(self._segs):
                    self._segs.append(np.zeros(_SEGMENTO, dtype=np.float32))
                k = min(_SEGMENTO - off, tam - i)
                self._segs[s][off:off + k] = a[i:i + k]
                self.n += k
                i += k

    def fatia(self, ini: int, fim: int) -> np.ndarray:
        with self._trava:
            if self._livre:
                raise AoVivoIndisponivel("áudio da sessão já foi liberado")
            fim = min(fim, self.n)
            if fim <= ini:
                return np.zeros(0, dtype=np.float32)
            partes = []
            for s in range(ini // _SEGMENTO, (fim - 1) // _SEGMENTO + 1):
                base = s * _SEGMENTO
                partes.append(self._segs[s][max(ini, base) - base:min(fim, base + _SEGMENTO) - base])
            return partes[0].copy() if len(partes) == 1 else np.concatenate(partes)

    def liberar(self) -> None:
        with self._trava:
            self._livre = True
            self._segs = []


class _PcmDuplo:
    """Dois canais (microfone, computador) lado a lado. Interface de `_Pcm` (`n`, `fatia`, `liberar`), em que `fatia` devolve
    a MISTURA (o que o Whisper transcreve); `canais` devolve os dois canais inteiros para a separação final."""

    def __init__(self) -> None:
        self.esq, self.dir = _Pcm(), _Pcm()

    @property
    def n(self) -> int:
        return min(self.esq.n, self.dir.n)

    def anexar(self, a: np.ndarray) -> None:   # `a`: amostras intercaladas (E, D, E, D...)
        self.esq.anexar(a[0::2])
        self.dir.anexar(a[1::2])

    def fatia(self, ini: int, fim: int) -> np.ndarray:
        fim = min(fim, self.n)
        # Mesma regra do caminho clássico (`canais.misturar`): se a soma passar de PICO_MIX, reduz o ganho do bloco inteiro
        # em vez de cortar a onda (clipping distorce). Limite conhecido: o clássico calcula o ganho sobre o arquivo inteiro e
        # o ao vivo só conhece o bloco; com áudio alto o ganho pode diferir entre blocos e do clássico (a forma de onda
        # não é recortada). Com pico até PICO_MIX a saída é idêntica à soma simples.
        return canais_mod.misturar(self.esq.fatia(ini, fim), self.dir.fatia(ini, fim))

    def canais(self) -> tuple[np.ndarray, np.ndarray]:
        n = self.n
        return self.esq.fatia(0, n), self.dir.fatia(0, n)

    def liberar(self) -> None:
        self.esq.liberar()
        self.dir.liberar()


def _fala_vad(audio: np.ndarray) -> list[tuple[int, int]]:
    """Trechos de fala (amostras) pelo Silero VAD do faster-whisper. Sem preenchimento: as lacunas são as pausas reais."""
    from faster_whisper.vad import VadOptions, get_speech_timestamps
    opc = VadOptions(min_silence_duration_ms=PAUSA_MIN_MS, speech_pad_ms=0, min_speech_duration_ms=250)
    return [(int(t["start"]), int(t["end"])) for t in get_speech_timestamps(audio, opc, sampling_rate=TAXA)]


def escolher_pausa(fala: list[tuple[int, int]], tam: int, minimo: int, ultimo: bool = False) -> int | None:
    """Corte (amostra) em uma pausa depois de `minimo`: meio da lacuna entre dois trechos de fala, ou do silêncio final da
    janela. Por padrão o primeiro; com `ultimo=True`, o último dentro da janela (enche o bloco). None se não houve pausa."""
    pausa_min = int(PAUSA_MIN_MS / 1000 * TAXA)
    lacunas = [(fala[i][1], fala[i + 1][0]) for i in range(len(fala) - 1)]
    if fala and tam - fala[-1][1] >= pausa_min:
        lacunas.append((fala[-1][1], tam))
    achado = None
    for a, b in lacunas:
        if b <= minimo:
            continue
        corte = max((a + b) // 2, minimo)
        if corte < b:
            if not ultimo:
                return corte
            achado = corte
    return achado


# -------------------------------------------------------------------------------------------------------- sessão
class Sessao:
    def __init__(self, id: str, arquivo: Path, idioma: str | None, locutores: int | None,
                 submeter: Callable[..., object], monotonico: Callable[[], float] = time.monotonic,
                 canais: str | None = None, glossario: str | None = None) -> None:
        self.id = id
        self.glossario = pipeline.normalizar_glossario(glossario)   # `hotwords` de cada bloco e da cauda ("" = como antes)
        self.canais = pipeline.normalizar_canais(canais)   # "mic+sistema" (arquivo estéreo) ou None (mono)
        self.arquivo = Path(arquivo)
        self.idioma = idioma
        self.locutores = locutores
        self._submeter = submeter         # submeter(fn): coloca `fn` na fila única do Whisper
        self._relogio = monotonico
        self.criada = self.ultimo_pedaco = self._relogio()

        # recepção
        self._trava_rec = threading.Lock()
        self._arq = open(self.arquivo, "wb")
        self._proximo = 0                 # próximo seq esperado
        self._adiantados: dict[int, bytes] = {}
        self.bytes_recebidos = 0
        self.pedacos = 0                  # pedaços contíguos já gravados

        # decodificação
        self._fluxo = _Fluxo()
        self.pcm: _Pcm | _PcmDuplo = _PcmDuplo() if self.canais else _Pcm()
        self.estereo = False              # com `canais`: o fluxo realmente tem 2 canais (senão decodifica como mono)
        self._eof = threading.Event()
        self.degradada = False
        self.motivo = ""
        self.liberada = False
        self.finalizando = False
        self.duracao_esperada = 0.0
        self._t_primeiro: float | None = None   # s da criação até a primeira palavra (métrica)
        self._t_fim: float | None = None        # instante do "parar" (métrica)

        # blocos / transcrição
        self._trava_estado = threading.Lock()
        self._trava_transcricao = threading.Lock()
        self._trava_agendar = threading.Lock()
        self._palavras: list[Palavra] = []
        self._lacunas: list[dict] = []    # lacunas com fala achadas nos blocos (tempo da gravação inteira; ver pipeline._palavras)
        self._cursor = 0                  # amostra onde começa o áudio ainda não entregue ao Whisper
        self._transcrito = 0
        self._bloco_em_fila = False
        self._ultima_avaliacao = 0
        self.blocos: list[tuple[float, float]] = []   # (início, fim) em s dos blocos transcritos (diagnóstico/testes)

        # diarização incremental (só mono; só mexida na fila do Whisper, sob _trava_transcricao: o objeto não é thread-safe)
        self._inc = None                  # AnaliseIncremental, criada na primeira vez que for alimentada
        self._inc_desligado = False       # sem incremental nesta sessão (fábrica devolveu None, falhou ou degradou)
        self._diar_cursor = 0             # amostras já entregues à diarização (contíguas desde 0)
        self._inc_janelas = 0             # janelas processadas durante a gravação

        self._decodificador = threading.Thread(target=self._decodificar, name=f"aovivo-{id[:6]}", daemon=True)
        self._decodificador.start()

    # ---- recepção ---------------------------------------------------------------------------------------
    def receber(self, seq: int, dados: bytes) -> dict:
        """Aceita um pedaço. Grava no arquivo estritamente em ordem de `seq`; adiantados esperam; repetido é ignorado."""
        if seq < 0 or len(dados) > MAX_PEDACO:
            raise ValueError("pedaço inválido")
        with self._trava_rec:
            if self.liberada:
                raise AoVivoIndisponivel("sessão encerrada")
            self.ultimo_pedaco = self._relogio()
            if seq < self._proximo or seq in self._adiantados:
                return self._estado_rec()
            if seq - self._proximo > MAX_ADIANTADOS:
                raise ValueError("pedaço longe demais da sequência")
            self._adiantados[seq] = dados
            while self._proximo in self._adiantados:
                d = self._adiantados.pop(self._proximo)
                self._arq.write(d)
                self._arq.flush()
                self.bytes_recebidos += len(d)
                self.pedacos += 1
                self._proximo += 1
                if not self.degradada:
                    self._fluxo.anexar(d)
            return self._estado_rec()

    def _estado_rec(self) -> dict:
        return {"proximo": self._proximo, "bytes": self.bytes_recebidos, "adiantados": len(self._adiantados)}

    def confere(self, total_bytes: int, total_pedacos: int) -> bool:
        with self._trava_rec:
            return (not self._adiantados and total_bytes == self.bytes_recebidos and total_pedacos == self.pedacos)

    def marcar_fim(self, duracao: float = 0.0) -> None:
        """O navegador terminou de enviar: o decodificador vê o fim do fluxo e para de agendar blocos."""
        with self._trava_rec:
            self.duracao_esperada = max(0.0, float(duracao or 0.0))
            self.finalizando = True
            self._t_fim = self._relogio()
            try:
                self._arq.close()
            except OSError:
                pass
        self._fluxo.encerrar()

    # ---- decodificação ----------------------------------------------------------------------------------
    def _decodificar(self) -> None:
        import av
        try:
            conteiner = av.open(self._fluxo, mode="r", options={"probesize": "32768", "analyzeduration": "0"})
            try:
                entrada = conteiner.streams.audio[0]
                reamostrador = None
                ultimo_agendado = 0
                for pacote in conteiner.demux(entrada):
                    if self.degradada or self.liberada:
                        return
                    for quadro in pacote.decode():
                        if reamostrador is None:   # decide pelo 1º quadro: dois canais pedidos E presentes, senão mono
                            self.estereo = bool(self.canais) and quadro.layout.nb_channels == 2
                            if self.canais and not self.estereo:
                                self.pcm = _Pcm()   # ainda vazio: volta ao mono (o decodificador é o único a anexar)
                            reamostrador = av.AudioResampler(format="flt", layout="stereo" if self.estereo else "mono", rate=TAXA)
                        for saida in reamostrador.resample(quadro):
                            self.pcm.anexar(saida.to_ndarray().reshape(-1))
                    if self.pcm.n - ultimo_agendado >= AVALIAR_A_CADA_S * TAXA:
                        ultimo_agendado = self.pcm.n
                        self._agendar()
                for saida in (reamostrador.resample(None) if reamostrador else []):
                    self.pcm.anexar(saida.to_ndarray().reshape(-1))
            finally:
                conteiner.close()
        except Exception as e:  # noqa: BLE001
            if self.liberada or self.degradada:
                pass
            elif self._fluxo.terminou and self.pcm.n > 0:
                print(f"[ao vivo {self.id[:6]}] fim do fluxo com erro (ignorado, áudio decodificado até aqui): {e}")
            else:
                self.degradar(f"decodificação falhou: {e}")
        finally:
            self._eof.set()
            if not self.degradada and not self.liberada:
                self._agendar()

    # ---- blocos -----------------------------------------------------------------------------------------
    def _planejar(self, final: bool) -> tuple[int, int] | None:
        """Próximo bloco (início, fim em amostras) a transcrever, ou None. Pula trechos sem fala (avança o cursor)."""
        minimo, maximo = int(BLOCO_MIN_S * TAXA), int(BLOCO_MAX_S * TAXA)
        while True:
            n, ini = self.pcm.n, self._cursor
            pend = n - ini
            if pend <= 0 or (not final and pend < minimo):
                return None
            fim_janela = min(n, ini + maximo)
            janela = self.pcm.fatia(ini, fim_janela)
            tam = len(janela)
            fala = _fala_vad(janela)
            total_fala = sum(b - a for a, b in fala)
            if total_fala < FALA_MIN_S * TAXA:
                # sem fala: não gasta Whisper. Fora da finalização mantém o final da janela (a fala pode estar começando).
                if final:
                    self._cursor = self._transcrito = fim_janela
                    continue
                self._cursor = self._transcrito = max(ini, fim_janela - int(RESTO_SEM_FALA_S * TAXA))
                return None
            if final and pend <= maximo:
                return ini, fim_janela
            corte = escolher_pausa(fala, tam, minimo, ultimo=BLOCO_ADAPTATIVO)   # também ao parar: o atraso pendente vira blocos cheios
            if corte is not None:
                return ini, ini + corte
            if tam >= maximo:
                alvo = _ponto_quieto(janela, CORTE_DURO_ALVO_S, CORTE_DURO_RAIO_S)
                corte = min(max(int(alvo * TAXA), minimo), maximo)
                return ini, ini + corte
            if final:
                return ini, fim_janela
            return None

    def _agendar(self) -> None:
        """Se há áudio suficiente e nenhum bloco desta sessão na fila, planeja o próximo e o entrega à fila do Whisper."""
        if self.degradada or self.liberada or self.finalizando:
            return
        if not self._trava_agendar.acquire(blocking=False):
            return
        try:
            with self._trava_estado:
                if self._bloco_em_fila or self.pcm.n - self._ultima_avaliacao < AVALIAR_A_CADA_S * TAXA:
                    return
                self._ultima_avaliacao = self.pcm.n
            try:
                plano = self._planejar(final=False)
            except AoVivoIndisponivel:
                return
            except Exception as e:  # noqa: BLE001
                return self.degradar(f"planejamento do bloco falhou: {e}")
            if plano is None:
                return
            with self._trava_estado:
                self._bloco_em_fila = True
            try:
                self._submeter(self._tarefa_bloco, plano)
            except Exception as e:  # noqa: BLE001
                with self._trava_estado:
                    self._bloco_em_fila = False
                self.degradar(f"não foi possível enfileirar o bloco: {e}")
        finally:
            self._trava_agendar.release()

    def _tarefa_bloco(self, plano: tuple[int, int]) -> None:
        """Roda na fila do Whisper."""
        try:
            if self.degradada or self.liberada:
                return
            with self._trava_transcricao:
                if self._cursor != plano[0]:   # a finalização já tratou este trecho
                    return
                self._transcrever_trecho(*plano)
                # o texto do bloco já saiu; só então paga a diarização (se o texto já está atrasado, deixa para o fim)
                if self.pcm.n - plano[1] <= ATRASO_DIAR_MAX_S * TAXA:
                    self._inc_janelas += self._alimentar_diarizacao(plano[1])
        except Exception as e:  # noqa: BLE001
            self.degradar(f"transcrição do bloco falhou: {e}")
        finally:
            with self._trava_estado:
                self._bloco_em_fila = False
                self._ultima_avaliacao = 0    # pode haver mais áudio pendente (atraso): reavalia já
            self._agendar()

    def _transcrever_trecho(self, ini: int, fim: int, aoPalavra: Callable[[], None] | None = None) -> None:
        bloco = self.pcm.fatia(ini, fim)
        dur = len(bloco) / TAXA
        desloc = ini / TAXA
        t0 = time.monotonic()
        if dur > 0:
            achadas: list[dict] = []   # tempos relativos ao bloco; as palavras recuperadas já saem em ordem de tempo
            try:
                extra = {"glossario": self.glossario} if self.glossario else {}   # vazio: chamada idêntica à de antes
                for p in pipeline._palavras(bloco, dur, self.idioma, lambda pr, m: None, lacunas=achadas, **extra):
                    with self._trava_estado:
                        self._palavras.append(Palavra(p.start + desloc, p.end + desloc, p.word))
                        if self._t_primeiro is None:
                            self._t_primeiro = self._relogio() - self.criada
                    if aoPalavra:
                        aoPalavra()
            finally:
                with self._trava_estado:
                    self._lacunas += [{**l, "inicio": round(l["inicio"] + desloc, 2), "fim": round(l["fim"] + desloc, 2)}
                                      for l in achadas]
        with self._trava_estado:
            self.blocos.append((ini / TAXA, fim / TAXA))
            self._cursor = fim
            self._transcrito = fim
        print(f"[ao vivo {self.id[:6]}] bloco {ini / TAXA:.1f}-{fim / TAXA:.1f}s ({dur:.1f}s) transcrito em {time.monotonic() - t0:.1f}s")

    def metricas(self) -> dict:
        """Números da sessão para `metricas.registrar_gravacao` (sem texto): duração, 1º texto, parar até pronto, blocos."""
        blocos = list(self.blocos)
        return dict(
            duracao_s=self.pcm.n / TAXA, primeiro_texto_s=self._t_primeiro, blocos=len(blocos),
            janelas_30s=sum(max(1, math.ceil((b - a) / 30.0)) for a, b in blocos),
            parar_ao_pronto_s=(self._relogio() - self._t_fim) if self._t_fim is not None else None)

    # ---- diarização incremental (só na fila do Whisper, sob _trava_transcricao) ---------------------------
    def _descartar_inc(self, motivo: str) -> None:
        """Abandona a diarização incremental desta sessão; a finalização usa `pipeline.diarizar` no áudio inteiro."""
        self._inc_desligado = True
        self._inc = None
        print(f"[ao vivo {self.id[:6]}] diarização incremental descartada: {motivo}; a diarização roda no fim.")

    def _alimentar_diarizacao(self, ate: int) -> int:
        """Entrega ao incremental o áudio de `_diar_cursor` até `ate` (contíguo, sem pular trechos sem fala). Cria o
        incremental na primeira vez. Nunca levanta: qualquer falha só descarta o incremental. Devolve janelas novas."""
        if self._inc_desligado or self.degradada or self.liberada:
            return 0
        if not isinstance(self.pcm, _Pcm):   # dois canais: o sistema é diarizado no fim, por `diarizar_remotos`
            self._inc_desligado = True
            return 0
        try:
            inc = self._inc
            if inc is None:
                inc = self._inc = diarizacao_mod.analise_incremental()
                if inc is None:
                    self._inc_desligado = True
                    return 0
            ate = min(ate, self.pcm.n)
            if ate <= self._diar_cursor:
                return 0
            novas = inc.alimentar(self.pcm.fatia(self._diar_cursor, ate))
            self._diar_cursor = ate
            return int(novas or 0)
        except Exception as e:  # noqa: BLE001
            self._descartar_inc(f"{type(e).__name__}: {e}")
            return 0

    def _concluir_incremental(self) -> list:
        """Alimenta o que falta e agrupa. Lista vazia = não há resultado aproveitável (quem chamou usa o caminho do lote)."""
        with self._trava_transcricao:
            try:
                t0 = time.monotonic()
                self._alimentar_diarizacao(self.pcm.n)
                inc = self._inc
                if inc is None or self._inc_desligado:
                    return []
                t1 = time.monotonic()
                trechos = inc.concluir(num_locutores=self.locutores)
                print(f"[ao vivo {self.id[:6]}] diarização incremental: {self._inc_janelas} janelas durante a gravação, "
                      f"resto alimentado em {t1 - t0:.1f}s, concluir em {time.monotonic() - t1:.1f}s")
                if not trechos:
                    print(f"[ao vivo {self.id[:6]}] diarização incremental sem trechos; usando a diarização do áudio inteiro.")
                return list(trechos or [])
            except Exception as e:  # noqa: BLE001
                self._descartar_inc(f"{type(e).__name__}: {e}")
                return []
            finally:
                self._inc = None
                self._inc_desligado = True

    # ---- consulta ---------------------------------------------------------------------------------------
    def falas(self) -> list:
        with self._trava_estado:
            palavras = list(self._palavras)
        return pipeline.falas_provisorias(palavras)

    def consultar(self, desde: int = 0) -> dict:
        falas = self.falas()
        desde = max(0, desde)
        return {
            "status": "degradada" if self.degradada else "ativa",
            "motivo": self.motivo,
            "falas": [asdict(f) for f in falas[desde:]],
            "total": len(falas),
            "recebido_s": round(self.pcm.n / TAXA, 2),
            "transcrito_s": round(self._transcrito / TAXA, 2),
        }

    # ---- finalização (roda na fila do Whisper) -----------------------------------------------------------
    def concluir(self, progresso: Callable[[float, str], None] | None = None,
                 parcial: Callable[[list], None] | None = None, meta: dict | None = None) -> list:
        """Transcreve a cauda, diariza o áudio inteiro e devolve as falas finais. Levanta AoVivoIndisponivel se o
        caminho ao vivo não serve: o chamador usa a transcrição clássica do arquivo."""
        emitir = progresso or (lambda p, m: None)
        ultimo = [0.0]

        def avisar(p: float, m: str) -> None:
            ultimo[0] = max(ultimo[0], p)
            emitir(ultimo[0], m)

        def faixa_transcricao() -> float:
            tot = max(self.pcm.n, 1)
            return pipeline._TRANSC_INI + (pipeline._TRANSC_FIM - pipeline._TRANSC_INI) * min(self._transcrito / tot, 1.0)

        if self.degradada:
            raise AoVivoIndisponivel(self.motivo or "sessão degradada")
        avisar(0.03, "Terminando de ler o áudio...")
        if not self._eof.wait(ESPERA_DECODIFICADOR_S):
            raise AoVivoIndisponivel("o decodificador não chegou ao fim do áudio")
        if self.degradada:
            raise AoVivoIndisponivel(self.motivo or "sessão degradada")
        decodificado = self.pcm.n / TAXA
        esperado = self.duracao_esperada
        if decodificado <= 0 or (esperado > 0 and decodificado < esperado - max(TOLERANCIA_DURACAO_S, 0.1 * esperado)):
            raise AoVivoIndisponivel(f"áudio decodificado ({decodificado:.1f}s) menor que o gravado ({esperado:.1f}s)")

        ultima = [0.0]

        def emitir_parcial() -> None:
            if parcial is None:
                return
            agora = time.monotonic()
            if agora - ultima[0] < pipeline._INTERVALO_PARCIAL:
                return
            ultima[0] = agora
            try:
                parcial(self.falas())
            except Exception as e:  # noqa: BLE001
                print(f"Callback parcial falhou (ignorado): {e}")

        avisar(faixa_transcricao(), "Transcrevendo o final da gravação...")
        if parcial is not None:
            try:
                parcial(self.falas())
            except Exception:  # noqa: BLE001
                pass
        with self._trava_transcricao:
            while True:
                plano = self._planejar(final=True)
                if plano is None:
                    break
                self._transcrever_trecho(*plano, aoPalavra=emitir_parcial)
                avisar(faixa_transcricao(), "Transcrevendo o final da gravação...")
        avisar(pipeline._TRANSC_FIM, "Transcrição concluída.")
        palavras = list(self._palavras)
        if meta is not None:   # lacunas com fala que a retranscrição não resolveu (tempos da gravação inteira)
            with self._trava_estado:
                achadas = list(self._lacunas)
            meta["lacunas"] = achadas
            meta.setdefault("avisos", []).extend(pipeline.avisos_de_lacunas(achadas))
        if parcial is not None and palavras:
            try:
                parcial(pipeline.falas_provisorias(palavras))
            except Exception:  # noqa: BLE001
                pass

        trechos = []
        separacao, sistema = None, None
        if palavras and self.estereo and isinstance(self.pcm, _PcmDuplo):
            try:   # dois canais: quem fala pelo microfone é o Locutor 1; só o canal do computador é diarizado
                esq, dir = self.pcm.canais()
                separacao = canais_mod.separar(esq, dir)
                if separacao.valido:
                    sistema = dir
                else:
                    print(f"[ao vivo {self.id[:6]}] dois canais ignorados ({separacao.motivo}): tratando como mono.")
                    separacao = None
            except AoVivoIndisponivel:
                raise
            except Exception as e:  # noqa: BLE001
                print(f"[ao vivo {self.id[:6]}] separação por canais falhou ({e}): tratando como mono.")
                separacao = None
        if separacao is not None:
            trechos = pipeline.diarizar_remotos(sistema, self.locutores, True, meta, avisar)
            if meta is not None:
                meta["canais"] = pipeline.CANAIS_MIC_SISTEMA
                meta["locutor_local"] = pipeline.LOCUTOR_LOCAL
            falas = pipeline.montar_falas(palavras, trechos, locais=separacao.trechos_locais)
            if meta is not None:
                meta["motor"] = "local"
                meta.setdefault("avisos", [])
            avisar(1.0, "Pronto.")
            return falas
        if palavras:
            avisar(pipeline._DIAR_INI, "Identificando locutores pelo timbre...")
            trechos = self._concluir_incremental()   # [] se não há incremental são: cai no áudio inteiro, como sempre
            if trechos:
                if meta is not None:
                    meta["diarizacao"] = "novo"
            else:
                try:
                    trechos = pipeline.diarizar(self.pcm.fatia(0, self.pcm.n), num_locutores=self.locutores)
                    pipeline.anotar_diarizacao(meta)
                except AoVivoIndisponivel:
                    raise
                except Exception as e:  # noqa: BLE001
                    print(f"Diarização falhou: {e}")
                    if meta is not None:
                        meta.setdefault("avisos", []).append(
                            f"A identificação de locutores falhou ({e}); todas as falas ficaram como Locutor 1.")
                    trechos = []
            avisar(pipeline._DIAR_FIM, "Locutores identificados.")
        falas = pipeline.montar_falas(palavras, trechos)
        if meta is not None:
            meta["motor"] = "local"
            meta.setdefault("avisos", [])
        avisar(1.0, "Pronto.")
        return falas

    # ---- fim de vida ------------------------------------------------------------------------------------
    def degradar(self, motivo: str) -> None:
        """Para o ao vivo desta sessão (sem mais texto ao vivo) e libera o áudio decodificado. O arquivo fica."""
        if not self.degradada:
            self.degradada = True
            self.motivo = motivo
            print(f"[ao vivo {self.id[:6]}] degradada: {motivo}")
        self._inc_desligado = True
        self._inc = None
        self._fluxo.cancelar()
        self.pcm.liberar()

    def liberar(self) -> None:
        self.liberada = True
        self._inc_desligado = True
        self._inc = None
        self._fluxo.cancelar()
        self.pcm.liberar()
        with self._trava_rec:
            try:
                self._arq.close()
            except OSError:
                pass

    def apagar(self) -> None:
        self.liberar()
        try:
            self.arquivo.unlink(missing_ok=True)
        except OSError as e:
            print(f"[ao vivo {self.id[:6]}] não foi possível apagar {self.arquivo}: {e}")


# ---------------------------------------------------------------------------------------------------- gerenciador
class Gerenciador:
    def __init__(self, submeter: Callable[..., object], relogio: Callable[[], float] = time.monotonic) -> None:
        self._submeter = submeter
        self._relogio = relogio
        self.sessoes: dict[str, Sessao] = {}
        self._trava = threading.Lock()
        self._varredor: threading.Thread | None = None

    def criar(self, arquivo: Path, idioma: str | None, locutores: int | None, canais: str | None = None,
              glossario: str | None = None) -> Sessao | None:
        """Nova sessão, ou None se já há sessões demais."""
        self.varrer()
        with self._trava:
            if sum(1 for s in self.sessoes.values() if not s.finalizando and not s.degradada) >= MAX_SESSOES:
                return None
            sid = uuid.uuid4().hex
            s = Sessao(sid, arquivo, idioma, locutores, self._submeter, self._relogio, canais, glossario)
            self.sessoes[sid] = s
        self._iniciar_varredor()
        return s

    def obter(self, sid: str) -> Sessao | None:
        return self.sessoes.get(sid)

    def remover(self, sid: str) -> Sessao | None:
        with self._trava:
            return self.sessoes.pop(sid, None)

    def varrer(self) -> int:
        """Sessões sem pedaço há OCIOSA_S (e não finalizadas) liberam o áudio da memória; o arquivo fica."""
        agora, n = self._relogio(), 0
        for s in list(self.sessoes.values()):
            if not s.finalizando and not s.degradada and not s.liberada and agora - s.ultimo_pedaco >= OCIOSA_S:
                s.degradar("ociosa")
                n += 1
        return n

    def _iniciar_varredor(self) -> None:
        if self._varredor and self._varredor.is_alive():
            return

        def laco() -> None:
            while True:
                time.sleep(VARREDURA_S)
                try:
                    self.varrer()
                except Exception as e:  # noqa: BLE001
                    print(f"Limpeza de sessões ao vivo falhou: {e}")

        self._varredor = threading.Thread(target=laco, name="aovivo-varredor", daemon=True)
        self._varredor.start()
