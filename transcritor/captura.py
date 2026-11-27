"""Captura do áudio no servidor, sem nenhum clique: microfone padrão (canal E) e áudio que sai do computador (canal D,
loopback WASAPI da saída padrão), em 16 kHz, intercalados em estéreo (E, D, E, D...).

Por que no servidor: o navegador só captura o áudio do computador (`getDisplayMedia`) depois de um clique. Aqui o
Windows entrega o loopback direto, e uma reunião agendada grava sozinha.

Pontos de projeto:
- O loopback não entrega pacote enquanto nada toca. O relógio de parede decide quanto áudio cada canal precisa ter; o que
  faltar é preenchido com silêncio (`Intercalador`), e o que sobrar além da folga é descartado, para os canais nunca
  se desalinharem nem derivarem do relógio.
- Cada dispositivo tem sua taxa (44,1 ou 48 kHz) e seu número de canais: o `Reamostrador` leva tudo a 16 kHz mono.
- Se o dispositivo padrão mudar (fone Bluetooth ligado ou desligado), a captura reabre os dispositivos. A lista de
  dispositivos do PortAudio só se renova ao reiniciá-lo, então a mudança é detectada por uma sonda em outro processo.

Diagnóstico: `python -m transcritor.captura --testar` abre os dois dispositivos por 2 s e imprime só taxas, canais e
se chegou amostra (nada é gravado).
"""
from __future__ import annotations

import json
import math
import os
import queue
import subprocess
import sys
import threading
import time
from typing import Callable, Protocol

import numpy as np

TAXA = 16000                 # Hz: o mesmo do Whisper e do `Sessao`
BLOCO_S = 1.0                # entrega blocos de 1 s (no máximo ~2 s se a entrega atrasar)
ATRASO_S = 0.5               # folga do relógio: dado atrasado até isto não vira silêncio
EXCESSO_S = 1.0              # acima disto de áudio adiantado num canal, descarta até MANTER_S (limita a deriva)
MANTER_S = 0.3
PASSO_S = 0.25               # intervalo do laço de captura
SONDA_S = 8.0                # intervalo da sonda de mudança do dispositivo padrão
REABRIR_S = 5.0              # intervalo das tentativas de abrir um dispositivo que falhou
LIMIAR_RMS_SILENCIO = 5e-4   # RMS abaixo disto no canal do computador conta como silêncio


# ----------------------------------------------------------------------------------------------- reamostragem
class Reamostrador:
    """Reamostra mono, em pedaços, para `TAXA`, sem emendas audíveis: guarda o contexto do pedaço anterior e segura
    ~10 ms de áudio para que a borda final de cada pedaço só seja emitida com o seguinte já conhecido. Só processa
    múltiplos inteiros do divisor, então a razão de saída é exata (sem deriva por arredondamento)."""

    def __init__(self, taxa_origem: int, taxa_destino: int = TAXA) -> None:
        self.origem, self.destino = int(taxa_origem), int(taxa_destino)
        g = math.gcd(self.origem, self.destino)
        self.up, self.down = self.destino // g, self.origem // g
        self._ov = self.down * max(1, math.ceil(64 / self.down))   # contexto: ~64+ amostras de entrada
        self._resto = np.zeros(0, dtype=np.float32)
        self._hist = np.zeros(0, dtype=np.float32)
        self._ja = 0   # amostras do início de `_hist` que já foram emitidas (contexto)

    def processar(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32).reshape(-1)
        if self.up == 1 and self.down == 1:
            return x
        buf = np.concatenate([self._resto, x])
        n = (len(buf) // self.down) * self.down
        self._resto = buf[n:]
        if n == 0:
            return np.zeros(0, dtype=np.float32)
        ent = np.concatenate([self._hist, buf[:n]])
        ini, fim = self._ja, len(ent) - self._ov
        if fim <= ini:
            self._hist = ent
            return np.zeros(0, dtype=np.float32)
        from scipy.signal import resample_poly
        y = resample_poly(ent, self.up, self.down).astype(np.float32)
        saida = y[ini * self.up // self.down: fim * self.up // self.down]
        ctx = min(self._ov, fim)
        self._hist = ent[fim - ctx:]
        self._ja = ctx
        return saida


# ------------------------------------------------------------------------------------------------ intercalação
class Intercalador:
    """Junta dois canais mono (0 = microfone, 1 = computador), já em 16 kHz, num fluxo estéreo alinhado pelo relógio."""

    def __init__(self, relogio: Callable[[], float] = time.monotonic, taxa: int = TAXA, bloco_s: float = BLOCO_S,
                 atraso_s: float = ATRASO_S) -> None:
        self._relogio = relogio
        self.taxa = taxa
        self._bloco = int(bloco_s * taxa)
        self._atraso = atraso_s
        self._t0 = relogio()
        self._emitido = 0
        self._buf = [np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32)]
        self.preenchido = [0, 0]   # amostras de silêncio inseridas por canal (diagnóstico)
        self.descartado = [0, 0]
        self._trava = threading.Lock()

    def anexar(self, canal: int, mono: np.ndarray) -> None:
        if len(mono):
            with self._trava:
                self._buf[canal] = np.concatenate([self._buf[canal], np.asarray(mono, dtype=np.float32).reshape(-1)])

    def _emitir(self, alvo: int) -> np.ndarray:
        saida = np.zeros((alvo, 2), dtype=np.float32)
        for c in (0, 1):
            b = self._buf[c]
            if len(b) < alvo:
                self.preenchido[c] += alvo - len(b)
                saida[:len(b), c] = b
                b = np.zeros(0, dtype=np.float32)
            else:
                saida[:, c] = b[:alvo]
                b = b[alvo:]
            exc = int(EXCESSO_S * self.taxa)
            if len(b) > exc:   # adiantado demais (o silêncio preenchido antes já cobria este trecho): descarta o excesso
                manter = int(MANTER_S * self.taxa)
                self.descartado[c] += len(b) - manter
                b = b[-manter:]
            self._buf[c] = b
        self._emitido += alvo
        return saida.reshape(-1)

    def coletar(self, agora: float | None = None) -> np.ndarray | None:
        """Bloco estéreo intercalado (float32, E D E D...) se já passou tempo para pelo menos `bloco_s`; senão None."""
        with self._trava:
            agora = self._relogio() if agora is None else agora
            alvo = int((agora - self._t0 - self._atraso) * self.taxa) - self._emitido
            if alvo < self._bloco:
                return None
            return self._emitir(alvo)

    def descarregar(self, agora: float | None = None) -> np.ndarray | None:
        """No fim: entrega o que o relógio já cobre, sem a folga."""
        with self._trava:
            agora = self._relogio() if agora is None else agora
            alvo = int((agora - self._t0) * self.taxa) - self._emitido
            return self._emitir(alvo) if alvo > 0 else None


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64)))) if len(x) else 0.0


# ------------------------------------------------------------------------------------------------ dispositivos
class Dispositivo(Protocol):
    nome: str
    taxa: int

    def ler(self) -> np.ndarray: ...      # mono float32 na taxa do dispositivo (o que chegou desde a última leitura)
    def vivo(self) -> bool: ...
    def fechar(self) -> None: ...


class Fabrica(Protocol):
    def abrir(self, tipo: str) -> Dispositivo | None: ...    # tipo: "mic" ou "sistema"
    def padroes(self) -> dict: ...                           # {"mic": nome|None, "saida": nome|None}
    def renovar(self) -> None: ...                           # reinicia o PortAudio (renova a lista de dispositivos)
    def encerrar(self) -> None: ...


_SONDA = (
    "import json, pyaudiowpatch as p\n"
    "pa = p.PyAudio()\n"
    "try:\n"
    "    w = pa.get_host_api_info_by_type(p.paWASAPI)\n"
    "    def nome(i):\n"
    "        return pa.get_device_info_by_index(i)['name'] if i is not None and i >= 0 else None\n"
    "    print(json.dumps({'mic': nome(w['defaultInputDevice']), 'saida': nome(w['defaultOutputDevice'])}))\n"
    "finally:\n"
    "    pa.terminate()\n"
)


class _DispositivoWasapi:
    def __init__(self, pa, info: dict, tipo: str) -> None:
        import pyaudiowpatch as pyaudio
        self.nome = str(info["name"])
        self.taxa = int(info["defaultSampleRate"])
        self.canais = max(1, int(info["maxInputChannels"]))
        self.tipo = tipo
        self._fila: queue.SimpleQueue[bytes] = queue.SimpleQueue()

        def retorno(dados, quadros, tempo, estado):
            self._fila.put(dados)
            return (None, pyaudio.paContinue)

        self._fluxo = pa.open(format=pyaudio.paFloat32, channels=self.canais, rate=self.taxa, input=True,
                              input_device_index=int(info["index"]), frames_per_buffer=max(256, self.taxa // 50),
                              stream_callback=retorno)

    def ler(self) -> np.ndarray:
        partes = []
        while True:
            try:
                partes.append(self._fila.get_nowait())
            except queue.Empty:
                break
        if not partes:
            return np.zeros(0, dtype=np.float32)
        x = np.frombuffer(b"".join(partes), dtype=np.float32)
        x = x[: (len(x) // self.canais) * self.canais].reshape(-1, self.canais)
        return x.mean(axis=1).astype(np.float32)

    def vivo(self) -> bool:
        try:
            return bool(self._fluxo.is_active())
        except Exception:  # noqa: BLE001
            return False

    def fechar(self) -> None:
        try:
            self._fluxo.stop_stream()
        except Exception:  # noqa: BLE001
            pass
        try:
            self._fluxo.close()
        except Exception:  # noqa: BLE001
            pass


class FabricaWasapi:
    """Dispositivos reais do Windows (PyAudioWPatch)."""

    def __init__(self) -> None:
        import pyaudiowpatch as pyaudio   # ImportError = sem captura nesta máquina (quem chama trata)
        self._m = pyaudio
        self._pa = pyaudio.PyAudio()

    def abrir(self, tipo: str) -> Dispositivo | None:
        try:
            wasapi = self._pa.get_host_api_info_by_type(self._m.paWASAPI)
            if tipo == "mic":
                i = wasapi["defaultInputDevice"]
                if i < 0:
                    return None
                info = self._pa.get_device_info_by_index(i)
            else:
                info = self._pa.get_default_wasapi_loopback()
            return _DispositivoWasapi(self._pa, info, tipo)
        except Exception as e:  # noqa: BLE001
            print(f"[captura] não abriu o dispositivo '{tipo}': {type(e).__name__}: {e}", flush=True)
            return None

    def padroes(self) -> dict:
        try:
            r = subprocess.run([sys.executable, "-c", _SONDA], capture_output=True, text=True, timeout=10)
            return json.loads(r.stdout.strip().splitlines()[-1])
        except Exception:  # noqa: BLE001
            return {}

    def renovar(self) -> None:
        try:
            self._pa.terminate()
        except Exception:  # noqa: BLE001
            pass
        self._pa = self._m.PyAudio()

    def encerrar(self) -> None:
        try:
            self._pa.terminate()
        except Exception:  # noqa: BLE001
            pass


def disponivel() -> tuple[bool, str]:
    """A captura funciona nesta máquina? (Windows com PyAudioWPatch instalado)."""
    if sys.platform != "win32":
        return False, "A captura automática só existe no Windows."
    try:
        import pyaudiowpatch  # noqa: F401
    except ImportError:
        return False, "Falta instalar o PyAudioWPatch (pip install PyAudioWPatch)."
    return True, ""


# ---------------------------------------------------------------------------------------------------- captura
class Captura:
    """Abre microfone e loopback, e entrega blocos estéreo a `ao_bloco(pcm_intercalado, rms_computador, rms_microfone)`."""

    def __init__(self, ao_bloco: Callable[[np.ndarray, float, float], None], relogio: Callable[[], float] = time.monotonic,
                 fabrica: Fabrica | None = None, sonda_s: float = SONDA_S, passo_s: float = PASSO_S) -> None:
        self._ao_bloco = ao_bloco
        self._relogio = relogio
        self._fabrica = fabrica
        self._sonda_s = sonda_s
        self._passo = passo_s
        self.avisos: list[str] = []
        self._disp: list[Dispositivo | None] = [None, None]
        self._amost: list[Reamostrador | None] = [None, None]
        self._tentou = [0.0, 0.0]
        self._padroes: dict = {}
        self._ultima_sonda = 0.0
        self.inter = Intercalador(relogio)
        self._parar = threading.Event()
        self._fio: threading.Thread | None = None
        self.reaberturas = 0

    # -- ciclo de vida
    def iniciar(self) -> None:
        if self._fabrica is None:
            self._fabrica = FabricaWasapi()
        self._padroes = self._fabrica.padroes()
        self._ultima_sonda = self._relogio()
        self.inter = Intercalador(self._relogio)
        self._abrir_todos()
        if not any(self._disp):
            self._fabrica.encerrar()
            raise RuntimeError("Nenhum dispositivo de áudio pôde ser aberto (microfone e saída do computador).")
        self._parar.clear()
        self._fio = threading.Thread(target=self._laco, name="aveia-captura", daemon=True)
        self._fio.start()

    def parar(self) -> None:
        self._parar.set()
        if self._fio is not None:
            self._fio.join(timeout=5)
        self._passo_leitura()
        resto = self.inter.descarregar()
        if resto is not None:
            self._entregar(resto)
        for i in (0, 1):
            self._fechar(i)
        if self._fabrica is not None:
            self._fabrica.encerrar()

    # -- internos
    def _abrir(self, i: int) -> None:
        d = self._fabrica.abrir("mic" if i == 0 else "sistema")
        self._tentou[i] = self._relogio()
        self._disp[i] = d
        self._amost[i] = Reamostrador(d.taxa) if d is not None else None

    def _abrir_todos(self) -> None:
        for i in (0, 1):
            self._abrir(i)
        rotulos = ("microfone", "áudio do computador")
        self.avisos = [f"Sem {rotulos[i]}: canal em silêncio." for i in (0, 1) if self._disp[i] is None]

    def _fechar(self, i: int) -> None:
        if self._disp[i] is not None:
            self._disp[i].fechar()
        self._disp[i] = self._amost[i] = None

    def _passo_leitura(self) -> None:
        for i in (0, 1):
            d = self._disp[i]
            if d is None:
                continue
            x = d.ler()
            if len(x):
                self.inter.anexar(i, self._amost[i].processar(x))
            if not d.vivo():   # fone desligado, dispositivo removido...
                self._fechar(i)
                self._tentou[i] = self._relogio()

    def _entregar(self, bloco: np.ndarray) -> None:
        try:
            self._ao_bloco(bloco, rms(bloco[1::2]), rms(bloco[0::2]))
        except Exception as e:  # noqa: BLE001
            print(f"[captura] o consumidor do bloco falhou: {type(e).__name__}: {e}", flush=True)

    def _verificar_mudanca(self) -> None:
        agora = self._relogio()
        if agora - self._ultima_sonda >= self._sonda_s:
            self._ultima_sonda = agora
            novos = self._fabrica.padroes()
            if novos and self._padroes and novos != self._padroes:
                print("[captura] o dispositivo padrão mudou: reabrindo os dispositivos.", flush=True)
                for i in (0, 1):
                    self._fechar(i)
                self._fabrica.renovar()
                self._abrir_todos()
                self.reaberturas += 1
            if novos:
                self._padroes = novos
        for i in (0, 1):   # um dispositivo que caiu: tenta de novo de tempos em tempos
            if self._disp[i] is None and agora - self._tentou[i] >= REABRIR_S:
                self._abrir(i)

    def _laco(self) -> None:
        while not self._parar.wait(self._passo):
            try:
                self._passo_leitura()
                bloco = self.inter.coletar()
                if bloco is not None:
                    self._entregar(bloco)
                self._verificar_mudanca()
            except Exception as e:  # noqa: BLE001
                print(f"[captura] erro no laço (segue): {type(e).__name__}: {e}", flush=True)


# ------------------------------------------------------------------------------------------------ diagnóstico
def _testar() -> int:
    ok, motivo = disponivel()
    if not ok:
        print(motivo)
        return 1
    fab = FabricaWasapi()
    try:
        for tipo, rotulo in (("mic", "microfone"), ("sistema", "loopback")):
            d = fab.abrir(tipo)
            if d is None:
                print(f"{rotulo}: não abriu")
                continue
            time.sleep(2.0)
            x = d.ler()
            print(f"{rotulo}: taxa={d.taxa} Hz, canais={d.canais}, amostras_recebidas={'sim' if len(x) else 'nao'} ({len(x)})")
            d.fechar()
    finally:
        fab.encerrar()
    return 0


if __name__ == "__main__":
    sys.exit(_testar() if "--testar" in sys.argv else 2)
