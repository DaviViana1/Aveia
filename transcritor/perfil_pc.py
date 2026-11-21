"""Sonda do computador: o que o PC tem e se ele aguenta o Whisper deste app.

Duas partes:
- `hardware()` e `avaliar_hardware()`: leitura rápida (processador, RAM, disco, Windows), só com a biblioteca padrão. Roda antes de
  instalar qualquer coisa (o `iniciar.bat` chama `python transcritor\\perfil_pc.py --rapido` com o Python do sistema).
- `medir_janela()`: teste real. Carrega o modelo e transcreve 10 s de ruído fraco, medindo o tempo de UMA janela de 30 s (o Whisper
  gasta a janela inteira mesmo com áudio curto). O fator é esse tempo dividido por 30. Resultado em `dados/computador.json`.

Faixas (hipótese, calibradas só com um PC; ver README): fator até 0,8 = ao vivo; até 2 = só depois da gravação; acima = lento.
Nada sai do computador e nada daqui guarda texto, áudio ou nome.
"""
from __future__ import annotations

import ctypes
import json
import os
import platform
import shutil
import struct
import sys
import time
from datetime import datetime
from pathlib import Path

RAM_MIN_GB = 4.0        # abaixo disso o modelo (int8) não cabe com folga
DISCO_MIN_GB = 6.0      # modelo (1,6 GB) + ambiente Python + gravações
FATOR_AO_VIVO = 0.8     # uma janela de 30 s processada em até 24 s: dá para acompanhar a fala
FATOR_DEPOIS = 2.0      # até 60 s por janela: transcrever depois da gravação ainda é razoável
JANELA_S = 30.0
TESTE_S = 10.0

_NOMES = {"ao_vivo": "Ao vivo", "depois": "Só depois da gravação", "lento": "Lento"}


def _pasta_dados() -> Path:
    return Path(os.environ.get("AVEIA_DADOS") or Path(__file__).resolve().parent.parent / "dados")


class _MemStatus(ctypes.Structure):
    _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong), ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong), ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong), ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong), ("sullAvailExtendedVirtual", ctypes.c_ulonglong)]


def _ram_gb() -> tuple[float | None, float | None]:
    """(total, livre) em GB, ou (None, None) fora do Windows ou se a leitura falhar."""
    try:
        m = _MemStatus()
        m.dwLength = ctypes.sizeof(_MemStatus)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):
            return None, None
        return round(m.ullTotalPhys / 2**30, 1), round(m.ullAvailPhys / 2**30, 1)
    except Exception:  # noqa: BLE001
        return None, None


def _avx2() -> bool | None:
    """True/False se o Windows diz que o processador tem AVX2 (PF_AVX2_INSTRUCTIONS_AVAILABLE = 40); None se não deu para saber."""
    try:
        return bool(ctypes.windll.kernel32.IsProcessorFeaturePresent(40))
    except Exception:  # noqa: BLE001
        return None


def _nome_cpu() -> str:
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as k:
            return str(winreg.QueryValueEx(k, "ProcessorNameString")[0]).strip()
    except Exception:  # noqa: BLE001
        return platform.processor() or "desconhecido"


def _windows() -> tuple[str, int | None]:
    """("Windows 11 (build 26200)", 26200). O Python mais antigo diz "10" também no Windows 11; o número da versão decide."""
    try:
        b = sys.getwindowsversion().build
        return f"{'Windows 11' if b >= 22000 else 'Windows 10'} (build {b})", b
    except Exception:  # noqa: BLE001
        return platform.platform(terse=True), None


def hardware(pasta: str | os.PathLike | None = None) -> dict:
    """O que o computador tem. Só números e nomes de componente; nada pessoal."""
    ram_total, ram_livre = _ram_gb()
    alvo = Path(pasta) if pasta else Path(__file__).resolve().parent.parent
    try:
        livre = round(shutil.disk_usage(alvo).free / 2**30, 1)
    except OSError:
        livre = None
    nome_windows, build = _windows()
    return {
        "windows": nome_windows,
        "windows_build": build,
        "python": platform.python_version(),
        "python_64_bits": struct.calcsize("P") == 8,
        "arquitetura": platform.machine(),
        "cpu": _nome_cpu(),
        "nucleos": os.cpu_count(),
        "avx2": _avx2(),
        "ram_total_gb": ram_total,
        "ram_livre_gb": ram_livre,
        "disco_livre_gb": livre,
    }


def avaliar_hardware(h: dict) -> dict:
    """`problemas` impedem (ou deixam o app inviável); `avisos` só pedem atenção. `ok` é True sem problemas."""
    problemas: list[str] = []
    avisos: list[str] = []
    if h.get("arquitetura") and str(h["arquitetura"]).upper() not in ("AMD64", "X86_64"):
        problemas.append(f"Processador {h['arquitetura']}: o Aveia foi feito e testado para Windows de 64 bits comum (AMD64).")
    if h.get("python_64_bits") is False:
        problemas.append("Python de 32 bits: instale o Python de 64 bits.")
    if h.get("windows_build") is not None and h["windows_build"] < 19041:
        problemas.append("Windows antigo demais: use o Windows 10 (versão 22H2) ou o Windows 11.")
    elif h.get("windows_build") is not None and h["windows_build"] < 19045:
        avisos.append("Atualize o Windows 10 para a versão 22H2 (Windows Update).")
    if h.get("avx2") is False:
        problemas.append("O processador não tem AVX2: o Whisper fica lento demais.")
    if h.get("ram_total_gb") is not None and h["ram_total_gb"] < RAM_MIN_GB:
        problemas.append(f"Só {h['ram_total_gb']} GB de memória (RAM): o mínimo é {RAM_MIN_GB:g} GB.")
    elif h.get("ram_total_gb") is not None and h["ram_total_gb"] < 8:
        avisos.append("Com menos de 8 GB de memória o computador pode ficar lento durante a transcrição.")
    if h.get("disco_livre_gb") is not None and h["disco_livre_gb"] < DISCO_MIN_GB:
        problemas.append(f"Só {h['disco_livre_gb']} GB livres no disco: precisa de pelo menos {DISCO_MIN_GB:g} GB.")
    if h.get("nucleos") is not None and h["nucleos"] < 4:
        avisos.append("Menos de 4 núcleos: a transcrição tende a ser mais lenta que a gravação.")
    return {"ok": not problemas, "problemas": problemas, "avisos": avisos}


def classificar(fator: float) -> str:
    """ao_vivo, depois ou lento, pelo fator (tempo de uma janela de 30 s dividido por 30)."""
    if fator <= FATOR_AO_VIVO:
        return "ao_vivo"
    if fator <= FATOR_DEPOIS:
        return "depois"
    return "lento"


def nome_da_classe(classe: str) -> str:
    return _NOMES.get(classe, classe)


def medir_janela(modelo=None) -> dict:
    """Teste real: tempo de uma janela de 30 s com o Whisper deste app. `modelo` (já carregado) é para testes."""
    import numpy as np

    t0 = time.monotonic()
    if modelo is None:
        from . import pipeline
        modelo = pipeline._carregar_modelo(pipeline.MODELO)
    carga_s = time.monotonic() - t0
    ruido = (np.random.default_rng(0).standard_normal(int(16000 * TESTE_S)) * 0.01).astype("float32")
    t1 = time.monotonic()
    segmentos, _ = modelo.transcribe(ruido, language="pt", beam_size=1, vad_filter=False,
                                     condition_on_previous_text=False, without_timestamps=True)
    list(segmentos)
    janela_s = time.monotonic() - t1
    fator = janela_s / JANELA_S
    return {"quando": datetime.now().isoformat(timespec="seconds"), "carga_modelo_s": round(carga_s, 1),
            "janela_s": round(janela_s, 1), "fator": round(fator, 2), "classe": classificar(fator)}


def salvar_teste(teste: dict) -> None:
    try:
        pasta = _pasta_dados()
        pasta.mkdir(parents=True, exist_ok=True)
        (pasta / "computador.json").write_text(json.dumps(teste, ensure_ascii=False), encoding="utf-8")
    except OSError:
        pass   # o resultado só some; nada quebra


def ultimo_teste() -> dict | None:
    try:
        d = json.loads((_pasta_dados() / "computador.json").read_text(encoding="utf-8"))
        return d if isinstance(d, dict) and "fator" in d else None
    except (OSError, ValueError):
        return None


def estado() -> dict:
    """Tudo o que a tela mostra: hardware, avaliação e o último teste real (se houver)."""
    h = hardware()
    t = ultimo_teste()
    return {"hardware": h, "avaliacao": avaliar_hardware(h), "teste": t,
            "classe_nome": nome_da_classe(t["classe"]) if t else None}


def _texto_rapido() -> tuple[str, int]:
    h = hardware()
    a = avaliar_hardware(h)
    linhas = [
        f"  Windows:   {h['windows']}, {h['arquitetura']}",
        f"  Python:    {h['python']}" + ("" if h["python_64_bits"] else " (32 bits)"),
        f"  Processador: {h['cpu']} ({h['nucleos']} núcleos, AVX2: {'sim' if h['avx2'] else 'não' if h['avx2'] is False else '?'})",
        f"  Memória:   {h['ram_total_gb']} GB no total, {h['ram_livre_gb']} GB livres",
        f"  Disco:     {h['disco_livre_gb']} GB livres",
    ]
    if a["problemas"]:
        linhas += ["", "  Este computador provavelmente NÃO aguenta o Aveia:"] + [f"   - {p}" for p in a["problemas"]]
    else:
        linhas += ["", "  Os requisitos básicos estão ok. O teste de velocidade fica em Ajustes, em Seu computador."]
    linhas += [f"   - {x}" for x in a["avisos"]]
    return "\n".join(linhas), (0 if a["ok"] else 2)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if "--testar" in sys.argv:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from transcritor import perfil_pc as _p
        r = _p.medir_janela()
        _p.salvar_teste(r)
        print(f"Janela de 30 s em {r['janela_s']} s (fator {r['fator']}): {_p.nome_da_classe(r['classe'])}")
        sys.exit(0)
    texto, codigo = _texto_rapido()
    print(texto)
    sys.exit(codigo)
