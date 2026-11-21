"""Monta o conjunto de teste de diarização (áudio + RTTM com verdade conhecida) em testes/diarizacao/dados/.

Fontes (todas abertas, CC-BY-4.0):
  * MLS português (Multilingual LibriSpeech, Facebook, CC-BY-4.0): 10 locutores reais lendo audiolivros.
    Com eles são montadas conversas SINTÉTICAS (2, 3 e 4 pessoas) com turnos curtos, pausas e sobreposição
    de tempos conhecidos (semente fixa, reproduzível).
  * AMI (reuniões reais em inglês, 4 pessoas, microfone distante "sdm", CC-BY-4.0), anotação do
    dataset `diarizers-community/ami` no Hugging Face: 3 trechos de 5 min, com sobreposição real.

Uso:  .venv\\Scripts\\python testes\\diarizacao\\preparar_dados.py   (AMI exige `pip install pyarrow`; se faltar, só o MLS é montado)
Baixa ~210 MB (MLS) + ~330 MB (AMI) para testes/diarizacao/cache/ (gitignored) e gera ~25 min de áudio.
"""
from __future__ import annotations

import io
import random
import sys
import tarfile
import urllib.request
import wave
from pathlib import Path

import numpy as np
from faster_whisper.audio import decode_audio

AQUI = Path(__file__).resolve().parent
CACHE = AQUI / "cache"
DADOS = AQUI / "dados"
SR = 16000

_HF = "https://huggingface.co/datasets"
MLS_BASE = f"{_HF}/facebook/multilingual_librispeech/resolve/main/data/mls_portuguese/test/audio"
MLS_ARQUIVOS = ["11247_10229_000", "12114_10229_000", "12287_12742_000", "12626_10229_000", "12670_13396_000",
                "13063_13511_000", "3050_2941_000", "4405_3604_000", "5739_4739_000", "7925_6390_000"]
AMI_URL = f"{_HF}/diarizers-community/ami/resolve/main/sdm/test-00000-of-00003.parquet"
# (linha da reunião no parquet, início s, duração s)
AMI_TRECHOS = [(5, 60, 300), (1, 400, 300), (0, 300, 300)]


def baixar(url: str, destino: Path) -> Path:
    if not destino.exists():
        destino.parent.mkdir(parents=True, exist_ok=True)
        print(f"baixando {url}")
        urllib.request.urlretrieve(url, destino)
    return destino


def salvar_wav(caminho: Path, audio: np.ndarray) -> None:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    pcm = (np.clip(audio, -1, 1) * 32767).astype("<i2")
    with wave.open(str(caminho), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


def salvar_rttm(caminho: Path, nome: str, trechos: list[tuple[float, float, str]]) -> None:
    linhas = [f"SPEAKER {nome} 1 {i:.3f} {f - i:.3f} <NA> <NA> {loc} <NA> <NA>"
              for i, f, loc in sorted(trechos) if f - i > 0.01]
    caminho.write_text("\n".join(linhas) + "\n", encoding="utf-8")


# ---------------------------------------------------------------- MLS (sintético)
def _regioes_fala(x: np.ndarray, pausa_min: float = 0.3) -> list[tuple[float, float]]:
    """VAD por energia (quadros de 20 ms): regiões de fala, juntando pausas menores que `pausa_min`."""
    q = int(0.02 * SR)
    n = len(x) // q
    rms = np.sqrt(np.mean(x[: n * q].reshape(n, q) ** 2, axis=1) + 1e-12)
    db = 20 * np.log10(rms)
    ativo = db > (np.percentile(db, 95) - 28)
    regioes, ini = [], None
    for i, a in enumerate(ativo):
        if a and ini is None:
            ini = i
        if not a and ini is not None:
            regioes.append([ini * 0.02, i * 0.02])
            ini = None
    if ini is not None:
        regioes.append([ini * 0.02, n * 0.02])
    fund: list[list[float]] = []
    for r in regioes:
        if fund and r[0] - fund[-1][1] < pausa_min:
            fund[-1][1] = r[1]
        else:
            fund.append(r)
    return [(a, b) for a, b in fund if b - a >= 0.25]


def montar_mls(n_locutores: int, duracao: float, semente: int, falantes: dict[str, list[Path]]) -> tuple[np.ndarray, list]:
    rng = random.Random(semente)
    ids = rng.sample(sorted(falantes), n_locutores)
    cache: dict[Path, np.ndarray] = {}

    def carregar(p: Path) -> np.ndarray:
        if p not in cache:
            cache[p] = decode_audio(str(p), sampling_rate=SR)
        return cache[p]

    saida = np.zeros(int((duracao + 30) * SR), dtype=np.float32)
    ref: list[tuple[float, float, str]] = []
    t, anterior, fim_ant, dur_ant = 0.5, None, 0.5, 0.0
    while t < duracao:
        loc = rng.choice([i for i in ids if i != anterior] if anterior else ids)
        x = carregar(rng.choice(falantes[loc]))
        dur = min(len(x) / SR - 0.1, rng.choice([1.5, 2.5, 4, 6, 9, 12]) * rng.uniform(0.7, 1.2))
        ini = rng.uniform(0, max(0.0, len(x) / SR - dur))
        clip = x[int(ini * SR): int((ini + dur) * SR)].copy()
        clip *= 10 ** (rng.uniform(-29, -23) / 20) / (np.sqrt(np.mean(clip ** 2)) + 1e-9)   # ~ -26 dBFS RMS
        # sobreposição (≈ 18% das trocas) ou pausa entre turnos
        if anterior is not None and dur_ant > 2.0 and rng.random() < 0.18:
            t = max(fim_ant - rng.uniform(0.4, 1.5), 0.0)
        else:
            t = fim_ant + rng.uniform(0.1, 1.2)
        a = int(t * SR)
        saida[a: a + len(clip)] += clip
        for r0, r1 in _regioes_fala(clip):
            ref.append((t + r0, t + r1, loc))
        anterior, fim_ant, dur_ant = loc, t + len(clip) / SR, len(clip) / SR
        t = fim_ant
    saida = saida[: int(fim_ant * SR) + SR]
    pico = np.abs(saida).max()
    if pico > 0.95:
        saida *= 0.95 / pico
    nomes = {loc: f"S{i + 1}" for i, loc in enumerate(ids)}
    return saida, [(a, b, nomes[l]) for a, b, l in ref]


def preparar_mls() -> None:
    pasta = CACHE / "mls"
    falantes: dict[str, list[Path]] = {}
    for nome in MLS_ARQUIVOS:
        tgz = baixar(f"{MLS_BASE}/{nome}.tar.gz", CACHE / "mls_tgz" / f"{nome}.tar.gz")
        destino = pasta / nome
        if not destino.exists():
            with tarfile.open(tgz) as t:
                t.extractall(destino, filter="data")
        loc = nome.split("_")[0]
        falantes.setdefault(loc, []).extend(sorted(destino.rglob("*.flac")))
    for n, dur, semente in [(2, 240, 11), (3, 270, 22), (4, 300, 33)]:
        audio, ref = montar_mls(n, dur, semente, falantes)
        nome = f"mls-pt-{n}loc"
        salvar_wav(DADOS / f"{nome}.wav", audio)
        salvar_rttm(DADOS / f"{nome}.rttm", nome, ref)
        print(f"{nome}: {len(audio) / SR:.0f}s, {n} locutores, {len(ref)} trechos")


# ---------------------------------------------------------------- AMI (real)
def preparar_ami() -> None:
    try:
        import pyarrow.parquet as pq
    except ImportError:
        print("pyarrow ausente: pulando AMI (pip install pyarrow)")
        return
    arq = baixar(AMI_URL, CACHE / "ami_sdm_test0.parquet")
    tabela = pq.read_table(arq)
    for linha, ini, dur in AMI_TRECHOS:
        r = tabela.slice(linha, 1).to_pylist()[0]
        audio = decode_audio(io.BytesIO(r["audio"]["bytes"]), sampling_rate=SR)
        trechos = []
        for a, b, s in zip(r["timestamps_start"], r["timestamps_end"], r["speakers"]):
            a, b = max(a, ini), min(b, ini + dur)
            if b - a > 0.01:
                trechos.append((a - ini, b - ini, s))
        nome = f"ami-sdm-{linha}"
        salvar_wav(DADOS / f"{nome}.wav", audio[int(ini * SR): int((ini + dur) * SR)])
        salvar_rttm(DADOS / f"{nome}.rttm", nome, trechos)
        print(f"{nome}: {dur}s, {len({s for _, _, s in trechos})} locutores, {len(trechos)} trechos")


if __name__ == "__main__":
    preparar_mls()
    preparar_ami()
    print("Pronto:", DADOS)
