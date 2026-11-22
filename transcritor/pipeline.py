"""Junta transcrição (faster-whisper) e diarização (sherpa-onnx)."""
from __future__ import annotations

import json
import math
import os
import re
import threading
import time
from dataclasses import asdict, dataclass
from typing import Callable, Iterable

import numpy as np
from faster_whisper import WhisperModel
from faster_whisper.audio import decode_audio

from . import canais as canais_mod
from . import motores
from .audio import preparar as preparar_audio
from .diarizacao import Trecho, diarizar, ultimo_motor

SAMPLE_RATE = 16000
MODELO = "large-v3-turbo"   # único modelo: no áudio de teste foi o mais preciso e tão rápido quanto o small
_modelos: dict[str, WhisperModel] = {}
_trava_modelo = threading.Lock()

LOCUTOR_PENDENTE = "Locutor ?"   # falas provisórias, ainda sem diarização
CANAIS_MIC_SISTEMA = "mic+sistema"   # arquivo estéreo: microfone à esquerda, áudio do computador à direita
LOCUTOR_LOCAL = "Locutor 1"      # com dois canais, o microfone é sempre o Locutor 1
_ID_LOCAL = -1                   # id interno do usuário (os da diarização são >= 0)
PAUSA_NOVA_FALA = 30.0           # s: só uma pausa de 30 s ou mais separa duas falas do mesmo locutor (mesmo timbre + pausa menor = mesma fala)
FALA_MAX_S = 300.0              # s: teto de uma fala (5 min); passou disso, a fala seguinte é do mesmo locutor
_JANELA_CORTE_TETO_S = 60.0     # s: o corte do teto cai na maior pausa dos últimos 60 s da fala
_PAUSA_NITIDA_S = 0.5           # s: pausa mínima para valer como ponto de corte do teto

# Suavização de "ilhas" (1–2 palavras atribuídas a outro locutor no meio da fala de alguém).
SUAVIZAR_MAX_PALAVRAS = 2        # tamanho máximo da ilha
SUAVIZAR_MAX_LACUNA = 0.3        # s: pausa máxima entre a ilha e cada palavra vizinha
_PONTUACAO_FINAL = (".", "?", "!", "…")
_FECHAMENTOS = "\"'”’»)]} "

# Faixas do progresso: leitura 0,02; transcrição 0,05→0,80; diarização 0,80→0,97; pronto 1,0.
_TRANSC_INI, _TRANSC_FIM = 0.05, 0.80
_DIAR_INI, _DIAR_FIM = 0.80, 0.97
_INTERVALO_PARCIAL = 1.0         # s entre chamadas de `parcial`


@dataclass
class Fala:
    inicio: float
    fim: float
    locutor: str
    texto: str


def _carregar_modelo(nome: str) -> WhisperModel:
    if nome in _modelos:
        return _modelos[nome]
    with _trava_modelo:   # nunca constrói o modelo duas vezes (aquecimento em paralelo com um trabalho)
        if nome not in _modelos:
            # cpu_threads: o padrão do CTranslate2 usa só 4 threads; o teste foi feito com todas.
            _modelos[nome] = WhisperModel(nome, device="auto", compute_type="int8", cpu_threads=os.cpu_count() or 0)
        return _modelos[nome]


def aquecer(em_segundo_plano: bool = True) -> None:
    """Carrega o modelo Whisper antes do primeiro trabalho. Nunca levanta exceção."""
    def _carregar() -> None:
        try:
            _carregar_modelo(MODELO)
            print(f"Modelo Whisper '{MODELO}' pronto.")
        except Exception as e:  # noqa: BLE001
            print(f"Aquecimento do modelo Whisper falhou: {e}")

    if em_segundo_plano:
        try:
            threading.Thread(target=_carregar, name="aquecer-whisper", daemon=True).start()
        except Exception as e:  # noqa: BLE001
            print(f"Não foi possível iniciar o aquecimento do modelo: {e}")
    else:
        _carregar()


# Ajuda o Whisper a acertar vocabulário e pontuação em reuniões em português (não é "texto a repetir").
DICA_PT = "Reunião de trabalho em português do Brasil, com números, orçamento e faturamento."


# ---- Glossário do usuário (`hotwords` do faster-whisper) --------------------------------------------------------------
# No faster-whisper o `hotwords` entra no prompt do decodificador como " termo, termo" depois do marcador "anterior"; ele
# corta em `max_length // 2 - 1` = 223 tokens (transcribe.py, get_prompt). Em português um token tem ~2 a 3 caracteres e
# um termo curto custa 2 a 5 tokens, então o corte é feito aqui, de forma determinística e sem carregar o modelo, por
# caracteres e número de termos, bem abaixo do limite de tokens. Não é correção de truncamento (ver RESULTADOS.md).
GLOSSARIO_MAX_TERMOS = 40
GLOSSARIO_MAX_CHARS = 300        # do texto final (termos juntados com ", ")
GLOSSARIO_TERMO_MAX_CHARS = 40   # termo mais comprido que isto não é "nome ou termo curto": é descartado


def normalizar_glossario(texto) -> str:
    """Glossário em texto livre (termos separados por vírgula, ponto e vírgula ou quebra de linha) -> "a, b, c".

    Apara e junta espaços, remove vazios e duplicados (sem diferenciar maiúsculas; vale a primeira grafia), descarta
    termos com mais de GLOSSARIO_TERMO_MAX_CHARS e para ao atingir GLOSSARIO_MAX_TERMOS ou GLOSSARIO_MAX_CHARS (os
    termos que não cabem são ignorados, na ordem dada). Devolve "" se não sobrar nada ou se a entrada não for texto."""
    if not isinstance(texto, str):
        return ""
    termos: list[str] = []
    vistos: set[str] = set()
    tamanho = 0
    for bruto in re.split(r"[,;\r\n]+", texto):
        termo = " ".join("".join(c for c in bruto if c.isprintable() or c.isspace()).split())
        if not termo or len(termo) > GLOSSARIO_TERMO_MAX_CHARS or termo.casefold() in vistos:
            continue
        novo = tamanho + len(termo) + (2 if termos else 0)
        if len(termos) >= GLOSSARIO_MAX_TERMOS or novo > GLOSSARIO_MAX_CHARS:
            break
        termos.append(termo)
        vistos.add(termo.casefold())
        tamanho = novo
    return ", ".join(termos)


def _opcoes_decodificacao(idioma: str | None, glossario: str | None = None) -> dict:
    """Parâmetros do `transcribe` (os mesmos para o áudio inteiro, para o bloco ao vivo e para a retranscrição de lacunas).

    `glossario` (já normalizado ou não: é normalizado de novo, de forma idempotente) vira `hotwords`; vazio não acrescenta
    nenhuma chave, e os kwargs ficam idênticos aos de antes do glossário."""
    opcoes = dict(
        language=idioma or None, word_timestamps=True, beam_size=5,
        vad_filter=True, vad_parameters=dict(min_silence_duration_ms=500, speech_pad_ms=300),
        # Contra alucinação e repetição em áudio ruim/sobreposto: não propaga um erro para o trecho seguinte.
        condition_on_previous_text=False, hallucination_silence_threshold=2.0,
        no_speech_threshold=0.5, compression_ratio_threshold=2.2,
        initial_prompt=DICA_PT if idioma == "pt" else None,
    )
    hotwords = normalizar_glossario(glossario)
    if hotwords:
        opcoes["hotwords"] = hotwords
    return opcoes


# ---- Defesa contra truncamento: retranscrição de lacunas ------------------------------------------------------------
# Em áudio de conversa (pausas, troca de locutor) o Whisper às vezes pula janelas inteiras (medido: 7 de 13 arquivos de
# conversa, 0 de 8 de fala contínua; testes/resultados-local/RESULTADOS.md). A rede de segurança: um trecho sem palavras
# onde o VAD vê fala é transcrito de novo, sozinho. `AVEIA_LACUNAS=0` desliga tudo.
LACUNA_MIN_S = 3.0           # s sem palavras (entre duas palavras, no começo ou no fim) que fazem suspeitar de perda
LACUNA_FALA_MIN_S = 2.0      # ...e o VAD precisa ver pelo menos isto de fala dentro dela
LACUNA_FOLGA_S = 0.3         # áudio extra de cada lado do trecho retranscrito (as palavras são filtradas por tempo)
# Piso de palavras por segundo de fala do VAD para aceitar o texto recuperado. Apoia-se em TRÊS pontos medidos: a falha
# ("Obrigado por assistir.", 3 palavras para 26,2 s de fala = 0,11) e dois acertos (48 palavras em 20,5 s = 2,3 e 31 em
# 26,2 s = 1,2). Qualquer valor entre 0,12 e 1,2 separa os três; 0,5 é só um ponto do meio, não uma calibração.
PALAVRAS_POR_S_MIN = 0.5
PALAVRAS_POR_S_PLENO = 1.0   # aceito, mas abaixo disto o trecho conta como "parcial" (mesma fragilidade: heurística)
PEDACO_MAX_S = 8.0           # 2ª tentativa: pedaços de fala de no máximo isto (VAD). Medido com 8 s: cv wav 3 -> 32 palavras
PEDACO_MIN_FALA_S = 0.4      # pedaço com menos fala que isto não é transcrito (o Whisper alucina em silêncio)
MAX_AVISOS_LACUNAS = 6       # avisos individuais por transcrição; o resto vira uma linha só

# Frases que o Whisper inventa em trechos que não entendeu (legendas de vídeo). Lista curta de propósito: só entram frases
# vistas nas medições ("obrigado por assistir" e "tchau" em testes/resultados-local/bruto/rec__*) ou muito conhecidas.
ALUCINACOES = (
    "obrigado por assistir",
    "legendas pela comunidade amara.org",
    "tchau",
)


def _lacunas_ligadas() -> bool:
    return os.environ.get("AVEIA_LACUNAS", "1").strip() != "0"


def _trechos_fala(audio: np.ndarray) -> list[tuple[float, float]]:
    """Trechos de fala (s) pelo Silero VAD do faster-whisper, com os mesmos parâmetros da transcrição."""
    from faster_whisper.vad import VadOptions, get_speech_timestamps
    opc = VadOptions(min_silence_duration_ms=500, speech_pad_ms=300)
    return [(t["start"] / SAMPLE_RATE, t["end"] / SAMPLE_RATE) for t in get_speech_timestamps(audio, opc)]


def _fala_em(fala: list[tuple[float, float]], a: float, b: float) -> float:
    return sum(max(0.0, min(f, b) - max(i, a)) for i, f in fala)


def _norm(texto: str) -> list[str]:
    """Palavras em minúsculas, sem pontuação (mesma ideia de testes/wer.py)."""
    t = texto.lower().replace("’", "").replace("'", "")
    return re.sub(r"[^\w\s]", " ", t).split()


def _eco_da_dica(palavras: list[str]) -> bool:
    """O texto recuperado repete a dica inicial (DICA_PT)? Texto curto: contido na dica. Texto longo: metade ou mais das
    palavras cai em trigramas da dica (um trigrama solto num texto longo pode ser fala de verdade)."""
    dica = _norm(DICA_PT)
    if len(palavras) < 3:
        return len(palavras) >= 2 and " ".join(palavras) in " ".join(dica)
    tri = {tuple(dica[i:i + 3]) for i in range(len(dica) - 2)}
    cobertas: set[int] = set()
    for i in range(len(palavras) - 2):
        if tuple(palavras[i:i + 3]) in tri:
            cobertas.update((i, i + 1, i + 2))
    return len(cobertas) >= 0.5 * len(palavras)


def _so_alucinacao(palavras: list[str]) -> bool:
    """O texto é só frase(s) de legenda inventada (ALUCINACOES), possivelmente repetida(s)?"""
    texto = " " + " ".join(palavras) + " "
    for frase in ALUCINACOES:
        texto = texto.replace(" " + " ".join(_norm(frase)) + " ", " ")
        texto = texto.replace(" " + " ".join(_norm(frase)) + " ", " ")   # 2x: ocorrências coladas compartilham o espaço
    return not texto.strip()


def _aceitavel(palavras: list, fala_s: float) -> bool:
    """Filtro do texto recuperado: não vazio, palavras por segundo de fala acima do piso, não é eco da dica nem só
    alucinação de legenda."""
    norm = [x for p in palavras for x in _norm(p.word)]
    if not norm or fala_s <= 0 or len(norm) / fala_s < PALAVRAS_POR_S_MIN:
        return False
    return not _eco_da_dica(norm) and not _so_alucinacao(norm)


def _buracos_com_fala(tempos: list[tuple[float, float]], ini: float, fim: float,
                      fala: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Lacunas (> LACUNA_MIN_S) entre `ini` e `fim` dadas as palavras (início, fim) com pelo menos LACUNA_FALA_MIN_S de fala."""
    pontos = [(ini, ini)] + sorted(tempos) + [(fim, fim)]
    return [(b0, a1) for (_a0, b0), (a1, _b1) in zip(pontos, pontos[1:])
            if a1 - b0 > LACUNA_MIN_S and _fala_em(fala, b0, a1) >= LACUNA_FALA_MIN_S]


def _cortes_em_pedacos(audio: np.ndarray, a: float, b: float, fala: list[tuple[float, float]],
                       maximo: float = PEDACO_MAX_S) -> list[float]:
    """Fronteiras [a, c1, ..., b] que dividem [a, b] em pedaços de fala de no máximo `maximo` s, cortando no meio das
    pausas do VAD (fala contínua mais longa que `maximo` é cortada no ponto mais quieto perto da divisão igual)."""
    segs: list[tuple[float, float]] = []
    for i, f in fala:
        i, f = max(i, a), min(f, b)
        if f - i <= 0:
            continue
        n = math.ceil((f - i) / maximo)
        pontos = [i]
        for k in range(1, n):
            alvo = i + (f - i) * k / n
            pontos.append(motores._ponto_quieto(audio, alvo, raio_s=min(1.0, (f - i) / (4 * n))))
        pontos.append(f)
        segs += [(x, y) for x, y in zip(pontos, pontos[1:]) if y > x]
    if not segs:
        return [a, b]
    pedacos = [list(segs[0])]
    for i, f in segs[1:]:
        if f - pedacos[-1][0] <= maximo:
            pedacos[-1][1] = f
        else:
            pedacos.append([i, f])
    meios = [(p[1] + q[0]) / 2 for p, q in zip(pedacos, pedacos[1:])]
    return [a] + meios + [b]


def _palavras_do_trecho(decodificar, audio: np.ndarray, ini: float, fim: float, de: float, ate: float) -> list:
    """Transcreve `audio[ini - folga : fim + folga]` e devolve as palavras (tempo absoluto) que COMEÇAM em [de, ate)."""
    total = len(audio) / SAMPLE_RATE
    a0, b0 = max(0.0, ini - LACUNA_FOLGA_S), min(total, fim + LACUNA_FOLGA_S)
    fatia = audio[int(a0 * SAMPLE_RATE):int(b0 * SAMPLE_RATE)]
    if len(fatia) == 0:
        return []
    return [motores.Palavra(w.start + a0, w.end + a0, w.word) for w in decodificar(fatia)
            if de <= w.start + a0 < ate]


def _recuperar_em_pedacos(decodificar, audio: np.ndarray, a: float, b: float, fala: list[tuple[float, float]],
                          maximo: float = PEDACO_MAX_S) -> list:
    """Retranscreve [a, b] em pedaços de fala de até `maximo` s, cortados nas pausas do VAD. Palavras em tempo absoluto,
    cada uma no pedaço onde começa (sem duplicar na emenda). Sem filtro: quem chama decide se aceita."""
    cortes = _cortes_em_pedacos(audio, a, b, fala, maximo)
    novas: list = []
    for ini, fim in zip(cortes, cortes[1:]):
        if _fala_em(fala, ini, fim) < PEDACO_MIN_FALA_S:
            continue
        de = a - 0.05 if ini == a else ini
        vozes = [(max(i, ini), min(f, fim)) for i, f in fala if min(f, fim) > max(i, ini)]
        # a fatia vai só de meio segundo antes da 1ª fala até meio segundo depois da última (sem silêncio inútil)
        novas += _palavras_do_trecho(decodificar, audio, max(ini, vozes[0][0] - 0.5), min(fim, vozes[-1][1] + 0.5), de, fim)
    return novas


def _recuperar_lacuna(decodificar, audio: np.ndarray, a: float, b: float, fala: list[tuple[float, float]]
                      ) -> tuple[list, dict]:
    """Retranscreve [a, b] (lacuna com fala). Devolve (palavras em ordem de tempo, diagnóstico). 1ª tentativa: o trecho
    inteiro como bloco independente. Se o texto não passa em `_aceitavel`, 2ª tentativa: pedaços de até PEDACO_MAX_S."""
    fala_s = _fala_em(fala, a, b)
    novas = _palavras_do_trecho(decodificar, audio, a, b, a - 0.05, b)
    aceito = _aceitavel(novas, fala_s)
    if not aceito:
        novas = _recuperar_em_pedacos(decodificar, audio, a, b, fala)
        aceito = _aceitavel(novas, fala_s)
    diag = {"inicio": round(float(a), 2), "fim": round(float(b), 2), "fala_s": round(float(fala_s), 1), "palavras_recuperadas": 0,
            "estado": "nao_recuperada"}
    if not aceito:
        return [], diag
    novas.sort(key=lambda p: p.start)
    n = sum(len(_norm(p.word)) for p in novas)
    diag["palavras_recuperadas"] = n
    sobra = _buracos_com_fala([(p.start, p.end) for p in novas], a, b, fala)
    diag["estado"] = "recuperada" if n / fala_s >= PALAVRAS_POR_S_PLENO and not sobra else "parcial"
    return novas, diag


def _palavras(audio: np.ndarray, duracao: float, idioma: str | None, avisar, lacunas: list | None = None,
              glossario: str | None = None):
    """Whisper local (faster-whisper). Gera palavras com tempo, atualizando o progresso.

    Rede de segurança contra truncamento (ver LACUNA_*): quando entre duas palavras (ou no começo/fim do áudio) há mais de
    LACUNA_MIN_S sem texto e o VAD vê pelo menos LACUNA_FALA_MIN_S de fala, o trecho é transcrito de novo e as palavras
    recuperadas saem NO LUGAR, antes da palavra seguinte: a saída continua em ordem de tempo. Só áudio de lacuna é
    retranscrito. Cada lacuna detectada é anotada em `lacunas` (se dada) como {inicio, fim, fala_s, palavras_recuperadas,
    estado} com estado recuperada | parcial | nao_recuperada; tempos relativos ao `audio` recebido.

    `glossario` (opcional) chega ao Whisper como `hotwords`, também na retranscrição de lacunas (mesmas opções)."""
    avisar(0.25, f"Carregando modelo Whisper '{MODELO}'...")
    whisper = _carregar_modelo(MODELO)
    opcoes = _opcoes_decodificacao(idioma, glossario)
    segmentos, _ = whisper.transcribe(audio, **opcoes)

    ligada = _lacunas_ligadas()
    total = len(audio) / SAMPLE_RATE
    fala_cache: list[list[tuple[float, float]]] = []   # VAD do áudio inteiro, só calculado se aparecer um candidato

    def vad() -> list[tuple[float, float]]:
        if not fala_cache:
            try:
                fala_cache.append(_trechos_fala(audio))
            except Exception as e:  # noqa: BLE001
                print(f"VAD das lacunas falhou (lacunas não verificadas): {e}")
                fala_cache.append([])
        return fala_cache[0]

    def decodificar(fatia: np.ndarray) -> list:
        segs, _ = whisper.transcribe(fatia, **opcoes)
        return [w for s in segs for w in (s.words or [])]

    def lacuna(a: float, b: float, progresso: float):
        """Palavras recuperadas de [a, b] se for lacuna com fala (e registra), senão nada."""
        fala = vad()
        if _fala_em(fala, a, b) < LACUNA_FALA_MIN_S:
            return []
        avisar(progresso, "Recuperando um trecho sem texto...")
        try:
            novas, diag = _recuperar_lacuna(decodificar, audio, a, b, fala)
        except Exception as e:  # noqa: BLE001  a defesa nunca derruba a transcrição
            print(f"Retranscrição da lacuna {a:.1f}-{b:.1f}s falhou: {e}")
            novas, diag = [], {"inicio": round(float(a), 2), "fim": round(float(b), 2), "fala_s": round(float(_fala_em(fala, a, b)), 1),
                               "palavras_recuperadas": 0, "estado": "nao_recuperada"}
        if lacunas is not None:
            lacunas.append(diag)
        if diag["estado"] != "recuperada":
            print(f"Lacuna {a:.1f}-{b:.1f}s ({diag['fala_s']} s de fala): {diag['estado']}, "
                  f"{diag['palavras_recuperadas']} palavras.")
        return novas

    fim_anterior = 0.0
    for seg in segmentos:
        avisar(0.25 + 0.7 * min(seg.end / duracao, 1.0), "Transcrevendo...")
        for w in seg.words or []:
            if ligada and w.start - fim_anterior > LACUNA_MIN_S:
                yield from lacuna(fim_anterior, w.start, 0.25 + 0.7 * min(w.start / duracao, 1.0))
            yield w
            fim_anterior = w.end
    if ligada and total - fim_anterior > LACUNA_MIN_S:
        yield from lacuna(fim_anterior, total, 0.95)


def avisos_de_lacunas(lacunas: list[dict]) -> list[str]:
    """Avisos (pt-BR) das lacunas não resolvidas por completo, com o intervalo de tempo da gravação."""
    def mmss(s: float, para_cima: bool = False) -> str:
        s = int(math.ceil(s) if para_cima else s)
        h, resto = divmod(s, 3600)
        m, seg = divmod(resto, 60)
        return f"{h}:{m:02d}:{seg:02d}" if h else f"{m:02d}:{seg:02d}"

    ruins = [l for l in lacunas if l.get("estado") != "recuperada"]
    avisos = []
    for l in ruins[:MAX_AVISOS_LACUNAS]:
        faixa = f"{mmss(l['inicio'])}–{mmss(l['fim'], True)}"
        resto = ("havia fala e o texto não foi recuperado." if l["estado"] == "nao_recuperada"
                 else "havia fala e só parte do texto foi recuperada.")
        avisos.append(f"Trecho {faixa} pode estar incompleto: {resto}")
    if len(ruins) > MAX_AVISOS_LACUNAS:
        avisos.append(f"Mais {len(ruins) - MAX_AVISOS_LACUNAS} trechos podem estar incompletos (havia fala sem texto).")
    return avisos


def _locutor_da_palavra(inicio: float, fim: float, trechos: list[Trecho]) -> int | None:
    melhor, melhor_ov = None, 0.0
    for t in trechos:
        ov = min(fim, t.fim) - max(inicio, t.inicio)
        if ov > melhor_ov:
            melhor, melhor_ov = t.locutor, ov
    if melhor is not None:
        return melhor
    # palavra fora de qualquer trecho: usa o trecho mais próximo
    meio = (inicio + fim) / 2
    if not trechos:
        return None
    return min(trechos, key=lambda t: min(abs(meio - t.inicio), abs(meio - t.fim))).locutor


def _termina_frase(palavra: str) -> bool:
    return palavra.rstrip(_FECHAMENTOS).endswith(_PONTUACAO_FINAL)


def _corte_do_teto(g: list) -> int:
    """Índice (>= 1) em que partir a fala `g` que estourou FALA_MAX_S: a maior pausa entre palavras dos últimos 60 s;
    sem pausa nítida, o último fim de frase dessa janela; senão `len(g)` (a palavra nova abre a fala seguinte)."""
    de = g[0].start + FALA_MAX_S - _JANELA_CORTE_TETO_S
    candidatos = [k for k in range(1, len(g)) if g[k].start >= de]
    if candidatos:
        k = max(candidatos, key=lambda i: g[i].start - g[i - 1].end)
        if g[k].start - g[k - 1].end >= _PAUSA_NITIDA_S:
            return k
        for k in reversed(candidatos):
            if _termina_frase(g[k - 1].word):
                return k
    return len(g)


def _agrupar(palavras: list, nomes: list[str]) -> list[Fala]:
    """Junta palavras consecutivas com o mesmo nome em falas. Uma fala só termina por troca de locutor (timbre), por pausa
    de PAUSA_NOVA_FALA ou mais, ou ao passar do teto FALA_MAX_S (a próxima fala segue com o mesmo locutor)."""
    grupos: list[tuple[str, list]] = []
    for p, nome in zip(palavras, nomes):
        if grupos and grupos[-1][0] == nome and p.start - grupos[-1][1][-1].end < PAUSA_NOVA_FALA:
            g = grupos[-1][1]
            if p.end - g[0].start > FALA_MAX_S:
                k = _corte_do_teto(g)
                grupos.append((nome, g[k:]))
                del g[k:]
            grupos[-1][1].append(p)
        else:
            grupos.append((nome, [p]))
    return [Fala(g[0].start, g[-1].end, nome, "".join(w.word for w in g).strip()) for nome, g in grupos]


def falas_provisorias(palavras: Iterable) -> list[Fala]:
    """Agrupa só por pausa, sem locutor (todas com LOCUTOR_PENDENTE)."""
    lista = list(palavras)
    return _agrupar(lista, [LOCUTOR_PENDENTE] * len(lista))


def _suavizar_ilhas(palavras: list, ids: list) -> list:
    """Reatribui ilhas de 1–2 palavras ao locutor que as cerca. Conservador: na dúvida, não mexe.

    Regra: ilha = 1–2 palavras seguidas do locutor X; a palavra antes e a depois são do mesmo outro locutor Y;
    pausas até as duas vizinhas < SUAVIZAR_MAX_LACUNA; e a ilha não é uma fala completa (não vale quando a palavra
    anterior E a última da ilha terminam em pontuação final: ex. "Sim." depois de uma frase concluída).
    """
    n = len(palavras)
    saida = list(ids)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and ids[j + 1] == ids[i]:
            j += 1
        if (j - i + 1 <= SUAVIZAR_MAX_PALAVRAS and i > 0 and j + 1 < n
                and ids[i - 1] == ids[j + 1] and ids[i - 1] != ids[i]):
            lacuna_ant = palavras[i].start - palavras[i - 1].end
            lacuna_dep = palavras[j + 1].start - palavras[j].end
            autonoma = _termina_frase(palavras[i - 1].word) and _termina_frase(palavras[j].word)
            if lacuna_ant < SUAVIZAR_MAX_LACUNA and lacuna_dep < SUAVIZAR_MAX_LACUNA and not autonoma:
                for k in range(i, j + 1):
                    saida[k] = ids[i - 1]
        i = j + 1
    return saida


def normalizar_canais(canais: str | None) -> str | None:
    """Só o valor conhecido vale; qualquer outro é tratado como mono (None)."""
    return CANAIS_MIC_SISTEMA if (canais or "").strip().lower() == CANAIS_MIC_SISTEMA else None


def _locutor_com_locais(inicio: float, fim: float, trechos: list[Trecho], locais: list[tuple[float, float]]) -> int:
    """Dois canais: a palavra é do usuário se sobrepõe mais os trechos locais (soma) que qualquer trecho remoto; senão
    é do trecho remoto de maior sobreposição. Fora de todos: o mais próximo (sem trecho remoto nenhum: remoto 0)."""
    ov_local = sum(max(0.0, min(fim, b) - max(inicio, a)) for a, b in locais)
    ov_remoto = max((min(fim, t.fim) - max(inicio, t.inicio) for t in trechos), default=0.0)
    if ov_local > 0 and ov_local > ov_remoto:
        return _ID_LOCAL
    if ov_remoto > 0:
        return _locutor_da_palavra(inicio, fim, trechos)
    meio = (inicio + fim) / 2
    inf = float("inf")
    d_local = min((min(abs(meio - a), abs(meio - b)) for a, b in locais), default=inf)
    d_remoto = min((min(abs(meio - t.inicio), abs(meio - t.fim)) for t in trechos), default=inf)
    if trechos:
        if d_local <= d_remoto:
            return _ID_LOCAL
    elif d_local <= 0.5:   # sem remotos (diarização vazia/falhou): só a folga de tempo da palavra conta como local
        return _ID_LOCAL
    return _locutor_da_palavra(inicio, fim, trechos) if trechos else 0


def montar_falas(palavras: Iterable, trechos: list[Trecho], suavizar: bool = True,
                 locais: list[tuple[float, float]] | None = None) -> list[Fala]:
    """Atribui locutor palavra a palavra, suaviza ilhas, agrupa e nomeia "Locutor N" por ordem de aparição.

    `locais` (gravação com dois canais): trechos (início, fim) em que o USUÁRIO fala, vindos do canal do microfone. Com
    ele, as palavras do usuário são sempre o "Locutor 1" (primeiro id fixo) e os locutores remotos de `trechos`
    seguem como "Locutor 2…" por ordem de aparição."""
    lista = list(palavras)
    if locais is None:
        if not trechos:
            return _agrupar(lista, ["Locutor 1"] * len(lista))
        ids = [_locutor_da_palavra(p.start, p.end, trechos) for p in lista]
    else:
        ids = [_locutor_com_locais(p.start, p.end, trechos, locais) for p in lista]
    if suavizar and os.environ.get("AVEIA_SUAVIZAR", "1").strip() != "0":
        ids = _suavizar_ilhas(lista, ids)
    ordem: dict[int, str] = {_ID_LOCAL: LOCUTOR_LOCAL} if locais is not None else {}
    nomes = [ordem.setdefault(i, f"Locutor {len(ordem) + 1}") for i in ids]
    return _agrupar(lista, nomes)


def transcrever(
    caminho_audio: str,
    idioma: str | None = "pt",
    diarizacao: bool = True,
    num_locutores: int | None = None,
    progresso: Callable[[float, str], None] | None = None,
    motor: str | None = None,       # só "local"; ver motores.py
    meta: dict | None = None,       # recebe {"motor": usado, "avisos": [...], "lacunas": [...]}
    parcial: Callable[[list[Fala]], None] | None = None,   # recebe falas provisórias durante a transcrição
    canais: str | None = None,      # "mic+sistema": arquivo estéreo (esq = microfone, dir = áudio do computador); ver canais.py
    glossario: str | None = None,   # nomes e termos curtos para o Whisper local (`hotwords`)
) -> list[Fala]:
    emitir = progresso or (lambda p, m: None)
    ultimo = [0.0]

    def avisar(p: float, m: str) -> None:   # progresso sempre monotônico
        ultimo[0] = max(ultimo[0], p)
        emitir(ultimo[0], m)

    def avisar_motor(p: float, m: str) -> None:   # os motores usam 0,25–0,95: remapeia para a faixa da transcrição
        f = min(max((p - 0.25) / 0.70, 0.0), 1.0)
        avisar(_TRANSC_INI + f * (_TRANSC_FIM - _TRANSC_INI), m)

    def enviar_parcial(palavras: list) -> None:
        if parcial is None:
            return
        try:
            parcial(falas_provisorias(palavras))
        except Exception as e:  # noqa: BLE001
            print(f"Callback parcial falhou (ignorado): {e}")

    avisar(0.02, "Lendo áudio...")
    separacao: canais_mod.Canais | None = None
    sistema: np.ndarray | None = None
    if normalizar_canais(canais):
        try:
            esq, dir = decode_audio(caminho_audio, sampling_rate=SAMPLE_RATE, split_stereo=True)
            separacao = canais_mod.separar(esq, dir)
            if separacao.valido:
                sistema = dir[:len(separacao.mix)]
                if meta is not None:
                    meta["canais"] = CANAIS_MIC_SISTEMA
                    meta["locutor_local"] = LOCUTOR_LOCAL
            else:
                print(f"Dois canais ignorados ({separacao.motivo}): tratando como mono.")
                separacao = None
        except Exception as e:  # noqa: BLE001
            print(f"Separação por canais falhou ({e}): tratando como mono.")
            separacao = None
    if separacao is not None:
        audio = separacao.mix
    else:
        audio: np.ndarray = decode_audio(caminho_audio, sampling_rate=SAMPLE_RATE)
    duracao = len(audio) / SAMPLE_RATE
    if meta is not None:
        meta["duracao_s"] = round(duracao, 1)
    audio = preparar_audio(audio)   # passa-altas + normalização de volume (transcritor/audio.py; AVEIA_PREPROC)

    palavras: list = []
    ultima_parcial = time.monotonic()
    extra = {"glossario": glossario} if normalizar_glossario(glossario) else {}   # vazio: chamada idêntica à de antes dele
    for p in motores.palavras(audio, duracao, idioma, avisar_motor, motor=motor, meta=meta, **extra):
        palavras.append(p)
        agora = time.monotonic()
        if parcial is not None and agora - ultima_parcial >= _INTERVALO_PARCIAL:
            ultima_parcial = agora
            enviar_parcial(palavras)
    avisar(_TRANSC_FIM, "Transcrição concluída.")
    if meta is not None:   # lacunas com fala que o Whisper pulou e a retranscrição não resolveu: nunca em silêncio
        meta.setdefault("avisos", []).extend(avisos_de_lacunas(meta.get("lacunas") or []))
    if palavras:
        enviar_parcial(palavras)

    trechos: list[Trecho] = []
    if separacao is not None and palavras:
        # Dois canais: só o canal do computador é diarizado (os remotos); o usuário vem do canal do microfone.
        trechos = diarizar_remotos(sistema, num_locutores, diarizacao, meta, avisar)
        falas = montar_falas(palavras, trechos, locais=separacao.trechos_locais)
        avisar(1.0, "Pronto.")
        return falas
    if diarizacao and palavras:
        avisar(_DIAR_INI, "Identificando locutores pelo timbre...")
        try:
            trechos = diarizar(audio, num_locutores=num_locutores)
            anotar_diarizacao(meta)
        except Exception as e:  # noqa: BLE001
            print(f"Diarização falhou: {e}")
            if meta is not None:
                meta.setdefault("avisos", []).append(
                    f"A identificação de locutores falhou ({e}); todas as falas ficaram como Locutor 1.")
            trechos = []
        avisar(_DIAR_FIM, "Locutores identificados.")

    falas = montar_falas(palavras, trechos)
    avisar(1.0, "Pronto.")
    return falas


def anotar_diarizacao(meta: dict | None) -> None:
    """Registra em `meta` qual método separou os locutores; se o novo falhou e o antigo assumiu, avisa na interface (o
    antigo erra mais a contagem de pessoas, e antes isso só aparecia no console)."""
    if meta is None:
        return
    motor, motivo = ultimo_motor()
    meta["diarizacao"] = motor
    if motor == "antigo" and motivo:
        meta.setdefault("avisos", []).append(
            f"Os locutores foram separados pelo método antigo ({motivo}). A contagem de pessoas pode estar errada.")


def diarizar_remotos(sistema: np.ndarray, num_locutores: int | None, diarizacao: bool, meta: dict | None,
                     avisar: Callable[[float, str], None]) -> list[Trecho]:
    """Diariza só o canal do computador. `num_locutores` conta todo mundo, então os remotos são `num_locutores - 1`
    (mínimo 1). Falha da diarização não derruba: os remotos viram um só locutor, com aviso."""
    if not diarizacao:
        return []
    avisar(_DIAR_INI, "Identificando os outros participantes pelo timbre...")
    try:
        remotos = max(1, num_locutores - 1) if num_locutores else None
        return diarizar(preparar_audio(sistema), num_locutores=remotos)
    except Exception as e:  # noqa: BLE001
        print(f"Diarização falhou: {e}")
        if meta is not None:
            meta.setdefault("avisos", []).append(
                f"A identificação de locutores falhou ({e}); os participantes remotos ficaram como um só locutor.")
        return []
    finally:
        avisar(_DIAR_FIM, "Locutores identificados.")


def _tempo(s: float, sep: str = ".") -> str:
    h, resto = divmod(s, 3600)
    m, seg = divmod(resto, 60)
    return f"{int(h):02d}:{int(m):02d}:{seg:06.3f}".replace(".", sep)


def para_texto(falas: list[Fala]) -> str:
    return "\n\n".join(f"[{_tempo(f.inicio)[:8]}] {f.locutor}: {f.texto}" for f in falas)


def para_srt(falas: list[Fala]) -> str:
    blocos = [
        f"{i}\n{_tempo(f.inicio, ',')} --> {_tempo(f.fim, ',')}\n{f.locutor}: {f.texto}\n"
        for i, f in enumerate(falas, 1)
    ]
    return "\n".join(blocos)


def para_json(falas: list[Fala]) -> str:
    return json.dumps([asdict(f) for f in falas], ensure_ascii=False, indent=2)
