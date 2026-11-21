"""Motor de diarização próprio sobre ONNX (onnxruntime + sherpa-onnx), no estilo do pyannote 3.x.

Etapas (cada uma pode ser calibrada/medida separadamente; ver testes/diarizacao/):
  1. segmentação: pyannote-segmentation-3.0 em janelas de 10 s deslizantes -> até 3 locutores locais por janela
     (inclui sobreposição, "powerset");
  2. embedding de voz de cada (janela, locutor local), só com os trechos em que ele fala sozinho;
  3. agrupamento hierárquico -> locutores globais; com `num_locutores` corta em K grupos (Ward). Sem K (método 3,
     o padrão): Ward cortado onde a divisão explica ≥ `limiar_var` da variância dos embeddings centrados (não
     depende da duração do áudio), depois fusão de grupos com o mesmo timbre (cosseno dos centroides sem centrar)
     e absorção de quem fala quase nada (mesma ideia do `_consolidar` do método antigo);
  4. de volta ao tempo: votação entre as janelas que cobrem cada quadro (permite 2 vozes ao mesmo tempo).
`AnaliseIncremental` faz as etapas 1-2 aos poucos (durante a gravação) e só agrupa em `concluir()`.
Sem torch, sem token, só CPU. Os modelos ONNX são carregados uma vez por processo e reutilizados.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SR = 16000
JANELA = 160000          # 10 s (tamanho fixo do modelo de segmentação)
QUADRO_AMOSTRAS = 270    # passo entre quadros de saída do modelo (16,875 ms)
CAMPO = 991              # campo receptivo de cada quadro, em amostras
# classes "powerset" do pyannote 3.0: nenhuma, {0}, {1}, {2}, {0,1}, {0,2}, {1,2}
_POWERSET = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1]], dtype=bool)


@dataclass
class Params:
    passo_janela: int = 150 * QUADRO_AMOSTRAS   # deslocamento entre janelas (≈ 2,53 s)
    min_fala_emb: float = 1.0       # s de fala exclusiva do locutor local para gerar embedding
    min_fala_emb_ov: float = 0.6    # s mínimo se contar a fala sobreposta
    limiar: float = 0.7             # corte da ligação média (distância cosseno 1 - sim)
    min_cluster: int = 3            # grupos com menos embeddings que isso são absorvidos pelo mais próximo
    min_duracao_on: float = 0.25    # descarta trechos mais curtos (s)
    min_duracao_off: float = 0.35   # junta pausas menores do mesmo locutor (s)
    max_locutores_auto: int = 8
    metodo: int = 3                 # 0 = ligação média, 1 = Ward (corte por altura), 2 = espectral (NME-SC), 3 = Ward + variância + fusão (padrão)
    centrar: bool = True            # tira a média dos embeddings do arquivo (remove canal/sala comum) antes do cosseno
    # --- método 3 (estimativa automática do nº de locutores; calibrado em 7 arquivos, ver README) ---
    limiar_var: float = 0.05        # só divide se a divisão explica ≥ 5 % da variância dos embeddings centrados
    fusao: float = 0.7              # grupos com cosseno ≥ isto entre os centroides (sem centrar) são o mesmo locutor
    min_fala_locutor: float = 10.0  # s: quem fala menos que isto (e que `min_fracao_locutor` do total) é absorvido
    min_fracao_locutor: float = 0.04


@dataclass
class Analise:
    """Saída das etapas 1-2 (independe do agrupamento): permite varrer parâmetros de clustering sem recalcular."""
    n_amostras: int
    passo: int
    atividade: np.ndarray            # (n_janelas, n_quadros, 3) bool
    emb: np.ndarray                  # (m, d) normalizados
    emb_idx: list[tuple[int, int]]   # (janela, locutor local) de cada embedding
    emb_dur: np.ndarray              # fala usada em cada embedding (s)


# ---------------------------------------------------------------------------------------------- 1. segmentação
_SESSOES: dict = {}
_EXTRATORES: dict = {}
_TRAVA_CARGA = threading.Lock()


def _sessao(modelo: Path, threads: int):
    """Sessão ONNX da segmentação, criada uma vez por (modelo, threads) e reutilizada (run() é thread-safe)."""
    chave = (str(modelo), threads)
    with _TRAVA_CARGA:
        if chave not in _SESSOES:
            import onnxruntime as ort

            op = ort.SessionOptions()
            op.intra_op_num_threads = threads
            op.inter_op_num_threads = 1
            _SESSOES[chave] = ort.InferenceSession(str(modelo), sess_options=op, providers=["CPUExecutionProvider"])
        return _SESSOES[chave]


def _inicios_janelas(n: int, passo: int) -> list[int]:
    """Início (em amostras) de cada janela de 10 s sobre `n` amostras; a última é preenchida com zeros se sobrar áudio."""
    inicios = list(range(0, max(n - JANELA, 0) + 1, passo))
    if inicios[-1] + JANELA < n:
        inicios.append(inicios[-1] + passo)
    return inicios


def _atividade_janelas(sess, audio: np.ndarray, inicios: list[int]) -> np.ndarray:
    """Segmentação das janelas que começam em `inicios` (relativos a `audio`): (n_janelas, n_quadros, 3) booleano."""
    saidas = []
    for i in range(0, len(inicios), 16):
        lote = np.zeros((len(inicios[i:i + 16]), 1, JANELA), dtype=np.float32)
        for j, a in enumerate(inicios[i:i + 16]):
            trecho = audio[a:a + JANELA]
            lote[j, 0, :len(trecho)] = trecho
        saidas.append(sess.run(None, {"x": lote})[0])
    y = np.concatenate(saidas)                       # (J, T, 7) log-probabilidades
    return _POWERSET[y.argmax(-1)]                    # (J, T, 3)


def segmentar(audio: np.ndarray, modelo: Path, p: Params, threads: int = 4) -> np.ndarray:
    """Atividade binária por janela: (n_janelas, n_quadros, 3)."""
    return _atividade_janelas(_sessao(modelo, threads), audio, _inicios_janelas(len(audio), p.passo_janela))


# ---------------------------------------------------------------------------------------------- 2. embeddings
def _extrator(modelo: Path, threads: int):
    """(extrator, trava) criado uma vez por (modelo, threads); a trava serializa o uso (o extrator não é documentado como thread-safe)."""
    chave = (str(modelo), threads)
    with _TRAVA_CARGA:
        if chave not in _EXTRATORES:
            import sherpa_onnx

            ext = sherpa_onnx.SpeakerEmbeddingExtractor(
                sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(modelo), num_threads=threads))
            _EXTRATORES[chave] = (ext, threading.Lock())
        return _EXTRATORES[chave]


def _amostras_do_locutor(audio: np.ndarray, ini: int, ativ: np.ndarray, outros: np.ndarray, p: Params):
    """Áudio concatenado dos quadros em que o locutor fala (preferindo os sem sobreposição)."""
    def juntar(mask: np.ndarray) -> np.ndarray:
        pedacos = []
        for a, b in _runs(mask):
            s = ini + a * QUADRO_AMOSTRAS
            pedacos.append(audio[s: ini + b * QUADRO_AMOSTRAS + CAMPO - QUADRO_AMOSTRAS])
        return np.concatenate(pedacos) if pedacos else np.zeros(0, dtype=np.float32)

    x = juntar(ativ & ~outros)
    if len(x) < p.min_fala_emb * SR:
        x = juntar(ativ)
        if len(x) < p.min_fala_emb_ov * SR:
            return None
    return x[: 10 * SR]


def _runs(mask: np.ndarray):
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def _embeddings_da_janela(extrator, audio: np.ndarray, ini: int, a: np.ndarray, p: Params):
    """Embeddings dos locutores locais de uma janela que começa em `ini` (relativo a `audio`), com atividade `a` (T, 3).
    Retorna [(locutor_local, vetor_normalizado, segundos_de_fala)]."""
    ext, trava = extrator
    saida = []
    for s in range(3):
        if a[:, s].sum() * QUADRO_AMOSTRAS < 0.5 * SR:
            continue
        outros = a[:, [k for k in range(3) if k != s]].any(1)
        x = _amostras_do_locutor(audio, ini, a[:, s], outros, p)
        if x is None:
            continue
        with trava:
            st = ext.create_stream()
            st.accept_waveform(SR, x.astype(np.float32))
            st.input_finished()
            v = np.asarray(ext.compute(st), dtype=np.float64)
        saida.append((s, v / (np.linalg.norm(v) or 1.0), len(x) / SR))
    return saida


def _juntar_embeddings(embs: list, idx: list, durs: list):
    d = len(embs[0]) if embs else 0
    return np.array(embs).reshape(-1, d), idx, np.array(durs)


def extrair_embeddings(audio: np.ndarray, atividade: np.ndarray, modelo_emb: Path, p: Params, threads: int = 4):
    ext = _extrator(modelo_emb, threads)
    embs, idx, durs = [], [], []
    for j in range(atividade.shape[0]):
        for s, v, dur in _embeddings_da_janela(ext, audio, j * p.passo_janela, atividade[j], p):
            embs.append(v)
            idx.append((j, s))
            durs.append(dur)
    return _juntar_embeddings(embs, idx, durs)


def analisar(audio: np.ndarray, modelo_seg: Path, modelo_emb: Path, p: Params | None = None, threads: int = 4) -> Analise:
    p = p or Params()
    ativ = segmentar(audio, modelo_seg, p, threads)
    emb, idx, durs = extrair_embeddings(audio, ativ, modelo_emb, p, threads)
    return Analise(len(audio), p.passo_janela, ativ, emb, idx, durs)


# ---------------------------------------------------------------------------------------------- 3. agrupamento
def agrupar_embeddings(emb: np.ndarray, k: int | None, p: Params) -> np.ndarray:
    """Rótulo de grupo (0..G-1) para cada embedding."""
    from scipy.cluster.hierarchy import fcluster, linkage

    n = len(emb)
    if n == 0:
        return np.zeros(0, dtype=int)
    if n == 1:
        return np.zeros(1, dtype=int)
    if p.centrar and n > 2:
        emb = emb - emb.mean(0)
        emb = emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-9)
    if p.metodo == 2:
        rot = _espectral(emb, k, p)
    else:
        z = linkage(emb, method="average" if p.metodo == 0 else "ward", metric="cosine" if p.metodo == 0 else "euclidean")
        if k:
            rot = fcluster(z, t=min(k, n), criterion="maxclust")
        elif p.metodo == 3:
            # altura de Ward^2 / 2 = aumento da soma dos quadrados; dividida pela soma total = fração da variância explicada
            sse = float(((emb - emb.mean(0)) ** 2).sum())
            rot = fcluster(z, t=np.sqrt(2.0 * p.limiar_var * sse), criterion="distance") if sse > 1e-9 else np.ones(n, dtype=int)
        else:
            rot = fcluster(z, t=p.limiar, criterion="distance")
    rot = _reindexar(rot - 1)
    if not k and p.metodo != 3:   # no método 3 a limpeza é `consolidar_locutores` (usa o tempo de fala, não a contagem)
        rot = _absorver_pequenos(emb, rot, p)
        # limite de segurança: no máximo N grupos
        while rot.max() + 1 > p.max_locutores_auto:
            rot = _absorver_pequenos(emb, rot, Params(min_cluster=int(np.bincount(rot).min()) + 1))
    return rot


def _kmeans(x: np.ndarray, k: int, reinicios: int = 8, semente: int = 0) -> np.ndarray:
    rng = np.random.default_rng(semente)
    melhor, melhor_inercia = None, np.inf
    for _ in range(reinicios):
        c = [x[rng.integers(len(x))]]
        for _ in range(1, k):                                   # k-means++
            d = np.min([((x - ci) ** 2).sum(1) for ci in c], axis=0)
            c.append(x[rng.choice(len(x), p=d / d.sum())] if d.sum() > 0 else x[rng.integers(len(x))])
        c = np.array(c)
        for _ in range(50):
            rot = np.argmin(((x[:, None] - c[None]) ** 2).sum(-1), axis=1)
            novo = np.array([x[rot == g].mean(0) if (rot == g).any() else c[g] for g in range(k)])
            if np.allclose(novo, c):
                break
            c = novo
        inercia = ((x - c[rot]) ** 2).sum()
        if inercia < melhor_inercia:
            melhor, melhor_inercia = rot, inercia
    return melhor


def _espectral(emb: np.ndarray, k: int | None, p: Params) -> np.ndarray:
    """Agrupamento espectral com escolha automática do nº de vizinhos e de locutores (NME-SC, Park et al. 2020)."""
    n = len(emb)
    sim = (1.0 + emb @ emb.T) / 2.0
    maxk = min(p.max_locutores_auto, n - 1)
    candidatos = sorted({int(round(v)) for v in np.linspace(max(3, 0.01 * n), max(4, 0.2 * n), 12)})
    candidatos = [c for c in candidatos if c < n] or [max(1, n - 1)]
    melhor = None
    for pv in candidatos:
        a = np.zeros_like(sim)
        idx = np.argsort(-sim, axis=1)[:, :pv]
        np.put_along_axis(a, idx, np.take_along_axis(sim, idx, 1), 1)
        a = (a + a.T) / 2
        d = np.maximum(a.sum(1), 1e-9)
        lap = np.eye(n) - a / np.sqrt(d[:, None] * d[None])
        val, vec = np.linalg.eigh(lap)
        gaps = np.diff(val[: maxk + 1])
        kk = int(np.argmax(gaps)) + 1
        g = (pv / n) / max(gaps.max(), 1e-9)
        if melhor is None or g < melhor[0]:
            melhor = (g, kk, vec)
    _, kk, vec = melhor
    kk = min(k, n) if k else max(kk, 1)
    if kk == 1:
        return np.zeros(n, dtype=int)
    f = vec[:, :kk]
    f = f / np.maximum(np.linalg.norm(f, axis=1, keepdims=True), 1e-9)
    return _kmeans(f, kk)


def _reindexar(rot: np.ndarray) -> np.ndarray:
    _, r = np.unique(rot, return_inverse=True)
    return r


def _centroides(emb: np.ndarray, rot: np.ndarray) -> np.ndarray:
    c = np.stack([emb[rot == g].mean(0) for g in range(rot.max() + 1)])
    return c / np.linalg.norm(c, axis=1, keepdims=True)


def _centrar(emb: np.ndarray) -> np.ndarray:
    c = emb - emb.mean(0)
    return c / np.maximum(np.linalg.norm(c, axis=1, keepdims=True), 1e-9)


def tempo_por_locutor(an: "Analise", rot: np.ndarray, p: Params) -> np.ndarray:
    """Segundos de fala atribuídos a cada grupo depois da votação (é o que o usuário vai ver)."""
    t = np.zeros(int(rot.max()) + 1)
    for a, b, g in montar_trechos(an, rot, p):
        t[g] += b - a
    return t


def consolidar_locutores(an: "Analise", rot: np.ndarray, p: Params) -> np.ndarray:
    """Limpeza do agrupamento sem K conhecido (equivale ao `_consolidar` do método antigo):
    1) funde grupos com o mesmo timbre: cosseno entre centroides dos embeddings SEM centrar ≥ `fusao` (a mesma pessoa
       em duas condições acústicas, ex.: virou o rosto; vozes diferentes ficam em ≤ 0,6 nos dados medidos);
    2) absorve, no grupo mais próximo (embeddings centrados), quem fala menos que `min_fala_locutor` s e
       `min_fracao_locutor` do total, e também o menor grupo enquanto houver mais que `max_locutores_auto`."""
    rot = rot.copy()
    if len(rot) < 2:
        return rot
    cen = _centrar(an.emb) if p.centrar and len(rot) > 2 else an.emb
    while rot.max() >= 1:
        g = int(rot.max()) + 1
        cr = _centroides(an.emb, rot)
        s = cr @ cr.T
        np.fill_diagonal(s, -9.0)
        a, b = np.unravel_index(int(s.argmax()), s.shape)
        if s[a, b] >= p.fusao:
            rot[rot == b] = a
            rot = _reindexar(rot)
            continue
        t = tempo_por_locutor(an, rot, p)
        m = int(t.argmin())
        if t[m] >= max(p.min_fala_locutor, p.min_fracao_locutor * t.sum()) and g <= p.max_locutores_auto:
            break
        sm = _centroides(cen, rot) @ _centroides(cen, rot)[m]
        sm[m] = -9.0
        rot[rot == m] = int(sm.argmax())
        rot = _reindexar(rot)
    return rot


def _absorver_pequenos(emb: np.ndarray, rot: np.ndarray, p: Params) -> np.ndarray:
    rot = rot.copy()
    while True:
        tam = np.bincount(rot)
        if len(tam) < 2 or tam.min() >= p.min_cluster:
            return _reindexar(rot)
        menor = int(tam.argmin())
        c = _centroides(emb, rot)
        sim = c @ c[menor]
        sim[menor] = -2
        rot[rot == menor] = int(sim.argmax())
        rot = _reindexar(rot)


# ---------------------------------------------------------------------------------------------- 4. de volta ao tempo
def montar_trechos(an: Analise, rot: np.ndarray, p: Params, num_fixo: int | None = None) -> list[tuple[float, float, int]]:
    J, T, _ = an.atividade.shape
    n_quadros = int(np.ceil(an.n_amostras / QUADRO_AMOSTRAS)) + 2
    shift_q = an.passo // QUADRO_AMOSTRAS
    n_loc = int(rot.max()) + 1 if len(rot) else 0
    if n_loc == 0:
        return []
    mapa = {(j, s): g for (j, s), g in zip(an.emb_idx, rot)}
    score = np.zeros((n_loc, n_quadros + T))
    contagem = np.zeros(n_quadros + T)
    soma_ativos = np.zeros(n_quadros + T)
    for j in range(J):
        a0 = j * shift_q
        contagem[a0: a0 + T] += 1
        soma_ativos[a0: a0 + T] += an.atividade[j].sum(1)
        for s in range(3):
            g = mapa.get((j, s))
            if g is not None:
                score[g, a0: a0 + T] += an.atividade[j][:, s]
    contagem = np.maximum(contagem, 1)
    score /= contagem
    n_ativos = np.rint(soma_ativos / contagem).astype(int)
    ativo = np.zeros_like(score, dtype=bool)
    ordem = np.argsort(-score, axis=0)
    for r in range(min(2, n_loc)):                  # no máximo 2 vozes simultâneas (como o modelo)
        g = ordem[r]
        quer = (n_ativos > r) & (np.take_along_axis(score, g[None], 0)[0] > 0.25)
        ativo[g[quer], np.flatnonzero(quer)] = True
    seg = QUADRO_AMOSTRAS / SR
    trechos = []
    for g in range(n_loc):
        runs = [[a, b] for a, b in _runs(ativo[g])]
        fund: list[list[int]] = []
        for r in runs:
            if fund and (r[0] - fund[-1][1]) * seg < p.min_duracao_off:
                fund[-1][1] = r[1]
            else:
                fund.append(r)
        for a, b in fund:
            if (b - a) * seg >= p.min_duracao_on:
                trechos.append((a * seg + CAMPO / 2 / SR, b * seg + CAMPO / 2 / SR, g))
    trechos.sort()
    return trechos


def diarizar_analise(an: Analise, k: int | None, p: Params | None = None):
    """Etapas 3-4 sobre uma `Analise`: (trechos [(início, fim, grupo)], rótulo de cada embedding)."""
    p = p or Params()
    rot = agrupar_embeddings(an.emb, k, p)
    if not k and p.metodo == 3 and len(rot):
        rot = consolidar_locutores(an, rot, p)
    return montar_trechos(an, rot, p), rot


def diarizar_audio(audio: np.ndarray, k: int | None, modelo_seg: Path, modelo_emb: Path, p: Params | None = None,
                   threads: int = 4) -> list[tuple[float, float, int]]:
    """Caminho em lote: áudio mono float32 16 kHz -> [(início, fim, locutor)] com locutores numerados pela ordem de aparição."""
    p = p or Params()
    trechos, _ = diarizar_analise(analisar(audio, modelo_seg, modelo_emb, p, threads), k, p)
    return _numerar_por_aparicao(trechos)


def _numerar_por_aparicao(trechos: list[tuple[float, float, int]]) -> list[tuple[float, float, int]]:
    ordem: dict[int, int] = {}
    for _, _, g in sorted(trechos):
        ordem.setdefault(g, len(ordem))
    return [(a, b, ordem[g]) for a, b, g in trechos]


# ---------------------------------------------------------------------------------------------- modo ao vivo
class AnaliseIncremental:
    """Etapas 1-2 aos poucos, para o modo ao vivo: `alimentar()` roda segmentação + embeddings (a parte cara) a cada
    janela de 10 s completa; `concluir()` trata a cauda, agrupa e devolve os trechos. Alimentar o áudio em pedaços
    quaisquer dá exatamente os mesmos trechos que o caminho em lote (`diarizar_audio`)."""

    def __init__(self, modelo_seg: Path, modelo_emb: Path, p: Params | None = None, threads: int = 4):
        self.p = p or Params()
        self._sess = _sessao(modelo_seg, threads)
        self._ext = _extrator(modelo_emb, threads)
        self._buf = np.zeros(0, dtype=np.float32)    # áudio ainda necessário (a partir de _base)
        self._base = 0                                # amostra absoluta do início de _buf
        self._total = 0                               # amostras recebidas
        self._ativ: list[np.ndarray] = []             # atividade de cada janela já processada
        self._embs: list[np.ndarray] = []
        self._idx: list[tuple[int, int]] = []
        self._durs: list[float] = []

    @property
    def duracao(self) -> float:
        return self._total / SR

    @property
    def janelas_prontas(self) -> int:
        return len(self._ativ)

    def alimentar(self, audio_novo: np.ndarray) -> int:
        """Acrescenta amostras (float32, mono, 16 kHz) e processa as janelas completas. Retorna quantas janelas novas."""
        audio_novo = np.asarray(audio_novo, dtype=np.float32).reshape(-1)
        self._buf = np.concatenate([self._buf, audio_novo])
        self._total += len(audio_novo)
        antes = len(self._ativ)
        pronto = max(0, (self._total - JANELA) // self.p.passo_janela + 1)   # janelas com início+10 s ≤ total
        if pronto > antes:
            self._processar(list(range(antes, pronto)), self._ativ, self._embs, self._idx, self._durs)
            # descarta o áudio que nenhuma janela futura usa
            livre = pronto * self.p.passo_janela
            if livre > self._base:
                self._buf = self._buf[livre - self._base:]
                self._base = livre
        return len(self._ativ) - antes

    def _processar(self, janelas: list[int], ativ: list, embs: list, idx: list, durs: list) -> None:
        passo = self.p.passo_janela
        locais = [j * passo - self._base for j in janelas]
        a = _atividade_janelas(self._sess, self._buf, locais)
        for j, loc, aj in zip(janelas, locais, a):
            ativ.append(aj)
            for s, v, dur in _embeddings_da_janela(self._ext, self._buf, loc, aj, self.p):
                embs.append(v)
                idx.append((j, s))
                durs.append(dur)

    def concluir(self, num_locutores: int | None = None) -> list:
        """Processa a cauda (janelas parciais, com zeros), agrupa e devolve `list[Trecho]` (locutores numerados pela
        ordem de aparição). Não altera o estado: dá para chamar de novo depois de alimentar mais áudio."""
        from .diarizacao import Trecho   # import tardio: diarizacao.py importa este módulo

        if self._total == 0:
            return []
        passo = self.p.passo_janela
        todas = _inicios_janelas(self._total, passo)
        ativ, embs, idx, durs = list(self._ativ), list(self._embs), list(self._idx), list(self._durs)
        faltam = list(range(len(ativ), len(todas)))
        if faltam:
            self._processar(faltam, ativ, embs, idx, durs)
        emb, idx, durs = _juntar_embeddings(embs, idx, durs)
        an = Analise(self._total, passo, np.stack(ativ), emb, idx, durs)
        trechos, _ = diarizar_analise(an, num_locutores, self.p)
        return [Trecho(a, b, g) for a, b, g in _numerar_por_aparicao(trechos)]
