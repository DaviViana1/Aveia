"""Testes da diarização: agrupamento, estimativa do nº de locutores, absorção/fusão, `montar_trechos`, chave
`AVEIA_DIARIZACAO` e fallback (offline, sobre embeddings/atividade sintéticos) + modelo real no áudio de referência
(pulado se faltar um arquivo de modelo; nada é baixado) + análise incremental == lote.

    .venv\\Scripts\\python -m unittest testes.test_diarizacao -v
"""
from __future__ import annotations

import contextlib
import dataclasses
import io
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from transcritor import diarizacao
from transcritor import diarizacao_onnx as D
from transcritor.diarizacao import Trecho

RAIZ = Path(__file__).resolve().parent.parent
REF = RAIZ / "testes" / "audio-referencia"
Q = D.QUADRO_AMOSTRAS / D.SR


# ------------------------------------------------------------------------------------------ dados sintéticos
def analise_sintetica(blocos: list[tuple[int, int, int]], d: int = 64, ruido: float = 0.04, comum: float = 0.6,
                      quadros: tuple[int, int] = (0, 589), semente: int = 0, centros=None) -> tuple[D.Analise, np.ndarray]:
    """Uma `Analise` com um locutor local ativo por janela. `blocos` = [(locutor, n_janelas, ...)] em sequência; cada
    janela gera um embedding = componente comum (canal/sala) + centro do locutor + ruído. Um terceiro item opcional no
    bloco limita a fala aos primeiros N quadros da janela. Devolve (análise, rótulo real)."""
    rng = np.random.default_rng(semente)
    if centros is None:
        centros = {}
    emb, verdade, idx = [], [], []
    ativ = []
    j = 0
    comum_v = rng.normal(size=d)
    comum_v = comum * comum_v / np.linalg.norm(comum_v)
    for item in blocos:
        g, n = item[0], item[1]
        fim_q = item[2] if len(item) > 2 else quadros[1]
        if g not in centros:
            c = rng.normal(size=d)
            centros[g] = c / np.linalg.norm(c)
        for _ in range(n):
            v = comum_v + centros[g] + ruido * rng.normal(size=d)
            emb.append(v / np.linalg.norm(v))
            verdade.append(g)
            idx.append((j, 0))
            a = np.zeros((589, 3), dtype=bool)
            a[quadros[0]:fim_q, 0] = True
            ativ.append(a)
            j += 1
    passo = D.Params().passo_janela
    an = D.Analise(n_amostras=(j - 1) * passo + D.JANELA, passo=passo, atividade=np.stack(ativ), emb=np.array(emb),
                   emb_idx=idx, emb_dur=np.full(len(emb), 9.0))
    return an, np.array(verdade)


def pureza(rot: np.ndarray, verdade: np.ndarray) -> float:
    """Fração dos embeddings no rótulo majoritário do seu grupo (1,0 = grupos perfeitos)."""
    ok = 0
    for g in np.unique(rot):
        ok += np.bincount(verdade[rot == g]).max()
    return ok / len(rot)


class AgrupamentoTest(unittest.TestCase):
    def test_estima_numero_de_locutores(self):
        for n_loc in (2, 3, 4):
            with self.subTest(locutores=n_loc):
                an, verdade = analise_sintetica([(g, 20) for g in range(n_loc)], semente=n_loc)
                tr, rot = D.diarizar_analise(an, None, D.Params())
                self.assertEqual(len(set(rot)), n_loc)
                self.assertEqual(pureza(rot, verdade), 1.0)
                self.assertEqual(len({g for _, _, g in tr}), n_loc)

    def test_um_unico_locutor_nao_e_dividido(self):
        an, _ = analise_sintetica([(0, 60)], semente=7)
        _, rot = D.diarizar_analise(an, None, D.Params())
        self.assertEqual(len(set(rot)), 1)

    def test_num_locutores_dado_e_respeitado(self):
        an, verdade = analise_sintetica([(g, 20) for g in range(4)], semente=11)
        for k in (1, 2, 3, 4):
            with self.subTest(k=k):
                _, rot = D.diarizar_analise(an, k, D.Params())
                self.assertEqual(len(set(rot)), k)
        # com k certo o agrupamento é perfeito, mesmo que a estimativa automática dissesse outra coisa
        _, rot = D.diarizar_analise(an, 4, D.Params())
        self.assertEqual(pureza(rot, verdade), 1.0)

    def test_independe_da_duracao(self):
        """O corte é por fração da variância explicada: o mesmo grupo de vozes com 3x mais janelas dá o mesmo nº."""
        for por_loc in (12, 36, 100):
            with self.subTest(janelas_por_locutor=por_loc):
                an, _ = analise_sintetica([(g, por_loc) for g in range(3)], semente=5)
                _, rot = D.diarizar_analise(an, None, D.Params())
                self.assertEqual(len(set(rot)), 3)

    def test_fusao_junta_o_mesmo_timbre_em_duas_condicoes(self):
        """Um locutor que forma dois grupos bem separados depois de centrar, mas com timbre quase igual sem centrar."""
        rng = np.random.default_rng(3)
        d = 64
        u = rng.normal(size=d)
        u /= np.linalg.norm(u)
        centros = {}
        base = rng.normal(size=d)
        base /= np.linalg.norm(base)
        centros[2], centros[3] = base + 0.3 * u, base - 0.3 * u    # mesma pessoa, duas condições
        an, verdade = analise_sintetica([(0, 20), (1, 20), (2, 20), (3, 20)], semente=4, centros=centros)
        _, rot_sem = D.diarizar_analise(an, None, dataclasses.replace(D.Params(), fusao=2.0))
        _, rot_com = D.diarizar_analise(an, None, D.Params())
        self.assertEqual(len(set(rot_sem)), 4)       # sem fusão: 4 grupos
        self.assertEqual(len(set(rot_com)), 3)       # com fusão: as duas condições viram um locutor
        self.assertEqual(len(set(rot_com[verdade == 2]) | set(rot_com[verdade == 3])), 1)

    def test_absorve_locutor_de_fala_minima(self):
        """Dois locutores de 50 s e um que fala 8 s no total (4 janelas de 2 s): o pequeno é absorvido."""
        an2, _ = analise_sintetica([(0, 20), (1, 20), (5, 4, 120)], semente=9)   # 4 janelas de outra voz, 2 s de fala cada
        _, rot = D.diarizar_analise(an2, None, D.Params())
        self.assertEqual(len(set(rot)), 2)
        _, rot_livre = D.diarizar_analise(an2, None, dataclasses.replace(D.Params(), min_fala_locutor=0.0, min_fracao_locutor=0.0))
        self.assertEqual(len(set(rot_livre)), 3)     # sem a regra, o pequeno sobra

    def test_limite_maximo_de_locutores_automatico(self):
        an, _ = analise_sintetica([(g, 15) for g in range(6)], semente=2)
        p = dataclasses.replace(D.Params(), max_locutores_auto=4)
        _, rot = D.diarizar_analise(an, None, p)
        self.assertLessEqual(len(set(rot)), 4)

    def test_vazio_e_um_embedding(self):
        self.assertEqual(len(D.agrupar_embeddings(np.zeros((0, 8)), None, D.Params())), 0)
        self.assertEqual(list(D.agrupar_embeddings(np.ones((1, 8)), None, D.Params())), [0])
        self.assertEqual(list(D.agrupar_embeddings(np.ones((1, 8)), 3, D.Params())), [0])


class MontarTrechosTest(unittest.TestCase):
    def _an(self, ativ: np.ndarray, n_janelas: int = 1) -> D.Analise:
        passo = D.Params().passo_janela
        return D.Analise(n_amostras=D.JANELA + (n_janelas - 1) * passo, passo=passo, atividade=ativ,
                         emb=np.zeros((0, 4)), emb_idx=[], emb_dur=np.zeros(0))

    def test_dois_locutores_em_sequencia_e_tempos(self):
        a = np.zeros((1, 589, 3), dtype=bool)
        a[0, 100:300, 0] = True      # locutor local 0: ~3,4 s
        a[0, 350:500, 1] = True      # locutor local 1
        an = self._an(a)
        an.emb_idx = [(0, 0), (0, 1)]
        tr = D.montar_trechos(an, np.array([0, 1]), D.Params())
        self.assertEqual([g for _, _, g in tr], [0, 1])
        deslocamento = D.CAMPO / 2 / D.SR
        self.assertAlmostEqual(tr[0][0], 100 * Q + deslocamento, places=3)
        self.assertAlmostEqual(tr[0][1], 300 * Q + deslocamento, places=3)
        self.assertAlmostEqual(tr[1][0], 350 * Q + deslocamento, places=3)

    def test_sobreposicao_mantem_as_duas_vozes(self):
        a = np.zeros((1, 589, 3), dtype=bool)
        a[0, 100:300, 0] = True
        a[0, 200:260, 1] = True      # fala junto de 200 a 260
        an = self._an(a)
        an.emb_idx = [(0, 0), (0, 1)]
        tr = D.montar_trechos(an, np.array([0, 1]), D.Params())
        t1 = [t for t in tr if t[2] == 1]
        self.assertEqual(len(t1), 1)
        self.assertTrue(t1[0][0] < 260 * Q and t1[0][1] > 200 * Q)
        t0 = [t for t in tr if t[2] == 0]
        self.assertTrue(t0[0][0] < 200 * Q < 260 * Q < t0[-1][1])   # a voz 0 continua durante a sobreposição

    def test_pausa_curta_junta_e_trecho_curto_some(self):
        a = np.zeros((1, 589, 3), dtype=bool)
        a[0, 100:200, 0] = True
        a[0, 205:300, 0] = True      # lacuna de 5 quadros (~0,08 s) < min_duracao_off -> um trecho só
        a[0, 400:405, 0] = True      # 5 quadros (~0,08 s) < min_duracao_on -> descartado
        an = self._an(a)
        an.emb_idx = [(0, 0)]
        tr = D.montar_trechos(an, np.array([0]), D.Params())
        self.assertEqual(len(tr), 1)
        self.assertAlmostEqual(tr[0][1] - tr[0][0], 200 * Q, delta=0.05)

    def test_sem_grupos_sem_trechos(self):
        an = self._an(np.zeros((1, 589, 3), dtype=bool))
        self.assertEqual(D.montar_trechos(an, np.zeros(0, dtype=int), D.Params()), [])

    def test_numeracao_por_aparicao(self):
        tr = [(0.0, 1.0, 7), (1.0, 2.0, 3), (2.0, 3.0, 7)]
        self.assertEqual(D._numerar_por_aparicao(tr), [(0.0, 1.0, 0), (1.0, 2.0, 1), (2.0, 3.0, 0)])

    def test_janelas_do_lote(self):
        passo = D.Params().passo_janela
        self.assertEqual(D._inicios_janelas(D.JANELA, passo), [0])
        self.assertEqual(D._inicios_janelas(5 * D.SR, passo), [0])                       # menor que 1 janela
        self.assertEqual(D._inicios_janelas(D.JANELA + passo, passo), [0, passo])
        self.assertEqual(D._inicios_janelas(D.JANELA + passo + 10, passo), [0, passo, 2 * passo])   # cauda parcial


# ------------------------------------------------------------------------------------------ chave e fallback
class ChaveEFallbackTest(unittest.TestCase):
    AUDIO = np.zeros(16000, dtype=np.float32)

    def _rodar(self, env: str | None, novo=None, antigo=None, **kw):
        ambiente = {"AVEIA_DIARIZACAO": env} if env is not None else {}
        saida = io.StringIO()
        with mock.patch.dict(os.environ, ambiente, clear=False), contextlib.redirect_stdout(saida):
            if env is None:
                os.environ.pop("AVEIA_DIARIZACAO", None)
            with mock.patch.object(diarizacao, "_diarizar_novo", **(novo or {"return_value": [Trecho(0, 1, 0)]})) as n, \
                 mock.patch.object(diarizacao, "_diarizar_antigo", **(antigo or {"return_value": [Trecho(0, 2, 9)]})) as a:
                r = diarizacao.diarizar(self.AUDIO, **kw)
        return r, n, a, saida.getvalue()

    def test_padrao_e_novo(self):
        r, n, a, _ = self._rodar(None)
        self.assertEqual(r, [Trecho(0, 1, 0)])
        n.assert_called_once()
        a.assert_not_called()

    def test_antigo_por_variavel(self):
        r, n, a, _ = self._rodar("antigo")
        self.assertEqual(r, [Trecho(0, 2, 9)])
        n.assert_not_called()
        a.assert_called_once()

    def test_valor_desconhecido_vale_novo_e_avisa(self):
        r, n, a, texto = self._rodar("qualquer")
        self.assertEqual(r, [Trecho(0, 1, 0)])
        self.assertIn("desconhecido", texto)

    def test_maiusculas_e_espacos(self):
        _, n, a, _ = self._rodar("  ANTIGO ")
        a.assert_called_once()

    def test_novo_que_falha_cai_no_antigo(self):
        r, n, a, texto = self._rodar("novo", novo={"side_effect": RuntimeError("modelo ausente")})
        self.assertEqual(r, [Trecho(0, 2, 9)])
        a.assert_called_once()
        self.assertIn("falhou", texto)
        self.assertIn("modelo ausente", texto)

    def test_novo_sem_trechos_cai_no_antigo(self):
        r, n, a, texto = self._rodar("novo", novo={"return_value": []})
        self.assertEqual(r, [Trecho(0, 2, 9)])
        a.assert_called_once()
        self.assertIn("não devolveu", texto)

    def test_num_locutores_chega_aos_dois_motores(self):
        _, n, _, _ = self._rodar("novo", num_locutores=3)
        self.assertEqual(n.call_args.args[1], 3)
        _, _, a, _ = self._rodar("antigo", num_locutores=3)
        self.assertEqual(a.call_args.args[1], 3)

    def test_novo_usa_o_embedding_campp_baixavel(self):
        self.assertEqual(diarizacao.EMBEDDING_NOVO, "campp_zhen")
        url = diarizacao.EMBEDDINGS["campp_zhen"]
        self.assertTrue(url.startswith("https://github.com/k2-fsa/sherpa-onnx/releases/download/"))
        self.assertTrue(url.endswith("3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx"))

    def test_assinatura_publica_e_trecho(self):
        import inspect
        nomes = list(inspect.signature(diarizacao.diarizar).parameters)
        self.assertEqual(nomes[:2], ["audio", "num_locutores"])
        t = Trecho(1.0, 2.0, 3)
        self.assertEqual((t.inicio, t.fim, t.locutor), (1.0, 2.0, 3))


# ------------------------------------------------------------------------------------------ modelo real
def _modelos_presentes() -> bool:
    seg = diarizacao.MODELS_DIR / "sherpa-onnx-pyannote-segmentation-3-0" / "model.onnx"
    emb = diarizacao.MODELS_DIR / Path(diarizacao.EMBEDDINGS[diarizacao.EMBEDDING_NOVO]).name
    return seg.exists() and emb.exists() and (REF / "audio.wav").exists() and (REF / "referencia.rttm").exists()


@unittest.skipUnless(_modelos_presentes(), "faltam modelos ou o áudio de referência (nada é baixado nos testes)")
class ModeloRealTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from faster_whisper.audio import decode_audio
        sys.path.insert(0, str(RAIZ / "testes" / "diarizacao"))
        cls.audio = decode_audio(str(REF / "audio.wav"), sampling_rate=16000)
        cls.seg = diarizacao._modelo_segmentacao()
        cls.emb = diarizacao._modelo_embedding(diarizacao.EMBEDDING_NOVO)

    def _der(self, trechos):
        import benchmark
        from metricas import avaliar, ler_rttm
        ref = benchmark.refinar_ref(ler_rttm(REF / "referencia.rttm"), REF / "audio.wav")
        hip = [(t.inicio, t.fim, f"s{t.locutor}") for t in trechos]
        return avaliar(ref, hip, len(self.audio) / 16000)

    def test_audio_de_referencia_acha_dois_locutores(self):
        with mock.patch.dict(os.environ, {"AVEIA_DIARIZACAO": "novo"}):
            trechos = diarizacao.diarizar(self.audio)
        m = self._der(trechos)
        self.assertEqual(m["n_hip"], 2, f"locutores achados: {m['n_hip']}")
        self.assertLess(m["der"], 0.10, f"DER {m['der']:.3f} (medido: 0,038; método antigo: 0,31)")

    def test_num_locutores_dado_no_audio_real(self):
        trechos = diarizacao.diarizar(self.audio, num_locutores=2)
        self.assertEqual(self._der(trechos)["n_hip"], 2)
        um = diarizacao.diarizar(self.audio, num_locutores=1)
        self.assertEqual(len({t.locutor for t in um}), 1)

    def test_sessoes_sao_reutilizadas(self):
        self.assertIs(D._sessao(self.seg, 4), D._sessao(self.seg, 4))
        self.assertIs(D._extrator(self.emb, 4), D._extrator(self.emb, 4))

    def test_incremental_igual_ao_lote_em_pedacos_quaisquer(self):
        p = D.Params()
        lote = [Trecho(a, b, g) for a, b, g in D.diarizar_audio(self.audio, None, self.seg, self.emb, p)]
        self.assertTrue(lote)
        rng = np.random.default_rng(42)
        inc = D.AnaliseIncremental(self.seg, self.emb, p)
        pos = 0
        while pos < len(self.audio):
            n = int(rng.integers(1, 50000))          # de 1 amostra a ~3 s
            pos_antes = inc.janelas_prontas
            inc.alimentar(self.audio[pos:pos + n])
            self.assertGreaterEqual(inc.janelas_prontas, pos_antes)
            pos += n
        esperadas = max(0, (len(self.audio) - D.JANELA) // p.passo_janela + 1)
        self.assertEqual(inc.janelas_prontas, esperadas)      # o custo foi pago durante a alimentação
        self.assertAlmostEqual(inc.duracao, len(self.audio) / 16000)
        self.assertEqual(inc.concluir(), lote)
        self.assertEqual(inc.concluir(), lote)                # concluir não altera o estado

    def test_incremental_com_k_e_cauda_parcial(self):
        p = D.Params()
        audio = self.audio[: 16000 * 27 + 1234]               # termina no meio de uma janela
        for k in (None, 2):
            lote = [Trecho(a, b, g) for a, b, g in D.diarizar_audio(audio, k, self.seg, self.emb, p)]
            inc = D.AnaliseIncremental(self.seg, self.emb, p)
            for i in range(0, len(audio), 16000 * 5):
                inc.alimentar(audio[i:i + 16000 * 5])
            self.assertEqual(inc.concluir(k), lote, f"k={k}")

    def test_incremental_audio_curto_e_vazio(self):
        p = D.Params()
        inc = D.AnaliseIncremental(self.seg, self.emb, p)
        self.assertEqual(inc.concluir(), [])
        self.assertEqual(inc.alimentar(np.zeros(0, dtype=np.float32)), 0)
        curto = self.audio[: 16000 * 6]
        inc.alimentar(curto)
        self.assertEqual(inc.janelas_prontas, 0)              # ainda não há janela de 10 s completa
        lote = [Trecho(a, b, g) for a, b, g in D.diarizar_audio(curto, None, self.seg, self.emb, p)]
        self.assertEqual(inc.concluir(), lote)


if __name__ == "__main__":
    unittest.main()
