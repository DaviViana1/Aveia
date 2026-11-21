"""Testes offline do pipeline: agrupamento, suavização de ilhas, ordem/parcial/progresso e aquecimento."""
from __future__ import annotations

import os
import threading
import unittest
from collections import namedtuple
from unittest import mock

import numpy as np

from transcritor import pipeline
from transcritor.diarizacao import Trecho

P = namedtuple("P", "start end word")


def frase(texto: str, inicio: float = 0.0, passo: float = 0.3, dur: float = 0.25) -> list[P]:
    """Palavras com espaço à esquerda, uma a cada `passo` segundos."""
    return [P(inicio + i * passo, inicio + i * passo + dur, " " + w) for i, w in enumerate(texto.split())]


class FalasProvisoriasTest(unittest.TestCase):
    def test_agrupa_por_pausa(self):
        pal = frase("olá pessoal tudo bem") + frase("agora outra", inicio=40.0)   # pausa de 30 s ou mais separa
        falas = pipeline.falas_provisorias(pal)
        self.assertEqual(len(falas), 2)
        self.assertEqual(falas[0].texto, "olá pessoal tudo bem")
        self.assertEqual(falas[1].texto, "agora outra")
        self.assertTrue(all(f.locutor == pipeline.LOCUTOR_PENDENTE for f in falas))

    def test_pausa_menor_que_30s_junta_e_30s_separa(self):
        a = [P(0.0, 1.0, " a"), P(31.0, 31.5, " b")]     # lacuna 30,0 -> separa
        b = [P(0.0, 1.0, " a"), P(30.9, 31.5, " b")]     # lacuna 29,9 -> junta
        c = [P(0.0, 1.0, " a"), P(3.5, 4.0, " b")]       # lacuna 2,5 (antes separava) -> junta
        self.assertEqual(len(pipeline.falas_provisorias(a)), 2)
        self.assertEqual(len(pipeline.falas_provisorias(b)), 1)
        self.assertEqual(len(pipeline.falas_provisorias(c)), 1)

    def test_gravacao_real_pausas_de_2_e_3_segundos_e_uma_fala(self):
        """Caso do usuário (2026-11-21): uma pessoa, pausas de 2,1 s e 3,5 s no meio da leitura = 1 fala, não 3."""
        pal = (frase("a sílaba fina da flauta", 1.3) + frase("ecoa pela sala silenciosa", 6.8)
               + frase("e pelas risadas", 28.0) + frase("estridentes dos esquilos", 34.6))
        falas = pipeline.falas_provisorias(pal)
        self.assertEqual(len(falas), 1)
        self.assertEqual((falas[0].inicio, falas[0].fim), (pal[0].start, pal[-1].end))

    def test_troca_de_locutor_separa_mesmo_com_pausa_curta(self):
        pal = frase("bom dia a todos", 0.0) + frase("oi tudo bem", 4.5)
        falas = pipeline.montar_falas(pal, [Trecho(0, 4, 0), Trecho(4.4, 8, 1)])
        self.assertEqual([f.locutor for f in falas], ["Locutor 1", "Locutor 2"])

    def test_mesmo_locutor_pausa_curta_junta_e_30s_separa(self):
        curta = frase("primeira parte", 0.0) + frase("segunda parte", 9.0)
        longa = frase("primeira parte", 0.0) + frase("segunda parte", 40.0)
        self.assertEqual(len(pipeline.montar_falas(curta, [Trecho(0, 12, 0)])), 1)
        falas = pipeline.montar_falas(longa, [Trecho(0, 45, 0)])
        self.assertEqual([f.locutor for f in falas], ["Locutor 1", "Locutor 1"])

    def test_teto_de_5_minutos(self):
        seq = lambda n, passo: [P(i * passo, i * passo + passo * 0.8, " p") for i in range(n)]
        casos = [(299.0, 1), (301.0, 2)]
        for dur, esperado in casos:
            pal = seq(int(dur), 1.0)    # uma palavra por segundo, sem pausa nítida
            falas = pipeline.falas_provisorias(pal)
            self.assertEqual(len(falas), esperado, dur)
            self.assertTrue(all(f.fim - f.inicio <= pipeline.FALA_MAX_S for f in falas))

    def test_teto_corta_na_maior_pausa_dos_ultimos_60s(self):
        pal = [P(i * 1.0, i * 1.0 + 0.8, " p") for i in range(250)]            # 0 a 249,8 s
        pal += [P(255.0 + i * 1.0, 255.8 + i * 1.0, " q") for i in range(60)]   # pausa de 5,2 s em 249,8 -> 255
        falas = pipeline.falas_provisorias(pal)
        self.assertEqual(len(falas), 2)
        self.assertAlmostEqual(falas[0].fim, 249.8)
        self.assertAlmostEqual(falas[1].inicio, 255.0)

    def test_teto_mantem_o_locutor(self):
        pal = [P(i * 1.0, i * 1.0 + 0.8, " p") for i in range(700)]
        falas = pipeline.montar_falas(pal, [Trecho(0, 700, 4)])
        self.assertGreaterEqual(len(falas), 3)
        self.assertEqual({f.locutor for f in falas}, {"Locutor 1"})
        self.assertEqual(sum(len(f.texto.split()) for f in falas), 700)

    def test_vazio_e_lista_nova(self):
        self.assertEqual(pipeline.falas_provisorias([]), [])
        pal = frase("um dois")
        self.assertIsNot(pipeline.falas_provisorias(pal), pipeline.falas_provisorias(pal))


class MontarFalasTest(unittest.TestCase):
    def test_nomes_por_ordem_de_aparicao(self):
        pal = frase("bom dia a todos", 0.0) + frase("tudo certo por aqui", 5.0) + frase("ótimo então", 10.0)
        trechos = [Trecho(0, 4, 7), Trecho(5, 9, 3), Trecho(10, 14, 7)]
        falas = pipeline.montar_falas(pal, trechos)
        self.assertEqual([f.locutor for f in falas], ["Locutor 1", "Locutor 2", "Locutor 1"])
        self.assertEqual(falas[1].texto, "tudo certo por aqui")

    def test_sem_trechos_locutor_1(self):
        falas = pipeline.montar_falas(frase("só um locutor aqui"), [])
        self.assertEqual(len(falas), 1)
        self.assertEqual(falas[0].locutor, "Locutor 1")
        self.assertEqual(falas[0].texto, "só um locutor aqui")


class SuavizarIlhasTest(unittest.TestCase):
    def _montar(self, pal, trechos, **kw):
        return pipeline.montar_falas(pal, trechos, **kw)

    def _cenario(self, ilha_palavras, ilha_idx):
        """Frase contínua do locutor 0 cujas palavras `ilha_idx` caem em trecho do locutor 1."""
        pal = frase(" ".join(ilha_palavras))
        trechos = []
        for i, p in enumerate(pal):
            trechos.append(Trecho(p.start, p.end, 1 if i in ilha_idx else 0))
        return pal, trechos

    def test_ilha_de_uma_palavra_absorvida(self):
        pal, tr = self._cenario("eu acho que sim vamos fazer isso".split(), {3})
        falas = self._montar(pal, tr)
        self.assertEqual(len(falas), 1)
        self.assertEqual(falas[0].texto, "eu acho que sim vamos fazer isso")

    def test_ilha_de_duas_palavras_absorvida(self):
        pal, tr = self._cenario("eu acho que vamos fazer isso agora".split(), {3, 4})
        falas = self._montar(pal, tr)
        self.assertEqual(len(falas), 1)
        self.assertEqual(falas[0].locutor, "Locutor 1")

    def test_interjecao_apos_fim_de_frase_mantida(self):
        pal, tr = self._cenario("Isso está pronto. Sim. Então vamos seguir".split(), {3})
        falas = self._montar(pal, tr)
        self.assertEqual([f.texto for f in falas], ["Isso está pronto.", "Sim.", "Então vamos seguir"])
        self.assertEqual([f.locutor for f in falas], ["Locutor 1", "Locutor 2", "Locutor 1"])

    def test_ilha_com_pausa_mantida(self):
        pal = [P(0.0, 0.25, " eu"), P(0.3, 0.55, " acho"), P(1.2, 1.4, " sim"),   # pausa de 0,65 s antes
               P(1.5, 1.7, " mesmo"), P(1.8, 2.0, " assim")]
        tr = [Trecho(0, 0.6, 0), Trecho(1.2, 1.45, 1), Trecho(1.5, 2.1, 0)]
        falas = self._montar(pal, tr)
        self.assertEqual([f.locutor for f in falas], ["Locutor 1", "Locutor 2", "Locutor 1"])

    def test_sequencia_de_tres_palavras_mantida(self):
        pal, tr = self._cenario("eu acho que vamos fazer isso agora mesmo".split(), {3, 4, 5})
        falas = self._montar(pal, tr)
        self.assertEqual(len(falas), 3)
        self.assertEqual(falas[1].texto, "vamos fazer isso")

    def test_desligada_por_env(self):
        pal, tr = self._cenario("eu acho que sim vamos fazer isso".split(), {3})
        with mock.patch.dict(os.environ, {"AVEIA_SUAVIZAR": "0"}):
            falas = self._montar(pal, tr)
        self.assertEqual(len(falas), 3)

    def test_desligada_por_parametro(self):
        pal, tr = self._cenario("eu acho que sim vamos fazer isso".split(), {3})
        self.assertEqual(len(self._montar(pal, tr, suavizar=False)), 3)

    def test_vizinhas_de_locutores_diferentes_nao_reatribui(self):
        pal = frase("um dois tres quatro")
        tr = [Trecho(0, 0.3, 0), Trecho(0.3, 0.6, 1), Trecho(0.6, 0.9, 2), Trecho(0.9, 1.3, 2)]
        falas = self._montar(pal, tr)
        self.assertEqual(len(falas), 3)


class TranscreverTest(unittest.TestCase):
    def setUp(self):
        self.eventos: list[str] = []
        self.progresso: list[tuple[float, str]] = []
        self.palavras = frase("bom dia pessoal vamos começar", 0.0) + frase("tudo certo", 5.0)

        def decode(caminho, sampling_rate=16000):
            return np.zeros(sampling_rate * 10, dtype=np.float32)

        def palavras(audio, duracao, idioma, avisar, motor=None, meta=None):
            meta = meta if meta is not None else {}
            meta.update(motor="local", avisos=meta.get("avisos", []))
            avisar(0.25, "Carregando modelo Whisper...")
            for i, p in enumerate(self.palavras):
                avisar(0.25 + 0.7 * (i + 1) / len(self.palavras), "Transcrevendo...")
                yield p

        self.diarizar_chamou = False

        def diarizar(audio, num_locutores=None):
            self.diarizar_chamou = True
            self.eventos.append("diarizar")
            if self.diarizar_falha:
                raise RuntimeError("modelo ausente")
            return [Trecho(0, 4, 5), Trecho(5, 8, 9)]

        self.diarizar_falha = False
        for alvo, novo in [
            (pipeline, ("decode_audio", decode)),
            (pipeline, ("preparar_audio", lambda a: a)),
            (pipeline.motores, ("palavras", palavras)),
            (pipeline, ("diarizar", diarizar)),
        ]:
            p = mock.patch.object(alvo, novo[0], novo[1])
            p.start()
            self.addCleanup(p.stop)
        # sem esperar 1 s reais entre parciais
        p = mock.patch.object(pipeline, "_INTERVALO_PARCIAL", 0.0)
        p.start()
        self.addCleanup(p.stop)

    def _registrar_parcial(self, falas):
        self.eventos.append("parcial")
        self.parciais.append(falas)

    def test_parcial_antes_da_diarizacao_e_pendente(self):
        self.parciais: list = []
        falas = pipeline.transcrever("x.wav", parcial=self._registrar_parcial,
                                     progresso=lambda p, m: self.progresso.append((p, m)), motor="local")
        self.assertIn("parcial", self.eventos)
        self.assertLess(max(i for i, e in enumerate(self.eventos) if e == "parcial"), self.eventos.index("diarizar"))
        for lote in self.parciais:
            self.assertTrue(all(f.locutor == pipeline.LOCUTOR_PENDENTE for f in lote))
        ids = {id(l) for l in self.parciais}
        self.assertEqual(len(ids), len(self.parciais))   # lista nova a cada chamada
        # último parcial já tem todo o texto
        self.assertEqual(" ".join(f.texto for f in self.parciais[-1]), "bom dia pessoal vamos começar tudo certo")
        self.assertEqual([f.locutor for f in falas], ["Locutor 1", "Locutor 2"])
        self.assertEqual(" ".join(f.texto for f in falas), "bom dia pessoal vamos começar tudo certo")

    def test_progresso_monotonico_e_faixas(self):
        pipeline.transcrever("x.wav", progresso=lambda p, m: self.progresso.append((p, m)), motor="local")
        valores = [p for p, _ in self.progresso]
        self.assertEqual(valores, sorted(valores))
        self.assertEqual(valores[0], 0.02)
        self.assertEqual(valores[-1], 1.0)
        msgs = [m for _, m in self.progresso]
        self.assertIn("Carregando modelo Whisper...", msgs)
        i_diar = msgs.index("Identificando locutores pelo timbre...")
        self.assertAlmostEqual(valores[i_diar], 0.80)
        self.assertTrue(all(0.05 <= v <= 0.80 for m, v in zip(msgs[:i_diar], valores[:i_diar]) if m == "Transcrevendo..."))

    def test_excecao_no_parcial_nao_quebra(self):
        def ruim(falas):
            raise ValueError("boom")
        falas = pipeline.transcrever("x.wav", parcial=ruim, motor="local")
        self.assertEqual(len(falas), 2)

    def test_sem_diarizacao_locutor_1(self):
        falas = pipeline.transcrever("x.wav", diarizacao=False, motor="local")
        self.assertFalse(self.diarizar_chamou)
        self.assertTrue(all(f.locutor == "Locutor 1" for f in falas))

    def test_falha_na_diarizacao_preserva_texto_com_aviso(self):
        self.diarizar_falha = True
        meta: dict = {}
        falas = pipeline.transcrever("x.wav", motor="local", meta=meta)
        self.assertEqual(" ".join(f.texto for f in falas), "bom dia pessoal vamos começar tudo certo")
        self.assertTrue(all(f.locutor == "Locutor 1" for f in falas))
        self.assertEqual(len(meta["avisos"]), 1)
        self.assertIn("modelo ausente", meta["avisos"][0])

    def test_falha_na_diarizacao_sem_meta_nao_levanta(self):
        self.diarizar_falha = True
        # sem locutores distintos e com pausa de 5 s (< 30 s), o texto todo é uma fala só
        self.assertEqual(len(pipeline.transcrever("x.wav", motor="local")), 1)


class AquecerTest(unittest.TestCase):
    def setUp(self):
        antigo = dict(pipeline._modelos)
        pipeline._modelos.clear()
        self.addCleanup(lambda: (pipeline._modelos.clear(), pipeline._modelos.update(antigo)))

    def test_duas_chamadas_constroem_uma_vez(self):
        with mock.patch.object(pipeline, "WhisperModel") as modelo:
            pipeline.aquecer(em_segundo_plano=False)
            pipeline.aquecer(em_segundo_plano=False)
            self.assertEqual(modelo.call_count, 1)

    def test_concorrente_constroi_uma_vez(self):
        chamadas = []

        def lento(*a, **k):
            chamadas.append(1)
            threading.Event().wait(0.1)
            return object()

        with mock.patch.object(pipeline, "WhisperModel", side_effect=lento):
            ts = [threading.Thread(target=pipeline._carregar_modelo, args=(pipeline.MODELO,)) for _ in range(5)]
            for t in ts:
                t.start()
            pipeline.aquecer(em_segundo_plano=False)
            for t in ts:
                t.join()
        self.assertEqual(len(chamadas), 1)

    def test_em_segundo_plano_usa_thread_daemon_e_nao_levanta(self):
        with mock.patch.object(pipeline, "WhisperModel", side_effect=RuntimeError("sem modelo")):
            pipeline.aquecer(em_segundo_plano=False)   # não levanta
            with mock.patch.object(pipeline.threading, "Thread") as th:
                pipeline.aquecer()
                self.assertTrue(th.call_args.kwargs["daemon"])
                th.return_value.start.assert_called_once()


if __name__ == "__main__":
    unittest.main()
