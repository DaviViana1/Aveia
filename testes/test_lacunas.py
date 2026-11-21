"""Defesa contra truncamento (pipeline._palavras: retranscrição de lacunas). Sem modelo, sem internet, sem VAD real.

`pipeline._carregar_modelo` devolve um Whisper falso com respostas roteirizadas (1ª chamada = áudio inteiro; as seguintes =
fatias de lacuna, em ordem) e `pipeline._trechos_fala` devolve a fala do VAD que cada teste combina.
Rodar: .venv\\Scripts\\python -m unittest testes.test_lacunas -v
"""
from __future__ import annotations

import os
import tempfile
import unittest
from collections import namedtuple
from pathlib import Path
from unittest import mock

import numpy as np

from transcritor import aovivo, pipeline
from transcritor.diarizacao import Trecho

Pal = namedtuple("Pal", "start end word")
Seg = namedtuple("Seg", "start end words")
TAXA = 16000


def palavras_em(inicio: float, fim: float, n: int, texto: str = "fala") -> list[tuple[float, float, str]]:
    """n palavras (início, fim, " texto<i>") espalhadas entre `inicio` e `fim` (tempos relativos ao áudio que as contém)."""
    passo = (fim - inicio) / n
    return [(inicio + i * passo, inicio + i * passo + 0.8 * passo, f" {texto}{i}") for i in range(n)]


class ModeloFalso:
    def __init__(self, principal, fatias=()):
        self.principal = principal
        self.fatias = list(fatias)
        self.duracoes: list[float] = []      # duração de cada áudio recebido (1ª = o inteiro)

    def transcribe(self, audio, **opcoes):
        self.duracoes.append(len(audio) / TAXA)
        lista = self.principal if len(self.duracoes) == 1 else (self.fatias.pop(0) if self.fatias else [])
        return iter([Seg(x[0], x[1], [Pal(*x)]) for x in lista]), None


class Base(unittest.TestCase):
    def setUp(self):
        self.vad_chamadas = 0
        self.fala: list[tuple[float, float]] = []

        def vad(audio):
            self.vad_chamadas += 1
            return list(self.fala)

        for alvo, nome, novo in [(pipeline, "_trechos_fala", vad)]:
            p = mock.patch.object(alvo, nome, novo)
            p.start()
            self.addCleanup(p.stop)
        p = mock.patch.dict(os.environ, {"AVEIA_LACUNAS": "1"})
        p.start()
        self.addCleanup(p.stop)

    def rodar(self, total, principal, fatias=(), lacunas=None):
        modelo = ModeloFalso(principal, fatias)
        with mock.patch.object(pipeline, "_carregar_modelo", lambda nome: modelo):
            saida = list(pipeline._palavras(np.zeros(int(total * TAXA), dtype=np.float32), total, "pt",
                                            lambda p, m: None, lacunas=lacunas))
        self.modelo = modelo
        return saida

    @staticmethod
    def tempos(saida):
        return [round(p.start, 2) for p in saida]


class SemLacunaTest(Base):
    def test_sem_lacuna_nada_muda_e_nao_roda_vad_nem_retry(self):
        principal = palavras_em(0.5, 29.5, 30)   # uma palavra por segundo, nenhuma pausa > 3 s
        lacunas: list = []
        saida = self.rodar(30.0, principal, lacunas=lacunas)
        self.assertEqual([(p.start, p.end, p.word) for p in saida], principal)
        self.assertEqual(len(self.modelo.duracoes), 1)
        self.assertEqual(self.vad_chamadas, 0)
        self.assertEqual(lacunas, [])

    def test_lacuna_sem_fala_suficiente_no_vad_nao_retranscreve(self):
        principal = palavras_em(0.5, 5, 5) + palavras_em(12, 15, 3, "b")
        self.fala = [(0.4, 5.0), (5.0, 6.0), (11.8, 15.0)]    # só 1 s de fala dentro da lacuna 4,5-12
        lacunas: list = []
        saida = self.rodar(16.0, principal, lacunas=lacunas)
        self.assertEqual(len(saida), 8)
        self.assertEqual(len(self.modelo.duracoes), 1)
        self.assertEqual(lacunas, [])
        self.assertEqual(self.vad_chamadas, 1)


class RecuperacaoTest(Base):
    def test_lacuna_com_fala_recuperada_em_ordem_de_tempo(self):
        antes, depois = palavras_em(0.5, 5, 5, "a"), palavras_em(13, 16, 3, "c")
        a = antes[-1][1]                                      # a lacuna começa no fim da última palavra de antes
        self.fala = [(0.4, 5.0), (5.5, 12.5), (12.8, 16.0)]
        # a fatia começa em a - 0,3 s; as palavras recuperadas vêm com tempo relativo a ela
        recuperadas = palavras_em(0.6, 7.6, 14, "b")
        lacunas: list = []
        saida = self.rodar(17.0, antes + depois, [recuperadas], lacunas)
        self.assertEqual(len(saida), 5 + 14 + 3)
        self.assertEqual(self.tempos(saida), sorted(self.tempos(saida)))
        self.assertEqual([p.word for p in saida[:5]], [x[2] for x in antes])
        self.assertEqual([p.word for p in saida[5:19]], [x[2] for x in recuperadas])
        self.assertEqual([p.word for p in saida[19:]], [x[2] for x in depois])
        self.assertAlmostEqual(saida[5].start, 0.6 + a - 0.3, places=2)
        self.assertEqual(len(self.modelo.duracoes), 2)       # só a fatia da lacuna foi retranscrita
        self.assertLess(self.modelo.duracoes[1], 9.0)
        self.assertEqual(len(lacunas), 1)
        self.assertEqual(lacunas[0]["estado"], "recuperada")
        self.assertEqual(lacunas[0]["palavras_recuperadas"], 14)
        self.assertEqual(set(lacunas[0]), {"inicio", "fim", "fala_s", "palavras_recuperadas", "estado"})
        self.assertAlmostEqual(lacunas[0]["inicio"], a, places=1)
        self.assertAlmostEqual(lacunas[0]["fim"], 13.0, places=1)

    def test_palavra_recuperada_depois_do_fim_da_lacuna_e_descartada(self):
        antes, depois = palavras_em(0.5, 5, 5, "a"), palavras_em(13, 16, 3, "c")
        self.fala = [(0.4, 5.0), (5.5, 12.5), (12.8, 16.0)]
        extra = palavras_em(0.6, 7.6, 14, "b") + [(8.5, 8.8, " duplicada")]   # a fatia vai até 13,3 s: 8,5 + 4,7 = 13,2
        saida = self.rodar(17.0, antes + depois, [extra])
        self.assertNotIn(" duplicada", [p.word for p in saida])
        self.assertEqual(self.tempos(saida), sorted(self.tempos(saida)))

    def test_lacuna_inicial(self):
        principal = palavras_em(7.0, 10.0, 3, "b")
        self.fala = [(0.5, 6.8), (7.0, 10.0)]
        lacunas: list = []
        saida = self.rodar(11.0, principal, [palavras_em(0.6, 6.0, 12, "a")], lacunas)
        self.assertEqual(len(saida), 15)
        self.assertEqual(self.tempos(saida), sorted(self.tempos(saida)))
        self.assertTrue(saida[0].word.startswith(" a"))
        self.assertEqual((lacunas[0]["inicio"], lacunas[0]["estado"]), (0.0, "recuperada"))

    def test_lacuna_final(self):
        principal = palavras_em(0.5, 4.0, 4, "a")
        self.fala = [(0.4, 4.0), (5.0, 11.0)]
        lacunas: list = []
        saida = self.rodar(12.0, principal, [palavras_em(0.9, 6.5, 12, "z")], lacunas)
        self.assertEqual(len(saida), 16)
        self.assertEqual(self.tempos(saida), sorted(self.tempos(saida)))
        self.assertTrue(saida[-1].word.startswith(" z"))
        self.assertAlmostEqual(lacunas[0]["fim"], 12.0, places=1)
        self.assertEqual(lacunas[0]["estado"], "recuperada")

    def test_estado_parcial_quando_densidade_baixa(self):
        principal = palavras_em(0.5, 4.0, 4) + palavras_em(14, 16, 2, "c")
        self.fala = [(0.4, 4.0), (4.5, 13.5), (13.8, 16.0)]   # 9 s de fala na lacuna
        lacunas: list = []
        self.rodar(17.0, principal, [palavras_em(0.6, 8.0, 6, "b")], lacunas)   # 6 palavras / 9 s = 0,67: passa, mas é pouco
        self.assertEqual(lacunas[0]["estado"], "parcial")
        self.assertEqual(lacunas[0]["palavras_recuperadas"], 6)

    def test_excecao_na_retranscricao_nao_derruba_e_vira_nao_recuperada(self):
        principal = palavras_em(0.5, 4.0, 4) + palavras_em(14, 16, 2, "c")
        self.fala = [(0.4, 4.0), (4.5, 13.5), (13.8, 16.0)]

        class Quebra(ModeloFalso):
            def transcribe(self, audio, **o):
                if self.duracoes:
                    raise RuntimeError("boom")
                return super().transcribe(audio, **o)

        modelo = Quebra(principal)
        lacunas: list = []
        with mock.patch.object(pipeline, "_carregar_modelo", lambda n: modelo):
            saida = list(pipeline._palavras(np.zeros(17 * TAXA, dtype=np.float32), 17.0, "pt", lambda p, m: None, lacunas=lacunas))
        self.assertEqual(len(saida), 6)
        self.assertEqual(lacunas[0]["estado"], "nao_recuperada")


class FiltrosTest(Base):
    def setUp(self):
        super().setUp()
        self.principal = palavras_em(0.5, 4.0, 4) + palavras_em(40, 43, 3, "c")   # lacuna 3,7-40, fala em 3 trechos
        self.fala = [(0.4, 4.0), (10.0, 18.0), (18.6, 26.0), (26.6, 34.0), (39.8, 43.0)]   # 22,8 s de fala na lacuna
        self.total = 44.0

    def test_texto_curto_demais_rejeitado_e_segunda_tentativa_em_pedacos_aceita(self):
        lacunas: list = []
        curto = palavras_em(10.5, 12.0, 3, "x")                         # 3 palavras para 22,8 s: 0,13 por s
        pedacos = [palavras_em(1.0, 8.5, 8, "p"), palavras_em(1.0, 7.6, 8, "q"), palavras_em(1.0, 7.4, 8, "r")]
        saida = self.rodar(self.total, self.principal, [curto] + pedacos, lacunas)
        self.assertEqual(len(self.modelo.duracoes), 1 + 1 + 3)           # inteiro, lacuna, 3 pedaços
        self.assertTrue(all(d < 10.0 for d in self.modelo.duracoes[2:]))
        palavras = [p.word for p in saida]
        self.assertEqual(len(saida), 4 + 24 + 3)
        self.assertNotIn(" x0", palavras)
        self.assertEqual(self.tempos(saida), sorted(self.tempos(saida)))
        self.assertEqual(lacunas[0]["estado"], "recuperada")
        self.assertEqual(lacunas[0]["palavras_recuperadas"], 24)

    def test_segunda_tentativa_tambem_rejeitada_deixa_lacuna_nao_recuperada(self):
        lacunas: list = []
        curto = palavras_em(10.5, 12.0, 3, "x")
        pedacos = [palavras_em(1.0, 3.0, 1, "p"), [], palavras_em(1.0, 3.0, 1, "r")]
        saida = self.rodar(self.total, self.principal, [curto] + pedacos, lacunas)
        self.assertEqual(len(saida), 4 + 3)                              # nada entrou
        self.assertEqual(len(lacunas), 1)
        self.assertEqual((lacunas[0]["estado"], lacunas[0]["palavras_recuperadas"]), ("nao_recuperada", 0))
        self.assertAlmostEqual(lacunas[0]["fala_s"], 23.2, places=1)

    def test_eco_da_dica_descartado(self):
        eco = [(10.2 + i * 0.5, 10.6 + i * 0.5, " " + w) for i, w in enumerate(
            "Reunião de trabalho em português do Brasil, com números, orçamento e faturamento.".split())]
        lacunas: list = []
        saida = self.rodar(self.total, self.principal, [eco, [], [], []], lacunas)
        self.assertEqual(len(saida), 7)
        self.assertEqual(lacunas[0]["estado"], "nao_recuperada")

    def test_somente_alucinacao_de_legenda_descartada(self):
        self.fala = [(0.4, 4.0), (10.0, 13.0), (39.8, 43.0)]            # lacuna com 3 s de fala: 3 palavras passam o piso
        lacunas: list = []
        saida = self.rodar(self.total, self.principal,
                           [[(0.8, 1.0, " Obrigado"), (1.0, 1.3, " por"), (1.3, 1.8, " assistir.")], []], lacunas)
        self.assertEqual(len(saida), 7)
        self.assertEqual(lacunas[0]["estado"], "nao_recuperada")

    def test_funcoes_de_filtro(self):
        n = pipeline._norm
        self.assertTrue(pipeline._so_alucinacao(n("Obrigado por assistir. Obrigado por assistir.")))
        self.assertTrue(pipeline._so_alucinacao(n("Legendas pela comunidade Amara.org")))
        self.assertFalse(pipeline._so_alucinacao(n("Obrigado por assistir, vamos ao orçamento do trimestre")))
        self.assertTrue(pipeline._eco_da_dica(n("em português do Brasil, com números, orçamento")))
        self.assertTrue(pipeline._eco_da_dica(n("Reunião de trabalho")))
        self.assertFalse(pipeline._eco_da_dica(n(
            "bom, sobre o faturamento do mês passado, eu acho que a reunião de trabalho em português do Brasil "
            "não muda o que combinamos")))
        P = lambda *w: [Pal(0, 1, " " + x) for x in w]   # noqa: E731
        self.assertFalse(pipeline._aceitavel(P("um", "dois", "três"), 10.0))     # 0,3 palavra por s de fala
        self.assertTrue(pipeline._aceitavel(P("um", "dois", "três", "quatro", "cinco"), 10.0))
        self.assertFalse(pipeline._aceitavel([], 5.0))

    def test_pedacos_cortam_nas_pausas_e_respeitam_o_maximo(self):
        audio = np.zeros(45 * TAXA, dtype=np.float32)
        c = pipeline._cortes_em_pedacos(audio, 3.7, 40.0, self.fala[1:4])
        self.assertEqual(c[0], 3.7)
        self.assertEqual(c[-1], 40.0)
        self.assertEqual(len(c), 4)
        self.assertAlmostEqual(c[1], 18.3, places=1)
        # fala contínua de 24 s sem pausa: dividida em 3 pedaços (nenhum de fala acima de 10 s)
        c2 = pipeline._cortes_em_pedacos(audio, 0.0, 30.0, [(2.0, 26.0)])
        self.assertEqual(len(c2), 4)
        self.assertEqual(c2, sorted(c2))


class DesligadoTest(Base):
    def test_env_desliga_tudo(self):
        principal = palavras_em(0.5, 4.0, 4) + palavras_em(14, 16, 2, "c")
        self.fala = [(0.4, 4.0), (4.5, 13.5), (13.8, 16.0)]
        lacunas: list = []
        with mock.patch.dict(os.environ, {"AVEIA_LACUNAS": "0"}):
            saida = self.rodar(17.0, principal, [palavras_em(0.6, 8.0, 12, "b")], lacunas)
        self.assertEqual([(p.start, p.end, p.word) for p in saida], principal)
        self.assertEqual(len(self.modelo.duracoes), 1)
        self.assertEqual(self.vad_chamadas, 0)
        self.assertEqual(lacunas, [])


class AvisosTest(unittest.TestCase):
    def test_texto_do_aviso(self):
        av = pipeline.avisos_de_lacunas([
            {"inicio": 57.4, "fim": 99.2, "fala_s": 26.2, "palavras_recuperadas": 0, "estado": "nao_recuperada"},
            {"inicio": 3700.0, "fim": 3711.5, "fala_s": 9.0, "palavras_recuperadas": 6, "estado": "parcial"},
            {"inicio": 10.0, "fim": 20.0, "fala_s": 9.0, "palavras_recuperadas": 20, "estado": "recuperada"},
        ])
        self.assertEqual(av[0], "Trecho 00:57–01:40 pode estar incompleto: havia fala e o texto não foi recuperado.")
        self.assertEqual(av[1], "Trecho 1:01:40–1:01:52 pode estar incompleto: havia fala e só parte do texto foi recuperada.")
        self.assertEqual(len(av), 2)

    def test_muitos_avisos_viram_resumo(self):
        lac = [{"inicio": 10.0 * i, "fim": 10.0 * i + 5, "fala_s": 4.0, "palavras_recuperadas": 0, "estado": "nao_recuperada"}
               for i in range(9)]
        av = pipeline.avisos_de_lacunas(lac)
        self.assertEqual(len(av), pipeline.MAX_AVISOS_LACUNAS + 1)
        self.assertIn("Mais 3 trechos", av[-1])


class TranscreverTest(Base):
    """Caminho clássico completo (`transcrever` → `motores.palavras` → `_palavras`) com o Whisper falso."""

    def setUp(self):
        super().setUp()
        for alvo, nome, novo in [
            (pipeline, "decode_audio", lambda c, sampling_rate=16000: np.zeros(sampling_rate * 120, dtype=np.float32)),
            (pipeline, "preparar_audio", lambda a: a),
            (pipeline, "diarizar", lambda a, num_locutores=None: [Trecho(0, 120, 0)]),
        ]:
            p = mock.patch.object(alvo, nome, novo)
            p.start()
            self.addCleanup(p.stop)
        self.principal = palavras_em(0.5, 56, 40, "a") + [(56.9, 57.4, " z")] + palavras_em(99.2, 118, 30, "c")
        self.fala = [(0.4, 57.4), (58.0, 99.0), (99.0, 118.0)]

    def _transcrever(self, fatias, **kw):
        modelo = ModeloFalso(self.principal, fatias)
        meta: dict = {}
        with mock.patch.object(pipeline, "_carregar_modelo", lambda n: modelo):
            falas = pipeline.transcrever("x.wav", motor="local", meta=meta, **kw)
        return falas, meta

    def test_lacuna_nao_recuperada_vira_aviso_com_o_intervalo(self):
        falas, meta = self._transcrever([[], [], [], [], []])
        self.assertEqual(len(meta["lacunas"]), 1)
        self.assertEqual(meta["lacunas"][0]["estado"], "nao_recuperada")
        self.assertEqual(meta["avisos"],
                         ["Trecho 00:57–01:40 pode estar incompleto: havia fala e o texto não foi recuperado."])
        self.assertEqual(sum(len(f.texto.split()) for f in falas), 71)

    def test_lacuna_recuperada_nao_gera_aviso_e_falas_ficam_em_ordem(self):
        falas, meta = self._transcrever([palavras_em(0.8, 41.0, 60, "b")])
        self.assertEqual(meta["lacunas"][0]["estado"], "recuperada")
        self.assertEqual(meta["avisos"], [])
        self.assertEqual(sum(len(f.texto.split()) for f in falas), 131)
        inicios = [f.inicio for f in falas]
        self.assertEqual(inicios, sorted(inicios))
        self.assertTrue(all(a.fim <= b.inicio + 1e-6 for a, b in zip(falas, falas[1:])))

    def test_sem_lacuna_meta_traz_lista_vazia_e_nenhum_aviso(self):
        self.principal = palavras_em(0.5, 118, 118)
        falas, meta = self._transcrever([])
        self.assertEqual((meta["lacunas"], meta["avisos"]), ([], []))

    def test_desligado_nao_avisa(self):
        with mock.patch.dict(os.environ, {"AVEIA_LACUNAS": "0"}):
            falas, meta = self._transcrever([])
        self.assertEqual((meta["lacunas"], meta["avisos"]), ([], []))
        self.assertEqual(sum(len(f.texto.split()) for f in falas), 71)

    def test_groq_com_fallback_local_devolve_lacunas_na_linha_do_tempo_inteira(self):
        from transcritor import motores
        meta: dict = {}
        # o local assume a partir de 10 s: o Whisper falso vê só audio[10 s:] (10 s); lacuna relativa 2,8-7,5 vira 12,8-17,5
        modelo = ModeloFalso(palavras_em(0.5, 3.0, 3) + palavras_em(7.5, 9.5, 2, "c"), [[], []])
        self.fala = [(0.4, 3.0), (3.5, 7.0), (7.4, 9.5)]
        audio = np.zeros(20 * TAXA, dtype=np.float32)
        with mock.patch.object(pipeline, "_carregar_modelo", lambda n: modelo),                 mock.patch.object(motores, "groq_configurado", lambda: True),                 mock.patch.object(motores, "_groq_palavras", lambda a, d, i, p, av, m, local: local(10.0)):
            palavras = list(motores.palavras(audio, 20.0, "pt", lambda p, m: None, motor="groq", meta=meta))
        self.assertEqual(len(palavras), 5)
        self.assertGreaterEqual(palavras[0].start, 10.0)
        self.assertEqual(len(meta["lacunas"]), 1)
        self.assertAlmostEqual(meta["lacunas"][0]["inicio"], 12.83, places=1)
        self.assertAlmostEqual(meta["lacunas"][0]["fim"], 17.5, places=1)
        self.assertEqual(meta["lacunas"][0]["estado"], "nao_recuperada")


class AoVivoTest(Base):
    """Caminho ao vivo: o bloco é transcrito com `_palavras`; as lacunas voltam com o tempo da gravação."""

    def test_deslocamento_do_bloco_nas_lacunas_e_avisos(self):
        def falso(audio, duracao, idioma, avisar, lacunas=None):
            lacunas.append({"inicio": 2.0, "fim": 9.5, "fala_s": 6.0, "palavras_recuperadas": 0, "estado": "nao_recuperada"})
            yield pipeline.motores.Palavra(0.5, 0.9, " oi")

        with tempfile.TemporaryDirectory() as d, \
                mock.patch.object(pipeline, "_palavras", falso), \
                mock.patch.object(pipeline, "diarizar", lambda a, num_locutores=None: [Trecho(0, 100, 0)]), \
                mock.patch.object(aovivo.diarizacao_mod, "analise_incremental", lambda *a, **k: None):
            s = aovivo.Sessao("abc123", Path(d) / "a.webm", "pt", None, lambda fn, *a: None)
            try:
                s.pcm.anexar(np.zeros(60 * TAXA, dtype=np.float32))
                s._transcrever_trecho(30 * TAXA, 50 * TAXA)
                self.assertEqual(s._lacunas[0]["inicio"], 32.0)
                self.assertEqual(s._lacunas[0]["fim"], 39.5)
                self.assertEqual(s._palavras[0].start, 30.5)
                s.marcar_fim(60.0)
                self.assertTrue(s._eof.wait(10))
                s._cursor = s.pcm.n   # o resto é silêncio: nada a transcrever na finalização
                meta: dict = {}
                s.concluir(meta=meta)
                self.assertEqual(meta["lacunas"][0]["inicio"], 32.0)
                self.assertEqual(meta["avisos"],
                                 ["Trecho 00:32–00:40 pode estar incompleto: havia fala e o texto não foi recuperado."])
            finally:
                s.liberar()


if __name__ == "__main__":
    unittest.main()
