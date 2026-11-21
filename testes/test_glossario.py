"""Glossário do usuário (`hotwords` do Whisper local). Sem modelo e sem internet: o Whisper e o VAD são falsos.

    .venv\\Scripts\\python -m unittest testes.test_glossario -v

Cobre: normalização e limite, `_opcoes_decodificacao` (vazio = kwargs idênticos aos de antes), caminho clássico
(`transcrever` -> `motores.palavras` -> `_palavras` -> `transcribe`), caminho ao vivo (blocos da `Sessao`), Groq (não recebe)
e as rotas (campo ausente é aceito e ignorado). Nada é gravado em `gravacoes/`: `app.GRAVACOES` aponta para uma pasta temporária.
"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from collections import namedtuple
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("AVEIA_DADOS", __import__("tempfile").mkdtemp(prefix="aveia-dados-teste-"))   # métricas dos testes não vão para dados/
from fastapi.testclient import TestClient  # noqa: E402

import app as aplicacao  # noqa: E402
from transcritor import aovivo, motores, pipeline  # noqa: E402
from transcritor import diarizacao as diarizacao_mod  # noqa: E402
from transcritor.diarizacao import Trecho  # noqa: E402

Pal = namedtuple("Pal", "start end word")
Seg = namedtuple("Seg", "start end words")
TAXA = 16000

# Opções de antes do glossário (cópia literal do que `_opcoes_decodificacao` devolvia): a prova de "vazio não muda nada".
OPCOES_ANTIGAS_PT = dict(
    language="pt", word_timestamps=True, beam_size=5,
    vad_filter=True, vad_parameters=dict(min_silence_duration_ms=500, speech_pad_ms=300),
    condition_on_previous_text=False, hallucination_silence_threshold=2.0,
    no_speech_threshold=0.5, compression_ratio_threshold=2.2,
    initial_prompt=pipeline.DICA_PT,
)


class ModeloEspiao:
    """Whisper falso: uma palavra por chamada e guarda os kwargs recebidos."""

    def __init__(self):
        self.chamadas: list[dict] = []

    def transcribe(self, audio, **opcoes):
        self.chamadas.append(opcoes)
        return iter([Seg(0.5, 0.9, [Pal(0.5, 0.9, " oi")])]), None


class NormalizacaoTest(unittest.TestCase):
    def test_separadores_espacos_vazios_e_duplicados(self):
        n = pipeline.normalizar_glossario("  Aveia ,throughput;\n\n  deploy  \r\n cluster, ,Throughput,, aveia ")
        self.assertEqual(n, "Aveia, throughput, deploy, cluster")

    def test_junta_espacos_internos(self):
        self.assertEqual(pipeline.normalizar_glossario("APIs   RESTful,\tmicro  serviço"), "APIs RESTful, micro serviço")

    def test_vazio_nulo_e_tipo_errado(self):
        for entrada in ("", "   ", " , ;\n", None, 5, ["a"], {"a": 1}):
            self.assertEqual(pipeline.normalizar_glossario(entrada), "", repr(entrada))

    def test_idempotente(self):
        uma = pipeline.normalizar_glossario("b, a;\nb,c")
        self.assertEqual(pipeline.normalizar_glossario(uma), uma)

    def test_remove_caracteres_de_controle(self):
        self.assertEqual(pipeline.normalizar_glossario("a\x00b\x07, c"), "ab, c")

    def test_limite_de_termos(self):
        n = pipeline.normalizar_glossario(",".join(f"t{i}" for i in range(100)))
        termos = n.split(", ")
        self.assertEqual(len(termos), pipeline.GLOSSARIO_MAX_TERMOS)
        self.assertEqual(termos[0], "t0")   # corta no fim: os primeiros termos são os que valem

    def test_limite_de_caracteres(self):
        n = pipeline.normalizar_glossario(",".join(f"termo{i:03d}xyz" for i in range(60)))
        self.assertLessEqual(len(n), pipeline.GLOSSARIO_MAX_CHARS)
        self.assertGreater(len(n), pipeline.GLOSSARIO_MAX_CHARS - 20)   # aproveita o espaço: só deixa de fora o que não cabe
        self.assertTrue(n.startswith("termo000xyz, termo001xyz"))

    def test_termo_comprido_demais_e_descartado(self):
        longo = "x" * (pipeline.GLOSSARIO_TERMO_MAX_CHARS + 1)
        self.assertEqual(pipeline.normalizar_glossario(f"ok, {longo}, tambem"), "ok, tambem")

    def test_limite_de_caracteres_e_bem_menor_que_o_de_tokens(self):
        # faster-whisper corta o hotwords em 223 tokens; com no máximo 300 caracteres só se chega lá se cada token tiver
        # ~1,3 caractere, o que não acontece com nomes e termos curtos.
        self.assertLessEqual(pipeline.GLOSSARIO_MAX_CHARS, 300)


class OpcoesTest(unittest.TestCase):
    def test_vazio_nao_muda_os_kwargs(self):
        for vazio in (None, "", "  ", " ,; \n"):
            self.assertEqual(pipeline._opcoes_decodificacao("pt", vazio), OPCOES_ANTIGAS_PT, repr(vazio))
        self.assertEqual(pipeline._opcoes_decodificacao("pt"), OPCOES_ANTIGAS_PT)
        self.assertNotIn("hotwords", pipeline._opcoes_decodificacao("pt", ""))

    def test_vazio_em_outros_idiomas_tambem_nao_muda(self):
        sem = pipeline._opcoes_decodificacao("en")
        self.assertEqual(pipeline._opcoes_decodificacao("en", ""), sem)
        self.assertIsNone(sem["initial_prompt"])
        self.assertIsNone(pipeline._opcoes_decodificacao(None)["language"])

    def test_com_glossario_so_acrescenta_hotwords(self):
        com = pipeline._opcoes_decodificacao("pt", " deploy ; cluster,deploy")
        self.assertEqual(com.pop("hotwords"), "deploy, cluster")
        self.assertEqual(com, OPCOES_ANTIGAS_PT)   # nada mais muda: beam, VAD, prompt, condition_on_previous_text...
        self.assertIs(com["condition_on_previous_text"], False)


class ClassicoTest(unittest.TestCase):
    """`transcrever` completo (caminho clássico) com o Whisper falso."""

    def setUp(self):
        self.modelo = ModeloEspiao()
        for alvo, nome, novo in [
            (pipeline, "_carregar_modelo", lambda n: self.modelo),
            (pipeline, "_trechos_fala", lambda audio: []),
            (pipeline, "decode_audio", lambda c, sampling_rate=16000: np.zeros(sampling_rate * 5, dtype=np.float32)),
            (pipeline, "preparar_audio", lambda a: a),
            (pipeline, "diarizar", lambda a, num_locutores=None: [Trecho(0, 5, 0)]),
        ]:
            p = mock.patch.object(alvo, nome, novo)
            p.start()
            self.addCleanup(p.stop)

    def test_sem_glossario_kwargs_identicos_aos_de_antes(self):
        pipeline.transcrever("x.wav", motor="local")
        self.assertEqual(self.modelo.chamadas, [OPCOES_ANTIGAS_PT])

    def test_glossario_chega_ao_transcribe(self):
        falas = pipeline.transcrever("x.wav", motor="local", glossario="throughput, cluster\nSLA")
        self.assertEqual(len(self.modelo.chamadas), 1)
        self.assertEqual(self.modelo.chamadas[0]["hotwords"], "throughput, cluster, SLA")
        self.assertEqual(falas[0].texto, "oi")

    def test_glossario_so_com_separadores_equivale_a_vazio(self):
        pipeline.transcrever("x.wav", motor="local", glossario=" , ;\n ")
        self.assertEqual(self.modelo.chamadas, [OPCOES_ANTIGAS_PT])

    def test_retranscricao_de_lacunas_recebe_o_glossario(self):
        # 1ª chamada devolve palavras no começo e no fim; o buraco de 27 s com fala do VAD é retranscrito com as mesmas opções.
        chamadas: list[dict] = []

        def transcribe(audio, **opcoes):
            chamadas.append(opcoes)
            if len(chamadas) == 1:
                return iter([Seg(0.5, 0.9, [Pal(0.5, 0.9, " a")]), Seg(28.0, 28.4, [Pal(28.0, 28.4, " b")])]), None
            return iter([Seg(0.1, 0.5, [Pal(0.1, 0.5, " c")])]), None

        modelo = mock.Mock()
        modelo.transcribe = transcribe
        with mock.patch.object(pipeline, "_carregar_modelo", lambda n: modelo), \
                mock.patch.object(pipeline, "_trechos_fala", lambda audio: [(1.0, 27.5)]), \
                mock.patch.object(pipeline, "decode_audio", lambda c, sampling_rate=16000: np.zeros(sampling_rate * 30, dtype=np.float32)), \
                mock.patch.dict(os.environ, {"AVEIA_LACUNAS": "1"}):
            pipeline.transcrever("x.wav", motor="local", glossario="deploy")
        self.assertGreaterEqual(len(chamadas), 2)   # houve retranscrição
        self.assertTrue(all(c.get("hotwords") == "deploy" for c in chamadas))

    def test_groq_nao_recebe_o_glossario_e_o_fallback_local_recebe(self):
        enviados: list = []

        def groq_falso(audio, duracao, idioma, prompt, avisar, meta, local):
            enviados.append((prompt, idioma))   # a Groq só recebe `prompt` (AVEIA_GROQ_PROMPT), nunca o glossário
            return iter(())

        with mock.patch.object(motores, "groq_configurado", lambda: True), \
                mock.patch.object(motores, "_groq_palavras", groq_falso), \
                mock.patch.dict(os.environ, {"AVEIA_GROQ_PROMPT": ""}):
            list(motores.palavras(np.zeros(TAXA, dtype=np.float32), 1.0, "pt", lambda p, m: None, motor="groq",
                                  meta={}, glossario="deploy"))
        self.assertEqual(enviados, [(None, "pt")])
        self.assertEqual(self.modelo.chamadas, [])   # com a Groq ativa o Whisper local não rodou

        # Groq sem chave -> cai no local, que usa o glossário.
        with mock.patch.object(motores, "groq_configurado", lambda: False):
            list(motores.palavras(np.zeros(TAXA, dtype=np.float32), 1.0, "pt", lambda p, m: None, motor="groq",
                                  meta={}, glossario="deploy"))
        self.assertEqual(self.modelo.chamadas[-1]["hotwords"], "deploy")


class AoVivoTest(unittest.TestCase):
    """Blocos e cauda da `Sessao` chegam ao Whisper com o glossário."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.modelo = ModeloEspiao()
        for alvo, nome, novo in [
            (pipeline, "_carregar_modelo", lambda n: self.modelo),
            (pipeline, "_trechos_fala", lambda audio: []),
            (pipeline, "diarizar", lambda a, num_locutores=None: [Trecho(0, 100, 0)]),
            (diarizacao_mod, "analise_incremental", lambda *a, **k: None),
        ]:
            p = mock.patch.object(alvo, nome, novo)
            p.start()
            self.addCleanup(p.stop)

    def _sessao(self, nome, **kw):
        s = aovivo.Sessao(nome * 32, Path(self.tmp.name) / f"{nome}.webm", "pt", None, lambda fn, *a: None, **kw)
        self.addCleanup(s.liberar)
        s.pcm.anexar(np.zeros(40 * TAXA, dtype=np.float32))
        return s

    def test_bloco_sem_glossario_usa_os_kwargs_de_antes(self):
        s = self._sessao("a")
        self.assertEqual(s.glossario, "")
        s._transcrever_trecho(0, 20 * TAXA)
        self.assertEqual(self.modelo.chamadas, [OPCOES_ANTIGAS_PT])

    def test_cada_bloco_e_a_cauda_recebem_o_glossario(self):
        s = self._sessao("b", glossario=" cluster ; deploy,cluster")
        self.assertEqual(s.glossario, "cluster, deploy")   # normalizado uma vez, ao criar
        s._transcrever_trecho(0, 15 * TAXA)
        s._transcrever_trecho(15 * TAXA, 40 * TAXA)   # a "cauda" usa o mesmo método
        self.assertEqual(len(self.modelo.chamadas), 2)
        self.assertTrue(all(c["hotwords"] == "cluster, deploy" for c in self.modelo.chamadas))

    def test_gerenciador_repassa_o_glossario(self):
        g = aovivo.Gerenciador(lambda fn, *a: None)
        s = g.criar(Path(self.tmp.name) / "g.webm", "pt", None, glossario="SLA, SLA, nodes")
        self.addCleanup(s.liberar)
        self.assertEqual(s.glossario, "SLA, nodes")
        s2 = g.criar(Path(self.tmp.name) / "h.webm", "pt", None)
        self.addCleanup(s2.liberar)
        self.assertEqual(s2.glossario, "")


class RotasTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        for p in (mock.patch.object(aplicacao, "GRAVACOES", Path(self.tmp.name)),
                  mock.patch.object(diarizacao_mod, "analise_incremental", lambda *a, **k: None),
                  mock.patch.dict(os.environ, {"AVEIA_PREPROC": "", "AVEIA_MOTOR": "local"})):
            p.start()
            self.addCleanup(p.stop)
        self.chamadas: list[tuple] = []
        p = mock.patch.object(aplicacao.fila, "submit", lambda fn, *a: self.chamadas.append((fn, a)))   # não executa nada
        p.start()
        self.addCleanup(p.stop)
        self.cliente = TestClient(aplicacao.app, base_url="http://127.0.0.1:8765")

    def _classico(self, **dados):
        r = self.cliente.post("/api/transcrever", files={"audio": ("a.webm", b"x")}, data=dados)
        self.assertEqual(r.status_code, 200, r.text)
        fn, args = self.chamadas[-1]
        self.assertIs(fn, aplicacao._executar)
        return args   # (job_id, caminho, idioma, locutores, motor, canais, glossario)

    def test_transcrever_sem_o_campo_e_aceito(self):
        self.assertEqual(self._classico(idioma="pt")[6], "")

    def test_transcrever_normaliza_o_glossario(self):
        self.assertEqual(self._classico(idioma="pt", glossario=" deploy ;cluster\n deploy")[6], "deploy, cluster")

    def test_aovivo_sem_o_campo_e_com_o_campo(self):
        r = self.cliente.post("/api/aovivo", json={"idioma": "pt", "ext": "webm"})
        self.assertTrue(r.json()["ao_vivo"], r.text)
        s1 = aplicacao.aovivo_sessoes.obter(r.json()["sessao"])
        self.addCleanup(s1.apagar)
        self.assertEqual(s1.glossario, "")
        r = self.cliente.post("/api/aovivo", json={"idioma": "pt", "ext": "webm", "glossario": "SLA, nodes ;SLA"})
        s2 = aplicacao.aovivo_sessoes.obter(r.json()["sessao"])
        self.addCleanup(s2.apagar)
        self.assertEqual(s2.glossario, "SLA, nodes")

    def test_aovivo_glossario_de_tipo_errado_e_ignorado(self):
        r = self.cliente.post("/api/aovivo", json={"idioma": "pt", "ext": "webm", "glossario": ["a", "b"]})
        self.assertTrue(r.json()["ao_vivo"], r.text)
        s = aplicacao.aovivo_sessoes.obter(r.json()["sessao"])
        self.addCleanup(s.apagar)
        self.assertEqual(s.glossario, "")


if __name__ == "__main__":
    unittest.main()
