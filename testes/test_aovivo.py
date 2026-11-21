"""Testes offline da transcrição durante a gravação (transcritor/aovivo.py + rotas /api/aovivo).

    .venv\\Scripts\\python -m unittest testes.test_aovivo -v

Sem modelo e sem internet: `pipeline._palavras` e `pipeline.diarizar` são trocados por falsos. O decodificador (PyAV), o
VAD (Silero, empacotado com o faster-whisper) e as rotas (TestClient) são reais. O áudio é `testes/audio-referencia/audio.webm`
fatiado em pedaços como os do MediaRecorder. Nada é gravado em `gravacoes/`: `app.GRAVACOES` aponta para uma pasta temporária.
"""
import json
import os
import random
import sys
import tempfile
import threading
import time
import unittest
from collections import namedtuple
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("AVEIA_DADOS", __import__("tempfile").mkdtemp(prefix="aveia-dados-teste-"))   # métricas dos testes não vão para dados/
from fastapi.testclient import TestClient  # noqa: E402

import app as aplicacao  # noqa: E402
from transcritor import aovivo, pipeline  # noqa: E402
from transcritor import diarizacao as diarizacao_mod  # noqa: E402
from transcritor.diarizacao import Trecho  # noqa: E402

AUDIO = Path(__file__).resolve().parent / "audio-referencia" / "audio.webm"
_FABRICA_REAL = diarizacao_mod.analise_incremental   # guardada antes de o Base trocá-la por um falso
P = namedtuple("P", "start end word")


def esperar(cond, limite=30.0, passo=0.05):
    fim = time.time() + limite
    while time.time() < fim:
        v = cond()
        if v:
            return v
        time.sleep(passo)
    return cond()


def palavras_falsas(chamadas):
    """`_palavras` falso: uma palavra a cada 2 s do bloco (com tempo relativo ao bloco)."""
    def falso(audio, duracao, idioma, avisar, lacunas=None):
        chamadas.append(duracao)
        assert audio.dtype == np.float32 and audio.ndim == 1
        t = 0.5
        while t < duracao - 0.3:
            yield P(t, t + 0.4, " palavra")
            t += 2.0
    return falso


def diarizar_falso(audio, num_locutores=None):
    return [Trecho(0.0, len(audio) / 16000, 0)]


@unittest.skipUnless(AUDIO.exists(), "falta testes/audio-referencia/audio.webm (áudio de referência não vai no repositório público)")
class Base(unittest.TestCase):
    def setUp(self):
        self.dados = AUDIO.read_bytes()
        self.tmp = tempfile.TemporaryDirectory()
        self.pasta = Path(self.tmp.name)
        self.chamadas: list[float] = []
        patches = [
            mock.patch.object(aplicacao, "GRAVACOES", self.pasta),
            mock.patch.object(pipeline, "_palavras", palavras_falsas(self.chamadas)),
            mock.patch.object(pipeline, "diarizar", diarizar_falso),
            mock.patch.object(diarizacao_mod, "analise_incremental", lambda *a, **k: None),   # padrão: sem incremental (offline)
            mock.patch.dict(os.environ, {"AVEIA_PREPROC": "", "AVEIA_MOTOR": "local"}),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        aplicacao.aovivo_sessoes.sessoes.clear()
        self.cli = TestClient(aplicacao.app, base_url="http://127.0.0.1:8765")
        self.addCleanup(self.limpar)

    def limpar(self):
        for s in list(aplicacao.aovivo_sessoes.sessoes.values()):
            s.liberar()
        aplicacao.aovivo_sessoes.sessoes.clear()
        self.cli.close()
        try:
            self.tmp.cleanup()
        except OSError:
            pass

    # ---- utilitários
    def fatias(self, n=30):
        tam = len(self.dados) // n + 1
        return [self.dados[i:i + tam] for i in range(0, len(self.dados), tam)]

    def abrir(self, **extra):
        r = self.cli.post("/api/aovivo", json={"idioma": "pt", "locutores": 0, "motor": "local", "ext": "webm", **extra})
        self.assertEqual(r.status_code, 200)
        return r.json()

    def enviar(self, sessao, seq, dados):
        return self.cli.post(f"/api/aovivo/{sessao}/pedaco", params={"seq": seq}, content=dados)

    def job_pronto(self, job, limite=60.0):
        def estado():
            r = self.cli.get(f"/api/jobs/{job}").json()
            return r if r["status"] != "processando" else None
        r = esperar(estado, limite)
        self.assertIsNotNone(r, "job não terminou")
        return r


class FluxoEPcmTest(unittest.TestCase):
    def test_fluxo_devolve_o_que_houver_e_bloqueia_ate_o_fim(self):
        f = aovivo._Fluxo()
        f.anexar(b"abc")
        self.assertEqual(f.read(10), b"abc")
        saida = []
        t = threading.Thread(target=lambda: saida.append(f.read(10)))
        t.start()
        time.sleep(0.1)
        self.assertTrue(t.is_alive())      # bloqueado: não há bytes
        f.anexar(b"xyz")
        t.join(2)
        self.assertEqual(saida, [b"xyz"])
        f.encerrar()
        self.assertEqual(f.read(10), b"")  # fim

    def test_pcm_atravessa_segmentos(self):
        pcm = aovivo._Pcm()
        a = np.arange(aovivo._SEGMENTO * 2 + 1000, dtype=np.float32)
        pcm.anexar(a[:5000])
        pcm.anexar(a[5000:])
        self.assertEqual(pcm.n, len(a))
        ini = aovivo._SEGMENTO - 10
        np.testing.assert_array_equal(pcm.fatia(ini, ini + 50), a[ini:ini + 50])
        np.testing.assert_array_equal(pcm.fatia(0, pcm.n), a)

    def _duplo(self, esq, dir):
        d = aovivo._PcmDuplo()
        inter = np.empty(2 * len(esq), dtype=np.float32)
        inter[0::2], inter[1::2] = esq, dir
        d.anexar(inter)
        return d

    def test_mistura_dupla_normal_e_identica_a_soma_simples(self):
        # Pico da soma dentro de PICO_MIX: saída bit a bit igual à de antes (soma, com o corte que então era inócuo).
        rnd = np.random.default_rng(3)
        esq = (rnd.uniform(-0.45, 0.45, 8000)).astype(np.float32)
        dir = (rnd.uniform(-0.45, 0.45, 8000)).astype(np.float32)
        antes = np.clip(esq + dir, -aovivo.canais_mod.PICO_MIX, aovivo.canais_mod.PICO_MIX)
        np.testing.assert_array_equal(self._duplo(esq, dir).fatia(0, 8000), antes)
        np.testing.assert_array_equal(self._duplo(esq, dir).fatia(100, 700), antes[100:700])

    def test_mistura_dupla_alta_reduz_ganho_sem_saturar(self):
        pico = aovivo.canais_mod.PICO_MIX
        t = np.arange(8000, dtype=np.float32) / 16000
        esq = (0.9 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
        dir = (0.8 * np.sin(2 * np.pi * 330 * t)).astype(np.float32)
        soma = esq + dir
        assert float(np.abs(soma).max()) > pico   # premissa: a soma estouraria
        mix = self._duplo(esq, dir).fatia(0, 8000)
        self.assertLessEqual(float(np.abs(mix).max()), pico + 1e-6)
        # forma de onda proporcional à soma (sem recorte): razão constante entre mix e soma
        ok = np.abs(soma) > 0.05
        razao = mix[ok] / soma[ok]
        self.assertLess(float(razao.max() - razao.min()), 1e-5)
        # sem topo achatado: o corte duro antigo deixava sequências de amostras coladas em +-PICO_MIX
        colado = (np.abs(mix) >= pico - 1e-6).astype(int)
        maior_seq = max((len(s) for s in "".join(map(str, colado)).split("0")), default=0)
        self.assertLessEqual(maior_seq, 2)
        antigo = np.clip(soma, -pico, pico)
        self.assertGreater(int(np.sum(np.abs(antigo) >= pico - 1e-6)), 10)   # premissa: o corte antigo achatava
        # igual ao que o caminho clássico faz sobre o mesmo trecho
        np.testing.assert_array_equal(mix, aovivo.canais_mod.misturar(esq, dir))

    def test_escolher_pausa(self):
        t = aovivo.TAXA
        minimo = 15 * t
        # fala contínua, depois pausa entre 18 s e 20 s: corta no meio da pausa
        fala = [(0, 18 * t), (20 * t, 28 * t)]
        self.assertEqual(aovivo.escolher_pausa(fala, 29 * t, minimo), 19 * t)
        # pausa antes do mínimo é ignorada
        self.assertIsNone(aovivo.escolher_pausa([(0, 5 * t), (7 * t, 29 * t)], 29 * t, minimo))
        # silêncio final da janela conta como pausa
        self.assertEqual(aovivo.escolher_pausa([(0, 16 * t)], 17 * t, minimo), int(16.5 * t))
        # pausa que cruza o mínimo: corta em max(meio, mínimo)
        self.assertEqual(aovivo.escolher_pausa([(0, 10 * t), (20 * t, 25 * t)], 26 * t, minimo), 15 * t)
        self.assertIsNone(aovivo.escolher_pausa([], 20 * t, minimo))


class PlanejamentoTest(unittest.TestCase):
    """Regra dos blocos (15–30 s) com o VAD trocado por um roteiro, direto na sessão."""

    def sessao(self, segundos, fala):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        s = aovivo.Sessao("x" * 32, Path(self.tmp.name) / "a.webm", "pt", None, lambda fn, *a: None)
        self.addCleanup(s.liberar)
        s.liberada = True          # o decodificador desiste; os testes alimentam `s.pcm` à mão
        s._fluxo.cancelar()
        s.pcm.anexar(np.zeros(int(segundos * aovivo.TAXA), dtype=np.float32))
        p = mock.patch.object(aovivo, "_fala_vad", lambda audio: [(int(a * aovivo.TAXA), int(b * aovivo.TAXA)) for a, b in fala if a * aovivo.TAXA < len(audio)])
        p.start()
        self.addCleanup(p.stop)
        return s

    def test_menos_de_15s_espera(self):
        s = self.sessao(14.0, [(0, 13.9)])
        self.assertIsNone(s._planejar(final=False))

    def test_corta_na_primeira_pausa_depois_de_15s(self):
        s = self.sessao(25.0, [(0, 8), (8.8, 17), (18.2, 24.9)])
        ini, fim = s._planejar(final=False)
        self.assertEqual(ini, 0)
        self.assertAlmostEqual(fim / aovivo.TAXA, 17.6, places=2)

    def test_sem_pausa_ate_30s_corta_duro_entre_28_e_30(self):
        s = self.sessao(40.0, [(0, 40)])
        ini, fim = s._planejar(final=False)
        self.assertEqual(ini, 0)
        self.assertGreaterEqual(fim / aovivo.TAXA, 28.0)
        self.assertLessEqual(fim / aovivo.TAXA, 30.0)

    def test_sem_pausa_e_menos_de_30s_espera(self):
        s = self.sessao(22.0, [(0, 22)])
        self.assertIsNone(s._planejar(final=False))

    def test_bloco_sem_fala_e_pulado(self):
        s = self.sessao(20.0, [])
        self.assertIsNone(s._planejar(final=False))
        self.assertAlmostEqual(s._cursor / aovivo.TAXA, 19.0, places=2)   # pulou, mantendo 1 s de segurança

    def test_finalizacao_pega_o_resto_inteiro(self):
        s = self.sessao(9.0, [(0, 8.5)])
        self.assertEqual(s._planejar(final=True), (0, 9 * aovivo.TAXA))
        s2 = self.sessao(3.0, [])
        self.assertIsNone(s2._planejar(final=True))
        self.assertEqual(s2._cursor, 3 * aovivo.TAXA)


class RotasTest(Base):
    def test_recusas(self):
        self.assertFalse(self.abrir(motor="groq")["ao_vivo"])
        with mock.patch.dict(os.environ, {"AVEIA_PREPROC": "hpf,norm"}):
            r = self.abrir()
            self.assertFalse(r["ao_vivo"])
            self.assertIn("AVEIA_PREPROC", r["motivo"])
        self.assertTrue(self.abrir()["ao_vivo"])
        self.assertEqual(self.cli.get("/api/aovivo/inexistente").status_code, 404)
        self.assertEqual(self.enviar("inexistente", 0, b"x").status_code, 404)
        self.assertEqual(self.cli.post("/api/aovivo/inexistente/finalizar", json={"bytes": 1, "pedacos": 1}).status_code, 404)

    def test_chunks_em_ordem_ao_vivo_e_final(self):
        sid = self.abrir()["sessao"]
        s = aplicacao.aovivo_sessoes.obter(sid)
        fatias = self.fatias(30)
        metade = int(len(fatias) * 0.6)
        for i, f in enumerate(fatias[:metade]):
            self.assertEqual(self.enviar(sid, i, f).status_code, 200)
        # decodificação incremental: o áudio da primeira parte chega sem ter terminado o envio
        self.assertTrue(esperar(lambda: self.cli.get(f"/api/aovivo/{sid}").json()["recebido_s"] > 20), "decodificação não é incremental")
        # blocos ao vivo: com mais de 15 s decodificados já houve ao menos um bloco transcrito antes de finalizar
        self.assertTrue(esperar(lambda: s.blocos), "nenhum bloco ao vivo")
        st = self.cli.get(f"/api/aovivo/{sid}").json()
        self.assertEqual(st["status"], "ativa")
        self.assertGreater(st["total"], 0)
        self.assertGreater(st["transcrito_s"], 14)
        total_meio = st["total"]
        for i, f in enumerate(fatias[metade:], start=metade):
            self.enviar(sid, i, f)
        # o texto provisório cresce
        self.assertTrue(esperar(lambda: self.cli.get(f"/api/aovivo/{sid}").json()["recebido_s"] > 55))
        self.assertEqual(self.pasta.joinpath(Path(s.arquivo).name).read_bytes(), self.dados)   # arquivo = original
        # `desde`: só o que vem depois
        tot = self.cli.get(f"/api/aovivo/{sid}").json()["total"]
        self.assertEqual(len(self.cli.get(f"/api/aovivo/{sid}", params={"desde": max(tot - 1, 0)}).json()["falas"]), min(tot, 1))
        self.assertTrue(all(f["locutor"] == pipeline.LOCUTOR_PENDENTE for f in self.cli.get(f"/api/aovivo/{sid}").json()["falas"]))

        r = self.cli.post(f"/api/aovivo/{sid}/finalizar", json={"bytes": len(self.dados), "pedacos": len(fatias), "duracao": 57.6})
        self.assertEqual(r.status_code, 200)
        job = r.json()["job"]
        self.assertIn("parciais", self.cli.get(f"/api/jobs/{job}").json())
        pronto = self.job_pronto(job)
        self.assertEqual(pronto["status"], "pronto", pronto)
        self.assertTrue(pronto["falas"])
        self.assertTrue(all(f["locutor"] == "Locutor 1" for f in pronto["falas"]))
        self.assertEqual(pronto["parciais"], [])
        self.assertEqual(pronto["motor"], "local")
        json_ao_lado = Path(pronto["arquivo"]).with_suffix(".json")
        self.assertTrue(json_ao_lado.exists())
        self.assertEqual(json.loads(json_ao_lado.read_text(encoding="utf-8")), pronto["falas"])
        # blocos: contíguos, cobrem tudo, e os fechados ao vivo respeitam 15–30 s
        self.assertGreaterEqual(len(s.blocos), 2)
        self.assertAlmostEqual(s.blocos[0][0], 0.0, places=3)
        for (_, fim), (ini, _) in zip(s.blocos, s.blocos[1:]):
            self.assertAlmostEqual(fim, ini, places=3)
        self.assertAlmostEqual(s.blocos[-1][1], s.pcm.n / 16000 if s.pcm.n else s.blocos[-1][1], places=1)
        for ini, fim in s.blocos[:-1]:
            self.assertGreaterEqual(fim - ini, 15.0 - 1e-3, s.blocos)
            self.assertLessEqual(fim - ini, 30.0 + 1e-3, s.blocos)
        self.assertLessEqual(s.blocos[-1][1] - s.blocos[-1][0], 30.0 + 1e-3)
        self.assertGreater(s.blocos[-1][1], 55.0)
        self.assertGreaterEqual(total_meio, 1)
        # duração decodificada ≈ a do clipe
        self.assertAlmostEqual(sum(f - i for i, f in s.blocos), 57.7, delta=3.0)
        self.assertNotIn(sid, aplicacao.aovivo_sessoes.sessoes)    # sessão encerrada depois do job

    def test_fora_de_ordem_e_duplicados(self):
        sid = self.abrir()["sessao"]
        s = aplicacao.aovivo_sessoes.obter(sid)
        fatias = self.fatias(24)
        ordem = list(range(len(fatias)))
        rnd = random.Random(7)
        for i in range(0, len(ordem) - 3, 4):      # embaralha em janelas de 4
            jan = ordem[i:i + 4]
            rnd.shuffle(jan)
            ordem[i:i + 4] = jan
        for i in ordem:
            self.assertEqual(self.enviar(sid, i, fatias[i]).status_code, 200)
            if rnd.random() < 0.4:
                self.assertEqual(self.enviar(sid, i, fatias[i]).status_code, 200)   # duplicado: idempotente
        self.assertEqual(self.enviar(sid, 3, b"lixo diferente").status_code, 200)   # seq já aceito: ignorado
        self.assertEqual((s.bytes_recebidos, s.pedacos), (len(self.dados), len(fatias)))
        self.assertTrue(esperar(lambda: s.pcm.n > 57 * 16000 or s._eof.is_set()))
        r = self.cli.post(f"/api/aovivo/{sid}/finalizar", json={"bytes": len(self.dados), "pedacos": len(fatias), "duracao": 57.6})
        self.assertEqual(r.status_code, 200)
        arquivo = Path(self.cli.get(f"/api/jobs/{r.json()['job']}").json()["arquivo"])
        self.assertEqual(arquivo.read_bytes(), self.dados)
        self.assertEqual(self.job_pronto(r.json()["job"])["status"], "pronto")

    def test_pedaco_com_lacuna_nao_confere(self):
        sid = self.abrir()["sessao"]
        fatias = self.fatias(10)
        for i in (0, 1, 3):                         # falta o 2
            self.enviar(sid, i, fatias[i])
        s = aplicacao.aovivo_sessoes.obter(sid)
        self.assertEqual(s.pedacos, 2)
        r = self.cli.post(f"/api/aovivo/{sid}/finalizar", json={"bytes": sum(len(fatias[i]) for i in (0, 1, 3)), "pedacos": 3})
        self.assertEqual(r.status_code, 409)

    def test_finalizar_com_contagem_errada_da_409(self):
        sid = self.abrir()["sessao"]
        fatias = self.fatias(10)
        for i, f in enumerate(fatias):
            self.enviar(sid, i, f)
        for corpo in ({"bytes": len(self.dados) - 1, "pedacos": len(fatias)}, {"bytes": len(self.dados), "pedacos": len(fatias) + 1}):
            self.assertEqual(self.cli.post(f"/api/aovivo/{sid}/finalizar", json=corpo).status_code, 409)
        # a sessão continua utilizável: com a contagem certa funciona
        r = self.cli.post(f"/api/aovivo/{sid}/finalizar", json={"bytes": len(self.dados), "pedacos": len(fatias), "duracao": 57.6})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(self.job_pronto(r.json()["job"])["status"], "pronto")

    def test_finalizar_repetido_devolve_o_mesmo_job_e_so_um_trabalho(self):
        sid = self.abrir()["sessao"]
        fatias = self.fatias(10)
        for i, f in enumerate(fatias):
            self.enviar(sid, i, f)
        corpo = {"bytes": len(self.dados), "pedacos": len(fatias), "duracao": 57.6}
        enviados = []
        original = aplicacao.fila.submit
        with mock.patch.object(aplicacao.fila, "submit", lambda fn, *a, **k: (enviados.append(fn), original(fn, *a, **k))[1]):
            r1 = self.cli.post(f"/api/aovivo/{sid}/finalizar", json=corpo)
            r2 = self.cli.post(f"/api/aovivo/{sid}/finalizar", json=corpo)   # a resposta da primeira "se perdeu"
            self.assertEqual((r1.status_code, r2.status_code), (200, 200))
            self.assertEqual(r1.json()["job"], r2.json()["job"])
            self.assertEqual(self.job_pronto(r1.json()["job"])["status"], "pronto")
            # sessão já concluída e removida: a repetição ainda recupera o mesmo job
            self.assertNotIn(sid, aplicacao.aovivo_sessoes.sessoes)
            r3 = self.cli.post(f"/api/aovivo/{sid}/finalizar", json=corpo)
            self.assertEqual(r3.json()["job"], r1.json()["job"])
            # contagem diferente da registrada continua dando 409
            self.assertEqual(self.cli.post(f"/api/aovivo/{sid}/finalizar", json={**corpo, "pedacos": len(fatias) + 1}).status_code, 409)
        self.assertEqual(sum(1 for fn in enviados if fn is aplicacao._executar_aovivo), 1)

    def test_decodificador_falha_degrada_e_finaliza_pelo_classico(self):
        sid = self.abrir()["sessao"]
        s = aplicacao.aovivo_sessoes.obter(sid)
        lixo = random.Random(1).randbytes(1_200_000)   # mais que o limite de sondagem do FFmpeg (1 MB): o formato não é reconhecido
        total = 0
        for i in range(300):
            self.enviar(sid, i, lixo[i * 4000:(i + 1) * 4000])
            total += 4000
        self.assertTrue(esperar(lambda: self.cli.get(f"/api/aovivo/{sid}").json()["status"] == "degradada"), "não degradou")
        self.assertIn("decodificação", self.cli.get(f"/api/aovivo/{sid}").json()["motivo"])
        # chegam mais pedaços depois de degradada: continuam sendo gravados (o arquivo é a rede de segurança)
        self.enviar(sid, 300, b"mais")
        total += 4
        chamadas = []

        def classico(caminho, **kw):
            chamadas.append((caminho, kw))
            kw["progresso"](0.5, "meio")
            return [pipeline.Fala(0.0, 1.0, "Locutor 1", "texto clássico")]
        with mock.patch.object(aplicacao, "transcrever", classico):
            r = self.cli.post(f"/api/aovivo/{sid}/finalizar", json={"bytes": total, "pedacos": 301, "duracao": 5})
            self.assertEqual(r.status_code, 200)
            pronto = self.job_pronto(r.json()["job"])
        self.assertEqual(pronto["status"], "pronto", pronto)
        self.assertEqual(pronto["falas"][0]["texto"], "texto clássico")
        self.assertEqual(len(chamadas), 1)
        self.assertEqual(Path(chamadas[0][0]).read_bytes(), lixo + b"mais")
        self.assertEqual(chamadas[0][1]["motor"], "local")
        self.assertTrue(Path(pronto["arquivo"]).with_suffix(".json").exists())

    def test_lixo_curto_so_falha_ao_finalizar_e_cai_no_classico(self):
        sid = self.abrir()["sessao"]
        self.enviar(sid, 0, b"isto nao e audio")
        with mock.patch.object(aplicacao, "transcrever", lambda caminho, **kw: [pipeline.Fala(0.0, 1.0, "Locutor 1", "clássico")]):
            r = self.cli.post(f"/api/aovivo/{sid}/finalizar", json={"bytes": 16, "pedacos": 1, "duracao": 2})
            pronto = self.job_pronto(r.json()["job"])
        self.assertEqual(pronto["status"], "pronto")
        self.assertEqual(pronto["falas"][0]["texto"], "clássico")

    def test_duracao_decodificada_menor_que_a_gravada_cai_no_classico(self):
        sid = self.abrir()["sessao"]
        fatias = self.fatias(10)
        for i, f in enumerate(fatias):
            self.enviar(sid, i, f)
        chamadas = []

        def classico(caminho, **kw):
            chamadas.append(caminho)
            return [pipeline.Fala(0.0, 1.0, "Locutor 1", "clássico")]
        with mock.patch.object(aplicacao, "transcrever", classico):
            r = self.cli.post(f"/api/aovivo/{sid}/finalizar", json={"bytes": len(self.dados), "pedacos": len(fatias), "duracao": 300})
            pronto = self.job_pronto(r.json()["job"])
        self.assertEqual(pronto["status"], "pronto")
        self.assertEqual(len(chamadas), 1)

    def test_delete_apaga_arquivo_parcial(self):
        sid = self.abrir()["sessao"]
        s = aplicacao.aovivo_sessoes.obter(sid)
        self.enviar(sid, 0, self.dados[:5000])
        arquivo = Path(s.arquivo)
        self.assertTrue(arquivo.exists())
        self.assertEqual(self.cli.delete(f"/api/aovivo/{sid}").status_code, 200)
        self.assertFalse(arquivo.exists())
        self.assertEqual(self.cli.get(f"/api/aovivo/{sid}").status_code, 404)
        self.assertEqual(self.cli.delete(f"/api/aovivo/{sid}").status_code, 200)   # idempotente
        self.assertEqual(list(self.pasta.glob("*.webm")), [])

    def test_sessoes_demais_sao_recusadas(self):
        for _ in range(aovivo.MAX_SESSOES):
            self.assertTrue(self.abrir()["ao_vivo"])
        r = self.abrir()
        self.assertFalse(r["ao_vivo"])


class OciosaTest(Base):
    def test_sessao_ociosa_libera_memoria_e_mantem_arquivo(self):
        relogio = [1000.0]
        g = aovivo.Gerenciador(lambda fn, *a: None, relogio=lambda: relogio[0])
        s = g.criar(self.pasta / "x.webm", "pt", None)
        s.receber(0, self.dados[:20000])
        self.assertTrue(esperar(lambda: s.pcm.n > 0))
        relogio[0] += aovivo.OCIOSA_S - 1
        self.assertEqual(g.varrer(), 0)
        relogio[0] += 2
        self.assertEqual(g.varrer(), 1)
        self.assertTrue(s.degradada)
        self.assertEqual(s.motivo, "ociosa")
        self.assertTrue(s.pcm._livre)
        self.assertEqual((self.pasta / "x.webm").read_bytes(), self.dados[:20000])   # o arquivo fica
        s.liberar()


class ClassicoParciaisTest(Base):
    def test_executar_expoe_parciais_e_limpa_ao_terminar(self):
        vistos = []

        def falso(caminho, parcial=None, **kw):
            parcial([pipeline.Fala(0.0, 1.0, pipeline.LOCUTOR_PENDENTE, "olá")])
            vistos.append([dict(p) for p in aplicacao.jobs["j1"]["parciais"]])
            return [pipeline.Fala(0.0, 1.0, "Locutor 1", "olá")]
        aplicacao.jobs["j1"] = {"status": "processando", "progresso": 0.0, "mensagem": "", "arquivo": str(self.pasta / "a.webm")}
        with mock.patch.object(aplicacao, "transcrever", falso):
            aplicacao._executar("j1", str(self.pasta / "a.webm"), "pt", None, "local")
        self.assertEqual(vistos[0][0]["texto"], "olá")
        self.assertEqual(vistos[0][0]["locutor"], pipeline.LOCUTOR_PENDENTE)
        j = aplicacao.jobs.pop("j1")
        self.assertEqual(j["status"], "pronto")
        self.assertEqual(j["parciais"], [])
        self.assertEqual(j["falas"][0]["locutor"], "Locutor 1")


class IncrementalFalso:
    """Registra o que recebe. `falha_em`: nº (1, 2...) da chamada de `alimentar` que levanta; `resultado`: o que `concluir` devolve."""

    def __init__(self, falha_em=None, resultado=None, concluir_levanta=False):
        self.chamadas: list[np.ndarray] = []
        self.concluiu: list = []
        self.falha_em, self.resultado, self.concluir_levanta = falha_em, resultado, concluir_levanta

    def alimentar(self, audio):
        self.chamadas.append(np.array(audio, copy=True))
        if self.falha_em == len(self.chamadas):
            raise RuntimeError("falha simulada no alimentar")
        return 1

    def concluir(self, num_locutores=None):
        self.concluiu.append(num_locutores)
        if self.concluir_levanta:
            raise RuntimeError("falha simulada no concluir")
        if self.resultado is not None:
            return self.resultado
        d = sum(len(a) for a in self.chamadas) / 16000
        return [Trecho(0.0, d / 2, 5), Trecho(d / 2, d, 7)]   # dois locutores: distingue do `diarizar_falso` (um só)


class IncrementalTest(Base):
    def rodar(self, inc, esperar_vivo=True, **extra):
        """Grava o clipe inteiro numa sessão com `inc` como fábrica, finaliza e devolve (sessão, áudio decodificado, job pronto)."""
        self.diar = []

        def espiao(audio, num_locutores=None):
            self.diar.append(len(audio))
            return diarizar_falso(audio, num_locutores)

        capturado = []
        original = aovivo.Sessao.concluir

        def concluir_espiao(self_, *a, **k):   # guarda o áudio decodificado inteiro (a sessão o libera depois do job)
            try:
                return original(self_, *a, **k)
            finally:
                capturado.append(self_.pcm.fatia(0, self_.pcm.n))
        patches = [mock.patch.object(diarizacao_mod, "analise_incremental", lambda *a, **k: inc),
                   mock.patch.object(pipeline, "diarizar", espiao),
                   mock.patch.object(aovivo.Sessao, "concluir", concluir_espiao)]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        sid = self.abrir(**extra)["sessao"]
        s = aplicacao.aovivo_sessoes.obter(sid)
        fatias = self.fatias(30)
        for i, f in enumerate(fatias):
            self.enviar(sid, i, f)
        self.assertTrue(esperar(lambda: s.pcm.n > 57 * 16000, 60), "decodificador não chegou ao fim")
        self.assertTrue(esperar(lambda: s.blocos), "nenhum bloco ao vivo")
        if esperar_vivo and inc is not None:
            self.assertTrue(esperar(lambda: inc.chamadas), "nada foi diarizado durante a gravação")
        r = self.cli.post(f"/api/aovivo/{sid}/finalizar", json={"bytes": len(self.dados), "pedacos": len(fatias), "duracao": 57.6})
        self.assertEqual(r.status_code, 200)
        pronto = self.job_pronto(r.json()["job"])
        self.assertEqual(pronto["status"], "pronto", pronto)
        self.assertEqual(len(capturado), 1)
        return s, capturado[0], pronto

    def test_audio_entregue_e_contiguo_e_completo_com_trecho_sem_fala_pulado(self):
        inc = IncrementalFalso()
        real = aovivo._fala_vad
        n = [0]

        def vad(audio):   # a 1ª avaliação não vê fala: o planejador pula o trecho (o cursor salta), e o áudio pulado tem de ser entregue
            n[0] += 1
            return [] if n[0] == 1 else real(audio)
        with mock.patch.object(aovivo, "_fala_vad", vad):
            s, ref, pronto = self.rodar(inc)
        self.assertGreater(s.blocos[0][0], 5.0, "o trecho sem fala não foi pulado; o teste não exercitou o caso")
        self.assertGreaterEqual(len(inc.chamadas), 2)
        np.testing.assert_array_equal(np.concatenate(inc.chamadas), ref)

    def test_audio_entregue_e_contiguo_e_completo(self):
        inc = IncrementalFalso()
        s, ref, pronto = self.rodar(inc)
        self.assertGreater(len(ref), 50 * 16000)
        np.testing.assert_array_equal(np.concatenate(inc.chamadas), ref)
        self.assertEqual(s._diar_cursor, len(ref))

    def test_concluir_do_incremental_e_usado_e_o_lote_nao(self):
        inc = IncrementalFalso()
        s, ref, pronto = self.rodar(inc)
        self.assertEqual(inc.concluiu, [None])      # locutores=0 no pedido -> None
        self.assertEqual(pronto["lacunas"], [])     # o job ao vivo expõe as lacunas (a interface as marca na transcrição)
        self.assertEqual(self.diar, [])
        self.assertEqual({f["locutor"] for f in pronto["falas"]}, {"Locutor 1", "Locutor 2"})
        self.assertFalse(s.degradada)
        self.assertIsNone(s._inc)                   # solto depois de usado

    def test_alimentar_que_falha_no_meio_cai_no_lote_sem_perder_texto(self):
        inc = IncrementalFalso(falha_em=2)
        s, ref, pronto = self.rodar(inc, esperar_vivo=False)
        self.assertEqual(inc.concluiu, [])
        self.assertEqual(len(self.diar), 1)
        self.assertEqual(self.diar[0], len(ref))     # lote no áudio inteiro
        self.assertFalse(s.degradada)
        self.assertTrue(all(f["locutor"] == "Locutor 1" for f in pronto["falas"]))
        self.assertGreaterEqual(len(s.blocos), 2)
        self.assertEqual(len(pronto["falas"]) > 0, True)
        self.assertGreater(len(s._palavras), 10)

    def test_concluir_que_devolve_vazio_cai_no_lote(self):
        inc = IncrementalFalso(resultado=[])
        s, ref, pronto = self.rodar(inc)
        self.assertEqual(len(inc.concluiu), 1)
        self.assertEqual(self.diar, [len(ref)])
        self.assertFalse(s.degradada)
        self.assertTrue(all(f["locutor"] == "Locutor 1" for f in pronto["falas"]))

    def test_concluir_que_levanta_cai_no_lote(self):
        inc = IncrementalFalso(concluir_levanta=True)
        s, ref, pronto = self.rodar(inc)
        self.assertEqual(len(inc.concluiu), 1)
        self.assertEqual(self.diar, [len(ref)])
        self.assertFalse(s.degradada)
        self.assertEqual(pronto["avisos"] if "avisos" in pronto else [], [])   # o lote funcionou: nenhum aviso de falha
        self.assertTrue(pronto["falas"])

    def test_fabrica_sem_incremental_e_o_comportamento_antigo(self):
        s, ref, pronto = self.rodar(None)
        self.assertEqual(self.diar, [len(ref)])
        self.assertTrue(s._inc_desligado)
        self.assertEqual(s._diar_cursor, 0)
        self.assertTrue(all(f["locutor"] == "Locutor 1" for f in pronto["falas"]))

    def test_dois_canais_nao_cria_incremental(self):
        chamou = []
        with mock.patch.object(diarizacao_mod, "analise_incremental", lambda *a, **k: chamou.append(1)):
            s = aovivo.Sessao("y" * 32, self.pasta / "c.webm", "pt", None, lambda fn, *a: None, canais="mic+sistema")
            self.addCleanup(s.liberar)
            s.estereo = True
            s.pcm = aovivo._PcmDuplo()
            s.pcm.anexar(np.zeros(2 * 16000, dtype=np.float32))
            s._tarefa_bloco((0, 16000))
            self.assertEqual(chamou, [])
            self.assertEqual(s._diar_cursor, 0)
            self.assertTrue(s._palavras)             # o texto do bloco saiu normalmente

    def test_atraso_grande_adia_a_diarizacao_para_a_finalizacao(self):
        inc = IncrementalFalso()
        with mock.patch.object(aovivo, "ATRASO_DIAR_MAX_S", -1.0):    # todo bloco parece atrasado
            s, ref, pronto = self.rodar(inc, esperar_vivo=False)
        self.assertEqual(len(inc.chamadas), 1)                          # tudo de uma vez, na finalização
        np.testing.assert_array_equal(inc.chamadas[0], ref)
        self.assertEqual(inc.concluiu, [None])
        self.assertEqual(self.diar, [])
        self.assertEqual({f["locutor"] for f in pronto["falas"]}, {"Locutor 1", "Locutor 2"})

    def test_degradar_e_liberar_soltam_o_incremental(self):
        for acao in ("degradar", "liberar"):
            s = aovivo.Sessao("z" * 32, self.pasta / f"{acao}.webm", "pt", None, lambda fn, *a: None)
            self.addCleanup(s.liberar)
            s._inc = IncrementalFalso()
            getattr(s, acao)(*(["teste"] if acao == "degradar" else []))
            self.assertIsNone(s._inc)
            self.assertTrue(s._inc_desligado)


class FabricaTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.pasta = Path(self.tmp.name)
        self.baixou = []
        for p in (mock.patch.object(diarizacao_mod, "MODELS_DIR", self.pasta),
                  mock.patch.object(diarizacao_mod, "_baixar", lambda *a: self.baixou.append(a)),
                  mock.patch.dict(os.environ, {"AVEIA_DIARIZACAO": "", "AVEIA_DIAR_AOVIVO": ""})):
            p.start()
            self.addCleanup(p.stop)

    def criar_modelos(self):
        seg = diarizacao_mod._caminho_segmentacao()
        emb = diarizacao_mod._caminho_embedding(diarizacao_mod.EMBEDDING_NOVO)
        seg.parent.mkdir(parents=True)
        seg.write_bytes(b"x")
        emb.write_bytes(b"x")
        return seg, emb

    def test_motor_antigo_devolve_none(self):
        self.criar_modelos()
        with mock.patch.dict(os.environ, {"AVEIA_DIARIZACAO": "antigo"}), \
                mock.patch.object(diarizacao_mod._onnx, "AnaliseIncremental", side_effect=AssertionError("não devia criar")):
            self.assertIsNone(_FABRICA_REAL())

    def test_interruptor_zero_devolve_none(self):
        self.criar_modelos()
        with mock.patch.dict(os.environ, {"AVEIA_DIAR_AOVIVO": "0"}), \
                mock.patch.object(diarizacao_mod._onnx, "AnaliseIncremental", side_effect=AssertionError("não devia criar")):
            self.assertIsNone(_FABRICA_REAL())

    def test_modelo_ausente_devolve_none_sem_baixar(self):
        with mock.patch.object(diarizacao_mod._onnx, "AnaliseIncremental", side_effect=AssertionError("não devia criar")):
            self.assertIsNone(_FABRICA_REAL())          # nenhum modelo
            seg, emb = self.criar_modelos()
            emb.unlink()
            self.assertIsNone(_FABRICA_REAL())          # só a segmentação
        self.assertEqual(self.baixou, [])

    def test_com_modelos_cria_com_os_mesmos_parametros_do_lote(self):
        seg, emb = self.criar_modelos()
        with mock.patch.object(diarizacao_mod._onnx, "AnaliseIncremental") as cls:
            self.assertIs(_FABRICA_REAL(), cls.return_value)
        args = cls.call_args.args
        self.assertEqual(args[:2], (seg, emb))
        self.assertEqual(args[2], diarizacao_mod._onnx.Params())
        self.assertEqual(args[3], 4)

    def test_erro_na_criacao_devolve_none(self):
        self.criar_modelos()
        with mock.patch.object(diarizacao_mod._onnx, "AnaliseIncremental", side_effect=RuntimeError("onnx quebrou")):
            self.assertIsNone(_FABRICA_REAL())


def _modelos_reais() -> bool:
    seg = diarizacao_mod._caminho_segmentacao()
    emb = diarizacao_mod._caminho_embedding(diarizacao_mod.EMBEDDING_NOVO)
    return seg.exists() and emb.exists() and AUDIO.exists()


@unittest.skipUnless(_modelos_reais(), "faltam os modelos de diarização (nada é baixado nos testes)")
class ModeloRealAoVivoTest(Base):
    def test_trechos_da_finalizacao_iguais_aos_do_lote(self):
        usados, lote = [], []
        montar = pipeline.montar_falas

        def espiao_montar(palavras, trechos, *a, **k):
            usados.append([(t.inicio, t.fim, t.locutor) for t in trechos])
            return montar(palavras, trechos, *a, **k)

        def nao_deve_rodar(audio, num_locutores=None):
            lote.append(len(audio))
            return diarizar_falso(audio, num_locutores)
        for p in (mock.patch.object(diarizacao_mod, "analise_incremental", _FABRICA_REAL),
                  mock.patch.object(pipeline, "diarizar", nao_deve_rodar),
                  mock.patch.object(pipeline, "montar_falas", espiao_montar)):
            p.start()
            self.addCleanup(p.stop)
        sid = self.abrir()["sessao"]
        s = aplicacao.aovivo_sessoes.obter(sid)
        fatias = self.fatias(30)
        for i, f in enumerate(fatias):
            self.enviar(sid, i, f)
        self.assertTrue(esperar(lambda: s.pcm.n > 57 * 16000, 60))
        capturado = []
        original = aovivo.Sessao.concluir

        def concluir_espiao(self_, *a, **k):
            try:
                return original(self_, *a, **k)
            finally:
                capturado.append(self_.pcm.fatia(0, self_.pcm.n))
        with mock.patch.object(aovivo.Sessao, "concluir", concluir_espiao):
            r = self.cli.post(f"/api/aovivo/{sid}/finalizar", json={"bytes": len(self.dados), "pedacos": len(fatias), "duracao": 57.6})
            self.assertEqual(r.status_code, 200)
            self.assertEqual(self.job_pronto(r.json()["job"], limite=600)["status"], "pronto")
        ref = capturado[0]
        self.assertEqual(lote, [], "o caminho do lote rodou: o incremental não foi usado")
        self.assertEqual(len(usados), 1)
        esperado = [(t.inicio, t.fim, t.locutor) for t in diarizacao_mod.diarizar(ref, s.locutores)]
        self.assertTrue(esperado)
        self.assertEqual(usados[0], esperado)


if __name__ == "__main__":
    unittest.main()
