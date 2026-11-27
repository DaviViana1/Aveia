"""Testes offline da gravação automática por controle (captura, sessão de PCM direto, rotas, parada, saída, alinhamento).

    .venv\\Scripts\\python -m unittest testes.test_controle -v

Sem dispositivo de áudio, sem modelo e sem internet: o áudio é PCM sintético, a captura é falsa, o relógio é simulado e a IA
fica desligada (AVEIA_IA=0). Nada é gravado em `saida/` nem em `gravacoes/` do repositório (pastas temporárias).
"""
import json
import os
import sys
import tempfile
import time
import unittest
import wave
from datetime import datetime, timedelta
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("AVEIA_DADOS", tempfile.mkdtemp(prefix="aveia-dados-teste-"))
os.environ["AVEIA_CONTROLE_DIR"] = tempfile.mkdtemp(prefix="aveia-controle-teste-")
from fastapi.testclient import TestClient  # noqa: E402

import app as aplicacao  # noqa: E402
from transcritor import alinhar_teams, aovivo, captura, controle as controle_mod, saida  # noqa: E402

TAXA = captura.TAXA


class Relogio:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t


class CapturaFalsa:
    """Guarda o callback e deixa o teste empurrar blocos como a captura real faria."""
    ultima = None

    def __init__(self, cb):
        self.cb = cb
        self.iniciada = self.parada = False
        CapturaFalsa.ultima = self

    def iniciar(self):
        self.iniciada = True

    def parar(self):
        self.parada = True


def bloco(segundos, e=0.1, d=0.1):
    n = int(segundos * TAXA)
    b = np.zeros((n, 2), dtype=np.float32)
    b[:, 0], b[:, 1] = e, d
    return b.reshape(-1)


# ------------------------------------------------------------------------------------------------ captura
class TestCaptura(unittest.TestCase):
    def test_reamostrador_razao_exata_e_continuidade(self):
        for taxa in (48000, 44100):
            t = np.arange(taxa * 3) / taxa
            x = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
            r = captura.Reamostrador(taxa)
            saida_ = np.concatenate([r.processar(x[i:i + 4800]) for i in range(0, len(x), 4800)])
            esperado = len(x) * 16000 // taxa
            self.assertLessEqual(esperado - len(saida_), 400, taxa)   # só a folga final (~10 a 25 ms) fica retida
            ref = 0.5 * np.sin(2 * np.pi * 440 * np.arange(len(saida_)) / 16000)
            meio = slice(2000, len(saida_) - 2000)   # fora das bordas do início e do fim
            self.assertLess(float(np.max(np.abs(saida_[meio] - ref[meio]))), 0.02, taxa)   # sem emendas entre pedaços

    def test_preenche_silencio_pelo_relogio(self):
        rel = Relogio()
        inter = captura.Intercalador(rel)
        rel.t += 2.5
        inter.anexar(0, np.full(int(2 * TAXA), 0.3, dtype=np.float32))   # só o microfone entregou (o loopback calou)
        b = inter.coletar()
        self.assertIsNotNone(b)
        self.assertEqual(len(b) // 2, int((2.5 - captura.ATRASO_S) * TAXA))
        self.assertTrue(np.all(b[1::2] == 0.0))
        self.assertTrue(np.all(b[0::2] == np.float32(0.3)))
        self.assertGreater(inter.preenchido[1], 0)
        self.assertEqual(inter.preenchido[0], 0)

    def test_intercala_e_d_alinhados(self):
        rel = Relogio()
        inter = captura.Intercalador(rel)
        rel.t += 2.0
        inter.anexar(0, np.full(2 * TAXA, 0.5, dtype=np.float32))
        inter.anexar(1, np.full(2 * TAXA, -0.25, dtype=np.float32))
        b = inter.coletar()
        self.assertTrue(np.all(b[0::2] == np.float32(0.5)))
        self.assertTrue(np.all(b[1::2] == np.float32(-0.25)))
        self.assertEqual(len(b) % 2, 0)

    def test_sem_bloco_antes_do_tempo_e_deriva_limitada(self):
        rel = Relogio()
        inter = captura.Intercalador(rel)
        rel.t += 0.8
        self.assertIsNone(inter.coletar())   # 0,8 s menos a folga de 0,5 s: pouco para um bloco de 1 s
        inter.anexar(0, np.ones(10 * TAXA, dtype=np.float32))   # canal adiantado demais: o excesso é descartado
        inter.anexar(1, np.ones(10 * TAXA, dtype=np.float32))
        rel.t += 1.0
        inter.coletar()
        self.assertGreater(inter.descartado[0], 0)
        self.assertLessEqual(len(inter._buf[0]), int(captura.MANTER_S * TAXA))

    def test_energia_do_canal_do_computador(self):
        b = bloco(1.0, e=0.5, d=0.0)
        self.assertLess(captura.rms(b[1::2]), captura.LIMIAR_RMS_SILENCIO)
        self.assertGreater(captura.rms(b[0::2]), 0.4)


# ------------------------------------------------------------------------------------------- sessão direta
class TestSessaoDireta(unittest.TestCase):
    def test_anexar_pcm_sem_decodificador_e_wav_em_disco(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            ger = aovivo.Gerenciador(lambda fn, *a: None)   # nada roda na fila: só a recepção
            s = ger.criar_direta(Path(tmp) / "a.wav", "pt")
            self.addCleanup(s.liberar)
            self.assertTrue(s.estereo and s.pcm_direto)
            s.anexar_pcm(bloco(3.0, e=0.2, d=0.4))
            self.assertEqual(s.pcm.n, 3 * TAXA)
            e, d = s.pcm.canais()
            self.assertAlmostEqual(float(e[0]), 0.2, places=5)
            self.assertAlmostEqual(float(d[0]), 0.4, places=5)
            s.marcar_fim()
            s.anexar_pcm(bloco(1.0))   # depois do fim: ignorado
            self.assertEqual(s.pcm.n, 3 * TAXA)
            self.assertTrue(s._eof.is_set())
            with wave.open(str(s.arquivo), "rb") as w:
                self.assertEqual((w.getnchannels(), w.getframerate(), w.getnframes()), (2, TAXA, 3 * TAXA))
            outra = aovivo.Sessao("x", Path(tmp) / "b.webm", "pt", None, lambda *a: None)
            self.addCleanup(outra.liberar)
            with self.assertRaises(aovivo.AoVivoIndisponivel):
                outra.anexar_pcm(bloco(0.1))
            s.liberar()
            outra.liberar()


# --------------------------------------------------------------------------------------------- controle
FALAS = [{"inicio": 1.0, "fim": 5.0, "locutor": "Locutor 1", "texto": "Bom dia a todos, vamos começar a reunião de hoje."},
         {"inicio": 6.0, "fim": 9.0, "locutor": "Locutor 2", "texto": "Bom dia, tenho os números do projeto prontos aqui."}]


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self.tmp.cleanup)
        self.raiz = Path(self.tmp.name)
        self.rel = Relogio()
        self.agora = datetime(2026, 12, 1, 10, 0, 0)
        self.jobs: dict = {}
        self.ger = aovivo.Gerenciador(lambda fn, *a: None)

        def executar(job_id, sessao):   # no lugar da transcrição: entrega as falas prontas
            self.jobs[job_id].update(status="pronto", progresso=1.0, falas=[dict(f) for f in FALAS])

        self.ctl = controle_mod.Controle(self.ger, executar, self.jobs, lambda fn, *a: fn(*a), self.raiz / "grav", self.raiz / "saida",
                                         CapturaFalsa, agora=lambda: self.agora, relogio=self.rel)
        p = mock.patch.dict(os.environ, {"AVEIA_IA": "0", "AVEIA_CONTROLE_GUARDAR_AUDIO": ""})
        p.start()
        self.addCleanup(p.stop)
        self.addCleanup(self.ctl.encerrar)
        self.addCleanup(lambda: self.ctl._pos and self.ctl._pos.update(saida_feita=True))   # solta a espera da thread de pós-processamento
        self.addCleanup(lambda: [x.liberar() for x in list(self.ger.sessoes.values())])

    def iniciar(self, fim_min=60):
        return self.ctl.iniciar({"evento_id": "ev-1", "titulo": "Reunião de Teste", "participantes": [{"nome": "Ana Teste", "email": "ana@exemplo.test"}],
                                 "fim_previsto": (self.agora + timedelta(minutes=fim_min)).isoformat()})

    def esperar_flag(self, pasta, limite=10.0):
        fim = time.time() + limite
        while time.time() < fim and not (Path(pasta) / "concluido.flag").exists():
            time.sleep(0.05)
        return (Path(pasta) / "concluido.flag").exists()


class TestRotas(Base):
    def setUp(self):
        super().setUp()
        p = mock.patch.object(aplicacao, "controle", self.ctl)
        p.start()
        self.addCleanup(p.stop)
        self.cli = TestClient(aplicacao.app, base_url="http://127.0.0.1:8765")
        self.addCleanup(self.cli.close)
        self.h = {"X-Aveia-Token": aplicacao.TOKEN_CONTROLE}

    def test_sem_token_ou_token_errado_401(self):
        for metodo, rota in (("get", "estado"), ("post", "iniciar"), ("post", "parar")):
            self.assertEqual(getattr(self.cli, metodo)(f"/api/controle/{rota}").status_code, 401, rota)
            self.assertEqual(getattr(self.cli, metodo)(f"/api/controle/{rota}", headers={"X-Aveia-Token": "errado"}).status_code, 401, rota)
        self.assertEqual(self.ctl.estado()["estado"], "ocioso")

    def test_estado_iniciar_parar_com_token(self):
        self.assertEqual(self.cli.get("/api/controle/estado", headers=self.h).json()["estado"], "ocioso")
        r = self.cli.post("/api/controle/iniciar", headers=self.h, json={
            "evento_id": "ev-1", "titulo": "Reunião de Teste", "participantes": [], "fim_previsto": (self.agora + timedelta(hours=1)).isoformat()})
        self.assertEqual(r.status_code, 200, r.text)
        self.assertTrue(r.json()["ok"] and r.json()["pasta_saida"].endswith("reuniao-de-teste"))
        est = self.cli.get("/api/controle/estado", headers=self.h).json()
        self.assertEqual((est["estado"], est["evento_id"], est["titulo"]), ("gravando", "ev-1", "Reunião de Teste"))
        self.assertTrue(CapturaFalsa.ultima.iniciada)
        self.assertEqual(self.cli.post("/api/controle/parar", headers=self.h, json={}).json()["ok"], True)
        self.assertTrue(CapturaFalsa.ultima.parada)
        self.assertTrue(self.esperar_flag(r.json()["pasta_saida"]))

    def test_dupla_gravacao_409(self):
        corpo = {"evento_id": "ev-1", "titulo": "A", "fim_previsto": (self.agora + timedelta(hours=1)).isoformat()}
        self.assertEqual(self.cli.post("/api/controle/iniciar", headers=self.h, json=corpo).status_code, 200)
        r = self.cli.post("/api/controle/iniciar", headers=self.h, json={**corpo, "evento_id": "ev-2"})
        self.assertEqual(r.status_code, 409)
        self.assertEqual((r.json()["motivo"], r.json()["evento_id"]), ("ja-gravando", "ev-1"))

    def test_painel_sem_token_so_com_campos_permitidos(self):
        self.assertEqual(self.cli.get("/api/controle/painel").json()["estado"], "ocioso")
        self.cli.post("/api/controle/iniciar", headers=self.h, json={
            "evento_id": "ev-1", "titulo": "Reunião de Teste", "participantes": [{"nome": "Ana", "email": "ana@exemplo.test"}],
            "fim_previsto": (self.agora + timedelta(hours=1)).isoformat()})
        self.agora += timedelta(seconds=75)
        r = self.cli.get("/api/controle/painel")
        self.assertEqual(r.status_code, 200)
        p = r.json()
        self.assertEqual(set(p), {"estado", "titulo", "inicio", "fim_previsto", "segundos", "evento_id", "sessao", "job", "assumida", "minha"})
        self.assertEqual((p["estado"], p["titulo"], p["segundos"]), ("gravando", "Reunião de Teste", 75))
        self.assertEqual((p["evento_id"], p["job"], p["assumida"]), ("ev-1", None, False))
        self.assertEqual(p["sessao"], list(self.ger.sessoes)[0])
        self.assertNotIn("ana@exemplo.test", r.text)

    def test_parar_tela_exige_origin_local(self):
        self.cli.post("/api/controle/iniciar", headers=self.h, json={"evento_id": "e", "titulo": "A", "fim_previsto": (self.agora + timedelta(hours=1)).isoformat()})
        self.assertEqual(self.cli.post("/api/controle/parar-tela").status_code, 403)   # sem Origin
        self.assertEqual(self.cli.post("/api/controle/parar-tela", headers={"Origin": "http://127.0.0.1:3000"}).status_code, 403)
        self.assertEqual(self.cli.post("/api/controle/parar-tela", headers={"Origin": "https://exemplo.com"}).status_code, 403)
        self.assertEqual(self.ctl.estado()["estado"], "gravando")
        r = self.cli.post("/api/controle/parar-tela", headers={"Origin": "http://127.0.0.1:8765"})
        self.assertEqual((r.status_code, r.json()["ok"]), (200, True))
        self.assertNotEqual(self.ctl.estado()["estado"], "gravando")

    def test_iniciar_409_com_gravacao_da_tela(self):
        s = self.ger.criar(self.raiz / "tela.wav", "pt", None)
        self.assertIsNotNone(s)
        r = self.cli.post("/api/controle/iniciar", headers=self.h, json={"evento_id": "e", "titulo": "A", "fim_previsto": (self.agora + timedelta(hours=1)).isoformat()})
        self.assertEqual((r.status_code, r.json()["motivo"]), (409, "tela-gravando"))
        self.assertEqual(self.ctl.estado()["estado"], "ocioso")

    # ---- a tela assume a gravação ------------------------------------------------------------------------
    ORIGEM = {"Origin": "http://127.0.0.1:8765"}

    def iniciar_http(self, evento="ev-1"):
        r = self.cli.post("/api/controle/iniciar", headers=self.h, json={
            "evento_id": evento, "titulo": "Reunião de Teste", "participantes": [{"nome": "Ana Teste", "email": "ana@exemplo.test"}],
            "usuario": "Davi", "fim_previsto": (self.agora + timedelta(hours=1)).isoformat()})
        self.assertEqual(r.status_code, 200, r.text)
        return r.json()

    def test_assumir_primeira_aceita_segunda_409_sem_origin_403(self):
        ini = self.iniciar_http()
        sessao = ini["sessao"]
        self.assertEqual(self.cli.post("/api/controle/assumir", json={"sessao": sessao, "pagina": "a"}).status_code, 403)   # sem Origin
        r = self.cli.post("/api/controle/assumir", headers=self.ORIGEM, json={"sessao": sessao, "pagina": "a"})
        self.assertEqual(r.status_code, 200, r.text)
        j = r.json()
        self.assertEqual((j["evento_id"], j["titulo"], j["participantes"], j["usuario"]), ("ev-1", "Reunião de Teste", ["Ana Teste"], "Davi"))
        self.assertNotIn("ana@exemplo.test", r.text)
        self.assertEqual(self.cli.post("/api/controle/assumir", headers=self.ORIGEM, json={"sessao": sessao, "pagina": "b"}).status_code, 409)
        self.assertEqual(self.cli.post("/api/controle/assumir", headers=self.ORIGEM, json={"sessao": sessao, "pagina": "a"}).status_code, 200)   # recarregar
        self.assertEqual(self.cli.post("/api/controle/assumir", headers=self.ORIGEM, json={"sessao": "outra", "pagina": "a"}).status_code, 404)
        self.assertTrue(self.cli.get("/api/controle/painel").json()["assumida"])
        self.rel.t += controle_mod.DONO_SUMIDO_S + 1   # a dona sumiu: outra página pode assumir
        self.assertEqual(self.cli.post("/api/controle/assumir", headers=self.ORIGEM, json={"sessao": sessao, "pagina": "b"}).status_code, 200)

    def test_parar_devolve_job(self):
        self.iniciar_http()
        r = self.cli.post("/api/controle/parar-tela", headers=self.ORIGEM)
        self.assertEqual(r.status_code, 200)
        job = r.json()["job"]
        self.assertTrue(job.startswith("controle-") and job in self.jobs)
        self.assertEqual(self.cli.post("/api/controle/parar-tela", headers=self.ORIGEM).json()["job"], job)   # repetido: o mesmo job

    def test_painel_expoe_o_job_depois_da_parada_automatica(self):
        self.iniciar_http()
        self.ctl.parar("silencio")
        p = self.cli.get("/api/controle/painel").json()
        self.assertTrue(p["job"].startswith("controle-"))
        self.assertIsNone(p["sessao"])

    def test_servidor_nao_posprocessa_quando_assumido(self):
        ini = self.iniciar_http()
        self.cli.post("/api/controle/assumir", headers=self.ORIGEM, json={"sessao": ini["sessao"], "pagina": "a"})
        with mock.patch.object(self.ctl, "nomear_e_resumir", side_effect=AssertionError("o servidor processou")):
            self.cli.post("/api/controle/parar-tela", headers=self.ORIGEM)
            time.sleep(0.6)
            self.assertFalse((Path(ini["pasta_saida"]) / "concluido.flag").exists())
            self.assertEqual(self.ctl.estado()["estado"], "processando")

    def test_pagina_que_some_cai_na_reserva_do_servidor(self):
        ini = self.iniciar_http()
        self.cli.post("/api/controle/assumir", headers=self.ORIGEM, json={"sessao": ini["sessao"], "pagina": "a"})
        self.cli.post("/api/controle/parar-tela", headers=self.ORIGEM)
        time.sleep(0.4)
        self.assertFalse((Path(ini["pasta_saida"]) / "concluido.flag").exists())
        self.rel.t += controle_mod.PAGINA_SUMIU_S + 1
        self.assertTrue(self.esperar_flag(ini["pasta_saida"]))
        meta = json.loads((Path(ini["pasta_saida"]) / "meta.json").read_text(encoding="utf-8"))
        self.assertEqual(meta["processado_por"], "servidor")

    def test_consulta_do_painel_mantem_a_pagina_viva(self):
        ini = self.iniciar_http()
        self.cli.post("/api/controle/assumir", headers=self.ORIGEM, json={"sessao": ini["sessao"], "pagina": "a"})
        self.cli.post("/api/controle/parar-tela", headers=self.ORIGEM)
        for _ in range(3):
            self.rel.t += 40
            self.cli.get("/api/controle/painel?pagina=a")
            time.sleep(0.3)
        self.assertFalse((Path(ini["pasta_saida"]) / "concluido.flag").exists())   # 120 s no total, mas a página consultou sempre

    def corpo_saida(self, **extra):
        return {"evento_id": "ev-1", "pagina": "a", "titulo": "Título final da tela", "ia": True,
                "falas": [dict(f) for f in FALAS], "locutores": {"Locutor 1": "Davi", "Locutor 2": "Ana Teste"},
                "resumo": {"resumo": "Resumo da tela.", "topicos": ["Ponto A"], "decisoes": ["Decisão B"],
                           "acoes": [{"tarefa": "Fazer C", "responsavel": "Ana Teste", "prazo": "sexta"}]}, **extra}

    def test_saida_da_pagina_grava_a_pasta_com_flag_por_ultimo_e_recusa_a_segunda(self):
        ini = self.iniciar_http()
        self.cli.post("/api/controle/assumir", headers=self.ORIGEM, json={"sessao": ini["sessao"], "pagina": "a"})
        self.assertEqual(self.cli.post("/api/controle/saida", json=self.corpo_saida()).status_code, 403)   # sem Origin
        self.cli.post("/api/controle/parar-tela", headers=self.ORIGEM)
        pasta = Path(ini["pasta_saida"])
        ordem = []
        real = saida._atomico

        def espiao(destino, conteudo):
            ordem.append(destino.name)
            real(destino, conteudo)

        with mock.patch.object(saida, "_atomico", espiao):
            r = self.cli.post("/api/controle/saida", headers=self.ORIGEM, json=self.corpo_saida())
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(ordem[-1], "concluido.flag")
        self.assertTrue((pasta / "concluido.flag").exists())
        meta = json.loads((pasta / "meta.json").read_text(encoding="utf-8"))
        self.assertEqual((meta["titulo"], meta["ia"], meta["processado_por"], meta["locutores"]["Locutor 2"]), ("Título final da tela", True, "tela", "Ana Teste"))
        self.assertIn("] Ana Teste: Bom dia, tenho", (pasta / "transcricao.md").read_text(encoding="utf-8"))
        self.assertEqual(json.loads((pasta / "falas.json").read_text(encoding="utf-8"))[1]["locutor"], "Locutor 2")   # rótulos originais
        resumo = (pasta / "resumo.md").read_text(encoding="utf-8")
        for trecho in ("## Resumo\nResumo da tela.", "- Ponto A", "- Decisão B", "- Fazer C (Ana Teste) (sexta)"):
            self.assertIn(trecho, resumo)
        # segunda saída do mesmo evento: recusada; a pasta não muda
        antes = (pasta / "meta.json").read_text(encoding="utf-8")
        r2 = self.cli.post("/api/controle/saida", headers=self.ORIGEM, json=self.corpo_saida(titulo="Outro"))
        self.assertEqual((r2.status_code, r2.json()["motivo"]), (409, "saida-ja-gravada"))
        self.assertEqual((pasta / "meta.json").read_text(encoding="utf-8"), antes)
        self.assertTrue(self.ctl._eventos_gravados)
        for _ in range(40):
            if self.ctl.estado()["estado"] == "ocioso":
                break
            time.sleep(0.05)
        self.assertEqual(self.ctl.estado()["estado"], "ocioso")

    def test_saida_nao_vira_duas_com_a_reserva(self):
        ini = self.iniciar_http()
        self.cli.post("/api/controle/assumir", headers=self.ORIGEM, json={"sessao": ini["sessao"], "pagina": "a"})
        self.cli.post("/api/controle/parar-tela", headers=self.ORIGEM)
        self.rel.t += controle_mod.PAGINA_SUMIU_S + 1
        self.assertTrue(self.esperar_flag(ini["pasta_saida"]))   # a reserva gravou
        r = self.cli.post("/api/controle/saida", headers=self.ORIGEM, json=self.corpo_saida())
        self.assertEqual((r.status_code, r.json()["motivo"]), (409, "saida-ja-gravada"))

    def test_saida_sem_gravacao_ou_de_outra_pagina(self):
        self.assertEqual(self.cli.post("/api/controle/saida", headers=self.ORIGEM, json=self.corpo_saida()).status_code, 404)
        ini = self.iniciar_http()
        self.cli.post("/api/controle/assumir", headers=self.ORIGEM, json={"sessao": ini["sessao"], "pagina": "a"})
        self.cli.post("/api/controle/parar-tela", headers=self.ORIGEM)
        self.assertEqual(self.cli.post("/api/controle/saida", headers=self.ORIGEM, json=self.corpo_saida(pagina="b")).status_code, 403)
        self.assertEqual(self.cli.post("/api/controle/saida", headers=self.ORIGEM, json=self.corpo_saida(falas=[{"x": 1}])).status_code, 400)
        self.assertFalse((Path(ini["pasta_saida"]) / "concluido.flag").exists())

    def test_desistir_passa_para_a_reserva_do_servidor(self):
        ini = self.iniciar_http()
        self.cli.post("/api/controle/assumir", headers=self.ORIGEM, json={"sessao": ini["sessao"], "pagina": "a"})
        self.cli.post("/api/controle/parar-tela", headers=self.ORIGEM)
        r = self.cli.post("/api/controle/saida", headers=self.ORIGEM, json={"evento_id": "ev-1", "pagina": "a", "desistir": True})
        self.assertEqual(r.json(), {"ok": True, "desistiu": True})
        self.assertTrue(self.esperar_flag(ini["pasta_saida"]))
        self.assertEqual(json.loads((Path(ini["pasta_saida"]) / "meta.json").read_text(encoding="utf-8"))["processado_por"], "servidor")

    def test_pagina_aberta_no_estado(self):
        self.assertIs(self.cli.get("/api/controle/estado", headers=self.h).json()["pagina_aberta"], False)
        # consulta sem identificação (aba antiga ou script) não conta como página aberta
        self.cli.get("/api/controle/painel")
        self.assertIs(self.cli.get("/api/controle/estado", headers=self.h).json()["pagina_aberta"], False)
        self.cli.get("/api/controle/painel?pagina=aba1")
        self.assertIs(self.cli.get("/api/controle/estado", headers=self.h).json()["pagina_aberta"], True)
        self.rel.t += controle_mod.PAGINA_ABERTA_S + 1
        self.assertIs(self.cli.get("/api/controle/estado", headers=self.h).json()["pagina_aberta"], False)

    def test_apos_saida_roda_uma_vez_com_a_pasta(self):
        marca = self.raiz / "marca.txt"
        script = self.raiz / "apos.py"
        script.write_text("import os\nopen(os.environ['MARCA'], 'a').write(os.environ['AVEIA_SAIDA_PASTA'] + '\\n')\n", encoding="utf-8")
        with mock.patch.dict(os.environ, {"AVEIA_APOS_SAIDA": f'"{sys.executable}" "{script}"', "MARCA": str(marca)}):
            ini = self.iniciar_http()
            self.cli.post("/api/controle/assumir", headers=self.ORIGEM, json={"sessao": ini["sessao"], "pagina": "a"})
            self.cli.post("/api/controle/parar-tela", headers=self.ORIGEM)
            self.assertEqual(self.cli.post("/api/controle/saida", headers=self.ORIGEM, json=self.corpo_saida()).status_code, 200)
            self.assertEqual(self.cli.post("/api/controle/saida", headers=self.ORIGEM, json=self.corpo_saida()).status_code, 409)
            fim = time.time() + 15
            while time.time() < fim and not marca.exists():
                time.sleep(0.1)
            time.sleep(1.0)   # uma segunda execução (indevida) teria tempo de aparecer
        self.assertEqual(marca.read_text(encoding="utf-8").splitlines(), [ini["pasta_saida"]])

    def test_apos_saida_que_falha_so_vai_para_o_log(self):
        with mock.patch.dict(os.environ, {"AVEIA_APOS_SAIDA": "x"}), mock.patch("subprocess.Popen", side_effect=OSError("sem comando")):
            ini = self.iniciar_http()
            self.cli.post("/api/controle/parar-tela", headers=self.ORIGEM)
            self.assertTrue(self.esperar_flag(ini["pasta_saida"]))   # a saída existe mesmo com o comando falhando

    def test_delete_aovivo_nao_derruba_a_sessao_do_controle(self):
        ini = self.iniciar_http()
        with mock.patch.object(aplicacao, "aovivo_sessoes", self.ger):
            self.assertEqual(self.cli.delete(f"/api/aovivo/{ini['sessao']}").json().get("controlada"), True)
        self.assertIn(ini["sessao"], self.ger.sessoes)

    def test_fim_previsto_invalido_400(self):
        r = self.cli.post("/api/controle/iniciar", headers=self.h, json={"evento_id": "x", "fim_previsto": "amanhã"})
        self.assertEqual(r.status_code, 400)


class TestParada(Base):
    def alimentar(self, segundos, d):
        """Empurra áudio pelo callback da captura (1 s por bloco), com o canal D no nível `d`."""
        for _ in range(int(segundos)):
            CapturaFalsa.ultima.cb(bloco(1.0, e=0.1, d=d), captura.rms(bloco(1.0, d=d)[1::2]), 0.1)

    def test_nao_para_antes_do_fim_previsto(self):
        self.assertEqual(self.iniciar(fim_min=60)[0], 200)
        self.alimentar(700, d=0.0)   # silêncio longo, mas ainda antes do fim
        self.assertIsNone(self.ctl.tick())
        self.assertEqual(self.ctl.estado()["estado"], "gravando")

    def test_para_por_silencio_depois_do_fim(self):
        self.iniciar(fim_min=60)
        self.alimentar(5, d=0.3)
        self.alimentar(601, d=0.0)
        self.agora += timedelta(minutes=61)
        self.assertEqual(self.ctl.tick(), "silencio")
        self.assertTrue(CapturaFalsa.ultima.parada)

    def test_som_no_computador_zera_a_contagem_do_silencio(self):
        self.iniciar(fim_min=1)
        self.alimentar(500, d=0.0)
        self.alimentar(2, d=0.3)
        self.alimentar(200, d=0.0)
        self.agora += timedelta(minutes=2)
        self.assertIsNone(self.ctl.tick())   # só 200 s de silêncio seguido

    def test_para_por_horario_limite(self):
        self.iniciar(fim_min=60)
        self.alimentar(5, d=0.3)   # ainda há som: o silêncio não manda parar
        self.agora += timedelta(minutes=89)
        self.assertIsNone(self.ctl.tick())
        self.agora += timedelta(minutes=2)   # 91 min: passou de fim_previsto + 30 min
        self.assertEqual(self.ctl.tick(), "limite")

    def test_parada_manual_gera_a_saida_com_flag_e_sem_ia(self):
        _, r = self.iniciar()
        self.alimentar(3, d=0.2)
        self.agora += timedelta(minutes=5)
        self.ctl.parar("manual")
        self.assertTrue(self.esperar_flag(r["pasta_saida"]))
        pasta = Path(r["pasta_saida"])
        meta = json.loads((pasta / "meta.json").read_text(encoding="utf-8"))
        self.assertIs(meta["ia"], False)
        self.assertEqual((meta["evento_id"], meta["origem_nomes"]), ("ev-1", "canal+pistas"))
        self.assertEqual(meta["locutores"], {"Locutor 1": None, "Locutor 2": None})
        self.assertAlmostEqual(meta["duracao_s"], 3.0, places=1)
        self.assertIn("[00:00:01] Locutor 1: Bom dia", (pasta / "transcricao.md").read_text(encoding="utf-8"))
        resumo = (pasta / "resumo.md").read_text(encoding="utf-8")
        for secao in ("## Resumo", "## Pontos-chave", "## Decisões", "## Próximos passos"):
            self.assertIn(secao, resumo)
        self.assertFalse(list(pasta.glob("*.tmp")))
        self.assertFalse(list((self.raiz / "grav").glob("*.wav")))   # sucesso: o áudio não fica no disco
        self.assertEqual(self.ctl.estado()["estado"], "ocioso")

    def test_ia_desligada_nao_chama_nada(self):
        g = {"titulo": "T", "participantes": [{"nome": "Ana Teste", "email": ""}], "usuario": "", "duracao_s": 9}
        with mock.patch("transcritor.resumo.gerar_resumo", side_effect=AssertionError("chamou a IA")), \
                mock.patch("transcritor.locutores.identificar", side_effect=AssertionError("chamou a IA")):
            nomes, origem, usou, resumo, _ = self.ctl.nomear_e_resumir(g, [dict(f) for f in FALAS])
        self.assertEqual((usou, resumo, origem), (False, None, "canal+pistas"))
        self.assertEqual(nomes, {"Locutor 1": None, "Locutor 2": None})


# ---------------------------------------------------------------------------------------------- saída
class TestSaida(unittest.TestCase):
    def test_pasta_estavel_e_unica(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            quando = datetime(2026, 12, 1, 9, 30)
            p1 = saida.pasta_para(Path(tmp), quando, "Reunião Semanal: Ações & Metas!")
            self.assertEqual(p1.name, "2026-12-01_0930_reuniao-semanal-acoes-metas")
            p1.mkdir()
            self.assertEqual(saida.pasta_para(Path(tmp), quando, "Reunião Semanal: Ações & Metas!").name, p1.name + "-2")
            self.assertEqual(saida.slug("???"), "reuniao")

    def test_flag_por_ultimo_e_atomico(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            pasta = Path(tmp) / "x"
            ordem = []
            real = saida._atomico

            def espiao(destino, conteudo):
                ordem.append(destino.name)
                real(destino, conteudo)

            with mock.patch.object(saida, "_atomico", espiao):
                saida.gravar(pasta, {"titulo": "T"}, [dict(f) for f in FALAS], {"Locutor 1": "Ana", "Locutor 2": None}, saida.montar_resumo(None))
            self.assertEqual(ordem[-1], "concluido.flag")
            self.assertEqual(sorted(p.name for p in pasta.iterdir()),
                             ["concluido.flag", "falas.json", "locutores.json", "meta.json", "resumo.md", "transcricao.md"])
            self.assertIn("Ana: Bom dia", (pasta / "transcricao.md").read_text(encoding="utf-8"))


# ----------------------------------------------------------------------------------------- alinhamento
VTT = """WEBVTT

00:00:10.000 --> 00:00:14.500
<v Ana Teste>Bom dia a todos, vamos começar a reunião de hoje.</v>

00:00:15.000 --> 00:00:18.000
<v Bruno Exemplo>Bom dia, tenho os números do projeto prontos aqui.</v>
"""


class TestAlinhar(unittest.TestCase):
    def test_le_vtt_do_teams(self):
        cortes = alinhar_teams.ler_vtt(VTT)
        self.assertEqual([(c["nome"], c["inicio"], c["fim"]) for c in cortes], [("Ana Teste", 10.0, 14.5), ("Bruno Exemplo", 15.0, 18.0)])

    def test_estima_deslocamento_e_nomeia_pela_sobreposicao(self):
        cortes = alinhar_teams.ler_vtt(VTT)
        desloc, pares = alinhar_teams.estimar_deslocamento(FALAS, cortes)
        self.assertEqual((desloc, pares), (9.0, 2))
        self.assertEqual(alinhar_teams.nomear(FALAS, cortes, desloc), {"Locutor 1": "Ana Teste", "Locutor 2": "Bruno Exemplo"})

    def test_reescreve_os_arquivos_do_contrato(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            pasta = Path(tmp) / "reuniao"
            saida.gravar(pasta, {"titulo": "T", "origem_nomes": "canal+pistas", "ia": False}, [dict(f) for f in FALAS],
                         {"Locutor 1": None, "Locutor 2": None}, saida.montar_resumo(None))
            vtt = Path(tmp) / "a.vtt"
            vtt.write_text(VTT, encoding="utf-8")
            r = alinhar_teams.alinhar(pasta, vtt)
            self.assertEqual((r["nomeados"], r["locutores"]), (2, 2))
            meta = json.loads((pasta / "meta.json").read_text(encoding="utf-8"))
            self.assertEqual((meta["origem_nomes"], meta["locutores"]["Locutor 2"]), ("teams", "Bruno Exemplo"))
            self.assertEqual(json.loads((pasta / "locutores.json").read_text(encoding="utf-8"))["Locutor 1"], "Ana Teste")
            self.assertIn("] Bruno Exemplo: Bom dia", (pasta / "transcricao.md").read_text(encoding="utf-8"))
            self.assertTrue((pasta / "concluido.flag").exists())


if __name__ == "__main__":
    unittest.main()
