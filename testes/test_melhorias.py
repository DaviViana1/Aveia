"""Testes offline das melhorias de 2026-11-20: corte adaptativo dos blocos ao vivo, métricas locais sem conteúdo, teto de IA,
servidor só local (Host e Origin), privacidade e apagar tudo, aviso de método antigo de diarização.

    .venv\\Scripts\\python -m unittest testes.test_melhorias -v

Sem modelo, sem internet e sem tocar em `gravacoes/` nem em `dados/` (tudo em pastas temporárias).
"""
import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fastapi.testclient import TestClient  # noqa: E402

import app as aplicacao  # noqa: E402
from transcritor import aovivo, arquivos, locutores, metricas, pipeline, resumo, titulo  # noqa: E402


class CorteAdaptativoTest(unittest.TestCase):
    def test_ultimo_corte_enche_o_bloco(self):
        t = aovivo.TAXA
        minimo = 15 * t
        fala = [(0, 16 * t), (17 * t, 24 * t), (25 * t, 29 * t)]   # pausas em 16-17 s e 24-25 s
        self.assertEqual(aovivo.escolher_pausa(fala, 29 * t, minimo), int(16.5 * t))                 # antigo: a primeira
        self.assertEqual(aovivo.escolher_pausa(fala, 29 * t, minimo, ultimo=True), int(24.5 * t))   # novo: a última
        self.assertIsNone(aovivo.escolher_pausa([(0, 29 * t)], 29 * t, minimo, ultimo=True))       # sem pausa: nada

    def test_sem_atraso_nada_muda(self):
        """Com só o mínimo na janela (início da gravação), a única pausa possível é a mesma nos dois modos."""
        t = aovivo.TAXA
        fala = [(0, 15 * t)]
        self.assertEqual(aovivo.escolher_pausa(fala, 16 * t, 15 * t), aovivo.escolher_pausa(fala, 16 * t, 15 * t, ultimo=True))


class MetricasTest(unittest.TestCase):
    def setUp(self):
        self.pasta = tempfile.mkdtemp(prefix="aveia-met-")
        self.env = mock.patch.dict(os.environ, {"AVEIA_DADOS": self.pasta})
        self.env.start()

    def tearDown(self):
        self.env.stop()

    def test_registrar_so_aceita_numeros_e_campos_conhecidos(self):
        metricas.registrar_gravacao(via="aovivo", motor="local", duracao_s=58.14159, texto="SEGREDO DA REUNIAO",
                                    nome_arquivo="a.webm", locutor_nome="Maria", primeiro_texto_s=26.1, locutores=2)
        linhas = (Path(self.pasta) / "metricas.jsonl").read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(linhas), 1)
        d = json.loads(linhas[0])
        self.assertEqual((d["via"], d["duracao_s"], d["locutores"]), ("aovivo", 58.14, 2))
        for proibido in ("SEGREDO", "Maria", "a.webm", '"texto"', "nome_arquivo", "locutor_nome"):
            self.assertNotIn(proibido, linhas[0])

    def test_texto_longo_nao_entra(self):
        metricas.registrar_gravacao(via="x" * 200, motor="local")
        d = json.loads((Path(self.pasta) / "metricas.jsonl").read_text(encoding="utf-8"))
        self.assertNotIn("via", d)

    def test_teto_de_ia(self):
        with mock.patch.dict(os.environ, {"AVEIA_IA_TETO_MES": "3"}):
            metricas.contar_ia("resumo")
            metricas.contar_ia("titulo")
            self.assertFalse(metricas.uso_ia()["aviso"] and metricas.uso_ia()["bloqueada"])
            metricas.contar_ia("titulo")                       # 3 de 3: aviso (>= 80%) e bloqueada
            u = metricas.uso_ia()
            self.assertEqual((u["chamadas"], u["teto"], u["aviso"], u["bloqueada"]), (3, 3, True, True))
            self.assertEqual(u["por_servico"], {"resumo": 1, "titulo": 2})
            with self.assertRaises(metricas.TetoIA):
                metricas.contar_ia("resumo")
            self.assertEqual(metricas.uso_ia()["chamadas"], 3)   # a chamada barrada não conta

    def test_teto_zero_bloqueia_e_nenhum_modulo_chama_a_rede(self):
        env = {"AVEIA_IA_TETO_MES": "0", "AI_GATEWAY_API_KEY": "chave-falsa"}
        with mock.patch.dict(os.environ, env), mock.patch("urllib.request.urlopen", side_effect=AssertionError("saiu para a rede")):
            with self.assertRaises(resumo.FalhaIA):
                resumo._chamar("s", "u", 10)
            with self.assertRaises(resumo.FalhaIA):
                locutores._chamar("s", "u", 10)
            with self.assertRaises(titulo._Falha):
                titulo._chamar("s", "u", 10)
            r = titulo.gerar_titulo(["uma nota com palavras suficientes para tentar gerar um titulo agora"], [])
            self.assertEqual(r.motivo, "falha")

    def test_mes_novo_zera(self):
        (Path(self.pasta)).mkdir(exist_ok=True)
        (Path(self.pasta) / "ia_uso.json").write_text(json.dumps({"mes": "2020-01", "chamadas": 999, "por_servico": {}}), encoding="utf-8")
        self.assertEqual(metricas.uso_ia()["chamadas"], 0)


class ServidorLocalTest(unittest.TestCase):
    def setUp(self):
        self.pasta = Path(tempfile.mkdtemp(prefix="aveia-grav-"))
        arquivos.criar_pastas(self.pasta)
        self.patch = mock.patch.object(aplicacao, "GRAVACOES", self.pasta)
        self.patch.start()
        self.cli = TestClient(aplicacao.app, base_url="http://127.0.0.1:8765")

    def tearDown(self):
        self.patch.stop()

    def test_host_estranho_e_recusado(self):
        for host in ("evil.com", "evil.com:8765", "127.0.0.1.evil.com"):
            r = self.cli.get("/api/ia", headers={"Host": host})
            self.assertEqual(r.status_code, 403, host)
        for host in ("127.0.0.1:8765", "localhost:8765", "[::1]:8765", "localhost"):
            self.assertEqual(self.cli.get("/api/ia", headers={"Host": host}).status_code, 200, host)

    def test_origem_estranha_nao_muda_nada(self):
        corpo = {"confirmar": False}
        for origem in ("http://evil.com", "https://evil.com:8765", "null", "http://127.0.0.1.evil.com", "http://localhost:3000",
                       "http://127.0.0.1:3000", "http://localhost:8765"):   # as três últimas: outra porta e outra grafia (outra origem)
            r = self.cli.post("/api/apagar-tudo", json=corpo, headers={"Origin": origem})
            self.assertEqual(r.status_code, 403, origem)
        for host in ("127.0.0.1:8765", "localhost:8765", "[::1]:8765"):   # Origin igual ao Host: mesma origem
            r = self.cli.post("/api/apagar-tudo", json=corpo, headers={"Host": host, "Origin": "http://" + host})
            self.assertEqual(r.status_code, 400, host)   # passou pelo filtro; falta a confirmação

    def test_pedido_de_outro_site_ou_do_mesmo_site_e_recusado(self):
        for valor in ("cross-site", "same-site"):
            r = self.cli.post("/api/apagar-tudo", json={"confirmar": False}, headers={"Sec-Fetch-Site": valor})
            self.assertEqual(r.status_code, 403, valor)
        for valor in ("same-origin", "none"):
            r = self.cli.post("/api/apagar-tudo", json={"confirmar": False}, headers={"Sec-Fetch-Site": valor})
            self.assertEqual(r.status_code, 400, valor)

    def test_testserver_nao_e_host_local_em_producao(self):
        self.assertEqual(self.cli.get("/api/ia", headers={"Host": "testserver"}).status_code, 403)

    def test_tamanho_do_corpo_tem_teto(self):
        r = self.cli.post("/api/aovivo/x/pedaco?seq=0", content=b"0" * (aplicacao.aovivo.MAX_PEDACO + 1))
        self.assertEqual(r.status_code, 413)
        r = self.cli.post("/api/apagar-tudo", content=b"{" + b" " * (aplicacao.MAX_JSON + 1) + b"}", headers={"Content-Type": "application/json"})
        self.assertEqual(r.status_code, 413)

    def test_cabecalhos_de_seguranca_e_sem_docs(self):
        r = self.cli.get("/")
        self.assertEqual(r.headers["X-Frame-Options"], "DENY")
        self.assertIn("frame-ancestors 'none'", r.headers["Content-Security-Policy"])
        for rota in ("/docs", "/redoc", "/openapi.json"):
            self.assertEqual(self.cli.get(rota).status_code, 404, rota)

    def test_job_nao_devolve_caminho_do_disco(self):
        aplicacao.jobs["j1"] = {"status": "pronto", "arquivo": "C:/x/y.webm", "falas": []}
        r = self.cli.get("/api/jobs/j1").json()
        self.assertNotIn("arquivo", r)
        self.assertEqual(r["status"], "pronto")

    def test_jobs_antigos_terminados_sao_descartados(self):
        for i in range(aplicacao._Jobs.MAX + 20):
            aplicacao.jobs[f"t{i}"] = {"status": "pronto"}
        self.assertLessEqual(len(aplicacao.jobs), aplicacao._Jobs.MAX)
        aplicacao.jobs["vivo"] = {"status": "processando"}
        for i in range(30):
            aplicacao.jobs[f"u{i}"] = {"status": "pronto"}
        self.assertIn("vivo", aplicacao.jobs, "o que está processando nunca é descartado")

    def test_extensao_do_upload_so_de_lista(self):
        for ent, esperado in ((".MP3", "mp3"), (".wav", "wav"), (".bat", "webm"), (".lnk", "webm"), ("", "webm"), (".mp3:evil", "webm")):
            self.assertEqual(aplicacao._ext_upload(ent), esperado, ent)

    def test_nomes_reservados_do_windows_nao_viram_base(self):
        for ruim in ("CON", "nul", "Aux.txt", "COM1", "lpt9.x"):
            self.assertFalse(arquivos.base_valida(ruim), ruim)
        for bom in ("nota-1", "console", "20261121-aveia", "conta.1"):
            self.assertTrue(arquivos.base_valida(bom), bom)

    def test_get_com_origem_estranha_continua_lendo_a_pagina(self):
        self.assertEqual(self.cli.get("/", headers={"Origin": "http://evil.com"}).status_code, 200)

    def test_privacidade_nao_vaza_chave(self):
        with mock.patch.dict(os.environ, {"AI_GATEWAY_API_KEY": "SEGREDO-123", "GROQ_API_KEY": "SEGREDO-456"}):
            r = self.cli.get("/api/privacidade")
            self.assertEqual(r.status_code, 200)
            self.assertNotIn("SEGREDO", r.text)
            ids = {x["id"]: x for x in r.json()["recursos"]}
            self.assertEqual(ids["transcricao"]["onde"], "local")
            self.assertEqual(ids["resumo"]["onde"], "nuvem")
            self.assertEqual(ids["groq"]["envia"], "áudio")
            self.assertEqual(self.cli.get("/api/ia").json()["uso"]["teto"], metricas.teto_mes())
            self.assertNotIn("SEGREDO", self.cli.get("/api/ia").text)

    def test_apagar_tudo_exige_confirmacao_e_apaga(self):
        (self.pasta / "pendente.webm").write_bytes(b"x")
        (self.pasta / "pendente.json").write_text("[]", encoding="utf-8")
        (self.pasta / arquivos.PASTAS["audio"] / "a.webm").write_bytes(b"x")
        (self.pasta / arquivos.PASTAS["transcricao"] / "a.txt").write_text("t", encoding="utf-8")
        self.assertEqual(self.cli.post("/api/apagar-tudo", json={}).status_code, 400)
        self.assertEqual(len(arquivos._arquivos_da_raiz(self.pasta)), 4)
        r = self.cli.post("/api/apagar-tudo", json={"confirmar": True})
        self.assertEqual((r.status_code, r.json()["removidos"]), (200, 4))
        self.assertEqual(arquivos._arquivos_da_raiz(self.pasta), [])
        for nome in arquivos.PASTAS.values():
            self.assertTrue((self.pasta / nome).is_dir())   # as pastas ficam

    def test_situacao_conta_o_que_passou_do_prazo(self):
        velho = self.pasta / arquivos.PASTAS["audio"] / "velho.webm"
        novo = self.pasta / arquivos.PASTAS["audio"] / "novo.webm"
        velho.write_bytes(b"x")
        novo.write_bytes(b"x")
        antes = time.time() - 40 * 86400
        os.utime(velho, (antes, antes))
        s = arquivos.situacao(self.pasta, 30)
        self.assertEqual((s["arquivos"], s["passaram_do_prazo"], s["prazo_dias"]), (2, 1, 30))
        self.assertGreaterEqual(s["mais_antigo_dias"], 39)


class AvisoDiarizacaoTest(unittest.TestCase):
    def test_metodo_antigo_por_falha_avisa(self):
        meta: dict = {}
        with mock.patch.object(pipeline, "ultimo_motor", return_value=("antigo", "ModuleNotFoundError: No module named 'scipy'")):
            pipeline.anotar_diarizacao(meta)
        self.assertEqual(meta["diarizacao"], "antigo")
        self.assertIn("método antigo", meta["avisos"][0])
        self.assertIn("scipy", meta["avisos"][0])

    def test_escolha_do_antigo_e_motor_novo_nao_avisam(self):
        for retorno in (("antigo", ""), ("novo", "")):
            meta: dict = {}
            with mock.patch.object(pipeline, "ultimo_motor", return_value=retorno):
                pipeline.anotar_diarizacao(meta)
            self.assertEqual(meta["diarizacao"], retorno[0])
            self.assertNotIn("avisos", meta)

    def test_sem_meta_nao_levanta(self):
        pipeline.anotar_diarizacao(None)


if __name__ == "__main__":
    unittest.main()
