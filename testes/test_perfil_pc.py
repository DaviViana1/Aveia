"""Sonda do computador (`transcritor/perfil_pc.py`) e rotas `/api/computador`.

    .venv\\Scripts\\python -m unittest testes.test_perfil_pc -v

Sem baixar modelo e sem rede: o Whisper é falso. A medida real de velocidade fica fora da suíte (depende do computador).
"""
import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
os.environ.setdefault("AVEIA_DADOS", tempfile.mkdtemp(prefix="aveia-dados-teste-"))
os.environ.setdefault("AVEIA_SEM_AQUECER", "1")
HTML = (RAIZ / "static" / "index.html").read_text(encoding="utf-8")
from transcritor import perfil_pc as pc  # noqa: E402

BOM = {"arquitetura": "AMD64", "python_64_bits": True, "avx2": True, "ram_total_gb": 16.0, "disco_livre_gb": 50.0, "nucleos": 8}


class ClassificarTest(unittest.TestCase):
    def test_faixas(self):
        self.assertEqual(pc.classificar(0.57), "ao_vivo")
        self.assertEqual(pc.classificar(0.8), "ao_vivo")
        self.assertEqual(pc.classificar(0.81), "depois")
        self.assertEqual(pc.classificar(2.0), "depois")
        self.assertEqual(pc.classificar(2.01), "lento")


class AvaliarTest(unittest.TestCase):
    def test_pc_bom_nao_tem_problema(self):
        a = pc.avaliar_hardware(BOM)
        self.assertTrue(a["ok"])
        self.assertEqual((a["problemas"], a["avisos"]), ([], []))

    def test_cada_requisito_fraco_vira_problema(self):
        for mudanca, trecho in (({"ram_total_gb": 3.5}, "memória"), ({"disco_livre_gb": 2.0}, "disco"), ({"avx2": False}, "AVX2"),
                                ({"python_64_bits": False}, "32 bits"), ({"arquitetura": "ARM64"}, "ARM64")):
            a = pc.avaliar_hardware({**BOM, **mudanca})
            self.assertFalse(a["ok"], mudanca)
            self.assertTrue(any(trecho in p for p in a["problemas"]), (mudanca, a["problemas"]))

    def test_windows_antigo(self):
        self.assertFalse(pc.avaliar_hardware({**BOM, "windows_build": 18363})["ok"])
        a = pc.avaliar_hardware({**BOM, "windows_build": 19044})
        self.assertTrue(a["ok"])
        self.assertTrue(any("22H2" in x for x in a["avisos"]))
        self.assertEqual(pc.avaliar_hardware({**BOM, "windows_build": 26200})["avisos"], [])

    def test_pouca_ram_ou_poucos_nucleos_so_avisa(self):
        a = pc.avaliar_hardware({**BOM, "ram_total_gb": 6.0, "nucleos": 2})
        self.assertTrue(a["ok"])
        self.assertEqual(len(a["avisos"]), 2)

    def test_leitura_que_falhou_nao_vira_problema(self):
        a = pc.avaliar_hardware({"arquitetura": "AMD64", "python_64_bits": True, "avx2": None, "ram_total_gb": None,
                                 "disco_livre_gb": None, "nucleos": None})
        self.assertTrue(a["ok"], "sem leitura não se reprova o PC")


class HardwareTest(unittest.TestCase):
    def test_campos_e_nada_pessoal(self):
        h = pc.hardware()
        for k in ("windows", "python", "cpu", "nucleos", "avx2", "ram_total_gb", "ram_livre_gb", "disco_livre_gb", "arquitetura"):
            self.assertIn(k, h)
        texto = json.dumps(h, ensure_ascii=False).lower()
        for pessoal in (os.environ.get("USERNAME", "").lower() or "x@x", os.environ.get("COMPUTERNAME", "").lower() or "x@x"):
            self.assertNotIn(pessoal, texto, "nome de usuário ou do computador não pode aparecer")


class MedirTest(unittest.TestCase):
    def test_mede_uma_janela_e_classifica(self):
        class Falso:
            def transcribe(self, audio, **kw):
                assert len(audio) == 160000 and kw["vad_filter"] is False and kw["language"] == "pt"
                time.sleep(0.05)
                return iter([]), None
        r = pc.medir_janela(modelo=Falso())
        self.assertEqual(r["classe"], "ao_vivo")
        self.assertLess(r["fator"], 0.1)
        for k in ("quando", "carga_modelo_s", "janela_s", "fator"):
            self.assertIn(k, r)

    def test_salvar_e_ler_o_ultimo_teste(self):
        with tempfile.TemporaryDirectory() as d, mock.patch.dict(os.environ, {"AVEIA_DADOS": d}):
            self.assertIsNone(pc.ultimo_teste())
            pc.salvar_teste({"fator": 1.2, "classe": "depois", "janela_s": 36.0, "quando": "2026-11-21T10:00:00"})
            self.assertEqual(pc.ultimo_teste()["classe"], "depois")
            self.assertEqual(pc.estado()["classe_nome"], "Só depois da gravação")
            (Path(d) / "computador.json").write_text("lixo", encoding="utf-8")
            self.assertIsNone(pc.ultimo_teste(), "arquivo corrompido não derruba")


class RotasTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from fastapi.testclient import TestClient
        import app as servidor
        cls.servidor = servidor
        cls.c = TestClient(servidor.app, base_url="http://127.0.0.1:8765")

    def test_get_devolve_hardware_e_avaliacao(self):
        r = self.c.get("/api/computador")
        self.assertEqual(r.status_code, 200)
        d = r.json()
        self.assertIn("hardware", d)
        self.assertIn("ok", d["avaliacao"])

    def test_post_testar_roda_na_fila_e_grava_o_resultado(self):
        falso = {"quando": "2026-11-21T10:00:00", "carga_modelo_s": 1.0, "janela_s": 12.0, "fator": 0.4, "classe": "ao_vivo"}
        with tempfile.TemporaryDirectory() as d, mock.patch.dict(os.environ, {"AVEIA_DADOS": d}), \
                mock.patch.object(pc, "medir_janela", return_value=falso):
            job = self.c.post("/api/computador/testar").json()["job"]
            for _ in range(100):
                j = self.c.get(f"/api/jobs/{job}").json()
                if j["status"] != "processando":
                    break
                time.sleep(0.05)
            self.assertEqual(j["status"], "pronto", j)
            self.assertEqual(j["teste"]["classe"], "ao_vivo")
            self.assertEqual(self.c.get("/api/computador").json()["teste"]["fator"], 0.4)

    def test_falha_do_teste_vira_erro_no_job(self):
        with mock.patch.object(pc, "medir_janela", side_effect=RuntimeError("sem modelo")):
            job = self.c.post("/api/computador/testar").json()["job"]
            for _ in range(100):
                j = self.c.get(f"/api/jobs/{job}").json()
                if j["status"] != "processando":
                    break
                time.sleep(0.05)
        self.assertEqual(j["status"], "erro")
        self.assertIn("sem modelo", j["erro"])

    def test_origem_de_fora_nao_dispara_o_teste(self):
        r = self.c.post("/api/computador/testar", headers={"Origin": "https://exemplo.com"})
        self.assertEqual(r.status_code, 403)


class TelaTest(unittest.TestCase):
    def test_secao_e_funcoes_na_interface(self):
        for trecho in ('id="pc"', 'id="pc-corpo"', "carregarComputador", "testarComputador", "/api/computador/testar"):
            self.assertIn(trecho, HTML)

    def test_sem_emoji_nem_travessao_longo(self):
        for ch in ("—", "–"):
            self.assertNotIn(ch, (RAIZ / "transcritor" / "perfil_pc.py").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
