"""Provedor de IA (Gemini por padrão, Gateway só como reserva), com um servidor HTTP falso em 127.0.0.1. Sem rede, sem chave real.

    .venv\\Scripts\\python -m unittest testes.test_ia_gemini -v
"""
import json
import os
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("AVEIA_DADOS", tempfile.mkdtemp(prefix="aveia-dados-teste-"))
from transcritor import ia, locutores, resumo, titulo  # noqa: E402


class Falso:
    """Registra cada pedido e responde conforme `acoes` (uma por pedido; "ok" se acabarem)."""

    def __init__(self):
        self.pedidos: list[dict] = []
        self.acoes: list[str] = []
        self.texto = "Resposta de teste."
        gw = self

        class H(BaseHTTPRequestHandler):
            def do_POST(self):
                corpo = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                gw.pedidos.append({"caminho": self.path, "auth": self.headers.get("Authorization"), "corpo": corpo})
                acao = gw.acoes.pop(0) if gw.acoes else "ok"
                if acao.startswith("http:"):
                    self.send_response(int(acao[5:])); self.send_header("Content-Length", "0"); self.end_headers(); return
                dados = json.dumps({"choices": [{"message": {"content": gw.texto}}]}).encode()
                self.send_response(200); self.send_header("Content-Length", str(len(dados))); self.end_headers(); self.wfile.write(dados)

            def log_message(self, *a):
                pass

        self.srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.srv.server_address[1]}/v1beta/openai/chat/completions"
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()


def ambiente(**extra):
    """Ambiente limpo de chaves e de variáveis do Aveia, com o que o teste quiser por cima."""
    base = {k: v for k, v in os.environ.items() if not (k.startswith("AVEIA_MODELO") or k in (
        "GEMINI_API_KEY", "AI_GATEWAY_API_KEY", "VERCEL_OIDC_TOKEN", "AVEIA_IA_URL", "AVEIA_IA_RACIOCINIO", "AVEIA_IA_TETO_MES"))}
    base.update(extra)
    return mock.patch.dict(os.environ, base, clear=True)


class Escolha(unittest.TestCase):
    def test_sem_chave(self):
        with ambiente():
            self.assertIsNone(ia.provedor()); self.assertFalse(ia.configurada()); self.assertIsNone(ia.chave())
            self.assertIn("GEMINI_API_KEY", ia.SEM_CHAVE)
            with self.assertRaises(resumo.IANaoConfigurada) as c:
                resumo._chamar("s", "u", 10)
            self.assertIn("GEMINI_API_KEY", str(c.exception))

    def test_gemini_tem_prioridade_sobre_o_gateway(self):
        with ambiente(GEMINI_API_KEY="g", AI_GATEWAY_API_KEY="v"):
            self.assertEqual(ia.provedor(), "gemini"); self.assertEqual(ia.chave(), "g")
            self.assertEqual(ia.url(), ia.GEMINI_URL)
            self.assertEqual(ia.modelo("AVEIA_MODELO_IA", "gw/modelo"), "gemini-3.1-flash-lite")
            self.assertIn("Google", ia.aviso_privacidade()); self.assertEqual(ia.plano(), "pago")
            self.assertIn("Plano pago", ia.aviso_privacidade()); self.assertIn("não é usado", ia.aviso_privacidade())
        with ambiente(GEMINI_API_KEY="g", AVEIA_GEMINI_PLANO="gratuito"):
            self.assertEqual(ia.plano(), "gratuito"); self.assertIn("camada gratuita", ia.aviso_privacidade()); self.assertIn("faturamento", ia.aviso_privacidade())

    def test_gateway_so_sem_chave_do_gemini(self):
        with ambiente(AI_GATEWAY_API_KEY="v"):
            self.assertEqual(ia.provedor(), "gateway"); self.assertEqual(ia.url(), ia.GATEWAY_URL)
            self.assertEqual(ia.modelo("AVEIA_MODELO_IA", "gw/modelo"), "gw/modelo")
            self.assertEqual(ia.parametros(), {"reasoning": {"enabled": False}})
        with ambiente(VERCEL_OIDC_TOKEN="t"):
            self.assertEqual(ia.provedor(), "gateway")

    def test_modelos_e_raciocinio_sobrescrevem_por_variavel(self):
        with ambiente(GEMINI_API_KEY="g", AVEIA_MODELO_GEMINI="gemini-3.5-flash-lite", AVEIA_IA_RACIOCINIO="low"):
            self.assertEqual(ia.modelo(), "gemini-3.5-flash-lite"); self.assertEqual(ia.parametros(), {"reasoning_effort": "low"})
            self.assertEqual(resumo.modelo(), "gemini-3.5-flash-lite")
            self.assertEqual(titulo.modelo(), "gemini-3.5-flash-lite")
            self.assertEqual(locutores.modelo(), "gemini-3.5-flash-lite")
        with ambiente(GEMINI_API_KEY="g", AVEIA_MODELO_TITULO="outro", AVEIA_IA_RACIOCINIO="invalido"):
            self.assertEqual(titulo.modelo(), "outro"); self.assertEqual(resumo.modelo(), "gemini-3.1-flash-lite")
            self.assertEqual(ia.raciocinio(), "minimal")

    def test_chave_em_branco_conta_como_ausente(self):
        with ambiente(GEMINI_API_KEY="   ", AI_GATEWAY_API_KEY=""):
            self.assertIsNone(ia.provedor())


class Chamadas(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.gw = Falso()

    def setUp(self):
        self.gw.pedidos.clear(); self.gw.acoes.clear(); self.gw.texto = "Resposta de teste."
        self.tmp = tempfile.mkdtemp(prefix="aveia-ia-")

    def amb(self, **extra):
        return ambiente(GEMINI_API_KEY="chave-falsa", AVEIA_IA_URL=self.gw.url, AVEIA_DADOS=self.tmp, **extra)

    def test_resumo_titulo_e_locutores_mandam_o_mesmo_pedido_do_gemini(self):
        with self.amb(), mock.patch.object(resumo.time, "sleep"):
            self.assertEqual(resumo._chamar("sis", "usu", 50), "Resposta de teste.")
            self.assertEqual(locutores._chamar("sis", "usu", 50), "Resposta de teste.")
            self.assertEqual(titulo._chamar("sis", "usu", 50), "Resposta de teste.")
        self.assertEqual(len(self.gw.pedidos), 3)
        for p in self.gw.pedidos:
            self.assertEqual(p["auth"], "Bearer chave-falsa")
            c = p["corpo"]
            self.assertEqual(c["model"], "gemini-3.1-flash-lite")
            self.assertEqual(c["reasoning_effort"], "minimal")
            self.assertNotIn("reasoning", c)
            self.assertEqual([m["role"] for m in c["messages"]], ["system", "user"])

    def test_gateway_de_reserva_manda_reasoning_e_o_modelo_dele(self):
        with ambiente(AI_GATEWAY_API_KEY="v", AVEIA_IA_URL=self.gw.url, AVEIA_DADOS=self.tmp):
            resumo._chamar("sis", "usu", 50)
        c = self.gw.pedidos[0]["corpo"]
        self.assertEqual(self.gw.pedidos[0]["auth"], "Bearer v")
        self.assertEqual(c["model"], resumo.MODELO_PADRAO); self.assertEqual(c["reasoning"], {"enabled": False}); self.assertNotIn("reasoning_effort", c)

    def test_400_refaz_sem_os_parametros_opcionais(self):
        self.gw.acoes = ["http:400"]
        with self.amb(), mock.patch.object(resumo.time, "sleep"):
            resumo._chamar("sis", "usu", 50)
        self.assertEqual(len(self.gw.pedidos), 2)
        self.assertIn("reasoning_effort", self.gw.pedidos[0]["corpo"]); self.assertNotIn("reasoning_effort", self.gw.pedidos[1]["corpo"])
        self.gw.pedidos.clear(); self.gw.acoes = ["http:400"]
        with self.amb(), mock.patch.object(titulo.time, "sleep"):
            titulo._chamar("sis", "usu", 50)
        self.assertNotIn("reasoning_effort", self.gw.pedidos[1]["corpo"])
        self.gw.pedidos.clear(); self.gw.acoes = ["http:400"]
        with self.amb(), mock.patch.object(locutores.time, "sleep"):
            locutores._chamar("sis", "usu", 50)
        self.assertNotIn("reasoning_effort", self.gw.pedidos[1]["corpo"])

    def test_429_tenta_de_novo_e_depois_desiste_com_mensagem_clara(self):
        self.gw.acoes = ["http:429"]
        with self.amb(), mock.patch.object(resumo.time, "sleep") as dorme:
            self.assertEqual(resumo._chamar("sis", "usu", 50), "Resposta de teste.")
        self.assertEqual(len(self.gw.pedidos), 2); dorme.assert_called()
        self.gw.pedidos.clear(); self.gw.acoes = ["http:429"] * 5
        with self.amb(), mock.patch.object(resumo.time, "sleep"):
            with self.assertRaises(resumo.FalhaIA) as c:
                resumo._chamar("sis", "usu", 50)
        self.assertIn("429", str(c.exception))
        with self.amb(), mock.patch.object(titulo.time, "sleep"):
            self.gw.acoes = ["http:429"] * 5
            with self.assertRaises(titulo._Falha):
                titulo._chamar("sis", "usu", 50)

    def test_teto_mensal_barra_a_chamada_antes_de_sair(self):
        with self.amb(AVEIA_IA_TETO_MES="0"):
            with self.assertRaises(resumo.FalhaIA):
                resumo._chamar("sis", "usu", 50)
            with self.assertRaises(resumo.FalhaIA):
                locutores._chamar("sis", "usu", 50)
            with self.assertRaises(titulo._Falha):
                titulo._chamar("sis", "usu", 50)
        self.assertEqual(self.gw.pedidos, [])

    def test_cada_chamada_conta_no_teto(self):
        from transcritor import metricas
        with self.amb(AVEIA_IA_TETO_MES="2"):
            resumo._chamar("s", "u", 10); locutores._chamar("s", "u", 10)
            self.assertEqual(metricas.uso_ia()["chamadas"], 2)
            with self.assertRaises(resumo.FalhaIA):
                resumo._chamar("s", "u", 10)
        self.assertEqual(len(self.gw.pedidos), 2)

    def test_titulo_de_ponta_a_ponta_com_o_gemini_falso(self):
        self.gw.texto = "Revisão do orçamento de marketing"
        falas = [{"inicio": 0, "fim": 9, "locutor": "Locutor 1", "texto": "Vamos revisar o orçamento de marketing do quarto trimestre com o time."}]
        with self.amb():
            r = titulo.gerar_titulo([], falas)
        self.assertEqual((r.titulo, r.motivo, r.modelo), ("Revisão do orçamento de marketing", "", "gemini-3.1-flash-lite"))

    def test_resposta_inesperada_vira_falha(self):
        self.gw.texto = None
        with self.amb():
            self.assertEqual(resumo._chamar("s", "u", 10), "")   # conteúdo vazio: quem chama decide (resumo refaz com folga)


class Interface(unittest.TestCase):
    def test_privacidade_e_api_ia_dizem_para_onde_o_texto_vai(self):
        with ambiente(GEMINI_API_KEY="chave-falsa", AVEIA_DADOS=tempfile.mkdtemp(prefix="aveia-ia-"), AVEIA_SEM_AQUECER="1"):
            import app
            from fastapi.testclient import TestClient
            c = TestClient(app.app, base_url="http://127.0.0.1:8765")
            p = c.get("/api/privacidade").json()
            ia_rec = [r for r in p["recursos"] if r["id"] == "ia"]   # título, resumo e nomes: uma linha só, com interruptor
            self.assertEqual(len(ia_rec), 1)
            self.assertTrue(ia_rec[0]["interruptor"])
            self.assertTrue(all(r["para"] == "Google (Gemini)" and r["envia"] == "texto" for r in ia_rec))
            self.assertIn("Plano pago", p["ia_aviso"]); self.assertIn("Google", p["ia_aviso"])
            j = c.get("/api/ia").json()
            self.assertEqual((j["configurada"], j["provedor"], j["modelo"]), (True, "gemini", "gemini-3.1-flash-lite"))
            self.assertNotIn("chave-falsa", json.dumps(p) + json.dumps(j))
        with ambiente(AVEIA_DADOS=tempfile.mkdtemp(prefix="aveia-ia-")):
            r = c.post("/api/resumo", json={"titulo": "", "notas": [], "falas": [], "duracao": 1})
            self.assertEqual((r.status_code, r.json()["codigo"]), (503, "sem_chave"))
            self.assertIn("GEMINI_API_KEY", r.json()["erro"])


if __name__ == "__main__":
    unittest.main()
