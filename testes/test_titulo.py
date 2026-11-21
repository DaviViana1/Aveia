"""Testes offline do título automático (`transcritor/titulo.py` e `/api/titulo`), com um gateway falso local.

    .venv\\Scripts\\python -m unittest testes.test_titulo -v

Não usam a chave real nem a internet: `AVEIA_IA_URL` aponta para um servidor HTTP em 127.0.0.1.
"""
import json
import os
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("AVEIA_DADOS", __import__("tempfile").mkdtemp(prefix="aveia-dados-teste-"))   # métricas dos testes não vão para dados/
from transcritor import titulo  # noqa: E402


class Gateway:
    """Servidor falso: cada requisição consome a próxima ação da fila ('ok:texto', 'http:500', 'vazio', 'dorme')."""

    def __init__(self):
        self.acoes: list[str] = []
        self.pedidos: list[dict] = []
        gw = self

        class H(BaseHTTPRequestHandler):
            def do_POST(self):
                corpo = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                gw.pedidos.append(corpo)
                acao = gw.acoes.pop(0) if gw.acoes else "ok:Revisão do orçamento de marketing"
                if acao.startswith("http:"):
                    self.send_response(int(acao[5:])); self.send_header("Retry-After", "0"); self.end_headers(); self.wfile.write(b"{}"); return
                if acao == "dorme":
                    import time; time.sleep(3)
                if acao == "lixo":
                    self.send_response(200); self.end_headers(); self.wfile.write(b"nao e json"); return
                texto = "" if acao in ("vazio", "dorme") else acao[3:]
                self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers()
                self.wfile.write(json.dumps({"choices": [{"message": {"content": texto}}]}).encode())

            def log_message(self, *a):
                pass

        self.srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.srv.handle_error = lambda *a: None   # o cliente que estoura o timeout fecha a conexão: não é erro do teste
        self.url = f"http://127.0.0.1:{self.srv.server_port}/v1/chat/completions"
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()


FALAS = [{"inicio": 0, "fim": 9, "locutor": "Locutor 1",
          "texto": "Vamos rever o orçamento de marketing do quarto trimestre e decidir a verba de mídia paga."},
         {"inicio": 9, "fim": 18, "locutor": "Locutor 2", "texto": "Eu proponho mover quarenta mil de eventos para mídia paga."}]


class Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.gw = Gateway()

    def setUp(self):
        self._env = dict(os.environ)
        os.environ.pop("GEMINI_API_KEY", None)   # estes testes usam o Gateway falso (reserva); o provedor padrão é o Gemini
        os.environ.update(AI_GATEWAY_API_KEY="chave-falsa", AVEIA_IA_URL=self.gw.url)
        os.environ.pop("AVEIA_MODELO_TITULO", None)
        self.gw.acoes.clear(); self.gw.pedidos.clear()
        self._timeout = titulo.TIMEOUT

    def tearDown(self):
        titulo.TIMEOUT = self._timeout
        os.environ.clear(); os.environ.update(self._env)


class TestFormato(unittest.TestCase):
    def test_limpar(self):
        casos = {
            '"Revisão do orçamento de marketing."': "Revisão do orçamento de marketing",
            "Título: Plano de migração do data center": "Plano de migração do data center",
            "**Roadmap do app móvel**\nporque ...": "Roadmap do app móvel",
            "📊 Resultados de vendas da região norte!": "Resultados de vendas da região norte",
            "  “Acordo sobre dinheiro com o pai”  ": "Acordo sobre dinheiro com o pai",
            "- Entrevista para vaga de back-end": "Entrevista para vaga de back-end",
            "": "",
        }
        for bruto, esperado in casos.items():
            self.assertEqual(titulo.limpar(bruto), esperado, bruto)

    def test_validar_aceita(self):
        for t in ["Revisão do orçamento de marketing do 4º trimestre", "Bug do login com Google em produção",
                  "Roteiro do episódio 12 sobre trabalho remoto", "Contrato de manutenção da ACME"]:
            self.assertEqual(titulo.validar(t), "", t)

    def test_validar_recusa(self):
        casos = {"": "vazio", "Reunião": "generico", "Nota de voz": "generico", "Gravação": "generico", "Nova nota": "generico",
                 "Teste": "generico", "Orçamento": "tamanho", "Reunião de equipe": "generico",
                 "Revisão do orçamento de marketing do quarto trimestre com a diretoria financeira": "longo",
                 "Reunião de 5 de novembro sobre orçamento": "data", "Orçamento de marketing 05/11/2026": "data",
                 "Gravação das 15:39 sobre vendas": "data", "Weekly meeting about the marketing budget": "idioma",
                 "Project review and planning": "idioma", "Um dois três quatro cinco seis sete oito nove dez": "tamanho"}
        for t, motivo in casos.items():
            self.assertEqual(titulo.validar(t), motivo, t)

    def test_entrada_limitada(self):
        longas = [{"inicio": i, "fim": i + 1, "locutor": "Maria", "texto": "palavra " * 30} for i in range(2000)]
        pedido = titulo.montar_pedido(["a" * 5000], longas)
        self.assertLess(len(pedido), titulo.ENTRADA_INICIO + titulo.ENTRADA_FIM + titulo.ENTRADA_NOTAS + 200)
        self.assertIn("[...]", pedido)
        self.assertIn("Maria:", pedido)
        self.assertNotIn("Locutor 1:", titulo.montar_pedido([], [{"inicio": 0, "fim": 1, "locutor": "Locutor 1", "texto": "oi tudo bem"}]))


class TestGerar(Base):
    def test_sem_chave(self):
        os.environ.pop("AI_GATEWAY_API_KEY", None); os.environ.pop("VERCEL_OIDC_TOKEN", None)
        r = titulo.gerar_titulo([], FALAS)
        self.assertEqual((r.titulo, r.motivo), (None, "sem_chave"))
        self.assertEqual(self.gw.pedidos, [])

    def test_ok(self):
        self.gw.acoes = ['ok:"Revisão do orçamento de marketing."']
        r = titulo.gerar_titulo([], FALAS)
        self.assertEqual((r.titulo, r.motivo), ("Revisão do orçamento de marketing", ""))
        self.assertEqual(len(self.gw.pedidos), 1)

    def test_curto_nao_chama_gateway(self):
        r = titulo.gerar_titulo([], [{"inicio": 0, "fim": 1, "locutor": "Locutor 1", "texto": "Alô teste"}])
        self.assertEqual(r.motivo, "curto"); self.assertEqual(self.gw.pedidos, [])
        r = titulo.gerar_titulo(["Contrato ACME"], [{"inicio": 0, "fim": 1, "locutor": "Locutor 1", "texto": "Alô"}])
        self.assertEqual(r.motivo, "")   # anotações bastam para tentar

    def test_gateway_fora(self):
        self.gw.acoes = ["http:500"] * 5
        r = titulo.gerar_titulo([], FALAS)
        self.assertEqual((r.titulo, r.motivo), (None, "falha"))

    def test_429_depois_ok(self):
        self.gw.acoes = ["http:429", "ok:Plano de migração para a nuvem"]
        self.assertEqual(titulo.gerar_titulo([], FALAS).titulo, "Plano de migração para a nuvem")

    def test_erro_4xx_nao_insiste(self):
        self.gw.acoes = ["http:403"]
        r = titulo.gerar_titulo([], FALAS)
        self.assertEqual(r.motivo, "falha"); self.assertEqual(len(self.gw.pedidos), 1)

    def test_resposta_vazia_tenta_com_mais_folga(self):
        self.gw.acoes = ["vazio", "ok:Orçamento de marketing do trimestre"]
        r = titulo.gerar_titulo([], FALAS)
        self.assertEqual(r.titulo, "Orçamento de marketing do trimestre")
        self.assertGreater(self.gw.pedidos[1]["max_tokens"], self.gw.pedidos[0]["max_tokens"])

    def test_resposta_ruim_refaz_uma_vez(self):
        self.gw.acoes = ["ok:Reunião", "ok:Revisão do orçamento de marketing"]
        r = titulo.gerar_titulo([], FALAS)
        self.assertEqual(r.titulo, "Revisão do orçamento de marketing")
        self.assertIn("recusada", self.gw.pedidos[1]["messages"][1]["content"])

    def test_duas_respostas_ruins_desiste(self):
        self.gw.acoes = ["ok:Reunião", "ok:Weekly meeting about the marketing budget"]
        r = titulo.gerar_titulo([], FALAS)
        self.assertEqual(r.titulo, None); self.assertTrue(r.motivo.startswith("invalido"))

    def test_sem_assunto(self):
        self.gw.acoes = ["ok:SEM_TITULO"]
        self.assertEqual(titulo.gerar_titulo([], FALAS).motivo, "sem_assunto")

    def test_json_invalido_e_timeout(self):
        self.gw.acoes = ["lixo"] * 3
        self.assertEqual(titulo.gerar_titulo([], FALAS).motivo, "falha")
        titulo.TIMEOUT = 1
        self.gw.acoes = ["dorme"] * 3
        self.assertEqual(titulo.gerar_titulo([], FALAS).motivo, "falha")

    def test_modelo_configuravel(self):
        os.environ["AVEIA_MODELO_TITULO"] = "mistral/mistral-small"
        titulo.gerar_titulo([], FALAS)
        self.assertEqual(self.gw.pedidos[0]["model"], "mistral/mistral-small")


class TestApi(Base):
    def test_endpoint(self):
        from fastapi.testclient import TestClient
        import app as servidor
        c = TestClient(servidor.app, base_url="http://127.0.0.1:8765")
        corpo = {"notas": [], "falas": FALAS}
        self.gw.acoes = ["ok:Revisão do orçamento de marketing"]
        r = c.post("/api/titulo", json=corpo)
        self.assertEqual((r.status_code, r.json()["titulo"]), (200, "Revisão do orçamento de marketing"))
        self.gw.acoes = ["http:500"] * 5
        r = c.post("/api/titulo", json=corpo)
        self.assertEqual((r.status_code, r.json()["titulo"], r.json()["motivo"]), (200, None, "falha"))
        os.environ.pop("AI_GATEWAY_API_KEY", None); os.environ.pop("VERCEL_OIDC_TOKEN", None)
        r = c.post("/api/titulo", json=corpo)
        self.assertEqual((r.status_code, r.json()["titulo"], r.json()["motivo"]), (200, None, "sem_chave"))


if __name__ == "__main__":
    unittest.main()
