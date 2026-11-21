"""Interruptor único da IA: desligado = ZERO chamadas ao provedor, para qualquer recurso; ligado (padrão) = tudo ativo.

    .venv\\Scripts\\python -m unittest testes.test_interruptor_ia -v

Servidor: `AVEIA_IA=0` com um provedor falso local que conta os pedidos. Navegador: a estrutura do `static/index.html` (a conferência
na tela fica na sessão de trabalho). Sem rede, sem chave real.
"""
import json
import os
import re
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
os.environ.setdefault("AVEIA_DADOS", tempfile.mkdtemp(prefix="aveia-dados-teste-"))
HTML = (RAIZ / "static" / "index.html").read_text(encoding="utf-8")
from transcritor import ia, locutores, resumo, titulo  # noqa: E402

FALAS = [{"inicio": 0, "fim": 9, "locutor": "Locutor 1", "texto": "Bom dia, aqui é o Marcos. Vamos revisar o orçamento de marketing do quarto trimestre."},
         {"inicio": 10, "fim": 15, "locutor": "Locutor 2", "texto": "Bom dia, Marcos. Tenho os números prontos."}]


class Provedor:
    def __init__(self):
        self.pedidos = 0
        p = self

        class H(BaseHTTPRequestHandler):
            def do_POST(self):
                self.rfile.read(int(self.headers["Content-Length"]))
                p.pedidos += 1
                dados = json.dumps({"choices": [{"message": {"content": '{"indices": [1]}'}}]}).encode()
                self.send_response(200); self.send_header("Content-Length", str(len(dados))); self.end_headers(); self.wfile.write(dados)

            def log_message(self, *a):
                pass

        self.srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.srv.server_address[1]}/v1beta/openai/chat/completions"
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()


class Servidor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prov = Provedor()

    def amb(self, **extra):
        base = {k: v for k, v in os.environ.items() if k not in ("GEMINI_API_KEY", "AI_GATEWAY_API_KEY", "VERCEL_OIDC_TOKEN", "AVEIA_IA", "AVEIA_IA_TETO_MES")}
        base.update(GEMINI_API_KEY="chave-falsa", AI_GATEWAY_API_KEY="outra-falsa", AVEIA_IA_URL=self.prov.url,
                    AVEIA_DADOS=tempfile.mkdtemp(prefix="aveia-i-"), AVEIA_SEM_AQUECER="1", **extra)
        return mock.patch.dict(os.environ, base, clear=True)

    def cliente(self):
        import app
        from fastapi.testclient import TestClient
        return TestClient(app.app, base_url="http://127.0.0.1:8765")

    def test_padrao_e_ligada(self):
        with self.amb():
            self.assertTrue(ia.ligada()); self.assertTrue(ia.configurada()); self.assertEqual(ia.provedor(), "gemini")
            self.assertTrue(self.cliente().get("/api/ia").json()["ligada"])

    def test_desligada_nao_chama_o_provedor_em_nenhum_recurso(self):
        for valor in ("0", "false", "off", "nao", "OFF"):
            with self.amb(AVEIA_IA=valor):
                antes = self.prov.pedidos
                c = self.cliente()
                self.assertFalse(ia.ligada()); self.assertFalse(ia.configurada()); self.assertIsNone(ia.provedor()); self.assertIsNone(ia.chave())
                j = c.get("/api/ia").json()
                self.assertEqual((j["configurada"], j["ligada"], j["provedor"]), (False, False, None))
                # resumo
                r = c.post("/api/resumo", json={"titulo": "", "notas": [], "falas": FALAS, "duracao": 15})
                self.assertEqual((r.status_code, r.json()["codigo"]), (503, "desligada"))
                # título
                r = c.post("/api/titulo", json={"notas": [], "falas": FALAS})
                self.assertEqual((r.status_code, r.json()["titulo"]), (200, None))
                # nomes dos locutores
                r = c.post("/api/locutores", json={"titulo": "", "notas": [], "falas": FALAS})
                self.assertEqual(r.json()["sugestoes"], {})
                # trecho do destaque
                r = c.post("/api/destaque-escopo", json={"destaque": "orçamento", "candidatas": [{"i": 1, "texto": "Vamos revisar o orçamento."}]}).json()
                self.assertEqual((r["modo"], r["motivo"]), ("regra", "desligada"))
                # chamadas diretas aos módulos
                with self.assertRaises(resumo.IANaoConfigurada):
                    resumo._chamar("s", "u", 10)
                with self.assertRaises(locutores.IANaoConfigurada):
                    locutores._chamar("s", "u", 10)
                self.assertEqual(titulo.gerar_titulo([], FALAS).motivo, "sem_chave")
                with self.assertRaises(ia.Falha) as e:
                    ia.chamar("destaques", "s", "u", 10)
                self.assertEqual(e.exception.motivo, "desligada")
                self.assertEqual(self.prov.pedidos, antes, f"AVEIA_IA={valor}: nenhum pedido pode sair")

    def test_ligada_chama_o_provedor(self):
        with self.amb():
            antes = self.prov.pedidos
            r = self.cliente().post("/api/destaque-escopo", json={"destaque": "orçamento", "candidatas": [{"i": 1, "texto": "Vamos revisar o orçamento."}]}).json()
            self.assertEqual((r["modo"], r["indices"]), ("ia", [1]))
            self.assertEqual(self.prov.pedidos, antes + 1)

    def test_desligada_tambem_desliga_a_reserva_do_gateway(self):
        with self.amb(AVEIA_IA="0"):
            os.environ.pop("GEMINI_API_KEY")
            self.assertIsNone(ia.provedor())   # há AI_GATEWAY_API_KEY, mas a IA está desligada

    def test_aviso_de_privacidade_quando_desligada(self):
        with self.amb(AVEIA_IA="0"):
            self.assertIn("desligada", ia.aviso_privacidade()); self.assertIn("nenhum texto sai", ia.aviso_privacidade())
            self.assertIn("AVEIA_IA=0", ia.sem_ia())


class Navegador(unittest.TestCase):
    def test_padrao_ligado_e_sem_ia_automatica_separada(self):
        self.assertRegex(HTML, r"const PADRAO = \{[^}]*\bia: true\b")
        self.assertNotRegex(HTML, r"\biaAuto\b.*:.*(false|true)", "a flag antiga iaAuto não existe mais")
        self.assertIn("function iaLigada() { return state.cfg.ia !== false; }", HTML)

    def test_todas_as_portas_da_ia_olham_o_interruptor(self):
        corpo = lambda nome: HTML[HTML.index(f"async function {nome}("):][:700]
        self.assertIn("if (!iaLigada()) return aviso(", corpo("gerarResumo"))
        self.assertIn("!iaLigada()", corpo("identificarLocutores"))
        self.assertIn("!iaLigada()", corpo("gerarTitulo"))
        self.assertIn("if (!iaPode()) return;", corpo("resolverDestaquesIA"))
        self.assertIn("const iaAutomatica = iaPode();", HTML)

    def test_todo_fetch_de_ia_esta_atras_de_uma_porta(self):
        # Os quatro endereços de IA só são chamados dentro das funções protegidas acima.
        donos = {"/api/resumo": "gerarResumo", "/api/titulo": "gerarTitulo", "/api/locutores": "identificarLocutores", "/api/destaque-escopo": "resolverDestaquesIA"}
        for url, dono in donos.items():
            usos = [m.start() for m in re.finditer(re.escape(f"fetch('{url}'"), HTML)]
            self.assertEqual(len(usos), 1, url)
            inicio = HTML.rindex(f"async function {dono}(", 0, usos[0])
            self.assertLess(usos[0] - inicio, 4500, f"{url} deve ficar dentro de {dono}")

    def test_aviso_de_privacidade_so_nos_ajustes_nunca_solto_na_tela(self):
        self.assertIn('id="iaauto-aviso"', HTML)   # dentro do painel de Ajustes
        ajustes = HTML[HTML.index('<header'):HTML.index("<!-- Gravações que não terminaram")]
        self.assertIn('id="iaauto-aviso"', ajustes, "o aviso fica no painel de Ajustes (ícone de configurações)")
        for banido in ('id="ia-aviso"', "mostrarAvisoIA", "avisoIAVisto", "desligarIADoAviso", "entendiAvisoIA"):
            self.assertNotIn(banido, HTML, "nenhum banner de aviso aparece sozinho")

    def test_botao_gerar_fica_inativo_com_a_ia_desligada(self):
        self.assertIn("btn.disabled = gerando || !iaLigada();", HTML)


if __name__ == "__main__":
    unittest.main()
