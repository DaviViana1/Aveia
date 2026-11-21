"""Testes offline dos "Destaques" (mensagens do usuário durante a gravação), com um gateway de IA falso local.

    .venv\\Scripts\\python -m unittest testes.test_destaques -v

Cobrem: posição dos destaques na transcrição, invariante "sem destaques = prompts de sempre", `gerar_resumo`
(chamada única, blocos, blocos com grupos, falhas), formatos de `arquivos.py` e o `/api/resumo`.
Não usam a chave real nem a internet: `AVEIA_IA_URL` aponta para um servidor HTTP em 127.0.0.1.
"""
import json
import os
import re
import sys
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("AVEIA_DADOS", __import__("tempfile").mkdtemp(prefix="aveia-dados-teste-"))   # métricas dos testes não vão para dados/
from transcritor import arquivos, resumo  # noqa: E402


class Gateway:
    """Modelo falso. Responde conforme a etapa (reconhecida pelo prompt de sistema) e registra todos os pedidos.

    `omitir`: números de destaque que o modelo "esquece" de devolver. `falhar_trecho`: nº do trecho que dá erro 403.
    """

    def __init__(self):
        self.pedidos: list[dict] = []
        self.omitir: set[int] = set()
        self.falhar_trecho: int | None = None
        self.lixo_nos_destaques = False
        self.trava = threading.Lock()
        gw = self

        class H(BaseHTTPRequestHandler):
            def do_POST(self):
                corpo = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                sistema, usuario = corpo["messages"][0]["content"], corpo["messages"][1]["content"]
                with gw.trava:
                    gw.pedidos.append({"sistema": sistema, "usuario": usuario})
                if gw.falhar_trecho and "UM TRECHO" in sistema and f"Trecho {gw.falhar_trecho} de" in usuario:
                    self.send_response(403); self.end_headers(); self.wfile.write(b"{}"); return
                texto = gw.responder(sistema, usuario)
                self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers()
                self.wfile.write(json.dumps({"choices": [{"message": {"content": texto}}]}).encode())

            def log_message(self, *a):
                pass

        self.srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.srv.handle_error = lambda *a: None
        self.url = f"http://127.0.0.1:{self.srv.server_port}/v1/chat/completions"
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    def responder(self, sistema: str, usuario: str) -> str:
        if "Segundo passo" in sistema:
            return "<resumo>\nNarrativa de teste.\n</resumo>\n<pontos>\n- Ponto um.\n- Ponto dois.\n</pontos>"
        if "Terceiro passo" in sistema:
            return "<resumo>\nNarrativa revisada de teste.\n</resumo>\n<pontos>\n- Ponto um.\n- Ponto dois.\n</pontos>"
        d: dict = {"qualidade": "boa", "tema": "Tema de teste", "fio": ["Primeiro.", "Segundo."], "fatos": [], "decisoes": [],
                   "acoes": [], "titulo": "Trecho de teste", "sintese": "Síntese.", "relato": "Relato do trecho."}
        if '"destaques"' in sistema:   # etapas plano e bloco com destaques
            ns = [int(n) for n in re.findall(r"DESTAQUE (\d+) DO USUÁRIO", usuario)]
            d["destaques"] = [{"n": n, "texto": f"Frase do modelo para o destaque {n}."} for n in ns if n not in self.omitir]
            if self.lixo_nos_destaques:
                d["destaques"] += ["lixo", {"n": "x", "texto": "sem número"}, {"n": 99, "texto": "número que não existe"}, {"n": 1}]
        return json.dumps(d)


def fala(inicio, texto="Fala de teste.", locutor="Ana", dur=5):
    return {"inicio": inicio, "fim": inicio + dur, "locutor": locutor, "texto": texto}


FALAS = [fala(0, "Vamos falar da integração com a API de pagamentos.", "Ana"),
         fala(30, "A segurança já aprovou a integração, com a condição de usar tokens de curta duração.", "Carlos"),
         fala(60, "Então fechamos o prazo para o dia 12.", "Ana")]

# Gravação de ~2 h 30 min espalhada em 3 blocos de 30 min (perfil "blocos"): 0–630 s, 1810–2430 s, 3610–4230 s.
FALAS_LONGAS = [fala(10, dur=20), fala(600, dur=30), fala(1810, dur=20), fala(2400, dur=30), fala(3610, dur=20), fala(4200, dur=30)]


class Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.gw = Gateway()

    def setUp(self):
        self._env = dict(os.environ)
        os.environ.pop("GEMINI_API_KEY", None)   # estes testes usam o Gateway falso (reserva); o provedor padrão é o Gemini
        os.environ.update(AI_GATEWAY_API_KEY="chave-falsa", AVEIA_IA_URL=self.gw.url)
        os.environ.pop("AVEIA_RESUMO_REVISAO", None)
        self.gw.pedidos.clear(); self.gw.omitir = set(); self.gw.falhar_trecho = None; self.gw.lixo_nos_destaques = False

    def tearDown(self):
        os.environ.clear(); os.environ.update(self._env)

    def prompts(self, etapa: str) -> list[dict]:
        chave = {"plano": "Primeiro passo: leia", "bloco": "UM TRECHO", "grupo": "Una-as em uma só ficha",
                 "plano_final": "fichas, em ordem cronológica", "redacao": "Segundo passo", "revisao": "Terceiro passo"}[etapa]
        return [p for p in self.gw.pedidos if chave in p["sistema"]]


class TestIntercalar(unittest.TestCase):
    def linhas(self, falas, destaques):
        marcas = resumo._sanear_destaques(destaques)
        return resumo._transcricao(falas, marcas).split("\n")

    def test_antes_da_primeira(self):
        l = self.linhas([fala(10), fala(40)], [{"t": -3, "texto": "início"}])   # t negativo vira 0
        self.assertEqual(l[0], '>>> DESTAQUE 1 DO USUÁRIO [00:00]: "início"')
        self.assertTrue(l[1].startswith("[00:10]"))
        l = self.linhas([fala(10)], [{"t": 4, "texto": "antes"}])
        self.assertEqual(l[0], '>>> DESTAQUE 1 DO USUÁRIO [00:04]: "antes"')

    def test_no_meio(self):
        l = self.linhas(FALAS, [{"t": 40, "texto": "Ponto importante"}])
        self.assertEqual(len(l), 4)
        self.assertTrue(l[1].startswith("[00:30] Carlos:"))
        self.assertEqual(l[2], '>>> DESTAQUE 1 DO USUÁRIO [00:40]: "Ponto importante"')
        self.assertTrue(l[3].startswith("[01:00] Ana:"))

    def test_depois_da_ultima(self):
        l = self.linhas(FALAS, [{"t": 500, "texto": "fim"}])
        self.assertEqual(l[-1], '>>> DESTAQUE 1 DO USUÁRIO [08:20]: "fim"')
        self.assertEqual(len(l), 4)

    def test_empate_com_o_inicio_de_uma_fala(self):
        # a regra é "antes da primeira fala com inicio > t": em t == inicio a marca fica DEPOIS dessa fala
        l = self.linhas(FALAS, [{"t": 30, "texto": "empate"}])
        self.assertTrue(l[1].startswith("[00:30] Carlos:"))
        self.assertIn("DESTAQUE 1", l[2])

    def test_mesmo_instante_mantem_a_ordem_de_envio(self):
        l = self.linhas(FALAS, [{"t": 40, "texto": "primeira"}, {"t": 40, "texto": "segunda"}, {"t": 10, "texto": "antes"}])
        marcas = [x for x in l if x.startswith(">>>")]
        self.assertEqual([m.split(":")[-1].strip() for m in marcas], ['"antes"', '"primeira"', '"segunda"'])
        self.assertIn("DESTAQUE 1 ", marcas[0]); self.assertIn("DESTAQUE 3 ", marcas[2])

    def test_sanear(self):
        longo = "x" * 400
        m = resumo._sanear_destaques([{"t": 5, "texto": "  com   espaços \n repetidos "}, {"t": 1, "texto": "   "}, {"t": "abc", "texto": "t ruim"},
                                      {"t": 2, "texto": longo}, "lixo", {"t": 3}])
        self.assertEqual([(x["n"], x["t"]) for x in m], [(1, 2.0), (2, 5.0)])
        self.assertEqual(len(m[0]["texto"]), 300)
        self.assertEqual(m[1]["texto"], "com espaços repetidos")
        muitas = resumo._sanear_destaques([{"t": i, "texto": f"m{i}"} for i in range(100, 0, -1)])
        self.assertEqual((len(muitas), muitas[0]["n"], muitas[0]["t"], muitas[-1]["n"]), (60, 1, 1.0, 60))
        self.assertEqual(resumo._sanear_destaques(None), []); self.assertEqual(resumo._sanear_destaques([]), [])


class TestSemDestaques(Base):
    def test_transcricao_igual_a_de_sempre(self):
        antigo = "\n".join(f"[{resumo._tempo(f['inicio'])}] {f['locutor']}: {f['texto']}" for f in FALAS)
        self.assertEqual(resumo._transcricao(FALAS), antigo)
        self.assertEqual(resumo._transcricao(FALAS, []), antigo)
        self.assertEqual(resumo.montar_pedido("T", ["nota"], FALAS), "Título: T\n\nAnotações do usuário (têm prioridade):\n- nota\n\nTranscrição:\n" + antigo)
        self.assertEqual(resumo._cabecalho("T", ["n"], []), resumo._cabecalho("T", ["n"]))

    def test_prompts_de_sistema(self):
        for p in resumo.PERFIS:
            for etapa in ("plano", "bloco", "grupo", "plano_final"):
                self.assertEqual(resumo._sistema(etapa, p), resumo._sistema(etapa, p, False))
                self.assertNotIn("estaque", resumo._sistema(etapa, p), etapa)
            for longa in (False, True):
                self.assertNotIn("estaque", resumo._sistema_redacao(p, longa))
                self.assertNotIn("estaque", resumo._sistema_revisao(p, longa))

    def test_nenhuma_chamada_menciona_destaque(self):
        for destaques in (None, [], [{"t": 3, "texto": "   "}]):
            self.gw.pedidos.clear()
            r = resumo.gerar_resumo("Reunião", ["nota"], FALAS, 90, destaques=destaques)
            self.assertEqual(r["destaques"], [])
            self.assertEqual(len(self.gw.pedidos), 3)   # plano, redação, revisão
            for p in self.gw.pedidos:
                self.assertNotIn("estaque", p["sistema"]); self.assertNotIn("DESTAQUE", p["usuario"]); self.assertNotIn("estaque", p["usuario"])
        r = resumo.gerar_resumo("Reunião", [], FALAS_LONGAS, 9000)   # blocos
        self.assertEqual(r["destaques"], [])
        for p in self.gw.pedidos:
            self.assertNotIn("estaque", p["sistema"]); self.assertNotIn("estaque", p["usuario"])


class TestPrompts(unittest.TestCase):
    def test_regras_e_campos(self):
        p = resumo.escolher_perfil(60)
        for etapa in ("plano", "bloco", "grupo", "plano_final"):
            s = resumo._sistema(etapa, p, True)
            self.assertIn("Destaques do usuário (prioridade máxima)", s)
            self.assertIn("NÃO foi dito na reunião", s)
            self.assertIn("nunca o atribua a um participante", s)
            self.assertEqual('"destaques": lista com um objeto' in s, etapa in ("plano", "bloco"), etapa)
        for longa in (False, True):
            for s in (resumo._sistema_redacao(p, longa, True), resumo._sistema_revisao(p, longa, True)):
                self.assertIn("sem expressões como", s)
                self.assertIn("NÃO devem repetir o que a seção Destaques já diz", s)

    def test_cabecalho_e_ficha(self):
        marcas = resumo._sanear_destaques([{"t": 125, "texto": "Ponto importante"}])
        cab = resumo._cabecalho("T", [], marcas)
        self.assertEqual(cab[-1], 'Destaques marcados pelo usuário durante a gravação (prioridade máxima):\n- Destaque 1 [02:05]: "Ponto importante"')
        ficha = resumo._secao("00:00–10:00", {"titulo": "X", "relato": "Y", "destaques": [{"n": 1, "texto": "Z"}, {"n": 2, "texto": ""}]})
        self.assertIn("Destaques do usuário: Destaque 1: Z", ficha)
        self.assertNotIn("Destaque 2", ficha)
        self.assertNotIn("Destaques do usuário", resumo._secao("00:00–10:00", {"titulo": "X", "relato": "Y"}))


class TestGerarChamadaUnica(Base):
    DESTAQUES = [{"t": 45, "texto": "Ponto importante"}, {"t": 70, "texto": "confirmar esse prazo"}, {"t": 5, "texto": "tokens"}]

    def test_uma_entrada_por_marca_em_ordem_de_tempo(self):
        r = resumo.gerar_resumo("Reunião", [], FALAS, 90, destaques=self.DESTAQUES)
        self.assertEqual([(d["t"], d["marca"]) for d in r["destaques"]], [(5.0, "tokens"), (45.0, "Ponto importante"), (70.0, "confirmar esse prazo")])
        self.assertEqual([d["texto"] for d in r["destaques"]], [f"Frase do modelo para o destaque {n}." for n in (1, 2, 3)])
        plano = self.prompts("plano")[0]
        self.assertIn('>>> DESTAQUE 2 DO USUÁRIO [00:45]: "Ponto importante"', plano["usuario"])
        self.assertIn('- Destaque 3 [01:10]: "confirmar esse prazo"', plano["usuario"])
        self.assertIn('"destaques"', plano["sistema"])
        # a linha do meio fica entre as falas de 30 s e 60 s
        t = plano["usuario"]
        self.assertTrue(t.index("[00:30] Carlos") < t.index("DESTAQUE 2 DO USUÁRIO [00:45]") < t.index("[01:00] Ana"))
        for etapa in ("redacao", "revisao"):
            (p,) = self.prompts(etapa)
            self.assertIn("Destaques do usuário (prioridade máxima)", p["sistema"])
            self.assertIn("Destaques marcados pelo usuário durante a gravação", p["usuario"])
            self.assertIn("DESTAQUE 2 DO USUÁRIO", p["usuario"])
        self.assertIn("- Destaque 2: Frase do modelo para o destaque 2.", self.prompts("redacao")[0]["usuario"])

    def test_modelo_omitiu_uma(self):
        self.gw.omitir = {2}
        r = resumo.gerar_resumo("Reunião", [], FALAS, 90, destaques=self.DESTAQUES)
        self.assertEqual(len(r["destaques"]), 3)
        self.assertEqual([d["texto"] for d in r["destaques"]], ["Frase do modelo para o destaque 1.", "", "Frase do modelo para o destaque 3."])
        self.assertEqual(r["destaques"][1]["marca"], "Ponto importante")

    def test_modelo_devolve_lixo(self):
        self.gw.lixo_nos_destaques = True
        r = resumo.gerar_resumo("Reunião", [], FALAS, 90, destaques=self.DESTAQUES[:1])
        self.assertEqual(len(r["destaques"]), 1)   # "lixo", n inválido, n inexistente e item sem texto não viram entradas
        self.assertEqual(r["destaques"][0]["marca"], "Ponto importante")

    def test_resposta_sem_o_campo(self):
        self.gw.omitir = {1, 2, 3}
        r = resumo.gerar_resumo("Reunião", [], FALAS, 90, destaques=self.DESTAQUES)
        self.assertEqual([d["texto"] for d in r["destaques"]], ["", "", ""])


class TestGerarBlocos(Base):
    # blocos: [10–630] [1810–2430] [3610–4230]
    DESTAQUES = [{"t": 620, "texto": "no fim do bloco 1"}, {"t": 2000, "texto": "no bloco 2"}, {"t": 3000, "texto": "na lacuna, mais perto do bloco 2"},
                 {"t": 9000, "texto": "depois de tudo"}, {"t": 5, "texto": "antes de tudo"}]

    def test_marcas_por_bloco(self):
        blocos = resumo._blocos(FALAS_LONGAS, 30)
        self.assertEqual(len(blocos), 3)
        por = resumo._marcas_por_bloco(blocos, resumo._sanear_destaques(self.DESTAQUES))
        self.assertEqual([[m["texto"] for m in b] for b in por],
                         [["antes de tudo", "no fim do bloco 1"], ["no bloco 2", "na lacuna, mais perto do bloco 2"], ["depois de tudo"]])

    def test_blocos_recebem_so_as_suas_marcas_e_todas_voltam(self):
        r = resumo.gerar_resumo("Reunião", [], FALAS_LONGAS, 9000, destaques=self.DESTAQUES)
        self.assertEqual(r["perfil"]["id"], "blocos")
        blocos = sorted(self.prompts("bloco"), key=lambda p: int(re.search(r"Trecho (\d+) de", p["usuario"]).group(1)))
        self.assertEqual(len(blocos), 3)
        esperado = [[1, 2], [3, 4], [5]]   # números na ordem de tempo: 5 s, 620 s, 2000 s, 3000 s, 9000 s
        for b, ns in zip(blocos, esperado):
            achados = [int(n) for n in re.findall(r"DESTAQUE (\d+) DO USUÁRIO", b["usuario"])]
            self.assertEqual(achados, ns)
            self.assertIn('"destaques"', b["sistema"])
        # a de 620 s entra depois da fala de 600 s (última do bloco 1); a de 9000 s vai para o fim do último bloco
        self.assertTrue(blocos[0]["usuario"].rstrip().endswith('"no fim do bloco 1"'))
        self.assertTrue(blocos[2]["usuario"].rstrip().endswith('"depois de tudo"'))
        self.assertEqual([d["marca"] for d in r["destaques"]],
                         ["antes de tudo", "no fim do bloco 1", "no bloco 2", "na lacuna, mais perto do bloco 2", "depois de tudo"])
        self.assertEqual([d["texto"] for d in r["destaques"]], [f"Frase do modelo para o destaque {n}." for n in range(1, 6)])
        (plano_final,) = self.prompts("plano_final")
        self.assertIn("Destaque 3: Frase do modelo para o destaque 3.", plano_final["usuario"])
        self.assertIn("Destaques marcados pelo usuário durante a gravação", plano_final["usuario"])
        self.assertNotIn('"destaques"', plano_final["sistema"])   # não são re-derivados no plano final
        self.assertIn("Destaque 2: Frase do modelo para o destaque 2.", self.prompts("redacao")[0]["usuario"])

    def test_bloco_sem_marcas_nao_recebe_instrucoes(self):
        resumo.gerar_resumo("Reunião", [], FALAS_LONGAS, 9000, destaques=[{"t": 2000, "texto": "só no bloco 2"}])
        por_trecho = {int(re.search(r"Trecho (\d+) de", p["usuario"]).group(1)): p for p in self.prompts("bloco")}
        self.assertNotIn("estaque", por_trecho[1]["sistema"] + por_trecho[1]["usuario"])
        self.assertNotIn("estaque", por_trecho[3]["sistema"] + por_trecho[3]["usuario"])
        self.assertIn("DESTAQUE 1 DO USUÁRIO", por_trecho[2]["usuario"])

    def test_bloco_que_falha_nao_perde_marcas(self):
        self.gw.falhar_trecho = 2
        r = resumo.gerar_resumo("Reunião", [], FALAS_LONGAS, 9000, destaques=self.DESTAQUES)
        self.assertEqual(len(r["destaques"]), 5)
        self.assertEqual([d["texto"] for d in r["destaques"]],
                         ["Frase do modelo para o destaque 1.", "Frase do modelo para o destaque 2.", "", "", "Frase do modelo para o destaque 5."])
        self.assertTrue(any("não pôde ser resumido" in a for a in r["perfil"]["avisos"]))

    def test_modelo_omitiu_em_bloco(self):
        self.gw.omitir = {4}
        r = resumo.gerar_resumo("Reunião", [], FALAS_LONGAS, 9000, destaques=self.DESTAQUES)
        self.assertEqual(r["destaques"][3]["texto"], ""); self.assertEqual(r["destaques"][2]["texto"], "Frase do modelo para o destaque 3.")

    def test_perfil_hierarquico_leva_as_marcas_pelos_grupos(self):
        # 4 h, blocos de 40 min, 3 blocos por grupo: 6 blocos → 2 grupos
        falas = [fala(i * 2400 + 10, dur=20) for i in range(6)]
        r = resumo.gerar_resumo("Reunião", [], falas, 14000, destaques=[{"t": 100, "texto": "primeiro"}, {"t": 12100, "texto": "do bloco 6"}])
        self.assertEqual(r["perfil"]["id"], "hierarquico")
        self.assertEqual(len(self.prompts("grupo")), 2)
        self.assertEqual([d["texto"] for d in r["destaques"]], ["Frase do modelo para o destaque 1.", "Frase do modelo para o destaque 2."])
        (g1, g2) = sorted(self.prompts("grupo"), key=lambda p: p["usuario"])
        self.assertIn("Destaque 1: Frase do modelo para o destaque 1.", g1["usuario"] + g2["usuario"])
        self.assertIn("Destaque 2: Frase do modelo para o destaque 2.", g1["usuario"] + g2["usuario"])
        (pf,) = self.prompts("plano_final")
        self.assertIn("Destaque 1: Frase do modelo para o destaque 1.", pf["usuario"])   # as fichas dos grupos carregam as marcas
        self.assertIn("Destaque 2: Frase do modelo para o destaque 2.", pf["usuario"])


class TestArquivos(unittest.TestCase):
    def test_texto_transcricao_sem_destaques_igual_ao_de_sempre(self):
        antigo = arquivos.texto_transcricao("Reunião", "base", 90, FALAS)
        self.assertEqual(arquivos.texto_transcricao("Reunião", "base", 90, FALAS, []), antigo)
        self.assertEqual(arquivos.texto_transcricao("Reunião", "base", 90, FALAS, None), antigo)
        self.assertNotIn("DESTAQUE", antigo)
        self.assertTrue(antigo.endswith("Ana: Então fechamos o prazo para o dia 12.\n"))

    def test_texto_transcricao_com_destaques(self):
        t = arquivos.texto_transcricao("Reunião", "base", 4000, FALAS, [
            {"t": 3725, "texto": "depois de tudo"}, {"t": 40.5, "texto": "Ponto   importante"}, {"t": 30, "texto": "empate"}, {"t": 1, "texto": "  "}])
        corpo = t.split("\n\n", 1)[1].rstrip("\n").split("\n\n")
        self.assertEqual(len(corpo), 6)
        self.assertTrue(corpo[0].startswith("[00:00:00] Ana:"))
        self.assertTrue(corpo[1].startswith("[00:00:30] Carlos:"))
        self.assertEqual(corpo[2], "[00:00:30] >>> DESTAQUE: empate")
        self.assertEqual(corpo[3], "[00:00:40] >>> DESTAQUE: Ponto importante")
        self.assertTrue(corpo[4].startswith("[00:01:00] Ana:"))
        self.assertEqual(corpo[5], "[01:02:05] >>> DESTAQUE: depois de tudo")

    def test_texto_analise(self):
        r = {"resumo": "Narrativa.", "topicos": ["a"], "modelo": "m",
             "destaques": [{"t": 45, "marca": "Ponto importante", "texto": "A integração foi aprovada."}, {"t": 3725, "marca": "Sem frase", "texto": ""}]}
        linhas = arquivos.texto_analise("Reunião", "base", r).split("\n")
        i = linhas.index("## Destaques")
        self.assertEqual(linhas[i - 2:i], ["Narrativa.", ""])   # logo depois do resumo
        self.assertEqual(linhas[i + 1], "- **[00:00:45] Ponto importante** — A integração foi aprovada.")
        self.assertEqual(linhas[i + 2], "- **[01:02:05] Sem frase**")
        self.assertEqual(linhas[i + 3], ""); self.assertEqual(linhas[i + 4], "## Pontos-chave")
        self.assertNotIn("## Destaques", arquivos.texto_analise("Reunião", "base", {"resumo": "x", "destaques": []}))
        self.assertNotIn("## Destaques", arquivos.texto_analise("Reunião", "base", {"resumo": "x"}))


class TestApi(Base):
    def _resumo(self, corpo: dict) -> dict:
        from fastapi.testclient import TestClient
        import app as servidor
        c = TestClient(servidor.app, base_url="http://127.0.0.1:8765")
        r = c.post("/api/resumo", json=corpo)
        self.assertEqual(r.status_code, 200, r.text)
        job = r.json()["job"]
        for _ in range(100):
            j = c.get(f"/api/jobs/{job}").json()
            if j["status"] != "processando":
                break
            time.sleep(0.1)
        self.assertEqual(j["status"], "pronto", j)
        return j["resultado"]

    def test_com_e_sem_destaques(self):
        base = {"titulo": "Reunião", "notas": [], "duracao": 90, "falas": FALAS}
        r = self._resumo(base)   # cliente antigo
        self.assertEqual(r["destaques"], [])
        r = self._resumo({**base, "destaques": [{"t": 45, "texto": "Ponto importante"}]})
        self.assertEqual(r["destaques"], [{"t": 45.0, "marca": "Ponto importante", "texto": "Frase do modelo para o destaque 1."}])
        self.assertIn("DESTAQUE 1 DO USUÁRIO", self.prompts("plano")[-1]["usuario"])

    def test_modelos_de_entrada(self):
        import app as servidor
        f = [{"inicio": 0, "locutor": "Ana", "texto": "oi"}]
        self.assertEqual(servidor.ConfirmarIn(falas=f).destaques, [])
        self.assertEqual(servidor.ArquivarIn(base="b", falas=f).destaques, [])
        self.assertEqual(servidor.ResumoIn(falas=f).destaques, [])
        d = [{"t": 3.5, "texto": "x"}]
        for m in (servidor.ConfirmarIn(falas=f, destaques=d), servidor.ArquivarIn(base="b", falas=f, destaques=d), servidor.ResumoIn(falas=f, destaques=d)):
            self.assertEqual([x.model_dump() for x in m.destaques], d)


if __name__ == "__main__":
    unittest.main()
