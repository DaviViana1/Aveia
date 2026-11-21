"""Troca de nome nos textos derivados e escolha do locutor pelo horário (`static/nomes.js`), executadas no Node.

    .venv\\Scripts\\python -m unittest testes.test_nomes_js -v

Cobre o item "renomear o locutor atualiza o resumo": "Locutor 1" vira o nome novo em todo o resumo, sem confundir com "Locutor 10",
sem pegar parte de outra palavra, preservando o que o usuário editou, e sem tocar na nota escrita à mão.
"""
import json
import shutil
import subprocess
import unittest
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
NODE = shutil.which("node")


def js(expr: str):
    codigo = f"const N = require({json.dumps(str(RAIZ / 'static' / 'nomes.js'))}); console.log(JSON.stringify({expr}));"
    r = subprocess.run([NODE, "-e", codigo], capture_output=True, text=True, timeout=30, encoding="utf-8")
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout)


RESUMO = {
    "resumo": "O Locutor 1 apresentou um crescimento de 12%. Taísa enfatizou a meta.\n\nO Locutor 1 validou a estratégia.",
    "topicos": ["Locutor 1 e Locutor 10 discutiram o mix", "Taísa explicou a carteira"],
    "capitulos": [{"inicio": 1, "titulo": "Abertura do Locutor 1", "resumo": "Locutor 1 apresenta"}],
    "decisoes": ["Pagamentos na primeira hora, validado pelo Locutor 1"],
    "acoes": [{"tarefa": "Locutor 1 alinha com a tesouraria", "responsavel": "Locutor 1", "prazo": "sexta"}],
    "destaques": [{"t": 5, "marca": "Ponto importante", "texto": "ouvir o Locutor 1 depois"}],
    "editado": True,
}


@unittest.skipUnless(NODE, "Node não encontrado")
class Troca(unittest.TestCase):
    def test_palavra_inteira_e_sem_confundir_locutor_1_com_10(self):
        r = js("N.trocar('Locutor 1 e Locutor 10 e locutor 1, Locutor 1.', ['Locutor 1'], 'Carlos')")
        self.assertEqual(r, "Carlos e Locutor 10 e Carlos, Carlos.")

    def test_nao_pega_parte_de_outra_palavra(self):
        self.assertEqual(js("N.trocar('Carlota e Carlos e Carlos-Mar', ['Carlos'], 'Dav')"), "Carlota e Dav e Dav-Mar")
        self.assertEqual(js("N.trocar('Anais viu a Ana', ['Ana'], 'Bia')"), "Anais viu a Bia")

    def test_acento_e_caracteres_especiais_do_nome(self):
        self.assertEqual(js("N.trocar('Taísa e Taisa', ['Taísa'], 'Marina')"), "Marina e Taisa", "grafia exata; sem confundir")
        self.assertEqual(js("N.trocar('fala o J. R. (diretor)', ['J. R.'], 'João')"), "fala o João (diretor)")

    def test_nome_novo_que_contem_o_antigo_nao_e_trocado_duas_vezes(self):
        self.assertEqual(js("N.trocar('Carlos falou. Carlos voltou', ['Carlos'], 'Carlos Viana')"), "Carlos Viana falou. Carlos Viana voltou")

    def test_varios_nomes_antigos_do_mesmo_locutor(self):
        self.assertEqual(js("N.trocar('Locutor 2 e Taísa e Tai', ['Locutor 2', 'Taísa', 'Locutor 2'], 'Marina')"), "Marina e Marina e Tai")

    def test_sem_nome_novo_ou_igual_nao_muda(self):
        self.assertEqual(js("N.trocar('Locutor 1 fala', ['Locutor 1'], '')"), "Locutor 1 fala")
        self.assertEqual(js("N.trocar('Carlos fala', ['Carlos'], 'Carlos')"), "Carlos fala")

    def test_resumo_todos_os_campos(self):
        r = js(f"N.trocarNoResumo({json.dumps(RESUMO)}, ['Locutor 1'], 'Carlos')")
        self.assertTrue(r["mudou"])
        x = r["resumo"]
        self.assertIn("O Carlos apresentou", x["resumo"]); self.assertIn("O Carlos validou", x["resumo"])
        self.assertIn("Taísa enfatizou", x["resumo"], "o nome de outro locutor não é tocado")
        self.assertEqual(x["topicos"][0], "Carlos e Locutor 10 discutiram o mix")
        self.assertEqual((x["capitulos"][0]["titulo"], x["capitulos"][0]["resumo"]), ("Abertura do Carlos", "Carlos apresenta"))
        self.assertEqual(x["decisoes"][0], "Pagamentos na primeira hora, validado pelo Carlos")
        self.assertEqual((x["acoes"][0]["tarefa"], x["acoes"][0]["responsavel"], x["acoes"][0]["prazo"]), ("Carlos alinha com a tesouraria", "Carlos", "sexta"))
        self.assertEqual(x["destaques"][0]["texto"], "ouvir o Carlos depois")
        self.assertEqual(x["destaques"][0]["marca"], "Ponto importante")
        self.assertTrue(x["editado"], "a marca de edição fica")
        self.assertEqual(x["capitulos"][0]["inicio"], 1)

    def test_resumo_editado_pelo_usuario_mantem_as_edicoes(self):
        editado = dict(RESUMO, resumo="Texto que EU escrevi sobre o Locutor 1, com 100% de certeza.\n\nParágrafo novo meu.")
        x = js(f"N.trocarNoResumo({json.dumps(editado)}, ['Locutor 1'], 'Carlos')")["resumo"]
        self.assertEqual(x["resumo"], "Texto que EU escrevi sobre o Carlos, com 100% de certeza.\n\nParágrafo novo meu.")

    def test_nao_altera_o_objeto_original_e_resumo_sem_nome_nao_muda(self):
        r = js(f"(() => {{ const o = {json.dumps(RESUMO)}; N.trocarNoResumo(o, ['Locutor 1'], 'Carlos'); return o.resumo; }})()")
        self.assertIn("Locutor 1", r)
        self.assertFalse(js(f"N.trocarNoResumo({json.dumps(RESUMO)}, ['Locutor 7'], 'Carlos')")["mudou"])
        self.assertEqual(js("N.trocarNoResumo(undefined, ['x'], 'y')"), {"mudou": False})

    def test_nomes_do_rotulo_na_ordem(self):
        n = {"nomes": {"Locutor 1": "Carlos"}, "apelidos": {"Locutor 1": ["Locutor 1", "Dav"]}}
        self.assertEqual(js(f"N.nomesDoRotulo({json.dumps(n)}, 'Locutor 1')"), ["Locutor 1", "Carlos", "Dav"])
        self.assertEqual(js("N.nomesDoRotulo({}, 'Locutor 2')"), ["Locutor 2"])

    def test_limpar_resumo_para_exportar(self):
        r = {"resumo": "um\n\n\n\n  \n\ndois", "topicos": ["a", "", "  "], "decisoes": [""], "capitulos": [{"titulo": "", "resumo": ""}, {"titulo": "x", "resumo": ""}],
             "acoes": [{"tarefa": ""}, {"tarefa": "fazer", "responsavel": "Carlos"}], "destaques": [{"marca": "", "texto": ""}]}
        x = js(f"N.limparResumo({json.dumps(r)})")
        self.assertEqual(x["resumo"], "um\n\ndois")
        self.assertEqual((x["topicos"], x["decisoes"]), (["a"], []))
        self.assertEqual(x["capitulos"], [{"titulo": "x", "resumo": ""}])
        self.assertEqual(x["acoes"], [{"tarefa": "fazer", "responsavel": "Carlos"}])
        self.assertEqual(x["destaques"], [])


def fala(i, f, loc):
    return {"inicio": i, "fim": f, "locutor": loc}


@unittest.skipUnless(NODE, "Node não encontrado")
class Locutor(unittest.TestCase):
    def achar(self, falas, t, quando="agora"):
        return js(f"N.acharLocutor({json.dumps(falas)}, {t}, {json.dumps(quando)})")

    def test_quem_falava_no_instante(self):
        falas = [fala(0, 20, "Locutor 1"), fala(20, 40, "Locutor 2")]
        self.assertEqual(self.achar(falas, 15), {"estado": "ok", "rotulo": "Locutor 1"})
        self.assertEqual(self.achar(falas, 30), {"estado": "ok", "rotulo": "Locutor 2"})

    def test_falou_agora_pega_a_ultima_fala_ate_o_instante(self):
        falas = [fala(0, 10, "Locutor 1"), fala(12, 14, "Locutor 2")]
        self.assertEqual(self.achar(falas, 20), {"estado": "ok", "rotulo": "Locutor 2"}, "silêncio depois da fala: a última")

    def test_dois_locutores_juntos_pergunta(self):
        falas = [fala(0, 20, "Locutor 1"), fala(20, 30, "Locutor 2")]
        r = self.achar(falas, 21)
        self.assertEqual(r["estado"], "ambiguo")
        self.assertEqual(set(r["candidatos"]), {"Locutor 1", "Locutor 2"})
        simultaneas = [fala(0, 30, "Locutor 1"), fala(0, 30, "Locutor 2")]
        self.assertEqual(self.achar(simultaneas, 10)["estado"], "ambiguo")

    def test_instante_sem_fala(self):
        self.assertEqual(self.achar([fala(0, 5, "Locutor 1")], 60), {"estado": "sem_fala"})
        self.assertEqual(self.achar([], 10), {"estado": "sem_fala"})

    def test_fala_longa_de_um_so_locutor(self):
        self.assertEqual(self.achar([fala(0, 300, "Locutor 1")], 150), {"estado": "ok", "rotulo": "Locutor 1"})

    def test_anterior(self):
        falas = [fala(0, 10, "Locutor 1"), fala(10, 20, "Locutor 2"), fala(20, 30, "Locutor 1")]
        self.assertEqual(self.achar(falas, 28, "anterior"), {"estado": "ok", "rotulo": "Locutor 2"})
        self.assertEqual(self.achar([fala(0, 30, "Locutor 1")], 20, "anterior"), {"estado": "sem_fala"})


if __name__ == "__main__":
    unittest.main()
