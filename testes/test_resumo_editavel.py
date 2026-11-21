"""Estrutura do resumo editável como bloco de notas, da identificação pelo chat e do cabeçalho flutuante (`static/index.html`).

    .venv\\Scripts\\python -m unittest testes.test_resumo_editavel -v

Sem navegador: garante o que quebra em silêncio ao mexer no HTML. A conferência na tela (digitar, Enter, apagar, desktop e 375 px) é manual.
"""
import re
import unittest
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
HTML = (RAIZ / "static" / "index.html").read_text(encoding="utf-8")


def funcao(nome: str) -> str:
    i = HTML.index(f"function {nome}(")
    return HTML[i:HTML.index("\n        }\n", i)]   # até o fecha-chaves da função (recuo de 8 espaços)


def render_resumo() -> str:
    a = HTML.index("        function renderResumo() {")
    b = HTML.index("        async function gerarResumo(")
    return HTML[a:b]


class ResumoEditavel(unittest.TestCase):
    def test_sem_controle_de_modo_edicao(self):
        for morto in ("editarResumo(", "salvarResumo(", "modoEdicao", "resumo-editar", "resumo-salvar", "<textarea id=\"resumo"):
            self.assertNotIn(morto, HTML, morto)
        self.assertEqual(len(re.findall(r'id="resumo-[a-z]+"', HTML)), 4, "só modelo, refazer, desfazer e corpo: nenhum botão de editar")

    def test_cada_texto_do_resumo_e_editavel_no_lugar(self):
        corpo = render_resumo()
        for chave in ("par:", "top", "dec", "aco", "cap:", "dest:"):
            self.assertIn(chave, corpo, chave)
        self.assertGreaterEqual(corpo.count("editavel(n,"), 9)
        self.assertIn("contentEditable = EDITAVEL ? 'plaintext-only' : 'true'", HTML, "texto simples: nada digitado vira HTML")
        self.assertNotIn("innerHTML", render_resumo() + funcao("editavel"), "o texto do usuário nunca entra como HTML")

    def test_colar_vira_texto_simples_e_enter_e_backspace_sao_de_bloco_de_notas(self):
        e = funcao("editavel")
        self.assertIn("getData('text/plain')", e)
        self.assertIn("ev.key === 'Enter'", e)
        self.assertIn("ev.key === 'Backspace'", e)
        self.assertIn("ev.isComposing", e, "não atrapalha acento e teclado de celular")
        i = funcao("opsItem")
        self.assertIn("arr.splice(i + 1, 0, novo(depois))", i, "Enter cria o item seguinte")
        self.assertIn("arr.splice(i, 1)", i, "item vazio some com Backspace")

    def test_sem_borda_nem_fundo_ao_focar(self):
        css = re.search(r"\.ed \{[^}]*\}", HTML).group(0)
        self.assertIn("outline: none", css)
        self.assertNotRegex(css, r"border|background|box-shadow|font")
        self.assertIn("[contenteditable]:focus { outline: none; }", HTML)

    def test_grava_sozinho_com_atraso_curto_e_marca_a_edicao(self):
        r = funcao("resumoEditado")
        self.assertIn("n.resumo.editado = true", r)
        self.assertIn("setTimeout(", r)
        self.assertIn("salvar(); sincronizarNomes(n)", r, "guarda no navegador e regrava o arquivo em Análises")
        self.assertIn("addEventListener('pagehide'", HTML, "fechar a aba não perde o que foi digitado")

    def test_ia_nao_sobrescreve_resumo_editado(self):
        g = funcao("gerarResumo")
        self.assertIn("if (alvo.resumo?.editado) {", g)
        self.assertIn("if (automatico) return;", g, "a IA automática nunca troca texto editado")
        self.assertIn("confirm('Você editou este resumo.", g, "Refazer pergunta antes")
        self.assertIn("alvo.resumoAnterior = alvo.resumo", g, "guarda a versão anterior")
        self.assertIn("gerarResumo(alvo, true)", HTML)
        self.assertIn("function desfazerRefazer()", HTML)
        self.assertIn('id="resumo-desfazer"', HTML)

    def test_exportacao_usa_o_texto_editado_sem_itens_vazios(self):
        i = HTML.index("const r = Nomes.limparResumo(n.resumo);")
        self.assertIn("RESUMO EXECUTIVO", HTML[i:i + 400])
        self.assertIn("resultado: Nomes.limparResumo(", HTML, "o arquivo em Análises também sai limpo")

    def test_linhas_da_linha_do_tempo_e_dos_destaques_nao_sao_mais_um_botao_inteiro(self):
        corpo = render_resumo()
        self.assertNotIn("row = el('button'", corpo, "o texto não pode estar dentro de um botão (não receberia o cursor)")
        self.assertIn("chip.onclick = () => irParaFala(", corpo, "a hora continua levando ao ponto da transcrição")


class RenomearAtualizaOResumo(unittest.TestCase):
    def test_todos_os_caminhos_de_renomear_propagam(self):
        self.assertIn("propagarNome(n, f.locutor, antigos, nomeDe(n, f.locutor))", HTML, "renomear no card da transcrição")
        self.assertIn("propagarNome(alvo, rotulo, antigos, s.nome)", HTML, "nome sugerido pela IA")
        self.assertIn("propagarNome(n, de, antigos, nomeDe(n, para))", HTML, "mesclar locutores")
        self.assertIn("propagarNome(n, rotulo, antigos, nome)", HTML, "identificação pelo chat")
        self.assertIn('<script src="/static/nomes.js"></script>', HTML)

    def test_propagar_so_mexe_no_resumo_e_no_titulo_automatico(self):
        f = funcao("propagarNome")
        self.assertIn("Nomes.trocarNoResumo(n.resumo, antigos, novo)", f)
        self.assertIn("n.tituloAuto && !n.tituloEditado", f, "título escrito por você nunca é tocado")
        self.assertNotIn("n.texto", f, "a nota escrita à mão nunca é tocada")
        self.assertNotIn("fetch(", f, "sem chamada de IA")


class IdentificacaoPeloChat(unittest.TestCase):
    def test_frase_de_identificacao_nao_vira_destaque(self):
        e = funcao("enviarDestaque")
        self.assertIn("comandoDoChat(texto)", e)
        self.assertIn("n.identificacoes", e)
        self.assertIn("/api/destaque-comando", HTML)
        self.assertIn("ia: iaPode()", HTML, "o interruptor único vale também aqui")

    def test_aplica_ao_fim_da_gravacao_antes_da_ia_e_do_resumo(self):
        a = HTML.index("await aplicarIdentificacoes(alvo, desloc)")
        self.assertLess(a, HTML.index("const iaAutomatica = iaPode();"))
        self.assertLess(a, HTML.index("identificarLocutores(alvo).finally"))

    def test_nome_definido_a_mao_pede_confirmacao_e_ambiguidade_pergunta(self):
        f = funcao("atribuirNome")
        self.assertIn("nome que você definiu. Trocar por", f)
        self.assertIn("mesclarLocutores(n, rotulo, outro)", f)
        self.assertIn("perguntarLocutor(", funcao("aplicarIdentificacoes"))


class CabecalhoFlutuante(unittest.TestCase):
    def header(self) -> str:
        return re.search(r'<header id="topo"[^>]*>', HTML).group(0)

    def test_cabecalho_fixo_com_vidro_fosco(self):
        h = self.header()
        for classe in ("fixed", "backdrop-blur-md", "bg-white/70", "border-b", "z-[25]"):
            self.assertIn(classe, h, classe)
        self.assertEqual(HTML.count("<header"), 1)

    def test_barra_encostada_no_topo_e_nas_laterais_sem_margem_nem_pilula(self):
        h = self.header()
        classes = h.split()
        self.assertIn("top-0", classes); self.assertIn("inset-x-0", classes, "largura total da janela (left 0, right 0)")
        for proibida in ("rounded-full", "rounded-2xl", "rounded-3xl", "rounded-xl", "rounded-lg", "mx-auto", "top-2", "top-4", "inset-x-4", "inset-x-8", "sm:top-4", "sm:inset-x-8", "m-2", "m-4"):
            self.assertNotIn(proibida, classes, proibida)
        self.assertNotRegex(h, r"mx-|mt-|ml-|mr-|px-|p-\d|rounded", "o próprio cabeçalho não tem margem, recuo nem canto")
        self.assertIn("shadow-[0_10px_24px_-14px", h, "sombra leve, deslocada para baixo (negativa nas laterais)")
        # o miolo (logotipo, selo, ícones) alinha com o conteúdo da página: mesmo recuo lateral do corpo e a largura máxima do layout + recuo
        miolo = re.search(r'<header id="topo"[^>]*>\s*<div class="([^"]*)"', HTML).group(1)
        self.assertIn("max-w-[1264px]", miolo)
        self.assertIn("px-4 sm:px-8", miolo)
        self.assertIn("h-14", miolo)

    def test_conteudo_reserva_o_espaco_do_cabecalho(self):
        corpo = re.search(r"<body class=\"([^\"]*)\"", HTML).group(1)
        self.assertIn("pt-[4.75rem]", corpo)
        self.assertIn("sm:pt-[5rem]", corpo)
        self.assertIn("scroll-padding-top: 5rem", HTML, "âncoras e rolagens automáticas não ficam cobertas")
        self.assertIn("top: 5rem; max-height: calc(100vh - 12rem)", HTML, "a transcrição fixa do desktop larga desce junto")

    def test_ajustes_alinha_com_o_miolo(self):
        tag = re.search(r'<div id="ajustes"[^>]*class="([^"]*)"', HTML).group(1)
        self.assertIn("right-4 sm:right-8", tag)
        self.assertIn("top-[3.75rem]", tag, "abre logo abaixo da barra de 3.5rem")

    def test_camadas_coerentes(self):
        # toast (40) e modal de pergunta (50) acima do cabeçalho (25), que fica acima do rodapé e do chat (20); menu (30) não o cobre.
        self.assertIn('class="fixed bottom-4 inset-x-0 z-20', HTML)
        self.assertIn('id="toast" class="fixed top-5 left-1/2 -translate-x-1/2 z-40', HTML)
        self.assertIn("fixed inset-0 z-50", HTML)
        self.assertIn('aside id="menu" class="fixed left-4 top-20 bottom-24 z-30', HTML, "o menu começa abaixo do cabeçalho")

    def test_sem_transparencia_e_sem_teclado(self):
        self.assertIn("prefers-reduced-transparency: reduce", HTML)
        self.assertIn("prefers-contrast: more", HTML)
        self.assertIn("backdrop-filter: none !important", HTML)
        self.assertNotIn("--kb", self.header(), "o cabeçalho não acompanha o teclado virtual")
        self.assertNotRegex(self.header(), r"transition|animate")


class Estilo(unittest.TestCase):
    def test_sem_travessao_longo_nem_emoji_no_que_foi_acrescentado(self):
        for nome in ("static/nomes.js", "transcritor/identificacao.py"):
            t = (RAIZ / nome).read_text(encoding="utf-8")
            self.assertNotRegex(t, "[\u2013\u2014]", nome)
            self.assertNotRegex(t, "[\U0001F300-\U0001FAFF☀-➿]", nome)
        i = HTML.index("// ---------- Resumo editável (como um bloco de notas)")
        self.assertNotRegex(HTML[i:i + 9000], "[\u2013\u2014]")


if __name__ == "__main__":
    unittest.main()
