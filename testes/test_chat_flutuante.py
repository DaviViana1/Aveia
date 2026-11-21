"""Estrutura do chat flutuante da gravação (static/index.html), sem navegador.

    .venv\Scripts\python -m unittest testes.test_chat_flutuante -v

O teste na tela (desktop e celular) é manual; aqui só se garante o que quebra em silêncio ao mexer no HTML.
"""
import re
import unittest
from pathlib import Path

HTML = (Path(__file__).resolve().parent.parent / "static" / "index.html").read_text(encoding="utf-8")


class ChatFlutuante(unittest.TestCase):
    def test_painel_fora_do_layout_e_antes_do_dock(self):
        i_chat = HTML.index('<section id="chat"')
        self.assertGreater(i_chat, HTML.index("</main>"), "o painel flutuante não pode ficar dentro da grade da nota")
        self.assertLess(i_chat, HTML.index("<footer"), "o painel vem antes do dock no documento")
        self.assertEqual(HTML.count('id="chat"'), 1)

    def test_painel_e_flutuante_com_vidro_fosco(self):
        tag = re.search(r'<section id="chat"[^>]*>', HTML).group(0)
        self.assertIn("fixed", tag)
        self.assertIn("hidden", tag, "começa oculto")
        corpo = HTML[HTML.index(tag):HTML.index("</section>", HTML.index(tag))]
        self.assertIn("backdrop-blur", corpo)

    def test_ids_usados_pelo_script_existem(self):
        for i in ("chat-lista", "chat-vazio", "chat-input", "chat-enviar", "chat-tempo"):
            self.assertEqual(HTML.count(f'id="{i}"'), 1, i)
            self.assertIn("$('" + i + "')", HTML, i + " deveria ser usado pelo script")

    def test_sem_botao_de_minimizar_nem_botao_do_chat_no_dock(self):
        for morto in ("alternarChat", "chat-btn", "chat-badge", "chatAberto"):
            self.assertNotIn(morto, HTML, morto)

    def test_painel_acompanha_o_recuo_do_menu_e_o_teclado(self):
        self.assertIn("'body, footer, #chat'", HTML)
        self.assertIn("--kb", HTML)

    def test_sem_travessao_longo_nem_emoji_no_painel(self):
        painel = HTML[HTML.index('<section id="chat"'):HTML.index("<!-- Dock -->")]
        self.assertNotRegex(painel, "[\u2013\u2014]")
        self.assertNotRegex(painel, "[\U0001F300-\U0001FAFF\u2600-\u27BF]")

    def test_feed_limita_10_linhas_e_apaga_os_antigos(self):
        css = re.search(r"\.chat-feed \{[^}]*\}", HTML).group(0)
        self.assertIn("max-height: calc(10 * 1.25rem", css, "o feed mostra no máximo 10 linhas de 1.25rem")
        self.assertRegex(HTML, r"const opacidadeDoFeed = k => Math\.max\(0\.\d+, 1 - k \* 0\.\d+\)")
        self.assertIn("c.style.opacity = opacidadeDoFeed(todos.length - 1 - i)", HTML, "o mais novo (k=0) fica com opacidade total")
        self.assertIn("lista.firstElementChild.remove()", HTML, "o corte é por item inteiro, medindo a altura")

    def test_cada_gravacao_comeca_com_o_chat_limpo(self):
        self.assertIn("chatDesde = nota().duracao || 0;", HTML)
        self.assertIn(">= chatDesde - 0.001", HTML)

    def test_linhas_em_tres_colunas_com_nome_so_quando_o_locutor_muda(self):
        self.assertIn("grid-template-columns: 2.25rem minmax(0, 1fr) 1.25rem", HTML)
        self.assertIn("it.fala.locutor !== locutorAnterior", HTML)

    def test_feed_junta_fala_e_destaque_por_tempo(self):
        self.assertIn("intercalar(n, falasAoVivo(n)).filter(", HTML)
        self.assertIn(".slice(-FEED_ITENS)", HTML)
        self.assertIn("function linhaChatFala(", HTML)
        self.assertIn("function linhaChatDestaque(", HTML)

    def test_transcricao_sai_da_pagina_quando_esta_no_painel(self):
        self.assertIn("const transcricaoNoPainel = () => emGravacao();", HTML)
        self.assertIn("|| noPainel);", HTML, "o cartão da transcrição fica oculto quando ela está no painel")
        self.assertIn("if (transcricaoNoPainelAntes) return renderChat();", HTML, "texto ao vivo atualiza o painel, não a página")

    def test_ao_parar_a_transcricao_vai_para_a_pagina_com_aviso(self):
        self.assertIn("const transcrevendoAgora = n => estado === 'processando' && processandoId === n.id;", HTML)
        self.assertIn("emGravacao() ? 'ao vivo' : 'Transcrevendo'", HTML)
        self.assertNotIn("sem locutores nem IA", HTML, "o aviso não diz o que falta, só que está transcrevendo")
        self.assertIn("@keyframes ponto-acende", HTML)
        self.assertIn("prefers-reduced-motion: reduce) { .pontos-anim i", HTML)
        self.assertIn("!temTranscricao(n) || noPainel", HTML)

    def test_falas_longas_sao_partidas_pelo_destaque(self):
        for f in ("function pontosDeCorte(", "function partirFala(", "function fronteirasDeFrase(", "function linhaContinuacao("):
            self.assertIn(f, HTML, f)
        self.assertIn(".flatMap(f => partirFala(f, dsOrdenados, usados))", HTML)

    def test_resumo_so_aparece_quando_existe_ou_esta_sendo_gerado(self):
        self.assertIn("!n.falas.length || (!r && !gerando)", HTML)

    def test_funcoes_chamadas_pelo_html_existem(self):
        for f in ("enviarDestaque", "renderChat", "abrirChat"):
            self.assertRegex(HTML, rf"function {f}\(")


if __name__ == "__main__":
    unittest.main()
