"""Testes offline (sem chamar a IA) das regras de precisão de transcritor/locutores.py.

    .venv\\Scripts\\python testes\\nomes\\test_validacao.py -v
"""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from transcritor import locutores as L  # noqa: E402


def falas(*pares):
    t, saida = 0.0, []
    for rot, texto in pares:
        saida.append({"inicio": t, "fim": t + 3, "locutor": f"Locutor {rot}", "texto": texto})
        t += 4
    return saida


def entrada(rot, nome, conf=0.95, tipo="autoapresentacao", trecho="", horario="00:00", **extra):
    return {"rotulo": f"Locutor {rot}", "nome": nome, "confianca": conf, "tipo": tipo,
            "evidencia": {"horario": horario, "trecho": trecho}, **extra}


def validar(resposta, fl, alvos=("Locutor 1", "Locutor 2"), fixos=None, recusados=None, notas=()):
    return L.validar({"locutores": resposta}, fl, list(alvos), fixos or {}, recusados or {}, list(notas), "")


class Validacao(unittest.TestCase):
    def setUp(self):
        self.fl = falas(("1", "Oi, Marcos, tudo bem? Conseguiu ver o contrato?"), ("2", "Tudo bem. Vi ontem."),
                        ("1", "Aqui é a Ana, do financeiro, só para constar."), ("2", "Certo."))

    def test_vocativo_vai_para_quem_responde(self):
        ok, _ = validar([entrada(2, "Marcos", tipo="vocativo", trecho="Oi, Marcos, tudo bem?")], self.fl)
        self.assertEqual(ok["Locutor 2"]["nome"], "Marcos")

    def test_vocativo_na_fala_do_proprio_locutor_cai(self):
        ok, d = validar([entrada(1, "Marcos", tipo="vocativo", trecho="Oi, Marcos, tudo bem?")], self.fl)
        self.assertEqual(ok, {})
        self.assertIn("vocativo", d[0]["motivo"])

    def test_autoapresentacao_de_outro_locutor_cai(self):
        ok, d = validar([entrada(2, "Ana", trecho="Aqui é a Ana, do financeiro")], self.fl)
        self.assertEqual(ok, {})
        self.assertIn("Locutor 1", d[0]["motivo"])

    def test_nome_inventado_cai(self):
        ok, d = validar([entrada(1, "Roberto", trecho="Aqui é a Ana, do financeiro")], self.fl)
        self.assertEqual(ok, {})
        self.assertIn("não consta", d[0]["motivo"])

    def test_sobrenome_inventado_cai(self):
        ok, _ = validar([entrada(1, "Ana Souza", trecho="Aqui é a Ana, do financeiro")], self.fl)
        self.assertEqual(ok, {})

    def test_sobrenome_das_notas_vale_e_usa_a_grafia_do_usuario(self):
        ok, _ = validar([entrada(1, "Ana Souza", trecho="Aqui é a Ana, do financeiro")], self.fl, notas=["Ana Souza (financeiro)"])
        self.assertEqual(ok["Locutor 1"]["nome"], "Ana Souza")
        fl = falas(("1", "Aqui é o Marcus, tudo certo?"), ("2", "Tudo."))
        ok, _ = validar([entrada(1, "Marcus", trecho="Aqui é o Marcus")], fl, notas=["Reunião com Marcos"])
        self.assertEqual(ok["Locutor 1"]["nome"], "Marcos")

    def test_evidencia_inexistente_cai(self):
        ok, d = validar([entrada(1, "Ana", trecho="eu sou a Ana e adoro planilhas")], self.fl)
        self.assertEqual(ok, {})
        self.assertIn("evidência", d[0]["motivo"])

    def test_confianca_baixa_cai(self):
        ok, _ = validar([entrada(1, "Ana", conf=0.79, trecho="Aqui é a Ana, do financeiro")], self.fl)
        self.assertEqual(ok, {})
        ok, _ = validar([entrada(1, "Ana", conf=80, trecho="Aqui é a Ana, do financeiro")], self.fl)   # 80% também é aceito
        self.assertEqual(ok["Locutor 1"]["nome"], "Ana")

    def test_cargo_papel_nunca_vira_nome(self):
        self.assertIsNone(L._limpar_nome("a doutora"))
        self.assertIsNone(L._limpar_nome("Cliente"))
        self.assertIsNone(L._limpar_nome("Locutor 3"))
        self.assertEqual(L._limpar_nome("Dra. Ana"), "Ana")
        self.assertEqual(L._limpar_nome("ana paula de souza"), "Ana Paula de Souza")

    def test_misturado_cai(self):
        ok, d = validar([entrada(1, "Ana", trecho="Aqui é a Ana, do financeiro", misturado=True)], self.fl)
        self.assertEqual(ok, {})
        self.assertIn("misturado", d[0]["motivo"])

    def test_regex_detecta_locutor_que_se_apresenta_com_dois_nomes(self):
        fl = falas(("1", "Eu sou o Marcos."), ("2", "Certo."), ("1", "Eu sou a Beatriz, pessoal."), ("2", "Ok."))
        ok, d = validar([entrada(1, "Marcos", trecho="Eu sou o Marcos.")], fl)
        self.assertEqual(ok, {})
        self.assertIn("misturado", d[0]["motivo"])

    def test_esse_aqui_e_nao_e_autoapresentacao(self):
        fl = falas(("1", "Eu sou o Pedro e esse aqui é o Lucas."), ("2", "Obrigado, Pedro. Bom, vamos lá."))
        self.assertEqual(L._intros_por_rotulo(fl), {"Locutor 1": {"Pedro"}})

    def test_nome_recusado_pelo_usuario_nao_volta(self):
        ok, d = validar([entrada(1, "Ana", trecho="Aqui é a Ana, do financeiro")], self.fl, recusados={"Locutor 1": ["Ana"]})
        self.assertEqual(ok, {})
        self.assertIn("recusado", d[0]["motivo"])

    def test_nome_ja_usado_por_locutor_fixo_cai_mas_homonimo_com_sobrenome_passa(self):
        ok, _ = validar([entrada(1, "Ana", trecho="Aqui é a Ana, do financeiro")], self.fl, alvos=("Locutor 1",), fixos={"Locutor 2": "Ana Lima"})
        self.assertEqual(ok, {})
        fl = falas(("1", "Eu sou a Ana Souza."), ("2", "Certo."))
        ok, _ = validar([entrada(1, "Ana Souza", trecho="Eu sou a Ana Souza.")], fl, alvos=("Locutor 1",), fixos={"Locutor 2": "Ana Lima"})
        self.assertEqual(ok["Locutor 1"]["nome"], "Ana Souza")

    def test_mesmo_nome_vocativo_perde_para_autoapresentacao(self):
        fl = falas(("1", "Oi, eu sou o Marcos."), ("2", "Prazer."), ("2", "Marcos, vamos começar?"), ("3", "Vamos."))
        ok, d = validar([entrada(1, "Marcos", trecho="eu sou o Marcos"),
                         entrada(3, "Marcos", tipo="vocativo", trecho="Marcos, vamos começar?")], fl, alvos=("Locutor 1", "Locutor 2", "Locutor 3"))
        self.assertEqual(list(ok), ["Locutor 1"])
        self.assertTrue(any("mais de um locutor" in x["motivo"] for x in d))

    def test_rotulo_desconhecido_e_json_torto_sao_ignorados(self):
        ok, _ = validar([entrada(9, "Ana", trecho="Aqui é a Ana"), "lixo", {"rotulo": "Locutor 1", "nome": None}], self.fl)
        self.assertEqual(ok, {})

    def test_vocativo_sem_resposta_imediata_cai(self):
        fl = falas(("1", "Marcos, você viu o e-mail?"), ("1", "Depois eu reenvio."), ("2", "Sobre o próximo assunto, preciso de dados."))
        ok, d = validar([entrada(2, "Marcos", tipo="vocativo", trecho="Marcos, você viu o e-mail?")], fl)
        self.assertEqual(ok, {})
        self.assertIn("sem resposta", d[0]["motivo"])

    def test_mesmo_locutor_chamado_por_dois_nomes_e_ambiguo(self):
        fl = falas(("1", "Oi, Marcos, tudo bem?"), ("2", "Tudo bem."), ("1", "E aí, Pedro, pronto para começar?"), ("2", "Pronto."))
        ok, d = validar([entrada(2, "Marcos", tipo="vocativo", trecho="Oi, Marcos, tudo bem?")], fl)
        self.assertEqual(ok, {})
        self.assertIn("ambíguo", d[0]["motivo"])

    def test_fala_com_dois_vocativos_vale_o_ultimo(self):
        fl = falas(("1", "Obrigado, Rodrigo. Marta, e o jurídico, como está?"), ("3", "Em dia."))
        ok, _ = validar([entrada(3, "Marta", tipo="vocativo", trecho="Marta, e o jurídico")], fl, alvos=("Locutor 3",))
        self.assertEqual(ok["Locutor 3"]["nome"], "Marta")
        ok, d = validar([entrada(3, "Rodrigo", tipo="vocativo", trecho="Obrigado, Rodrigo.")], fl, alvos=("Locutor 3",))
        self.assertEqual(ok, {})

    def test_tempo_e_segundos(self):
        self.assertEqual(L._segundos("1:02:05"), 3725)
        self.assertEqual(L._segundos("02:05"), 125)
        self.assertEqual(L._segundos(12), 12)
        self.assertIsNone(L._segundos("depois"))


class Janelas(unittest.TestCase):
    def test_gravacao_longa_envia_abertura_e_janelas_dos_candidatos(self):
        enchimento = "Então, sobre esse ponto específico, precisamos olhar os números do período com calma antes de decidir. " * 2
        fl = falas(*([("1", "Aqui é o Marcos."), ("2", "Eu sou a Ana.")] + [(str(1 + i % 2), enchimento) for i in range(600)]
                     + [("1", "Beatriz, pode resumir?"), ("3", "Posso sim.")] + [(str(1 + i % 2), enchimento) for i in range(50)]))
        indices, completo = L.selecionar_falas(fl, set(), limite=20000)
        self.assertFalse(completo)
        self.assertIn(0, indices)
        self.assertIn(602, indices)   # a fala do vocativo
        self.assertIn(603, indices)   # e a resposta
        self.assertLess(sum(L._tamanho(fl[i], True) for i in indices), 20000 + 1)
        self.assertLess(len(indices), len(fl) / 5)

    def test_gravacao_curta_vai_inteira(self):
        fl = falas(("1", "Oi."), ("2", "Olá."))
        self.assertEqual(L.selecionar_falas(fl, set()), ([0, 1], True))


class Degradacao(unittest.TestCase):
    FALAS = falas(("1", "Aqui é o Marcos."), ("2", "Oi, Marcos."))

    def test_sem_chave(self):
        with mock.patch.dict(os.environ, {"AI_GATEWAY_API_KEY": "", "VERCEL_OIDC_TOKEN": "", "GEMINI_API_KEY": ""}):
            with self.assertRaises(L.IANaoConfigurada):
                L.identificar("", [], self.FALAS)

    def test_falha_de_rede_vira_falhaia(self):
        with mock.patch.dict(os.environ, {"AI_GATEWAY_API_KEY": "x", "AVEIA_IA_URL": "http://127.0.0.1:9/v1/chat"}), \
                mock.patch.object(L.time, "sleep"):
            with self.assertRaises(L.FalhaIA):
                L.identificar("", [], self.FALAS)

    def test_resposta_nao_json_vira_falhaia(self):
        with mock.patch.dict(os.environ, {"AI_GATEWAY_API_KEY": "x"}), mock.patch.object(L, "_chamar", return_value="desculpe, não sei"):
            with self.assertRaises(L.FalhaIA):
                L.identificar("", [], self.FALAS)

    def test_todos_ja_nomeados_nao_chama_ia(self):
        with mock.patch.dict(os.environ, {"AI_GATEWAY_API_KEY": "x"}), mock.patch.object(L, "_chamar", side_effect=AssertionError("não devia chamar")):
            r = L.identificar("", [], self.FALAS, nomes_fixos={"Locutor 1": "A", "Locutor 2": "B"})
        self.assertEqual(r["sugestoes"], {})



class MensagemRecebida(unittest.TestCase):
    """Quem recebe uma carta/e-mail e a lê em voz alta: o nome da saudação é dele, mas só com a entrega na conversa."""

    def setUp(self):
        self.carta = falas(("1", "Aqui tem esta carta... é da caixa urbana."), ("2", "Uma carta?"), ("2", "De quem será? Obrigada."),
                           ("1", "Não há de quê, minha senhora."), ("2", "Ah! A carta é de Pedro."),
                           ("2", "Marta, peço desculpa por tudo o que aconteceu. Esquece este assunto, Pedro."))

    def teste(self, rot="2", nome="Marta", conf=0.85, fl=None, trecho="Marta, peço desculpa por tudo o que aconteceu"):
        return validar([entrada(rot, nome, conf, "mensagem_recebida", trecho)], fl or self.carta, alvos=("Locutor 1", "Locutor 2"))

    def test_entrega_na_conversa_e_saudacao_valem(self):
        ok, d = self.teste()
        self.assertEqual(ok["Locutor 2"]["nome"], "Marta", d)
        self.assertEqual(ok["Locutor 2"]["tipo"], "mensagem_recebida")

    def test_exige_confianca_maior_que_a_do_vocativo(self):
        ok, d = self.teste(conf=0.82)
        self.assertEqual(ok, {}); self.assertIn("0.85", d[0]["motivo"])

    def test_sem_entrega_ler_em_voz_alta_nao_basta(self):
        fl = falas(("2", "Pessoal, o cliente mandou este e-mail."), ("2", "Marta, peço que revise o orçamento até sexta."))
        ok, d = self.teste(fl=fl, trecho="Marta, peço que revise o orçamento")
        self.assertEqual(ok, {}); self.assertIn("não há entrega", d[0]["motivo"])

    def test_a_entrega_precisa_ser_de_outro_locutor_com_resposta(self):
        fl = falas(("1", "Chegou uma carta."), ("1", "Ainda bem."), ("2", "Marta, peço desculpa por tudo o que aconteceu."))
        ok, d = self.teste(fl=fl)
        self.assertEqual(ok, {}); self.assertIn("não há entrega", d[0]["motivo"])

    def test_o_nome_precisa_abrir_a_mensagem(self):
        fl = falas(("1", "Aqui tem esta carta."), ("2", "Obrigada."), ("2", "Ontem falei com a Marta sobre a carta e o prazo."))
        ok, d = self.teste(fl=fl, trecho="falei com a Marta sobre a carta")
        self.assertEqual(ok, {}); self.assertIn("saudação", d[0]["motivo"])

    def test_a_mensagem_lida_por_outro_locutor_nao_vale(self):
        ok, d = self.teste(rot="1", nome="Marta")
        self.assertEqual(ok, {})

    def test_chamado_por_outro_nome_cai(self):
        fl = falas(("1", "Aqui tem esta carta, Beatriz."), ("2", "Obrigada."), ("2", "Marta, peço desculpa por tudo o que aconteceu."))
        ok, d = self.teste(fl=fl)
        self.assertEqual(ok, {}); self.assertIn("chamado por outro nome", d[0]["motivo"])

    def test_prompt_ensina_o_tipo_e_o_validador_o_conhece(self):
        self.assertIn("mensagem_recebida", L.TIPOS); self.assertIn("MENSAGEM RECEBIDA", L.SISTEMA)


if __name__ == "__main__":
    unittest.main()
