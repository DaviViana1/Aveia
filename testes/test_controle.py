"""Testes offline da gravação automática por controle (captura, sessão de PCM direto, rotas, parada, saída, alinhamento).

    .venv\\Scripts\\python -m unittest testes.test_controle -v

Sem dispositivo de áudio, sem modelo e sem internet: o áudio é PCM sintético, a captura é falsa, o relógio é simulado e a IA
fica desligada (AVEIA_IA=0). Nada é gravado em `saida/` nem em `gravacoes/` do repositório (pastas temporárias).
"""
import json
import os
import sys
import tempfile
import time
import unittest
import wave
from datetime import datetime, timedelta
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("AVEIA_DADOS", tempfile.mkdtemp(prefix="aveia-dados-teste-"))
os.environ["AVEIA_CONTROLE_DIR"] = tempfile.mkdtemp(prefix="aveia-controle-teste-")
from fastapi.testclient import TestClient  # noqa: E402

import app as aplicacao  # noqa: E402
from transcritor import alinhar_teams, aovivo, captura, controle as controle_mod, saida  # noqa: E402

TAXA = captura.TAXA


class Relogio:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t


class CapturaFalsa:
    """Guarda o callback e deixa o teste empurrar blocos como a captura real faria."""
    ultima = None

    def __init__(self, cb):
        self.cb = cb
        self.iniciada = self.parada = False
        CapturaFalsa.ultima = self

    def iniciar(self):
        self.iniciada = True

    def parar(self):
        self.parada = True


def bloco(segundos, e=0.1, d=0.1):
    n = int(segundos * TAXA)
    b = np.zeros((n, 2), dtype=np.float32)
    b[:, 0], b[:, 1] = e, d
    return b.reshape(-1)


# ------------------------------------------------------------------------------------------------ captura
class TestCaptura(unittest.TestCase):
    def test_reamostrador_razao_exata_e_continuidade(self):
        for taxa in (48000, 44100):
            t = np.arange(taxa * 3) / taxa
            x = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
            r = captura.Reamostrador(taxa)
            saida_ = np.concatenate([r.processar(x[i:i + 4800]) for i in range(0, len(x), 4800)])
            esperado = len(x) * 16000 // taxa
            self.assertLessEqual(esperado - len(saida_), 400, taxa)   # só a folga final (~10 a 25 ms) fica retida
            ref = 0.5 * np.sin(2 * np.pi * 440 * np.arange(len(saida_)) / 16000)
            meio = slice(2000, len(saida_) - 2000)   # fora das bordas do início e do fim
            self.assertLess(float(np.max(np.abs(saida_[meio] - ref[meio]))), 0.02, taxa)   # sem emendas entre pedaços

    def test_preenche_silencio_pelo_relogio(self):
        rel = Relogio()
        inter = captura.Intercalador(rel)
        rel.t += 2.5
        inter.anexar(0, np.full(int(2 * TAXA), 0.3, dtype=np.float32))   # só o microfone entregou (o loopback calou)
        b = inter.coletar()
        self.assertIsNotNone(b)
        self.assertEqual(len(b) // 2, int((2.5 - captura.ATRASO_S) * TAXA))
        self.assertTrue(np.all(b[1::2] == 0.0))
        self.assertTrue(np.all(b[0::2] == np.float32(0.3)))
        self.assertGreater(inter.preenchido[1], 0)
        self.assertEqual(inter.preenchido[0], 0)

    def test_intercala_e_d_alinhados(self):
        rel = Relogio()
        inter = captura.Intercalador(rel)
        rel.t += 2.0
        inter.anexar(0, np.full(2 * TAXA, 0.5, dtype=np.float32))
        inter.anexar(1, np.full(2 * TAXA, -0.25, dtype=np.float32))
        b = inter.coletar()
        self.assertTrue(np.all(b[0::2] == np.float32(0.5)))
        self.assertTrue(np.all(b[1::2] == np.float32(-0.25)))
        self.assertEqual(len(b) % 2, 0)

    def test_sem_bloco_antes_do_tempo_e_deriva_limitada(self):
        rel = Relogio()
        inter = captura.Intercalador(rel)
        rel.t += 0.8
        self.assertIsNone(inter.coletar())   # 0,8 s menos a folga de 0,5 s: pouco para um bloco de 1 s
        inter.anexar(0, np.ones(10 * TAXA, dtype=np.float32))   # canal adiantado demais: o excesso é descartado
        inter.anexar(1, np.ones(10 * TAXA, dtype=np.float32))
        rel.t += 1.0
        inter.coletar()
        self.assertGreater(inter.descartado[0], 0)
        self.assertLessEqual(len(inter._buf[0]), int(captura.MANTER_S * TAXA))

    def test_energia_do_canal_do_computador(self):
        b = bloco(1.0, e=0.5, d=0.0)
        self.assertLess(captura.rms(b[1::2]), captura.LIMIAR_RMS_SILENCIO)
        self.assertGreater(captura.rms(b[0::2]), 0.4)


# ------------------------------------------------------------------------------------------- sessão direta
class TestSessaoDireta(unittest.TestCase):
    def test_anexar_pcm_sem_decodificador_e_wav_em_disco(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            ger = aovivo.Gerenciador(lambda fn, *a: None)   # nada roda na fila: só a recepção
            s = ger.criar_direta(Path(tmp) / "a.wav", "pt")
            self.addCleanup(s.liberar)
            self.assertTrue(s.estereo and s.pcm_direto)
            s.anexar_pcm(bloco(3.0, e=0.2, d=0.4))
            self.assertEqual(s.pcm.n, 3 * TAXA)
            e, d = s.pcm.canais()
            self.assertAlmostEqual(float(e[0]), 0.2, places=5)
            self.assertAlmostEqual(float(d[0]), 0.4, places=5)
            s.marcar_fim()
            s.anexar_pcm(bloco(1.0))   # depois do fim: ignorado
            self.assertEqual(s.pcm.n, 3 * TAXA)
            self.assertTrue(s._eof.is_set())
            with wave.open(str(s.arquivo), "rb") as w:
                self.assertEqual((w.getnchannels(), w.getframerate(), w.getnframes()), (2, TAXA, 3 * TAXA))
            outra = aovivo.Sessao("x", Path(tmp) / "b.webm", "pt", None, lambda *a: None)
            self.addCleanup(outra.liberar)
            with self.assertRaises(aovivo.AoVivoIndisponivel):
                outra.anexar_pcm(bloco(0.1))
            s.liberar()
            outra.liberar()


# --------------------------------------------------------------------------------------------- controle
FALAS = [{"inicio": 1.0, "fim": 5.0, "locutor": "Locutor 1", "texto": "Bom dia a todos, vamos começar a reunião de hoje."},
         {"inicio": 6.0, "fim": 9.0, "locutor": "Locutor 2", "texto": "Bom dia, tenho os números do projeto prontos aqui."}]


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self.tmp.cleanup)
        self.raiz = Path(self.tmp.name)
        self.rel = Relogio()
        self.agora = datetime(2026, 12, 1, 10, 0, 0)
        self.jobs: dict = {}
        self.ger = aovivo.Gerenciador(lambda fn, *a: None)

        def executar(job_id, sessao):   # no lugar da transcrição: entrega as falas prontas
            self.jobs[job_id].update(status="pronto", progresso=1.0, falas=[dict(f) for f in FALAS])

        self.ctl = controle_mod.Controle(self.ger, executar, self.jobs, lambda fn, *a: fn(*a), self.raiz / "grav", self.raiz / "saida",
                                         CapturaFalsa, agora=lambda: self.agora, relogio=self.rel)
        p = mock.patch.dict(os.environ, {"AVEIA_IA": "0", "AVEIA_CONTROLE_GUARDAR_AUDIO": ""})
        p.start()
        self.addCleanup(p.stop)
        self.addCleanup(self.ctl.encerrar)
        self.addCleanup(lambda: [x.liberar() for x in list(self.ger.sessoes.values())])

    def iniciar(self, fim_min=60):
        return self.ctl.iniciar({"evento_id": "ev-1", "titulo": "Reunião de Teste", "participantes": [{"nome": "Ana Teste", "email": "ana@exemplo.test"}],
                                 "fim_previsto": (self.agora + timedelta(minutes=fim_min)).isoformat()})

    def esperar_flag(self, pasta, limite=10.0):
        fim = time.time() + limite
        while time.time() < fim and not (Path(pasta) / "concluido.flag").exists():
            time.sleep(0.05)
        return (Path(pasta) / "concluido.flag").exists()


class TestRotas(Base):
    def setUp(self):
        super().setUp()
        p = mock.patch.object(aplicacao, "controle", self.ctl)
        p.start()
        self.addCleanup(p.stop)
        self.cli = TestClient(aplicacao.app, base_url="http://127.0.0.1:8765")
        self.addCleanup(self.cli.close)
        self.h = {"X-Aveia-Token": aplicacao.TOKEN_CONTROLE}

    def test_sem_token_ou_token_errado_401(self):
        for metodo, rota in (("get", "estado"), ("post", "iniciar"), ("post", "parar")):
            self.assertEqual(getattr(self.cli, metodo)(f"/api/controle/{rota}").status_code, 401, rota)
            self.assertEqual(getattr(self.cli, metodo)(f"/api/controle/{rota}", headers={"X-Aveia-Token": "errado"}).status_code, 401, rota)
        self.assertEqual(self.ctl.estado()["estado"], "ocioso")

    def test_estado_iniciar_parar_com_token(self):
        self.assertEqual(self.cli.get("/api/controle/estado", headers=self.h).json()["estado"], "ocioso")
        r = self.cli.post("/api/controle/iniciar", headers=self.h, json={
            "evento_id": "ev-1", "titulo": "Reunião de Teste", "participantes": [], "fim_previsto": (self.agora + timedelta(hours=1)).isoformat()})
        self.assertEqual(r.status_code, 200, r.text)
        self.assertTrue(r.json()["ok"] and r.json()["pasta_saida"].endswith("reuniao-de-teste"))
        est = self.cli.get("/api/controle/estado", headers=self.h).json()
        self.assertEqual((est["estado"], est["evento_id"], est["titulo"]), ("gravando", "ev-1", "Reunião de Teste"))
        self.assertTrue(CapturaFalsa.ultima.iniciada)
        self.assertEqual(self.cli.post("/api/controle/parar", headers=self.h, json={}).json()["ok"], True)
        self.assertTrue(CapturaFalsa.ultima.parada)
        self.assertTrue(self.esperar_flag(r.json()["pasta_saida"]))

    def test_dupla_gravacao_409(self):
        corpo = {"evento_id": "ev-1", "titulo": "A", "fim_previsto": (self.agora + timedelta(hours=1)).isoformat()}
        self.assertEqual(self.cli.post("/api/controle/iniciar", headers=self.h, json=corpo).status_code, 200)
        r = self.cli.post("/api/controle/iniciar", headers=self.h, json={**corpo, "evento_id": "ev-2"})
        self.assertEqual(r.status_code, 409)
        self.assertEqual((r.json()["motivo"], r.json()["evento_id"]), ("ja-gravando", "ev-1"))

    def test_fim_previsto_invalido_400(self):
        r = self.cli.post("/api/controle/iniciar", headers=self.h, json={"evento_id": "x", "fim_previsto": "amanhã"})
        self.assertEqual(r.status_code, 400)


class TestParada(Base):
    def alimentar(self, segundos, d):
        """Empurra áudio pelo callback da captura (1 s por bloco), com o canal D no nível `d`."""
        for _ in range(int(segundos)):
            CapturaFalsa.ultima.cb(bloco(1.0, e=0.1, d=d), captura.rms(bloco(1.0, d=d)[1::2]), 0.1)

    def test_nao_para_antes_do_fim_previsto(self):
        self.assertEqual(self.iniciar(fim_min=60)[0], 200)
        self.alimentar(700, d=0.0)   # silêncio longo, mas ainda antes do fim
        self.assertIsNone(self.ctl.tick())
        self.assertEqual(self.ctl.estado()["estado"], "gravando")

    def test_para_por_silencio_depois_do_fim(self):
        self.iniciar(fim_min=60)
        self.alimentar(5, d=0.3)
        self.alimentar(601, d=0.0)
        self.agora += timedelta(minutes=61)
        self.assertEqual(self.ctl.tick(), "silencio")
        self.assertTrue(CapturaFalsa.ultima.parada)

    def test_som_no_computador_zera_a_contagem_do_silencio(self):
        self.iniciar(fim_min=1)
        self.alimentar(500, d=0.0)
        self.alimentar(2, d=0.3)
        self.alimentar(200, d=0.0)
        self.agora += timedelta(minutes=2)
        self.assertIsNone(self.ctl.tick())   # só 200 s de silêncio seguido

    def test_para_por_horario_limite(self):
        self.iniciar(fim_min=60)
        self.alimentar(5, d=0.3)   # ainda há som: o silêncio não manda parar
        self.agora += timedelta(minutes=89)
        self.assertIsNone(self.ctl.tick())
        self.agora += timedelta(minutes=2)   # 91 min: passou de fim_previsto + 30 min
        self.assertEqual(self.ctl.tick(), "limite")

    def test_parada_manual_gera_a_saida_com_flag_e_sem_ia(self):
        _, r = self.iniciar()
        self.alimentar(3, d=0.2)
        self.agora += timedelta(minutes=5)
        self.ctl.parar("manual")
        self.assertTrue(self.esperar_flag(r["pasta_saida"]))
        pasta = Path(r["pasta_saida"])
        meta = json.loads((pasta / "meta.json").read_text(encoding="utf-8"))
        self.assertIs(meta["ia"], False)
        self.assertEqual((meta["evento_id"], meta["origem_nomes"]), ("ev-1", "canal+pistas"))
        self.assertEqual(meta["locutores"], {"Locutor 1": None, "Locutor 2": None})
        self.assertAlmostEqual(meta["duracao_s"], 3.0, places=1)
        self.assertIn("[00:00:01] Locutor 1: Bom dia", (pasta / "transcricao.md").read_text(encoding="utf-8"))
        resumo = (pasta / "resumo.md").read_text(encoding="utf-8")
        for secao in ("## Resumo", "## Pontos-chave", "## Decisões", "## Próximos passos"):
            self.assertIn(secao, resumo)
        self.assertFalse(list(pasta.glob("*.tmp")))
        self.assertFalse(list((self.raiz / "grav").glob("*.wav")))   # sucesso: o áudio não fica no disco
        self.assertEqual(self.ctl.estado()["estado"], "ocioso")

    def test_ia_desligada_nao_chama_nada(self):
        g = {"titulo": "T", "participantes": [{"nome": "Ana Teste", "email": ""}], "usuario": "", "duracao_s": 9}
        with mock.patch("transcritor.resumo.gerar_resumo", side_effect=AssertionError("chamou a IA")), \
                mock.patch("transcritor.locutores.identificar", side_effect=AssertionError("chamou a IA")):
            nomes, origem, usou, resumo, _ = self.ctl.nomear_e_resumir(g, [dict(f) for f in FALAS])
        self.assertEqual((usou, resumo, origem), (False, None, "canal+pistas"))
        self.assertEqual(nomes, {"Locutor 1": None, "Locutor 2": None})


# ---------------------------------------------------------------------------------------------- saída
class TestSaida(unittest.TestCase):
    def test_pasta_estavel_e_unica(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            quando = datetime(2026, 12, 1, 9, 30)
            p1 = saida.pasta_para(Path(tmp), quando, "Reunião Semanal: Ações & Metas!")
            self.assertEqual(p1.name, "2026-12-01_0930_reuniao-semanal-acoes-metas")
            p1.mkdir()
            self.assertEqual(saida.pasta_para(Path(tmp), quando, "Reunião Semanal: Ações & Metas!").name, p1.name + "-2")
            self.assertEqual(saida.slug("???"), "reuniao")

    def test_flag_por_ultimo_e_atomico(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            pasta = Path(tmp) / "x"
            ordem = []
            real = saida._atomico

            def espiao(destino, conteudo):
                ordem.append(destino.name)
                real(destino, conteudo)

            with mock.patch.object(saida, "_atomico", espiao):
                saida.gravar(pasta, {"titulo": "T"}, [dict(f) for f in FALAS], {"Locutor 1": "Ana", "Locutor 2": None}, saida.montar_resumo(None))
            self.assertEqual(ordem[-1], "concluido.flag")
            self.assertEqual(sorted(p.name for p in pasta.iterdir()),
                             ["concluido.flag", "falas.json", "locutores.json", "meta.json", "resumo.md", "transcricao.md"])
            self.assertIn("Ana: Bom dia", (pasta / "transcricao.md").read_text(encoding="utf-8"))


# ----------------------------------------------------------------------------------------- alinhamento
VTT = """WEBVTT

00:00:10.000 --> 00:00:14.500
<v Ana Teste>Bom dia a todos, vamos começar a reunião de hoje.</v>

00:00:15.000 --> 00:00:18.000
<v Bruno Exemplo>Bom dia, tenho os números do projeto prontos aqui.</v>
"""


class TestAlinhar(unittest.TestCase):
    def test_le_vtt_do_teams(self):
        cortes = alinhar_teams.ler_vtt(VTT)
        self.assertEqual([(c["nome"], c["inicio"], c["fim"]) for c in cortes], [("Ana Teste", 10.0, 14.5), ("Bruno Exemplo", 15.0, 18.0)])

    def test_estima_deslocamento_e_nomeia_pela_sobreposicao(self):
        cortes = alinhar_teams.ler_vtt(VTT)
        desloc, pares = alinhar_teams.estimar_deslocamento(FALAS, cortes)
        self.assertEqual((desloc, pares), (9.0, 2))
        self.assertEqual(alinhar_teams.nomear(FALAS, cortes, desloc), {"Locutor 1": "Ana Teste", "Locutor 2": "Bruno Exemplo"})

    def test_reescreve_os_arquivos_do_contrato(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            pasta = Path(tmp) / "reuniao"
            saida.gravar(pasta, {"titulo": "T", "origem_nomes": "canal+pistas", "ia": False}, [dict(f) for f in FALAS],
                         {"Locutor 1": None, "Locutor 2": None}, saida.montar_resumo(None))
            vtt = Path(tmp) / "a.vtt"
            vtt.write_text(VTT, encoding="utf-8")
            r = alinhar_teams.alinhar(pasta, vtt)
            self.assertEqual((r["nomeados"], r["locutores"]), (2, 2))
            meta = json.loads((pasta / "meta.json").read_text(encoding="utf-8"))
            self.assertEqual((meta["origem_nomes"], meta["locutores"]["Locutor 2"]), ("teams", "Bruno Exemplo"))
            self.assertEqual(json.loads((pasta / "locutores.json").read_text(encoding="utf-8"))["Locutor 1"], "Ana Teste")
            self.assertIn("] Bruno Exemplo: Bom dia", (pasta / "transcricao.md").read_text(encoding="utf-8"))
            self.assertTrue((pasta / "concluido.flag").exists())


if __name__ == "__main__":
    unittest.main()
