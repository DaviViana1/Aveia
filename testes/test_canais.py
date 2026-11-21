"""Testes offline do áudio do computador em canal separado (transcritor/canais.py + pipeline + ao vivo + rotas).

    .venv\\Scripts\\python -m unittest testes.test_canais -v

Sem modelo e sem internet. O sinal é sintético mas feito de FALA REAL: trechos de `testes/diarizacao/cache/mls/<locutor>/*.flac`
(um locutor "local" no canal esquerdo, dois "remotos" no direito), com lacunas de silêncio, um trecho de fala sobreposta e duas
variantes: fone (sem eco) e alto-falante (eco = 0,2 x canal do computador, atrasado 30 ms, somado ao microfone). O VAD (Silero,
empacotado com o faster-whisper) e o decodificador (PyAV) são reais; Whisper e diarização são falsos.
Nada é gravado em `gravacoes/`: `app.GRAVACOES` aponta para uma pasta temporária.
"""
import glob
import os
import sys
import tempfile
import time
import unittest
import wave
from collections import namedtuple
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("AVEIA_DADOS", __import__("tempfile").mkdtemp(prefix="aveia-dados-teste-"))   # métricas dos testes não vão para dados/
from faster_whisper.audio import decode_audio  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import app as aplicacao  # noqa: E402
from transcritor import aovivo, canais, pipeline  # noqa: E402
from transcritor.diarizacao import Trecho  # noqa: E402
from transcritor.motores import Palavra  # noqa: E402

MLS = Path(__file__).resolve().parent / "diarizacao" / "cache" / "mls"
LOCAL, REMOTO_B, REMOTO_C = "12114_10229_000", "11247_10229_000", "12287_12742_000"
T = 16000
P = namedtuple("P", "start end word")

# Linha do tempo (s) da cena sintética. Local = microfone (E); remotos = computador (D).
FALAS_LOCAIS = [(1.0, 0), (21.0, 1), (41.0, 2)]          # (início, índice do clipe)
FALAS_REMOTAS = [(11.0, REMOTO_B, 0), (31.0, REMOTO_C, 0), (43.0, REMOTO_B, 1)]   # a de 43 s sobrepõe a local de 41 s
DURACAO_CLIPE = 8.0
TOTAL_S = 54


def _clipes(pasta: str, n: int = 3) -> list[np.ndarray]:
    arqs = sorted(glob.glob(str(MLS / pasta / "*.flac")))[:n]
    return [decode_audio(a, sampling_rate=T)[:int(DURACAO_CLIPE * T)] for a in arqs]


def _colocar(canal: np.ndarray, clipe: np.ndarray, t: float) -> None:
    i = int(t * T)
    canal[i:i + len(clipe)] += clipe


_cena_cache: dict = {}


def cena(eco: float = 0.0, atraso: float = 0.030):
    """(esq, dir, so_local): esq = microfone (+ eco), dir = computador, so_local = o locutor local sozinho (gabarito)."""
    chave = (eco, atraso)
    if chave in _cena_cache:
        return _cena_cache[chave]
    if "clipes" not in _cena_cache:
        _cena_cache["clipes"] = {p: _clipes(p) for p in (LOCAL, REMOTO_B, REMOTO_C)}
    clipes = _cena_cache["clipes"]
    n = TOTAL_S * T
    esq, dir, so_local = (np.zeros(n, np.float32) for _ in range(3))
    for t, i in FALAS_LOCAIS:
        _colocar(so_local, clipes[LOCAL][i], t)
    for t, quem, i in FALAS_REMOTAS:
        _colocar(dir, clipes[quem][i], t)
    esq += so_local
    if eco:
        d = int(atraso * T)
        esq[d:] += eco * dir[:-d]
    rng = np.random.default_rng(7)   # ruído de fundo (-70 dBFS): os canais nunca são silêncio digital
    esq += 3e-4 * rng.standard_normal(n).astype(np.float32)
    dir += 3e-4 * rng.standard_normal(n).astype(np.float32)
    _cena_cache[chave] = (esq, dir, so_local)
    return _cena_cache[chave]


def quadros(trechos, n_quadros: int) -> np.ndarray:
    m = np.zeros(n_quadros, bool)
    for a, b in trechos:
        m[int(round(a / canais.QUADRO_S)):int(round(b / canais.QUADRO_S))] = True
    return m


def metricas(esq, dir, so_local):
    """Precisão/revocação por quadro dos trechos locais contra o gabarito (VAD do local sozinho, sem eco)."""
    r = canais.separar(esq, dir)
    nq = len(esq) // int(canais.QUADRO_S * T)
    pred = quadros(r.trechos_locais, nq)
    ver = canais._fala_quadros(so_local, nq)
    tp = int((pred & ver).sum())
    precisao, revocacao = tp / max(int(pred.sum()), 1), tp / max(int(ver.sum()), 1)
    # eco puro = quadros em que só o computador fala (o local está calado)
    remoto_sozinho = canais._fala_quadros(dir, nq) & ~_dilatar_ver(ver)
    falso_eco = float((pred & remoto_sozinho).sum()) / max(int(remoto_sozinho.sum()), 1)
    return r, precisao, revocacao, falso_eco


def _dilatar_ver(ver: np.ndarray) -> np.ndarray:
    return canais._dilatar(ver, 6) | canais._dilatar(ver[::-1], 6)[::-1]


@unittest.skipUnless(MLS.exists(), "faltam os clipes de fala em testes/diarizacao/cache/mls")
class SeparacaoTest(unittest.TestCase):
    def _checar(self, nome, eco, piso_prec, piso_rev):
        esq, dir, so_local = cena(eco)
        r, prec, rev, falso_eco = metricas(esq, dir, so_local)
        print(f"\n  [{nome}] válido={r.valido} eco={r.eco:.4f} precisão={prec:.3f} revocação={rev:.3f} "
              f"quadros de eco puro marcados como local={falso_eco:.3%} trechos={len(r.trechos_locais)}")
        self.assertTrue(r.valido, r.motivo)
        self.assertGreaterEqual(prec, piso_prec)
        self.assertGreaterEqual(rev, piso_rev)
        self.assertLess(falso_eco, 0.03)
        return r

    def _revocacao_fala_sobreposta(self, eco):
        """Revocação do local só no trecho de fala sobreposta (43–49 s: o remoto fala junto)."""
        esq, dir, so_local = cena(eco)
        r = canais.separar(esq, dir)
        nq = len(esq) // int(canais.QUADRO_S * T)
        pred, ver = quadros(r.trechos_locais, nq), canais._fala_quadros(so_local, nq)
        ini, fim = int(43 / canais.QUADRO_S), int(49 / canais.QUADRO_S)
        rev = float((pred[ini:fim] & ver[ini:fim]).sum()) / max(int(ver[ini:fim].sum()), 1)
        print(f"\n  [fala sobreposta, eco={eco}] revocação do local em 43–49 s = {rev:.3f}")
        return rev

    def test_fone_sem_eco(self):
        r = self._checar("fone", 0.0, 0.90, 0.95)
        self.assertLess(r.eco, 0.01)
        self.assertGreater(self._revocacao_fala_sobreposta(0.0), 0.90)

    def test_alto_falante_com_eco(self):
        r = self._checar("eco 0,2", 0.2, 0.90, 0.90)
        self.assertGreater(r.eco, 0.02)       # estimou o eco
        self.assertGreater(self._revocacao_fala_sobreposta(0.2), 0.85)

    def test_eco_forte(self):
        self._checar("eco 0,4", 0.4, 0.90, 0.80)

    def test_eco_puro_nao_e_local(self):
        esq, dir, _ = cena(0.2)
        r = canais.separar(esq, dir)
        for ini, fim in ((11.0, 19.0), (31.0, 39.0)):   # só o computador fala
            self.assertFalse([t for t in r.trechos_locais if t[0] < fim and t[1] > ini], (ini, fim, r.trechos_locais))

    def test_mix_protegida_contra_estouro(self):
        esq = np.full(T, 0.9, np.float32)
        mix = canais.misturar(esq, esq.copy())
        self.assertLessEqual(float(np.abs(mix).max()), canais.PICO_MIX + 1e-6)
        pequeno = canais.misturar(esq * 0.1, esq * 0.1)
        np.testing.assert_allclose(pequeno, esq * 0.2, atol=1e-6)   # sem estouro: soma pura

    def test_canais_identicos_nao_sao_validos(self):
        esq, _, _ = cena(0.0)
        r = canais.separar(esq, esq.copy())
        self.assertFalse(r.valido)
        self.assertEqual(r.trechos_locais, [])
        self.assertEqual(len(r.mix), len(esq))

    def test_copia_com_ganho_e_ruido_tambem_nao(self):
        esq, _, _ = cena(0.0)
        rng = np.random.default_rng(1)
        r = canais.separar(esq, 0.7 * esq + 1e-4 * rng.standard_normal(len(esq)).astype(np.float32))
        self.assertFalse(r.valido)

    def test_sistema_em_silencio_nao_e_valido(self):
        esq, dir, _ = cena(0.0)
        r = canais.separar(esq, np.zeros_like(dir))
        self.assertFalse(r.valido)
        self.assertIn("computador", r.motivo)

    def test_microfone_em_silencio_nao_e_valido(self):
        esq, dir, _ = cena(0.0)
        self.assertFalse(canais.separar(np.zeros_like(esq), dir).valido)

    def test_canal_so_com_ruido_nao_e_valido(self):
        esq, dir, _ = cena(0.0)
        ruido = 3e-4 * np.random.default_rng(3).standard_normal(len(dir)).astype(np.float32)
        r = canais.separar(esq, ruido)
        self.assertFalse(r.valido)   # sem fala no computador

    def test_audio_curto(self):
        r = canais.separar(np.zeros(100, np.float32), np.zeros(100, np.float32))
        self.assertFalse(r.valido)


# ------------------------------------------------------------------------------------------------ montar_falas
class MontarFalasLocaisTest(unittest.TestCase):
    def test_local_e_sempre_locutor_1(self):
        pal = [P(12.0, 12.4, " a"), P(22.0, 22.4, " b"), P(32.0, 32.4, " c")]
        trechos = [Trecho(11, 19, 0), Trecho(31, 39, 1)]
        falas = pipeline.montar_falas(pal, trechos, locais=[(21.0, 27.0)])
        self.assertEqual([f.locutor for f in falas], ["Locutor 2", "Locutor 1", "Locutor 3"])

    def test_sem_locais_o_comportamento_nao_muda(self):
        pal = [P(12.0, 12.4, " a"), P(32.0, 32.4, " c")]
        falas = pipeline.montar_falas(pal, [Trecho(11, 19, 5), Trecho(31, 39, 9)])
        self.assertEqual([f.locutor for f in falas], ["Locutor 1", "Locutor 2"])

    def test_maior_sobreposicao_decide(self):
        pal = [P(10.0, 11.0, " x")]
        # 0,3 s com o local e 0,7 s com o remoto: remoto
        falas = pipeline.montar_falas(pal, [Trecho(10.3, 12, 0)], locais=[(9.0, 10.3)])
        self.assertEqual(falas[0].locutor, "Locutor 2")
        # 0,7 s com o local e 0,3 s com o remoto: local
        falas = pipeline.montar_falas(pal, [Trecho(10.7, 12, 0)], locais=[(9.0, 10.7)])
        self.assertEqual(falas[0].locutor, "Locutor 1")

    def test_sem_trechos_remotos_o_que_nao_e_local_vira_remoto(self):
        pal = [P(2.0, 2.4, " a"), P(12.0, 12.4, " b")]
        falas = pipeline.montar_falas(pal, [], locais=[(1.0, 5.0)])
        self.assertEqual([f.locutor for f in falas], ["Locutor 1", "Locutor 2"])

    def test_palavra_fora_de_tudo_vai_ao_mais_proximo(self):
        pal = [P(6.0, 6.2, " a"), P(9.5, 9.7, " b")]
        falas = pipeline.montar_falas(pal, [Trecho(10, 15, 0)], locais=[(1.0, 5.5)], suavizar=False)
        self.assertEqual([f.locutor for f in falas], ["Locutor 1", "Locutor 2"])


# ------------------------------------------------------------------------------------------------ pipeline.transcrever
@unittest.skipUnless(MLS.exists(), "faltam os clipes de fala em testes/diarizacao/cache/mls")
class TranscreverCanaisTest(unittest.TestCase):
    def setUp(self):
        esq, dir, _ = cena(0.2)
        self.esq, self.dir = esq, dir
        self.stereo = (esq, dir)
        self.chamadas_decode: list[dict] = []
        self.diar: list[dict] = []
        # (início, fim, texto): local em 2–3 s e 22–23 s (dentro da fala do local); remotos em 12–13 s e 32–33 s
        self.palavras = [P(2.0, 3.0, " oi"), P(12.0, 13.0, " bom"), P(22.0, 23.0, " dia"), P(32.0, 33.0, " tchau")]

        def decode(caminho, sampling_rate=16000, split_stereo=False):
            self.chamadas_decode.append({"split_stereo": split_stereo})
            if isinstance(self.stereo, Exception):
                raise self.stereo
            if split_stereo:
                return self.stereo
            return (self.stereo[0] + self.stereo[1]) / 2

        def palavras(audio, duracao, idioma, avisar, motor=None, meta=None):
            meta = meta if meta is not None else {}
            meta.update(motor="local", avisos=meta.get("avisos", []))
            self.audio_whisper = audio
            yield from self.palavras

        def diarizar(audio, num_locutores=None):
            self.diar.append({"audio": audio, "n": num_locutores})
            return [Trecho(11, 19, 0), Trecho(31, 39, 1)]

        for alvo, nome, novo in [(pipeline, "decode_audio", decode), (pipeline, "preparar_audio", lambda a: a),
                                 (pipeline.motores, "palavras", palavras), (pipeline, "diarizar", diarizar)]:
            p = mock.patch.object(alvo, nome, novo)
            p.start()
            self.addCleanup(p.stop)

    def test_dois_canais_rotulam_pelo_canal(self):
        meta: dict = {}
        falas = pipeline.transcrever("x.webm", motor="local", meta=meta, canais="mic+sistema")
        self.assertEqual([(f.locutor, f.texto) for f in falas],
                         [("Locutor 1", "oi"), ("Locutor 2", "bom"), ("Locutor 1", "dia"), ("Locutor 3", "tchau")])
        self.assertEqual(meta["canais"], "mic+sistema")
        self.assertEqual(meta["locutor_local"], "Locutor 1")
        self.assertEqual(self.chamadas_decode, [{"split_stereo": True}])        # decodificou só uma vez, estéreo
        # Whisper transcreve a mistura; a diarização só o canal do computador
        np.testing.assert_allclose(self.audio_whisper, canais.misturar(self.esq, self.dir))
        self.assertEqual(len(self.diar), 1)
        np.testing.assert_array_equal(self.diar[0]["audio"], self.dir)

    def test_remoto_primeiro_e_local_continua_locutor_1(self):
        self.palavras = [P(12.0, 13.0, " bom"), P(22.0, 23.0, " dia")]
        falas = pipeline.transcrever("x.webm", motor="local", canais="mic+sistema")
        self.assertEqual([f.locutor for f in falas], ["Locutor 2", "Locutor 1"])

    def test_num_locutores_conta_todos(self):
        for informado, esperado in ((None, None), (3, 2), (2, 1), (1, 1)):
            self.diar.clear()
            pipeline.transcrever("x.webm", motor="local", num_locutores=informado, canais="mic+sistema")
            self.assertEqual(self.diar[0]["n"], esperado, informado)

    def test_canais_identicos_voltam_ao_mono(self):
        self.stereo = (self.esq, self.esq.copy())
        meta: dict = {}
        falas = pipeline.transcrever("x.webm", motor="local", meta=meta, num_locutores=3, canais="mic+sistema")
        self.assertNotIn("canais", meta)
        self.assertNotIn("locutor_local", meta)
        self.assertEqual([f.locutor for f in falas], ["Locutor 1", "Locutor 2"])   # diarização mono (fake); palavras do mesmo locutor com pausa < 30 s são uma fala só
        self.assertEqual(self.diar[0]["n"], 3)                                  # num_locutores intacto
        self.assertEqual(self.chamadas_decode, [{"split_stereo": True}, {"split_stereo": False}])

    def test_arquivo_que_nao_decodifica_em_estereo_cai_no_mono(self):
        meta: dict = {}
        with mock.patch.object(pipeline.canais_mod, "separar", side_effect=RuntimeError("boom")):
            falas = pipeline.transcrever("x.webm", motor="local", meta=meta, canais="mic+sistema")
        self.assertNotIn("canais", meta)
        self.assertEqual(len(falas), 2)   # cai no mono: Locutor 1 (3 palavras, pausas < 30 s) e Locutor 2

    def test_valor_desconhecido_e_mono(self):
        for valor in (None, "", "estereo", "MIC+SISTEMA2"):
            self.chamadas_decode.clear()
            meta: dict = {}
            pipeline.transcrever("x.webm", motor="local", meta=meta, canais=valor)
            self.assertEqual(self.chamadas_decode, [{"split_stereo": False}], valor)
            self.assertNotIn("canais", meta)

    def test_mono_sem_canais_nao_muda(self):
        falas = pipeline.transcrever("x.webm", motor="local")
        self.assertEqual([f.locutor for f in falas], ["Locutor 1", "Locutor 2"])   # pausa < 30 s do mesmo locutor = uma fala
        self.assertEqual(self.diar[0]["n"], None)

    def test_falha_da_diarizacao_dos_remotos_preserva_o_texto(self):
        meta: dict = {}
        with mock.patch.object(pipeline, "diarizar", side_effect=RuntimeError("modelo ausente")):
            falas = pipeline.transcrever("x.webm", motor="local", meta=meta, canais="mic+sistema")
        self.assertEqual([f.locutor for f in falas], ["Locutor 1", "Locutor 2", "Locutor 1", "Locutor 2"])
        self.assertIn("modelo ausente", meta["avisos"][0])
        self.assertEqual(meta["locutor_local"], "Locutor 1")


# ------------------------------------------------------------------------------------------------ ao vivo + rotas
def escrever_wav(caminho: Path, esq: np.ndarray, dir: np.ndarray) -> None:
    pcm = np.empty(len(esq) * 2, dtype=np.int16)
    pcm[0::2] = np.clip(esq, -1, 1) * 32767
    pcm[1::2] = np.clip(dir, -1, 1) * 32767
    with wave.open(str(caminho), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(T)
        w.writeframes(pcm.tobytes())


def esperar(cond, limite=30.0, passo=0.05):
    fim = time.time() + limite
    while time.time() < fim:
        v = cond()
        if v:
            return v
        time.sleep(passo)
    return cond()


@unittest.skipUnless(MLS.exists(), "faltam os clipes de fala em testes/diarizacao/cache/mls")
class AoVivoCanaisTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.pasta = Path(self.tmp.name)
        self.addCleanup(lambda: self._limpar())
        self.diar: list[dict] = []

        def diarizar(audio, num_locutores=None):
            self.diar.append({"audio": audio, "n": num_locutores})
            return [Trecho(11, 19, 0), Trecho(31, 39, 1)]

        for alvo, nome, novo in [(pipeline, "_palavras", lambda *a, **k: iter(())), (pipeline, "diarizar", diarizar)]:
            p = mock.patch.object(alvo, nome, novo)
            p.start()
            self.addCleanup(p.stop)
        self.palavras = [Palavra(2.0, 3.0, " oi"), Palavra(12.0, 13.0, " bom"), Palavra(22.0, 23.0, " dia"), Palavra(32.0, 33.0, " tchau")]
        self.sessoes: list = []

    def _limpar(self):
        for s in self.sessoes:
            s.liberar()
        try:
            self.tmp.cleanup()
        except OSError:
            pass

    def sessao(self, esq, dir, canais_pedido="mic+sistema", nome="a.wav"):
        arq = self.pasta / nome
        escrever_wav(arq, esq, dir)
        dados = arq.read_bytes()
        s = aovivo.Sessao("s" * 32, self.pasta / ("rec-" + nome), "pt", None, lambda fn, *a: None, canais=canais_pedido)
        self.sessoes.append(s)
        for i in range(0, len(dados), 32768):
            s.receber(i // 32768, dados[i:i + 32768])
        s.marcar_fim(len(esq) / T)
        self.assertTrue(s._eof.wait(60), "decodificador não terminou")
        s._palavras.extend(self.palavras)   # sem Whisper: as palavras entram prontas, com tempo absoluto
        return s

    def test_sessao_estereo_rotula_pelo_canal(self):
        esq, dir, _ = cena(0.2)
        s = self.sessao(esq, dir)
        self.assertTrue(s.estereo)
        self.assertIsInstance(s.pcm, aovivo._PcmDuplo)
        self.assertAlmostEqual(s.pcm.n / T, TOTAL_S, delta=0.1)
        meta: dict = {}
        falas = s.concluir(meta=meta)
        self.assertEqual([(f.locutor, f.texto) for f in falas],
                         [("Locutor 1", "oi"), ("Locutor 2", "bom"), ("Locutor 1", "dia"), ("Locutor 3", "tchau")])
        self.assertEqual(meta["canais"], "mic+sistema")
        self.assertEqual(meta["locutor_local"], "Locutor 1")
        self.assertEqual(meta["motor"], "local")
        # a diarização recebeu só o canal do computador (a menos da quantização de 16 bits do WAV)
        self.assertEqual(len(self.diar), 1)
        n = min(len(dir), len(self.diar[0]["audio"]))
        np.testing.assert_allclose(self.diar[0]["audio"][:n], dir[:n], atol=2e-4)
        # a mistura que o Whisper recebe é a soma dos canais
        mix = s.pcm.fatia(0, 10 * T)
        np.testing.assert_allclose(mix, np.clip(esq[:10 * T] + dir[:10 * T], -0.99, 0.99), atol=4e-4)

    def test_canais_identicos_viram_mono(self):
        esq, _, _ = cena(0.0)
        s = self.sessao(esq, esq.copy())
        meta: dict = {}
        falas = s.concluir(meta=meta)
        self.assertNotIn("canais", meta)
        self.assertEqual([f.locutor for f in falas], ["Locutor 1", "Locutor 2"])   # diarização mono (fake); palavras do mesmo locutor com pausa < 30 s são uma fala só

    def test_arquivo_mono_com_canais_pedidos_decodifica_como_mono(self):
        audio = decode_audio(str(Path(__file__).resolve().parent / "audio-referencia" / "audio.webm"), sampling_rate=T)[:20 * T]
        arq = self.pasta / "mono.wav"
        with wave.open(str(arq), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(T)
            w.writeframes((np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes())
        dados = arq.read_bytes()
        s = aovivo.Sessao("m" * 32, self.pasta / "rec-mono.wav", "pt", None, lambda fn, *a: None, canais="mic+sistema")
        self.sessoes.append(s)
        s.receber(0, dados)
        s.marcar_fim(20.0)
        self.assertTrue(s._eof.wait(60))
        self.assertFalse(s.estereo)
        self.assertIsInstance(s.pcm, aovivo._Pcm)
        self.assertAlmostEqual(s.pcm.n / T, 20.0, delta=0.1)
        s._palavras.extend(self.palavras)
        meta: dict = {}
        falas = s.concluir(meta=meta)
        self.assertNotIn("canais", meta)
        self.assertEqual(len(falas), 1)   # mono sem trechos de diarização: um locutor e pausas < 30 s = uma fala

    def test_sessao_mono_sem_canais_inalterada(self):
        esq, dir, _ = cena(0.0)
        s = self.sessao(esq, dir, canais_pedido=None, nome="b.wav")   # arquivo estéreo, mas ninguém pediu: decodifica como mono
        self.assertFalse(s.estereo)
        self.assertIsInstance(s.pcm, aovivo._Pcm)
        self.assertIsNone(s.canais)


@unittest.skipUnless(MLS.exists(), "faltam os clipes de fala em testes/diarizacao/cache/mls")
class RotasCanaisTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.pasta = Path(self.tmp.name)
        for p in (mock.patch.object(aplicacao, "GRAVACOES", self.pasta),
                  mock.patch.dict(os.environ, {"AVEIA_PREPROC": "", "AVEIA_MOTOR": "local"})):
            p.start()
            self.addCleanup(p.stop)
        aplicacao.aovivo_sessoes.sessoes.clear()
        self.cli = TestClient(aplicacao.app, base_url="http://127.0.0.1:8765")
        self.addCleanup(self._limpar)

    def _limpar(self):
        for s in list(aplicacao.aovivo_sessoes.sessoes.values()):
            s.liberar()
        aplicacao.aovivo_sessoes.sessoes.clear()
        self.cli.close()
        try:
            self.tmp.cleanup()
        except OSError:
            pass

    def test_aovivo_valida_canais(self):
        for enviado, esperado in (("mic+sistema", "mic+sistema"), ("outra-coisa", None), ("", None), (None, None)):
            corpo = {"idioma": "pt", "motor": "local", "ext": "webm"}
            if enviado is not None:
                corpo["canais"] = enviado
            sid = self.cli.post("/api/aovivo", json=corpo).json()["sessao"]
            self.assertEqual(aplicacao.aovivo_sessoes.obter(sid).canais, esperado, enviado)

    def test_transcrever_repassa_canais_e_expoe_no_job(self):
        vistos = []

        def falso(caminho, **kw):
            vistos.append(kw.get("canais"))
            kw["meta"].update(motor="local", avisos=[], canais=kw.get("canais"), locutor_local="Locutor 1" if kw.get("canais") else None)
            return [pipeline.Fala(0.0, 1.0, "Locutor 1", "oi")]

        for enviado, esperado in (("mic+sistema", "mic+sistema"), ("lixo", None), ("", None)):
            with mock.patch.object(aplicacao, "transcrever", falso):
                job = self.cli.post("/api/transcrever", files={"audio": ("g.webm", b"x")},
                                    data={"idioma": "pt", "canais": enviado}).json()["job"]
                r = esperar(lambda: (lambda j: j if j["status"] != "processando" else None)(self.cli.get(f"/api/jobs/{job}").json()), 20)
            self.assertEqual(r["status"], "pronto", r)
            self.assertEqual(vistos[-1], esperado)
            self.assertEqual(r["canais"], esperado)
            self.assertEqual(r["locutor_local"], "Locutor 1" if esperado else None)

    def test_job_classico_expoe_lacunas(self):
        lac = [{"inicio": 57.0, "fim": 100.0, "fala_s": 30.0, "palavras_recuperadas": 0, "estado": "nao_recuperada"}]

        def falso(caminho, **kw):
            kw["meta"].update(motor="local", avisos=["aviso"], lacunas=lac)
            return [pipeline.Fala(0.0, 1.0, "Locutor 1", "oi")]
        with mock.patch.object(aplicacao, "transcrever", falso):
            job = self.cli.post("/api/transcrever", files={"audio": ("g.webm", b"x")}, data={"idioma": "pt"}).json()["job"]
            r = esperar(lambda: (lambda j: j if j["status"] != "processando" else None)(self.cli.get(f"/api/jobs/{job}").json()), 20)
        self.assertEqual(r["status"], "pronto", r)
        self.assertEqual(r["lacunas"], lac)

    def test_job_sem_lacunas_devolve_lista_vazia(self):
        def falso(caminho, **kw):
            kw["meta"].update(motor="local", avisos=[])
            return []
        with mock.patch.object(aplicacao, "transcrever", falso):
            job = self.cli.post("/api/transcrever", files={"audio": ("g.webm", b"x")}, data={"idioma": "pt"}).json()["job"]
            r = esperar(lambda: (lambda j: j if j["status"] != "processando" else None)(self.cli.get(f"/api/jobs/{job}").json()), 20)
        self.assertEqual(r["lacunas"], [])

    def test_transcrever_sem_o_campo_e_mono(self):
        vistos = []

        def falso(caminho, **kw):
            vistos.append(kw.get("canais"))
            return []
        with mock.patch.object(aplicacao, "transcrever", falso):
            job = self.cli.post("/api/transcrever", files={"audio": ("g.webm", b"x")}, data={"idioma": "pt"}).json()["job"]
            esperar(lambda: self.cli.get(f"/api/jobs/{job}").json()["status"] != "processando", 20)
        self.assertEqual(vistos, [None])


if __name__ == "__main__":
    unittest.main()
