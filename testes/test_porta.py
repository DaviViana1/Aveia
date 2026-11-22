"""Escolha da porta ao iniciar: o Aveia não briga com outro programa nem abre duas vezes a mesma pasta.

    .venv\Scripts\python -m unittest testes.test_porta -v
"""
import io
import json
import os
import socket
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
os.environ.setdefault("AVEIA_DADOS", tempfile.mkdtemp(prefix="aveia-dados-teste-"))
os.environ.setdefault("AVEIA_SEM_AQUECER", "1")
import app  # noqa: E402


def _ocupar() -> tuple[socket.socket, int]:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    s.listen(5)
    return s, s.getsockname()[1]


class _Resposta(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class PortaTest(unittest.TestCase):
    def setUp(self):
        self._env = mock.patch.dict(os.environ, {}, clear=False)
        self._env.start()
        os.environ.pop("AVEIA_PORTA", None)

    def tearDown(self):
        self._env.stop()

    def test_porta_livre_e_usada(self):
        s, porta = _ocupar()
        s.close()
        with mock.patch.object(app, "PORTA", porta):
            self.assertEqual(app._escolher_porta(), (porta, False))

    def test_outro_programa_na_porta_manda_para_a_proxima(self):
        s, porta = _ocupar()
        try:
            with mock.patch.object(app, "PORTA", porta), \
                    mock.patch.object(app.urllib.request, "urlopen", side_effect=OSError("não é o Aveia")):
                escolhida, rodando = app._escolher_porta()
            self.assertFalse(rodando)
            self.assertGreater(escolhida, porta)
        finally:
            s.close()

    def test_mesma_pasta_ja_rodando_nao_abre_de_novo(self):
        s, porta = _ocupar()
        try:
            corpo = json.dumps({"app": "aveia", "pasta": app._id_da_pasta()}).encode()
            with mock.patch.object(app, "PORTA", porta), \
                    mock.patch.object(app.urllib.request, "urlopen", return_value=_Resposta(corpo)):
                self.assertEqual(app._escolher_porta(), (porta, True))
        finally:
            s.close()

    def test_outra_copia_do_aveia_na_porta_nao_conta_como_esta(self):
        s, porta = _ocupar()
        try:
            corpo = json.dumps({"app": "aveia", "pasta": "outra-pasta"}).encode()
            with mock.patch.object(app, "PORTA", porta), \
                    mock.patch.object(app.urllib.request, "urlopen", return_value=_Resposta(corpo)):
                escolhida, rodando = app._escolher_porta()
            self.assertFalse(rodando)
            self.assertGreater(escolhida, porta)
        finally:
            s.close()

    def test_porta_fixada_por_variavel_nao_procura_outra(self):
        os.environ["AVEIA_PORTA"] = "8799"
        with mock.patch.object(app, "PORTA", 8799):
            self.assertEqual(app._escolher_porta(), (8799, False))

    def test_rota_instancia(self):
        from fastapi.testclient import TestClient
        with mock.patch.dict(os.environ, {"AVEIA_TESTES": "1"}):
            r = TestClient(app.app, base_url="http://127.0.0.1:8765").get("/api/instancia")
        self.assertEqual(r.json(), {"app": "aveia", "pasta": app._id_da_pasta()})


if __name__ == "__main__":
    unittest.main()
