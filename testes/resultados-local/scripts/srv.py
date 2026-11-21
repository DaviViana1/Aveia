"""Servidor de teste do Aveia na porta 8799 (gravacoes em pasta temporaria; sem .env; sem IA externa)."""
import os, sys, tempfile, pathlib, time
RAIZ = str(pathlib.Path(__file__).resolve().parents[3])  # raiz do projeto
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["AI_GATEWAY_API_KEY"] = ""
os.environ["GROQ_API_KEY"] = ""
os.environ["AVEIA_SEM_NAVEGADOR"] = "1"
os.environ["AVEIA_SEM_AQUECER"] = "1"
os.chdir(RAIZ); sys.path.insert(0, RAIZ)
# o app lê o .env ao ser importado: impede (a leitura) escondendo o arquivo so durante o import
_ex = pathlib.Path.exists
pathlib.Path.exists = lambda self, *a, **k: False if self.name == ".env" else _ex(self, *a, **k)
import app
pathlib.Path.exists = _ex
tmp = pathlib.Path(tempfile.gettempdir()) / "aveia-gravacoes-teste"  # pasta temporaria, fora do projeto
tmp.mkdir(exist_ok=True)
app.GRAVACOES = tmp
from transcritor import pipeline
print("chaves:", repr(os.environ.get("GROQ_API_KEY")), repr(os.environ.get("AI_GATEWAY_API_KEY")), flush=True)
t0 = time.time(); pipeline.aquecer(em_segundo_plano=False); print(f"modelo carregado em {time.time()-t0:.1f}s", flush=True)
import uvicorn
uvicorn.run(app.app, host="127.0.0.1", port=8799, log_level="warning")
