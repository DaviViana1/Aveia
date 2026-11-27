"""Aveia: servidor local. Serve a interface e transcreve os áudios enviados."""
import hashlib
import hmac
import json
import os
import socket
import sys
import threading
import time
import urllib.request
import uuid
import webbrowser
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

PASTA = Path(__file__).parent


def _carregar_env(caminho: Path) -> None:
    """Lê KEY=valor de um .env (sem sobrescrever variáveis já definidas no sistema)."""
    if not caminho.exists():
        return
    for linha in caminho.read_text(encoding="utf-8").splitlines():
        linha = linha.strip()
        if not linha or linha.startswith("#") or "=" not in linha:
            continue
        chave, valor = linha.split("=", 1)
        os.environ.setdefault(chave.strip(), valor.strip().strip("\"'"))


_carregar_env(PASTA / ".env")

from transcritor import aovivo, arquivos, controle as controle_mod, ia, metricas, motores, perfil_pc, pipeline, resumo, titulo  # noqa: E402  (depois do .env, que define a chave)
from transcritor import audio as preproc  # noqa: E402  (pré-processamento: se ligado, o modo ao vivo é recusado)
from transcritor.pipeline import transcrever  # noqa: E402

ia.carregar_chaves_extras()   # opcional: AVEIA_ENV_EXTRA=<arquivo .env>; só as chaves de IA
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

PORTA = int(os.environ.get("AVEIA_PORTA") or 8765)   # fixa no uso normal; AVEIA_PORTA só para testes e servidores temporários
ESTATICOS = Path(__file__).parent / "static"
SAIDA = PASTA / "saida"   # reuniões gravadas por controle (fora do Git); não passa por transcritor/arquivos.py (que limita a 5 por pasta)
GRAVACOES = PASTA / "gravacoes"   # raiz: áudio pendente até a nota ser salva. Subpastas: ver transcritor/arquivos.py
IDIOMAS = {"pt": "pt", "en": "en", "es": "es", "auto": None}

class FalaIn(BaseModel):
    inicio: float
    fim: float | None = None
    locutor: str
    texto: str


class DestaqueIn(BaseModel):
    t: float        # segundos desde o início da nota
    texto: str


class ConfirmarIn(BaseModel):
    """Dados da nota salva: permitem guardar áudio e transcrição nas pastas, com o mesmo nome-base."""
    base: str | None = None
    parte: int = 1          # nº da gravação dentro da nota (1 = primeira)
    titulo: str = ""
    duracao: float = 0
    falas: list[FalaIn] = []   # a nota inteira, com os nomes atuais dos locutores
    destaques: list[DestaqueIn] = []


class ArquivarIn(BaseModel):
    base: str
    titulo: str = ""
    duracao: float = 0
    falas: list[FalaIn] = []
    destaques: list[DestaqueIn] = []
    resultado: dict = {}    # resumo gerado pela IA


app = FastAPI(title="Aveia", docs_url=None, redoc_url=None, openapi_url=None)   # sem /docs: não carrega script de CDN nem lista as rotas
arquivos.criar_pastas(GRAVACOES)

# O servidor é só desta máquina. Mesmo assim uma página qualquer aberta no navegador pode tentar falar com
# http://127.0.0.1:8765 (CSRF) ou apontar um nome de domínio para ele (DNS rebinding). Por isso: o cabeçalho Host precisa ser
# local; pedido que muda algo (não GET) só passa se o Origin, quando existe, for EXATAMENTE a mesma origem do Host (mesma porta:
# outra página em localhost:3000 não vale) e se o navegador não disser que o pedido veio de outro site (Sec-Fetch-Site); e o
# tamanho do corpo tem teto (upload, pedaço ao vivo e JSON).
_HOSTS_LOCAIS = {"127.0.0.1", "localhost", "[::1]"}
MAX_UPLOAD = int(float(os.environ.get("AVEIA_MAX_UPLOAD_MB") or 1024) * 1024 * 1024)   # /api/transcrever (padrão 1 GB)
MAX_JSON = 5 * 1024 * 1024
_CABECALHOS_SEGUROS = {"X-Frame-Options": "DENY", "Content-Security-Policy": "frame-ancestors 'none'",
                       "X-Content-Type-Options": "nosniff", "Referrer-Policy": "no-referrer"}


def _nome_do_host(valor: str) -> str:
    valor = (valor or "").strip().lower()
    if valor.startswith("["):   # IPv6: [::1]:8765
        return valor[:valor.find("]") + 1] if "]" in valor else valor
    return valor.split(":", 1)[0]


def _hosts_locais() -> set[str]:
    return _HOSTS_LOCAIS | ({"testserver"} if os.environ.get("AVEIA_TESTES") == "1" else set())   # `testserver` é do TestClient


def _recusa(msg: str, codigo: int = 403) -> JSONResponse:
    return JSONResponse({"erro": msg}, status_code=codigo, headers=_CABECALHOS_SEGUROS)


@app.middleware("http")
async def _so_local(request: Request, call_next):
    host = request.headers.get("host", "")
    if _nome_do_host(host) not in _hosts_locais():
        return _recusa("Acesso só pela própria máquina.")
    if request.method not in ("GET", "HEAD", "OPTIONS"):
        origem = request.headers.get("origin")
        if origem is not None and (origem == "null" or origem.split("://", 1)[-1].split("/", 1)[0].strip().lower() != host.strip().lower()):
            return _recusa("Origem não permitida.")
        if request.headers.get("sec-fetch-site") not in (None, "same-origin", "none"):
            return _recusa("Origem não permitida.")
        caminho = request.url.path
        pesado = caminho == "/api/transcrever" or caminho.endswith("/pedaco")
        limite = MAX_UPLOAD if caminho == "/api/transcrever" else aovivo.MAX_PEDACO if caminho.endswith("/pedaco") else MAX_JSON
        tamanho = request.headers.get("content-length")
        if tamanho is None and pesado:
            return _recusa("Envie o tamanho do corpo (Content-Length).", 411)
        try:
            if tamanho is not None and int(tamanho) > limite:
                return _recusa("Arquivo grande demais.", 413)
        except ValueError:
            return _recusa("Content-Length inválido.", 400)
    resposta = await call_next(request)
    for k, v in _CABECALHOS_SEGUROS.items():
        resposta.headers.setdefault(k, v)
    return resposta
# Whisper usa a CPU inteira; um trabalho por vez.
fila = ThreadPoolExecutor(max_workers=1)
class _Jobs(dict):
    """Trabalhos em memória. Passando de MAX, descarta os mais antigos que já terminaram (nunca o que está processando)."""
    MAX = 200

    def __setitem__(self, k, v):
        super().__setitem__(k, v)
        if len(self) > self.MAX:
            for velho in [c for c, j in list(self.items()) if j.get("status") != "processando"][: len(self) - self.MAX]:
                self.pop(velho, None)


jobs: dict[str, dict] = _Jobs()


def _executar(job_id: str, caminho: str, idioma: str | None, locutores: int | None, motor: str | None = None,
              canais: str | None = None, glossario: str | None = None):
    job = jobs[job_id]
    meta: dict = {}   # motor realmente usado e avisos (cota, fallback...), mostrados na interface
    inicio = time.monotonic()

    def progresso(p: float, msg: str):
        job.update(progresso=p, mensagem=msg, motor=meta.get("motor"), avisos=meta.get("avisos", []))

    def parcial(falas):   # texto provisório (sem locutor) para a interface mostrar enquanto o resto roda
        job["parciais"] = [asdict(f) for f in falas]

    try:
        falas = [asdict(f) for f in transcrever(
            caminho, idioma=idioma, diarizacao=True,
            num_locutores=locutores, progresso=progresso, motor=motor, meta=meta, parcial=parcial, canais=canais,
            glossario=glossario,
        )]
        job.update(motor=meta.get("motor"), avisos=meta.get("avisos", []), lacunas=meta.get("lacunas", []),
                   canais=meta.get("canais"), locutor_local=meta.get("locutor_local"))
        # O resultado fica em disco junto do áudio até a interface confirmar que salvou a nota.
        Path(caminho).with_suffix(".json").write_text(json.dumps(falas, ensure_ascii=False), encoding="utf-8")
        job.update(status="pronto", progresso=1.0, falas=falas, parciais=[])
        total, duracao = time.monotonic() - inicio, float(meta.get("duracao_s") or 0)
        metricas.registrar_gravacao(via="classico", motor=meta.get("motor"), total_s=total, duracao_s=duracao or None,
                                    fator_tempo=(total / duracao) if duracao else None, lacunas=len(meta.get("lacunas") or []),
                                    locutores=len({f["locutor"] for f in falas}), diarizacao=meta.get("diarizacao"),
                                    cpu_threads=os.cpu_count())
    except Exception as e:  # devolve o erro para a interface; o áudio é mantido
        job.update(status="erro", erro=str(e))
        metricas.registrar_gravacao(via="classico", erro=type(e).__name__)


# ---- transcrição durante a gravação (transcritor/aovivo.py) ----------------------------------------------------
# Os blocos ao vivo e os trabalhos clássicos dividem a mesma fila de um worker: um import longo atrasa o texto ao vivo.
aovivo_sessoes = aovivo.Gerenciador(lambda fn, *a: fila.submit(fn, *a))


def _executar_aovivo(job_id: str, sessao: "aovivo.Sessao"):
    """Fecha uma sessão ao vivo: cauda + diarização. Qualquer falha cai, aqui dentro, na transcrição clássica do arquivo."""
    job = jobs[job_id]
    caminho = str(sessao.arquivo)
    meta: dict = {}

    def progresso(p: float, msg: str):   # monotônico, mesmo se cair no clássico no meio do caminho
        job.update(progresso=max(p, job.get("progresso", 0.0)), mensagem=msg, motor=meta.get("motor"), avisos=meta.get("avisos", []))

    def parcial(falas):
        novas = [asdict(f) for f in falas]
        atual = job.get("parciais") or []
        if novas and (not atual or novas[-1]["fim"] >= atual[-1]["fim"]):   # o texto na tela nunca encolhe
            job["parciais"] = novas

    try:
        try:
            falas = [asdict(f) for f in sessao.concluir(progresso=progresso, parcial=parcial, meta=meta)]
        except Exception as e:  # noqa: BLE001  qualquer dúvida: transcrição clássica do mesmo arquivo
            print(f"Ao vivo: usando a transcrição clássica ({e})")
            meta = {}
            job.update(mensagem="Transcrevendo o áudio inteiro...")
            falas = [asdict(f) for f in transcrever(
                caminho, idioma=sessao.idioma, diarizacao=True, num_locutores=sessao.locutores,
                progresso=progresso, motor="local", meta=meta, parcial=parcial, canais=sessao.canais,
                glossario=sessao.glossario,
            )]
        job.update(motor=meta.get("motor"), avisos=meta.get("avisos", []), lacunas=meta.get("lacunas", []),
                   canais=meta.get("canais"), locutor_local=meta.get("locutor_local"))
        Path(caminho).with_suffix(".json").write_text(json.dumps(falas, ensure_ascii=False), encoding="utf-8")
        job.update(status="pronto", progresso=1.0, falas=falas, parciais=[])
        metricas.registrar_gravacao(via="aovivo", motor=meta.get("motor"), lacunas=len(meta.get("lacunas") or []),
                                    locutores=len({f["locutor"] for f in falas}), diarizacao=meta.get("diarizacao"),
                                    cpu_threads=os.cpu_count(), **sessao.metricas())
    except Exception as e:  # o áudio é mantido
        job.update(status="erro", erro=str(e))
        metricas.registrar_gravacao(via="aovivo", erro=type(e).__name__)
    finally:
        aovivo_sessoes.remover(sessao.id)
        sessao.liberar()


async def _campos(request: Request) -> dict:
    """Corpo do pedido como dict, seja JSON ou formulário."""
    tipo = request.headers.get("content-type", "")
    try:
        if "json" in tipo:
            dados = await request.json()
            return dados if isinstance(dados, dict) else {}
        return dict(await request.form())
    except Exception:  # noqa: BLE001
        return {}


# ---- gravação automática por controle (transcritor/controle.py): sem tela, sem clique ----------------------------------
TOKEN_CONTROLE = os.environ.get("AVEIA_CONTROLE_TOKEN") or controle_mod.novo_token()   # gravado em controle.json na subida
controle = controle_mod.Controle(aovivo_sessoes, _executar_aovivo, jobs, lambda fn, *a: fila.submit(fn, *a), GRAVACOES, SAIDA)


def _sem_token(request: Request) -> JSONResponse | None:
    dado = request.headers.get("x-aveia-token") or ""
    if not hmac.compare_digest(dado.encode("utf-8"), TOKEN_CONTROLE.encode("utf-8")):
        return JSONResponse({"ok": False, "motivo": "token"}, status_code=401)
    return None


@app.get("/api/controle/estado")
def api_controle_estado(request: Request):
    return _sem_token(request) or controle.estado()


@app.post("/api/controle/iniciar")
async def api_controle_iniciar(request: Request):
    """{evento_id, titulo, participantes:[{nome,email}], fim_previsto}: começa a gravar microfone e áudio do computador."""
    if (negado := _sem_token(request)) is not None:
        return negado
    codigo, corpo = await run_in_threadpool(controle.iniciar, await _campos(request))
    return JSONResponse(corpo, status_code=codigo)


@app.post("/api/controle/parar")
async def api_controle_parar(request: Request):
    if (negado := _sem_token(request)) is not None:
        return negado
    return await run_in_threadpool(controle.parar, "manual")


@app.post("/api/aovivo")
async def api_aovivo_criar(request: Request):
    """Abre uma sessão de transcrição durante a gravação. Recusa (`ao_vivo: false`) fora do motor local ou com pré-processamento."""
    c = await _campos(request)
    motor = motores.normalizar_motor(str(c.get("motor") or ""))
    motivo = aovivo.motivo_recusa(motor, preproc.configuracao())
    if motivo:
        return {"ao_vivo": False, "motivo": motivo}
    try:
        locutores = int(c.get("locutores") or 0) or None
    except (TypeError, ValueError):
        locutores = None
    GRAVACOES.mkdir(parents=True, exist_ok=True)
    nome = f"{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:8]}.{aovivo.extensao_valida(str(c.get('ext') or ''))}"
    try:
        s = aovivo_sessoes.criar(GRAVACOES / nome, IDIOMAS.get(str(c.get("idioma") or "pt")), locutores,
                                 pipeline.normalizar_canais(str(c.get("canais") or "")),
                                 pipeline.normalizar_glossario(c.get("glossario")))
    except OSError as e:
        return {"ao_vivo": False, "motivo": f"Não foi possível criar o arquivo: {e}"}
    if s is None:
        return {"ao_vivo": False, "motivo": "Já há gravações ao vivo demais em andamento."}
    return {"sessao": s.id, "ao_vivo": True}


@app.post("/api/aovivo/{sessao}/pedaco")
async def api_aovivo_pedaco(sessao: str, seq: int, request: Request):
    """Um pedaço do MediaRecorder (corpo cru). Entra no arquivo estritamente em ordem de `seq`; repetido é ignorado."""
    s = aovivo_sessoes.obter(sessao)
    if s is None:
        return JSONResponse({"erro": "Sessão não encontrada."}, status_code=404)
    dados = await request.body()
    try:
        return {"ok": True, **s.receber(seq, dados)}
    except aovivo.AoVivoIndisponivel as e:
        return JSONResponse({"erro": str(e)}, status_code=410)
    except (ValueError, OSError) as e:
        return JSONResponse({"erro": str(e)}, status_code=400)


@app.get("/api/aovivo/{sessao}")
def api_aovivo_consultar(sessao: str, desde: int = 0):
    """Falas provisórias (sem locutor) a partir do índice `desde`. A última fala ainda pode crescer."""
    s = aovivo_sessoes.obter(sessao)
    if s is None:
        return JSONResponse({"erro": "Sessão não encontrada."}, status_code=404)
    return s.consultar(desde)


class FinalizarAoVivoIn(BaseModel):
    bytes: int = -1       # total enviado pelo navegador
    pedacos: int = -1     # quantidade de pedaços enviados
    duracao: float = 0    # s gravados (sem pausas)


_trava_finalizar = threading.Lock()
_finalizadas: dict[str, tuple[str, int, int]] = {}   # sessão -> (job, bytes, pedaços) das sessões já finalizadas (as mais recentes)
_MAX_FINALIZADAS = 64


@app.post("/api/aovivo/{sessao}/finalizar")
def api_aovivo_finalizar(sessao: str, corpo: FinalizarAoVivoIn):
    """O navegador parou de gravar e enviou tudo. Se a contagem não bate com o que está em disco, 409 (o navegador
    envia o arquivo inteiro pelo caminho clássico). Senão devolve um job comum: cauda + diarização."""
    with _trava_finalizar:
        # Idempotente: se a resposta anterior se perdeu, o navegador repete o pedido e recebe o MESMO job (nunca dois jobs
        # por sessão). O vínculo sessão -> job sobrevive à remoção da sessão (que acontece ao fim do job).
        anterior = _finalizadas.get(sessao)
        if anterior is not None:
            job_ant, bytes_ant, pedacos_ant = anterior
            if (corpo.bytes, corpo.pedacos) == (bytes_ant, pedacos_ant):
                return {"job": job_ant}
            return JSONResponse({"erro": "O que chegou ao servidor não confere com o gravado."}, status_code=409)
        s = aovivo_sessoes.obter(sessao)
        if s is None:
            return JSONResponse({"erro": "Sessão não encontrada."}, status_code=404)
        if s.finalizando or not s.confere(corpo.bytes, corpo.pedacos):
            return JSONResponse({"erro": "O que chegou ao servidor não confere com o gravado."}, status_code=409)
        s.marcar_fim(corpo.duracao)
        job_id = uuid.uuid4().hex
        jobs[job_id] = {"status": "processando", "progresso": 0.0, "mensagem": "Na fila...", "arquivo": str(s.arquivo),
                        "parciais": [asdict(f) for f in s.falas()]}
        _finalizadas[sessao] = (job_id, corpo.bytes, corpo.pedacos)
        while len(_finalizadas) > _MAX_FINALIZADAS:   # só as mais recentes: o vínculo serve a repetições logo após a perda
            _finalizadas.pop(next(iter(_finalizadas)))
        fila.submit(_executar_aovivo, job_id, s)
        return {"job": job_id}


@app.delete("/api/aovivo/{sessao}")
def api_aovivo_apagar(sessao: str):
    """O navegador abandonou o ao vivo (vai enviar o áudio inteiro pelo caminho clássico): libera a memória e apaga o
    arquivo parcial. Sessão já em finalização não é mexida (o job é dono do arquivo)."""
    s = aovivo_sessoes.obter(sessao)
    if s is None:
        return {"ok": True}
    if s.finalizando:
        return {"ok": True, "finalizando": True}
    aovivo_sessoes.remover(sessao)
    s.apagar()
    return {"ok": True}


_EXT_UPLOAD = set(aovivo.EXTENSOES) | {"opus", "wma", "mkv", "mov", "3gp", "aif", "aiff", "amr", "mpga"}


def _ext_upload(sufixo: str) -> str:
    """Extensão do arquivo enviado: só de uma lista de formatos de áudio e vídeo (o nome vem do cliente); o resto vira webm."""
    ext = (sufixo or "").lstrip(".").lower()
    return ext if ext in _EXT_UPLOAD else "webm"


@app.post("/api/transcrever")
async def api_transcrever(
    audio: UploadFile = File(...),
    idioma: str = Form("pt"),
    locutores: int = Form(0),
    motor: str = Form(""),      # ignorado: a transcrição é sempre local (o campo fica por compatibilidade com versões antigas)
    canais: str = Form(""),     # "mic+sistema": arquivo estéreo (microfone à esquerda, áudio do computador à direita); outro valor = mono
    glossario: str = Form(""),  # nomes e termos curtos (vírgula ou linha) para o Whisper local; vazio = como antes
):
    job_id = uuid.uuid4().hex
    sufixo = "." + _ext_upload(Path(audio.filename or "").suffix)
    GRAVACOES.mkdir(exist_ok=True)
    arquivo = GRAVACOES / f"{datetime.now():%Y%m%d-%H%M%S}-{job_id[:8]}{sufixo}"
    arquivo.write_bytes(await audio.read())
    jobs[job_id] = {"status": "processando", "progresso": 0.0, "mensagem": "Na fila...", "arquivo": str(arquivo)}
    fila.submit(_executar, job_id, str(arquivo), IDIOMAS.get(idioma), locutores or None, motores.normalizar_motor(motor),
                pipeline.normalizar_canais(canais), pipeline.normalizar_glossario(glossario))
    return {"job": job_id}


@app.get("/api/computador")
def api_computador():
    """O que o computador tem, se aguenta o app e o último teste real de velocidade (`dados/computador.json`)."""
    return perfil_pc.estado()


def _executar_teste_pc(job_id: str):
    job = jobs[job_id]
    try:
        job.update(mensagem="Carregando o modelo e medindo uma janela de 30 s...", progresso=0.2)
        teste = perfil_pc.medir_janela()
        perfil_pc.salvar_teste(teste)
        job.update(status="pronto", progresso=1.0, teste=teste, classe_nome=perfil_pc.nome_da_classe(teste["classe"]))
    except Exception as e:  # noqa: BLE001
        job.update(status="erro", erro=str(e))


@app.post("/api/computador/testar")
def api_computador_testar():
    """Teste real: uma janela de 30 s no Whisper. Entra na fila do Whisper (um trabalho por vez), então a medida não disputa a CPU."""
    job_id = uuid.uuid4().hex
    jobs[job_id] = {"status": "processando", "progresso": 0.0, "mensagem": "Na fila..."}
    fila.submit(_executar_teste_pc, job_id)
    return {"job": job_id}


@app.get("/api/jobs/{job_id}")
def api_job(job_id: str):
    job = jobs.get(job_id)
    if job is None:
        return JSONResponse({"status": "erro", "erro": "Trabalho não encontrado."}, status_code=404)
    return {k: v for k, v in job.items() if k != "arquivo"}   # o caminho do arquivo no disco não sai do servidor


@app.post("/api/jobs/{job_id}/confirmar")
def api_confirmar(job_id: str, corpo: ConfirmarIn | None = None):
    """A interface salvou a nota. O áudio pendente vai para `Áudio/` e a transcrição para `Transcrições/`
    enquanto houver vaga (até 5 arquivos por pasta); o que não couber é apagado, como sempre foi."""
    job = jobs.get(job_id)
    salvos: dict[str, bool] = {}
    if job and job.get("arquivo"):
        arquivo = Path(job["arquivo"])
        arquivo.with_suffix(".json").unlink(missing_ok=True)
        if corpo and arquivos.base_valida(corpo.base):
            falas = [f.model_dump() for f in corpo.falas]
            try:
                salvos["audio"] = arquivos.guardar_audio(GRAVACOES, arquivo, corpo.base, corpo.parte)
                if falas:
                    salvos["transcricao"] = arquivos.guardar_transcricao(GRAVACOES, corpo.base, corpo.titulo, corpo.duracao, falas,
                                                                  [d.model_dump() for d in corpo.destaques])
            except OSError as e:
                print(f"Não foi possível gravar em {GRAVACOES}: {e}")
        arquivo.unlink(missing_ok=True)   # não coube (ou não foi pedido guardar): apaga
        job.pop("falas", None)
    return {"ok": True, "salvos": salvos, "limite": arquivos.LIMITE}


@app.post("/api/arquivar")
def api_arquivar(corpo: ArquivarIn):
    """Guarda a análise (resumo da IA) em `Análises/` e atualiza a transcrição com os nomes atuais dos locutores."""
    if not arquivos.base_valida(corpo.base):
        return JSONResponse({"erro": "Nome de arquivo inválido."}, status_code=400)
    salvos: dict[str, bool] = {}
    try:
        if corpo.falas:
            salvos["transcricao"] = arquivos.guardar_transcricao(
                GRAVACOES, corpo.base, corpo.titulo, corpo.duracao, [f.model_dump() for f in corpo.falas],
                [d.model_dump() for d in corpo.destaques])
        if corpo.resultado:
            salvos["analise"] = arquivos.guardar_analise(GRAVACOES, corpo.base, corpo.titulo, corpo.resultado)
    except OSError as e:
        print(f"Não foi possível gravar em {GRAVACOES}: {e}")
    return {"ok": True, "salvos": salvos, "limite": arquivos.LIMITE}


@app.delete("/api/arquivos/{base}")
def api_apagar_arquivos(base: str):
    """A nota foi excluída na interface: apaga o áudio, a transcrição e a análise dela (liberando as vagas)."""
    if not arquivos.base_valida(base):
        return JSONResponse({"erro": "Nome de arquivo inválido."}, status_code=400)
    try:
        return {"ok": True, "removidos": arquivos.apagar(GRAVACOES, base)}
    except OSError as e:
        print(f"Não foi possível apagar os arquivos de {base}: {e}")
        return JSONResponse({"erro": str(e)}, status_code=500)


class ResumoIn(BaseModel):
    titulo: str = ""
    notas: list[str] = []
    duracao: float = 0
    falas: list[FalaIn]
    destaques: list[DestaqueIn] = []


# Resumos de gravações longas levam minutos (vários blocos): rodam em segundo plano, com progresso.
fila_ia = ThreadPoolExecutor(max_workers=2)


def _executar_resumo(job_id: str, corpo: ResumoIn):
    job = jobs[job_id]
    try:
        r = resumo.gerar_resumo(
            corpo.titulo, corpo.notas, [f.model_dump() for f in corpo.falas], corpo.duracao,
            progresso=lambda p, m: job.update(progresso=p, mensagem=m),
            destaques=[d.model_dump() for d in corpo.destaques],
        )
        job.update(status="pronto", progresso=1.0, resultado=r)
    except resumo.IANaoConfigurada as e:
        job.update(status="erro", erro=str(e), codigo="sem_chave")
    except resumo.FalhaIA as e:
        job.update(status="erro", erro=str(e), codigo="falha")
    except Exception as e:
        job.update(status="erro", erro=str(e), codigo="falha")


@app.get("/api/ia")
def api_ia():
    return {"configurada": resumo.configurada(), "modelo": resumo.modelo(), "uso": metricas.uso_ia(),
            "provedor": ia.provedor(), "destino": ia.destino(), "aviso": ia.aviso_privacidade(), "sem_chave": ia.sem_ia(), "ligada": ia.ligada()}


RETENCAO_DIAS = int(os.environ.get("AVEIA_RETENCAO_DIAS", "30") or 30)


@app.get("/api/privacidade")
def api_privacidade():
    """O que fica nesta máquina e o que pode sair dela, por recurso (a interface mostra esta tabela), mais o uso da IA do mês
    e a situação dos arquivos guardados. Só números e nomes; nenhum valor de chave."""
    recursos = [
        {"id": "transcricao", "nome": "Transcrição (Whisper)", "onde": "local", "ativo": True},
        {"id": "locutores", "nome": "Separação de locutores", "onde": "local", "ativo": True},
        {"id": "ia", "nome": "IA (título, resumo e nomes)", "onde": "nuvem", "envia": "texto", "para": ia.destino(),
         "interruptor": True, "disponivel": resumo.configurada()},
    ]
    rede = [
        {"nome": "Modelos de transcrição e de locutores (primeiro uso)", "envia": "nada"},
        {"nome": "Estilos e fontes da interface (a cada abertura)", "envia": "só o endereço de rede"},
    ]
    return {"recursos": recursos, "rede": rede, "ia": metricas.uso_ia(), "ia_aviso": ia.aviso_privacidade(),
            "gravacoes": arquivos.situacao(GRAVACOES, RETENCAO_DIAS)}


class ApagarTudoIn(BaseModel):
    confirmar: bool = False


@app.post("/api/apagar-tudo")
def api_apagar_tudo(corpo: ApagarTudoIn):
    """Apaga todo o áudio e as transcrições guardados em `gravacoes/` (a interface limpa as notas do navegador). Só com
    `confirmar: true`."""
    if not corpo.confirmar:
        return JSONResponse({"erro": "Confirmação ausente."}, status_code=400)
    return {"ok": True, "removidos": arquivos.apagar_tudo(GRAVACOES)}


@app.post("/api/resumo")
def api_resumo(corpo: ResumoIn):
    if not resumo.configurada():
        return JSONResponse({"erro": ia.sem_ia(), "codigo": "sem_chave" if ia.ligada() else "desligada"}, status_code=503)
    job_id = uuid.uuid4().hex
    jobs[job_id] = {"status": "processando", "progresso": 0.0, "mensagem": "Iniciando..."}
    fila_ia.submit(_executar_resumo, job_id, corpo)
    return {"job": job_id}


class TituloIn(BaseModel):
    notas: list[str] = []
    falas: list[FalaIn] = []


@app.post("/api/titulo")
def api_titulo(corpo: TituloIn):
    """Título automático da nota (chamada curta, separada do resumo). Sempre responde 200: sem chave, falha ou
    título ruim viram `titulo: null` + `motivo`, e a interface só mantém "Nova nota"."""
    r = titulo.gerar_titulo(corpo.notas, [f.model_dump() for f in corpo.falas])
    return {"titulo": r.titulo, "motivo": r.motivo, "modelo": r.modelo}


from transcritor import locutores as ident_locutores  # noqa: E402  (identificação de nomes dos locutores)
from transcritor import destaques as escopo_destaque  # noqa: E402  (quais frases pertencem a um destaque)
from transcritor import identificacao as ident_comando  # noqa: E402  (frase do chat que dá nome a quem fala)


class CandidataIn(BaseModel):
    i: int
    inicio: float = 0
    fim: float = 0
    locutor: str = ""
    texto: str = ""


class DestaqueEscopoIn(BaseModel):
    destaque: str
    candidatas: list[CandidataIn] = []


@app.post("/api/destaque-escopo")
def api_destaque_escopo(corpo: DestaqueEscopoIn):
    """Quais das frases candidatas (ditas até 10 s antes de a pessoa começar a digitar o destaque) pertencem ao tópico dele.
    Sempre responde 200: sem chave, teto atingido ou erro vira `modo: "regra"` + `motivo` e o navegador usa a janela inteira."""
    if not ia.configurada():
        return {"modo": "regra", "motivo": "sem_chave" if ia.ligada() else "desligada"}
    try:
        r = escopo_destaque.escopo(corpo.destaque, [c.model_dump() for c in corpo.candidatas])
        return {"modo": "ia", "indices": r["indices"], "modelo": r["modelo"]}
    except ia.Falha as e:
        return {"modo": "regra", "motivo": e.motivo}
    except Exception:
        return {"modo": "regra", "motivo": "falha"}


class DestaqueComandoIn(BaseModel):
    texto: str
    ia: bool = True   # interruptor do navegador (Ajustes): desligado = só a regra de texto, nenhuma chamada


@app.post("/api/destaque-comando")
def api_destaque_comando(corpo: DestaqueComandoIn):
    """A frase do chat é um comando de identificação ("Marina que está falando") ou um destaque comum? Sempre responde 200.
    Regra de texto primeiro (sem rede); a IA só entra com o interruptor ligado e a frase parecendo identificação."""
    try:
        return ident_comando.interpretar(corpo.texto, usar_ia=corpo.ia)
    except Exception:   # noqa: BLE001  qualquer falha: a frase segue como destaque comum
        return {"comando": False, "motivo": "falha"}


class LocutoresIn(BaseModel):
    """`falas` com o rótulo original ("Locutor N"). `nomes_fixos`: rótulos que já têm nome (manual ou já identificado),
    que não são reavaliados. `recusados`: nomes que o usuário desfez, por rótulo (não voltam a ser sugeridos)."""
    titulo: str = ""
    notas: list[str] = []
    falas: list[FalaIn] = []
    nomes_fixos: dict[str, str] = {}
    recusados: dict[str, list[str]] = {}


@app.post("/api/locutores")
def api_locutores(corpo: LocutoresIn):
    """Descobre o nome real dos locutores (uma chamada barata à IA, independente do resumo). Roda em thread própria do
    FastAPI e a interface não espera por ela. Sempre responde 200: sem chave ou com falha vira `sugestoes: {}` + `motivo`,
    e os locutores seguem como "Locutor N"."""
    if not resumo.configurada():
        return {"sugestoes": {}, "motivo": "sem_chave"}
    try:
        r = ident_locutores.identificar(corpo.titulo, corpo.notas, [f.model_dump() for f in corpo.falas],
                                        corpo.nomes_fixos, corpo.recusados)
        return {"sugestoes": r["sugestoes"], "descartados": r["descartados"], "janelas": r["janelas"], "modelo": r["modelo"], "motivo": None}
    except resumo.IANaoConfigurada:
        return {"sugestoes": {}, "motivo": "sem_chave"}
    except Exception as e:   # noqa: BLE001  falha de rede/IA: degrada em silêncio
        print(f"Identificação de locutores falhou: {e}")
        return {"sugestoes": {}, "motivo": "falha"}


@app.get("/")
def raiz():
    return FileResponse(ESTATICOS / "index.html")


app.mount("/static", StaticFiles(directory=ESTATICOS), name="static")


def _id_da_pasta() -> str:
    """Identifica esta cópia do Aveia (pela pasta), para saber se a porta ocupada é de outra instância ou da mesma."""
    return hashlib.sha1(str(PASTA.resolve()).lower().encode("utf-8")).hexdigest()[:12]


@app.get("/api/instancia")
def api_instancia():
    return {"app": "aveia", "pasta": _id_da_pasta()}


def _porta_ocupada(porta: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.4)
        return s.connect_ex(("127.0.0.1", porta)) == 0


def _escolher_porta() -> tuple[int | None, bool]:
    """(porta, já_está_rodando). A 8765 é a padrão. Se estiver ocupada: se é esta mesma pasta, o app já está no ar (só abre o
    navegador); se é outro programa ou outra cópia do Aveia, usa a próxima porta livre (até 8799), em vez de falhar ou brigar."""
    if os.environ.get("AVEIA_PORTA"):
        return PORTA, False
    for porta in range(PORTA, PORTA + 35):
        if not _porta_ocupada(porta):
            return porta, False
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{porta}/api/instancia", timeout=1.5) as r:
                dados = json.loads(r.read().decode("utf-8"))
            if dados.get("app") == "aveia" and dados.get("pasta") == _id_da_pasta():
                return porta, True
        except Exception:  # noqa: BLE001  (outro programa, ou Aveia de versão sem esta rota)
            pass
    return None, False


def _sair_ja(codigo: int = 0) -> None:
    """Ctrl+C: o uvicorn para o servidor, mas as threads de trabalho (transcrição, IA) não são daemon e seguram o processo até o
    trabalho em curso acabar (minutos numa gravação longa), e o console parece travado. Cancela o que ainda está na fila e sai
    de uma vez. O que estava sendo transcrito se perde, mas o áudio fica guardado no navegador ("Tentar de novo")."""
    controle.encerrar()
    controle_mod.apagar_arquivo_controle()
    fila.shutdown(wait=False, cancel_futures=True)
    fila_ia.shutdown(wait=False, cancel_futures=True)
    print("Aveia encerrado.", flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(codigo)


def principal() -> None:
    global PORTA
    porta, ja_rodando = _escolher_porta()
    if porta is None:
        print(f"Aveia não iniciou: as portas {PORTA} a {PORTA + 34} estão ocupadas.", flush=True)
        _sair_ja(1)
    if ja_rodando:
        print(f"O Aveia desta pasta já está aberto em http://127.0.0.1:{porta}", flush=True)
        if not os.environ.get("AVEIA_SEM_NAVEGADOR"):
            webbrowser.open(f"http://127.0.0.1:{porta}")
        _sair_ja(0)
    if porta != PORTA:
        print(f"A porta {PORTA} está em uso por outro programa: o Aveia abriu em http://127.0.0.1:{porta}", flush=True)
    PORTA = porta
    try:
        controle_mod.gravar_arquivo_controle(porta, TOKEN_CONTROLE)
    except OSError as e:
        print(f"Não foi possível gravar o controle.json (a gravação por controle fica sem endereço): {e}", flush=True)
    # Aquecimento do modelo Whisper só ao iniciar o servidor de verdade (importar `app` em testes não carrega o modelo).
    # AVEIA_SEM_AQUECER=1 desliga. Assim o primeiro bloco ao vivo não espera o modelo carregar.
    if not os.environ.get("AVEIA_SEM_AQUECER"):
        pipeline.aquecer()
    if not os.environ.get("AVEIA_SEM_NAVEGADOR"):
        threading.Timer(1.2, lambda: webbrowser.open(f"http://127.0.0.1:{PORTA}")).start()
    # timeout_graceful_shutdown: o navegador pode manter conexões abertas; não espera por elas mais que alguns segundos.
    servidor = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=PORTA, log_level="warning", timeout_graceful_shutdown=3))
    codigo = 0
    try:
        servidor.run()
    except KeyboardInterrupt:
        pass
    except BaseException as e:  # noqa: BLE001  (porta ocupada etc.: mostra e sai com erro, sem prender o console)
        print(f"Aveia não iniciou: {e}", flush=True)
        codigo = 1
    _sair_ja(codigo)


if __name__ == "__main__":
    principal()
