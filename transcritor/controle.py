"""Gravação automática por controle: outro programa local manda iniciar e parar, sem nenhum clique na tela.

Fluxo: `iniciar` abre uma sessão ao vivo de PCM direto (`aovivo.Sessao`) e a captura do servidor (`captura.Captura`); cada bloco
de ~1 s vai para `Sessao.anexar_pcm`. A gravação para por `parar` ou sozinha: depois de `fim_previsto`, com 10 min seguidos de
silêncio no canal do computador, ou no máximo em `fim_previsto + 30 min`. Ao parar, a cauda e os locutores seguem em
segundo plano (job). Se uma página da tela "assumiu" a gravação (`assumir`), é ela quem faz nomes, resumo e título e manda o
resultado final por `saida`; sem página (ou se ela some por 4 min depois da parada) o servidor faz tudo sozinho (reserva).
Em qualquer caso a pasta `saida/...` sai uma vez só, com o `concluido.flag` por último; depois dela roda `AVEIA_APOS_SAIDA`.

IA: tudo respeita o interruptor existente (`ia.configurada()`): desligada = nenhuma chamada e `meta.ia = false`.
Nada aqui envia e-mail ou mensagem; o texto da reunião só é gravado em `saida/` (fora do Git).
"""
from __future__ import annotations

import json
import os
import secrets
import subprocess
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable

import numpy as np

from . import aovivo, captura, ia, saida

SILENCIO_PARADA_S = 600.0            # 10 min seguidos sem som no computador, depois do fim previsto
LIMITE_APOS_FIM = timedelta(minutes=30)
VIGIA_S = 5.0
PAGINA_ABERTA_S = 90.0              # alguma página consultou o painel há menos que isto (aba em segundo plano consulta ~1 vez por minuto)
DONO_SUMIDO_S = 90.0                # a página dona não consulta há isto: outra página pode assumir
PAGINA_SUMIU_S = 240.0              # depois da parada, sem consulta há isto: o servidor processa sozinho


def pasta_controle() -> Path:
    base = os.environ.get("AVEIA_CONTROLE_DIR") or os.path.join(os.environ.get("LOCALAPPDATA") or str(Path.home()), "Aveia")
    return Path(base)


def gravar_arquivo_controle(porta: int, token: str) -> Path:
    """`controle.json` = porta, token e pid desta instância (para os scripts locais acharem o servidor)."""
    pasta = pasta_controle()
    pasta.mkdir(parents=True, exist_ok=True)
    destino = pasta / "controle.json"
    tmp = destino.with_name("controle.json.tmp")
    tmp.write_text(json.dumps({"porta": porta, "token": token, "pid": os.getpid()}), encoding="utf-8")
    os.replace(tmp, destino)
    return destino


def apagar_arquivo_controle() -> None:
    """Remove o `controle.json` só se for desta instância."""
    destino = pasta_controle() / "controle.json"
    try:
        if json.loads(destino.read_text(encoding="utf-8")).get("pid") == os.getpid():
            destino.unlink()
    except Exception:  # noqa: BLE001
        pass


def novo_token() -> str:
    return secrets.token_hex(24)


def _participantes(v) -> list[dict]:
    saida_ = []
    for p in v if isinstance(v, list) else []:
        if isinstance(p, dict) and str(p.get("nome") or "").strip():
            saida_.append({"nome": str(p["nome"]).strip(), "email": str(p.get("email") or "").strip()})
    return saida_


class Controle:
    def __init__(self, gerenciador: aovivo.Gerenciador, executar: Callable[[str, aovivo.Sessao], None],
                 jobs: dict, submeter: Callable[..., object], gravacoes: Path, raiz_saida: Path,
                 fabrica_captura: Callable[[Callable], captura.Captura] | None = None,
                 agora: Callable[[], datetime] = datetime.now, relogio: Callable[[], float] = time.monotonic) -> None:
        self._ger = gerenciador
        self._executar = executar        # app._executar_aovivo(job_id, sessao): termina o ao vivo e põe as falas em jobs[job_id]
        self._jobs = jobs
        self._submeter = submeter        # fila do Whisper
        self._gravacoes = Path(gravacoes)
        self._raiz_saida = Path(raiz_saida)
        self._injetada = fabrica_captura is not None   # teste: dispensa a checagem da máquina
        self._fabrica = fabrica_captura or (lambda cb: captura.Captura(cb))
        self._agora = agora
        self._relogio = relogio
        self._trava = threading.RLock()
        self._atual: dict | None = None      # gravação em curso
        self._pos: dict | None = None        # última gravação em processamento (ou já processada)
        self._pos_ativos = 0
        self.ultimo_erro = ""
        self._vigia: threading.Thread | None = None
        self._encerrar_vigia = threading.Event()
        self._painel_visto: float | None = None   # última consulta ao painel (qualquer página)
        self._eventos_gravados: set[str] = set()  # evento_id com saída já gravada (nunca duas)

    # ---- estado ---------------------------------------------------------------------------------------
    def estado(self) -> dict:
        with self._trava:
            if self._atual is not None:
                g, est = self._atual, "gravando"
            elif self._pos_ativos and self._pos is not None:
                g, est = self._pos, "processando"
            else:
                g, est = (self._pos if self._pos else None), "ocioso"
            r = {"estado": est, "evento_id": None, "titulo": None, "inicio": None, "fim_previsto": None, "pasta_saida": None,
                 "sessao": None, "job": None, "assumida": False, "pagina_aberta": self.pagina_aberta()}
            if g is not None and est != "ocioso":
                r.update(evento_id=g["evento_id"], titulo=g["titulo"], inicio=g["inicio"].isoformat(timespec="seconds"),
                         fim_previsto=g["fim_previsto"].isoformat(timespec="seconds"), pasta_saida=str(g["pasta"]),
                         sessao=g["sessao"].id if est == "gravando" else None, job=g.get("job"), assumida=bool(g.get("assumida_por")))
            if self.ultimo_erro:
                r["ultimo_erro"] = self.ultimo_erro
            return r

    def pagina_aberta(self) -> bool:
        return self._painel_visto is not None and self._relogio() - self._painel_visto < PAGINA_ABERTA_S

    def painel(self, pagina: str = "") -> dict:
        """Só o que a tela mostra (sem participantes, e-mails, pasta nem token). Cada consulta vale como sinal de que a página
        está aberta; se `pagina` é a dona da gravação, renova o "visto" dela."""
        minha = False
        with self._trava:
            agora = self._relogio()
            self._painel_visto = agora
            if pagina:
                for g in (self._atual, self._pos):
                    if g is not None and g.get("assumida_por") == pagina:
                        g["pagina_vista"] = agora
                        minha = True
        e = self.estado()
        seg = 0
        if e["estado"] != "ocioso" and e["inicio"]:
            seg = max(0, int((self._agora() - datetime.fromisoformat(e["inicio"])).total_seconds()))
        return {"estado": e["estado"], "titulo": e["titulo"], "inicio": e["inicio"], "fim_previsto": e["fim_previsto"], "segundos": seg,
                "evento_id": e["evento_id"], "sessao": e["sessao"], "job": e["job"], "assumida": e["assumida"],
                "minha": minha and e["estado"] != "ocioso"}

    def _tela_gravando(self) -> bool:
        """Há gravação manual do navegador (sessão ao vivo que não é a captura do servidor)?"""
        return any(not s.pcm_direto and not (s.finalizando or s.degradada or s.liberada) for s in list(self._ger.sessoes.values()))

    # ---- iniciar --------------------------------------------------------------------------------------
    def iniciar(self, dados: dict) -> tuple[int, dict]:
        with self._trava:
            if self._atual is not None:
                return 409, {"ok": False, "motivo": "ja-gravando", "evento_id": self._atual["evento_id"]}
            if self._tela_gravando():
                return 409, {"ok": False, "motivo": "tela-gravando"}
            ok, motivo = (True, "") if self._injetada else captura.disponivel()
            if not ok:
                return 503, {"ok": False, "motivo": "captura-indisponivel", "detalhe": motivo}
            agora = self._agora()
            try:
                fim = datetime.fromisoformat(str(dados["fim_previsto"])) if dados.get("fim_previsto") else agora + timedelta(hours=1)
            except ValueError:
                return 400, {"ok": False, "motivo": "fim-previsto-invalido"}
            if fim.tzinfo is not None:
                fim = fim.astimezone().replace(tzinfo=None)
            titulo = " ".join(str(dados.get("titulo") or "").split()) or "Reunião"
            pasta = saida.pasta_para(self._raiz_saida, agora, titulo)
            self._gravacoes.mkdir(parents=True, exist_ok=True)
            arquivo = self._gravacoes / f"{agora:%Y%m%d-%H%M%S}-controle.wav"
            sessao = self._ger.criar_direta(arquivo, "pt")
            if sessao is None:
                return 409, {"ok": False, "motivo": "sessoes-demais"}
            g = {"evento_id": str(dados.get("evento_id") or "") or f"sessao-{sessao.id[:12]}",   # sem id de evento: a própria sessão identifica
                 "titulo": titulo, "participantes": _participantes(dados.get("participantes")),
                 "usuario": str(dados.get("usuario") or os.environ.get("AVEIA_NOME_USUARIO") or "").strip(),
                 "inicio": agora, "fim_previsto": fim, "pasta": pasta, "sessao": sessao, "captura": None,
                 "silencio_s": 0.0, "parando": False, "assumida_por": None, "pagina_vista": 0.0, "t_parada": 0.0,
                 "saida_feita": False, "trava_saida": threading.Lock(), "falas": None}
            try:
                g["captura"] = self._fabrica(lambda pcm, rms_d, rms_e, _g=g: self._ao_bloco(_g, pcm, rms_d))
                g["captura"].iniciar()
            except Exception as e:  # noqa: BLE001
                self._ger.remover(sessao.id)
                sessao.apagar()
                return 500, {"ok": False, "motivo": "captura-falhou", "detalhe": f"{type(e).__name__}: {e}"}
            self._atual = g
            self._iniciar_vigia()
            return 200, {"ok": True, "sessao": sessao.id, "pasta_saida": str(pasta)}

    def _ao_bloco(self, g: dict, pcm: np.ndarray, rms_d: float) -> None:
        if g["parando"]:
            return
        g["sessao"].anexar_pcm(pcm)
        dur = len(pcm) / 2 / aovivo.TAXA
        g["silencio_s"] = g["silencio_s"] + dur if rms_d < captura.LIMIAR_RMS_SILENCIO else 0.0

    # ---- página assume a gravação -----------------------------------------------------------------------
    def assumir(self, sessao: str, pagina: str = "") -> tuple[int, dict]:
        """Uma página da tela passa a conduzir a gravação (nota, ao vivo, parar, pós-processamento). Só uma por vez: a mesma
        página (mesmo id) pode assumir de novo ao recarregar; outra recebe 409, salvo se a dona sumiu (sem consultar o painel)."""
        pagina = str(pagina or "").strip()[:64]
        with self._trava:
            g = self._atual
            if g is None or g["sessao"].id != str(sessao or ""):
                return 404, {"ok": False, "motivo": "sem-gravacao"}
            dono = g.get("assumida_por")
            if dono and not (pagina and pagina == dono) and self._relogio() - g["pagina_vista"] < DONO_SUMIDO_S:
                return 409, {"ok": False, "motivo": "ja-assumida"}
            g["assumida_por"] = pagina or "anonima"
            g["pagina_vista"] = self._relogio()
            seg = max(0, int((self._agora() - g["inicio"]).total_seconds()))
            return 200, {"ok": True, "evento_id": g["evento_id"], "titulo": g["titulo"], "segundos": seg,
                         "participantes": [p["nome"] for p in g["participantes"]], "usuario": g["usuario"]}

    # ---- parada ---------------------------------------------------------------------------------------
    def tick(self) -> str | None:
        """Regra de parada automática. Devolve o motivo se parou."""
        with self._trava:
            g = self._atual
            if g is None:
                return None
            agora = self._agora()
            if agora >= g["fim_previsto"]:
                if agora >= g["fim_previsto"] + LIMITE_APOS_FIM:
                    motivo = "limite"
                elif g["silencio_s"] >= SILENCIO_PARADA_S:
                    motivo = "silencio"
                else:
                    return None
                self.parar(motivo)
                return motivo
        return None

    def parar(self, motivo: str = "manual") -> dict:
        with self._trava:
            g = self._atual
            if g is None:
                return {"ok": True, "estado": self.estado()["estado"], "ja_parado": True,
                        "job": (self._pos or {}).get("job")}
            self._atual = None
            g["parando"] = True
            g["motivo_parada"] = motivo
            g["t_parada"] = self._relogio()
            self._pos, self._pos_ativos = g, self._pos_ativos + 1
        try:
            g["captura"].parar()
        except Exception as e:  # noqa: BLE001
            print(f"[controle] erro ao parar a captura: {e}", flush=True)
        g["fim"] = self._agora()
        sessao = g["sessao"]
        duracao = sessao.pcm.n / aovivo.TAXA
        sessao.marcar_fim(duracao)
        job_id = f"controle-{sessao.id[:12]}"
        self._jobs[job_id] = {"status": "processando", "progresso": 0.0, "mensagem": "Na fila...", "arquivo": str(sessao.arquivo),
                              "parciais": [], "duracao_s": round(duracao, 2)}
        g["job"] = job_id
        g["duracao_s"] = duracao
        self._submeter(self._executar, job_id, sessao)
        threading.Thread(target=self._pos_processar, args=(g,), name="aveia-controle-pos", daemon=True).start()
        return {"ok": True, "job": job_id}

    def _iniciar_vigia(self) -> None:
        if self._vigia and self._vigia.is_alive():
            return
        self._encerrar_vigia.clear()

        def laco() -> None:
            while not self._encerrar_vigia.wait(VIGIA_S):
                try:
                    self.tick()
                except Exception as e:  # noqa: BLE001
                    print(f"[controle] vigia falhou (segue): {e}", flush=True)

        self._vigia = threading.Thread(target=laco, name="aveia-controle-vigia", daemon=True)
        self._vigia.start()

    def encerrar(self) -> None:
        """Ao desligar o servidor: para a captura, sem processar (o WAV em disco fica)."""
        self._encerrar_vigia.set()
        with self._trava:
            g, self._atual = self._atual, None
        if g is not None:
            try:
                g["captura"].parar()
            except Exception:  # noqa: BLE001
                pass

    # ---- processamento final -------------------------------------------------------------------------
    def _pos_processar(self, g: dict) -> None:
        """Espera a transcrição. Sem página dona: reserva na hora. Com página dona: espera a `saida` dela; se ela sumir
        (sem consultar o painel por 4 min depois da parada), faz a reserva."""
        try:
            job = self._jobs[g["job"]]
            while job.get("status") == "processando":
                time.sleep(0.1)
            if job.get("status") != "pronto":
                raise RuntimeError(job.get("erro") or "a transcrição falhou")
            g["falas"] = [dict(f) for f in job.get("falas") or []]   # a tela apaga as falas do job ao confirmar: guarda a cópia
            while g.get("assumida_por") and not g["saida_feita"]:
                if self._relogio() - max(g["pagina_vista"], g["t_parada"]) > PAGINA_SUMIU_S:
                    g["assumida_por"] = None
                    break
                time.sleep(0.1)
            if not g["saida_feita"]:
                self._reserva(g)
        except Exception as e:  # noqa: BLE001
            self.ultimo_erro = f"{type(e).__name__}: {e}"
            print(f"[controle] processamento falhou: {self.ultimo_erro}", flush=True)
            try:
                g["pasta"].mkdir(parents=True, exist_ok=True)
                (g["pasta"] / "meta.json").write_text(json.dumps({"evento_id": g["evento_id"], "titulo": g["titulo"], "erro": self.ultimo_erro},
                                                                   ensure_ascii=False), encoding="utf-8")
            except OSError:
                pass
        finally:
            with self._trava:
                self._pos_ativos = max(0, self._pos_ativos - 1)

    def _meta_base(self, g: dict) -> dict:
        return {"evento_id": g["evento_id"], "titulo": g["titulo"], "participantes": g["participantes"],
                "inicio": g["inicio"].isoformat(timespec="seconds"), "fim": g["fim"].isoformat(timespec="seconds"),
                "duracao_s": round(g["duracao_s"], 1), "motivo_parada": g.get("motivo_parada", "manual")}

    def _gravar_uma_vez(self, g: dict, escrever: Callable[[], None]) -> bool:
        """Grava a saída no máximo uma vez por gravação (e uma por evento). False se alguém já gravou."""
        with g["trava_saida"]:
            if g["saida_feita"] or (g["evento_id"] and g["evento_id"] in self._eventos_gravados):
                return False
            escrever()
            g["saida_feita"] = True
            if g["evento_id"]:
                self._eventos_gravados.add(g["evento_id"])
        self._apos_saida(g["pasta"])
        return True

    def _reserva(self, g: dict) -> None:
        """O servidor faz nomes, resumo e a saída sozinho (nenhuma página assumiu, ou a página sumiu)."""
        falas = g["falas"] or []
        nomes, origem, usou_ia, resumo_res, aviso = self.nomear_e_resumir(g, [dict(f) for f in falas])
        meta = {**self._meta_base(g), "ia": usou_ia, "origem_nomes": origem, "processado_por": "servidor"}
        if aviso:
            meta["aviso"] = aviso
        if self._gravar_uma_vez(g, lambda: saida.gravar(g["pasta"], meta, falas, nomes, saida.montar_resumo(resumo_res))):
            self._limpar_audio(g)

    # ---- saída enviada pela página ---------------------------------------------------------------------
    def saida(self, dados: dict) -> tuple[int, dict]:
        """Resultado final montado pela tela: {evento_id, pagina?, titulo, falas (rótulos originais), locutores {rótulo: nome|null},
        resumo (dict: resumo, topicos, decisoes, acoes), ia}. Ou {evento_id, desistir: true} para o servidor assumir a reserva."""
        evento = str(dados.get("evento_id") or "")
        with self._trava:
            g = self._pos
            if g is None or g["evento_id"] != evento or "fim" not in g:
                return 404, {"ok": False, "motivo": "sem-gravacao"}
            if g["saida_feita"] or (evento and evento in self._eventos_gravados):
                return 409, {"ok": False, "motivo": "saida-ja-gravada"}
            pagina = str(dados.get("pagina") or "").strip()[:64]
            if g.get("assumida_por") and pagina and pagina != g["assumida_por"]:
                return 403, {"ok": False, "motivo": "outra-pagina"}
        if dados.get("desistir"):
            g["assumida_por"] = None
            return 200, {"ok": True, "desistiu": True}
        falas = []
        for f in dados.get("falas") if isinstance(dados.get("falas"), list) else []:
            try:
                falas.append({"inicio": float(f["inicio"]), "fim": float(f["fim"]), "locutor": str(f["locutor"]), "texto": str(f["texto"])})
            except (KeyError, TypeError, ValueError):
                return 400, {"ok": False, "motivo": "falas-invalidas"}
        locs = dados.get("locutores") if isinstance(dados.get("locutores"), dict) else {}
        nomes: dict[str, str | None] = {}
        for f in falas:
            v = locs.get(f["locutor"])
            nomes[f["locutor"]] = str(v).strip() if isinstance(v, str) and v.strip() else None
        resumo_res = dados.get("resumo") if isinstance(dados.get("resumo"), dict) else None
        titulo = " ".join(str(dados.get("titulo") or "").split()) or g["titulo"]
        meta = {**self._meta_base(g), "titulo": titulo, "ia": bool(dados.get("ia")), "origem_nomes": "canal+pistas",
                "processado_por": "tela"}
        if not self._gravar_uma_vez(g, lambda: saida.gravar(g["pasta"], meta, falas, nomes, saida.montar_resumo(resumo_res))):
            return 409, {"ok": False, "motivo": "saida-ja-gravada"}
        self._limpar_audio(g)
        return 200, {"ok": True, "pasta_saida": str(g["pasta"])}

    def _apos_saida(self, pasta: Path) -> None:
        """`AVEIA_APOS_SAIDA`: comando de linha executado uma vez, desanexado e sem janela, com `AVEIA_SAIDA_PASTA`. Falha só vai ao log."""
        cmd = (os.environ.get("AVEIA_APOS_SAIDA") or "").strip()
        if not cmd:
            return
        try:
            flags = 0
            if os.name == "nt":
                flags = subprocess.CREATE_NO_WINDOW | subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
            subprocess.Popen(cmd, shell=True, env={**os.environ, "AVEIA_SAIDA_PASTA": str(pasta)}, stdin=subprocess.DEVNULL,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=flags, close_fds=True)
        except Exception as e:  # noqa: BLE001
            print(f"[controle] AVEIA_APOS_SAIDA falhou: {type(e).__name__}: {e}", flush=True)

    def nomear_e_resumir(self, g: dict, falas: list[dict]) -> tuple[dict, str, bool, dict | None, str]:
        """(nomes por rótulo, origem_nomes, ia usada, resumo ou None, aviso). Sem IA: só rótulos e resumo vazio."""
        rotulos: list[str] = []
        for f in falas:
            if f["locutor"] not in rotulos:
                rotulos.append(f["locutor"])
        nomes: dict[str, str | None] = {r: None for r in rotulos}
        usuario = g.get("usuario") or ""
        if usuario and "Locutor 1" in nomes:
            nomes["Locutor 1"] = usuario   # canal E = microfone = quem grava
        if not ia.configurada() or not falas:
            return nomes, "canal+pistas", False, None, ""
        from . import locutores as ident
        from . import resumo
        pistas = [f"Participantes convidados: {', '.join(p['nome'] for p in g['participantes'])}."] if g["participantes"] else []
        pistas.append("Locutor 1 é quem grava a reunião (microfone deste computador); os demais locutores vêm do áudio da chamada.")
        aviso = ""
        try:
            fixos = {r: n for r, n in nomes.items() if n}
            r = ident.identificar(g["titulo"], pistas, falas, fixos, {})
            for rot, s in (r.get("sugestoes") or {}).items():
                if rot in nomes and not nomes[rot]:
                    nomes[rot] = s["nome"]
        except Exception as e:  # noqa: BLE001
            aviso = f"nomes por IA falharam ({type(e).__name__})"
        falas_nomeadas = [{**f, "locutor": nomes.get(f["locutor"]) or f["locutor"]} for f in falas]
        try:
            res = resumo.gerar_resumo(g["titulo"], pistas, falas_nomeadas, g["duracao_s"])
        except Exception as e:  # noqa: BLE001
            return nomes, "canal+pistas", False, None, (aviso + "; " if aviso else "") + f"resumo por IA falhou ({type(e).__name__})"
        return nomes, "canal+pistas", True, res, aviso

    def _limpar_audio(self, g: dict) -> None:
        """Sucesso: apaga o WAV e o .json parcial (o áudio de reunião não fica no disco), salvo AVEIA_CONTROLE_GUARDAR_AUDIO=1."""
        if os.environ.get("AVEIA_CONTROLE_GUARDAR_AUDIO") == "1":
            return
        arq = Path(g["sessao"].arquivo)
        for p in (arq, arq.with_suffix(".json")):
            try:
                p.unlink(missing_ok=True)
            except OSError:
                pass
