"""Gravação automática por controle: outro programa local manda iniciar e parar, sem nenhum clique na tela.

Fluxo: `iniciar` abre uma sessão ao vivo de PCM direto (`aovivo.Sessao`) e a captura do servidor (`captura.Captura`); cada bloco
de ~1 s vai para `Sessao.anexar_pcm`. A gravação para por `parar` ou sozinha: depois de `fim_previsto`, com 10 min seguidos de
silêncio no canal do computador, ou no máximo em `fim_previsto + 30 min`. Ao parar, o processamento (cauda, locutores,
nomes, resumo) segue em segundo plano e termina com a pasta `saida/...` e o `concluido.flag`.

IA: tudo respeita o interruptor existente (`ia.configurada()`): desligada = nenhuma chamada e `meta.ia = false`.
Nada aqui envia e-mail ou mensagem; o texto da reunião só é gravado em `saida/` (fora do Git).
"""
from __future__ import annotations

import json
import os
import secrets
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

    # ---- estado ---------------------------------------------------------------------------------------
    def estado(self) -> dict:
        with self._trava:
            if self._atual is not None:
                g, est = self._atual, "gravando"
            elif self._pos_ativos and self._pos is not None:
                g, est = self._pos, "processando"
            else:
                g, est = (self._pos if self._pos else None), "ocioso"
            r = {"estado": est, "evento_id": None, "titulo": None, "inicio": None, "fim_previsto": None, "pasta_saida": None}
            if g is not None and est != "ocioso":
                r.update(evento_id=g["evento_id"], titulo=g["titulo"], inicio=g["inicio"].isoformat(timespec="seconds"),
                         fim_previsto=g["fim_previsto"].isoformat(timespec="seconds"), pasta_saida=str(g["pasta"]))
            if self.ultimo_erro:
                r["ultimo_erro"] = self.ultimo_erro
            return r

    # ---- iniciar --------------------------------------------------------------------------------------
    def iniciar(self, dados: dict) -> tuple[int, dict]:
        with self._trava:
            if self._atual is not None:
                return 409, {"ok": False, "motivo": "ja-gravando", "evento_id": self._atual["evento_id"]}
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
            g = {"evento_id": str(dados.get("evento_id") or ""), "titulo": titulo, "participantes": _participantes(dados.get("participantes")),
                 "usuario": str(dados.get("usuario") or os.environ.get("AVEIA_NOME_USUARIO") or "").strip(),
                 "inicio": agora, "fim_previsto": fim, "pasta": pasta, "sessao": sessao, "captura": None,
                 "silencio_s": 0.0, "parando": False}
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
                return {"ok": True, "estado": self.estado()["estado"], "ja_parado": True}
            self._atual = None
            g["parando"] = True
            g["motivo_parada"] = motivo
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
                              "parciais": []}
        g["job"] = job_id
        g["duracao_s"] = duracao
        self._submeter(self._executar, job_id, sessao)
        threading.Thread(target=self._pos_processar, args=(g,), name="aveia-controle-pos", daemon=True).start()
        return {"ok": True}

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
        try:
            job = self._jobs[g["job"]]
            while job.get("status") == "processando":
                time.sleep(0.5)
            if job.get("status") != "pronto":
                raise RuntimeError(job.get("erro") or "a transcrição falhou")
            falas = [dict(f) for f in job.get("falas") or []]
            nomes, origem, usou_ia, resumo_res, aviso = self.nomear_e_resumir(g, falas)
            meta = {"evento_id": g["evento_id"], "titulo": g["titulo"], "participantes": g["participantes"],
                    "inicio": g["inicio"].isoformat(timespec="seconds"), "fim": g["fim"].isoformat(timespec="seconds"),
                    "duracao_s": round(g["duracao_s"], 1), "ia": usou_ia, "origem_nomes": origem,
                    "motivo_parada": g.get("motivo_parada", "manual")}
            if aviso:
                meta["aviso"] = aviso
            saida.gravar(g["pasta"], meta, falas, nomes, saida.montar_resumo(resumo_res))
            self._limpar_audio(g)
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
