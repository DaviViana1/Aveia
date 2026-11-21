"""Qual IA na nuvem o Aveia usa (resumo, título, nomes dos locutores e destaques).

- Padrão: Gemini, pela API compatível com OpenAI do Google. Chave: `GEMINI_API_KEY`. Modelo padrão `gemini-3.1-flash-lite`
  (barato; nos Gemini 3 o raciocínio não desliga, então o padrão é `reasoning_effort = minimal`).
- Reserva: Vercel AI Gateway, só quando NÃO há `GEMINI_API_KEY` (e há `AI_GATEWAY_API_KEY` ou `VERCEL_OIDC_TOKEN`).

Interruptor único: `AVEIA_IA=0` (ou `false`, `nao`, `off`) desliga TODA a IA no servidor (nenhuma chamada sai, para nenhum recurso);
o padrão é tudo ligado. O navegador tem o interruptor dele nos Ajustes (`cfg.ia`, também padrão ligado).

Variáveis opcionais: `AVEIA_MODELO_GEMINI` (modelo do Gemini), `AVEIA_GEMINI_PLANO` (`pago`, o padrão, ou `gratuito`: só muda o aviso de privacidade), `AVEIA_IA_RACIOCINIO` (`minimal`, `low`, `medium`, `high`),
`AVEIA_IA_URL` (troca o endereço; os testes usam um servidor falso local) e, por serviço, `AVEIA_MODELO_IA`,
`AVEIA_MODELO_TITULO` e `AVEIA_MODELO_LOCUTORES` (valem nos dois provedores). Nunca lê nem imprime o valor de uma chave além de
mandá-lo no cabeçalho da chamada.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

from . import metricas

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
GEMINI_MODELO = "gemini-3.1-flash-lite"
GATEWAY_URL = "https://ai-gateway.vercel.sh/v1/chat/completions"
RACIOCINIOS = ("minimal", "low", "medium", "high")
SEM_CHAVE = "Defina GEMINI_API_KEY no arquivo .env."
# Parâmetros que só alguns modelos aceitam: se a chamada der 400, ela é refeita sem eles.
PARAMETROS_OPCIONAIS = ("reasoning_effort", "reasoning")


def _env(nome: str) -> str:
    return (os.environ.get(nome) or "").strip()


def ligada() -> bool:
    """Interruptor geral do servidor: `AVEIA_IA=0` desliga toda a IA."""
    return _env("AVEIA_IA").lower() not in ("0", "false", "nao", "não", "off", "desligada", "desligado")


def sem_ia() -> str:
    """Por que não há IA agora (mensagem para o usuário)."""
    return SEM_CHAVE if ligada() else "A IA está desligada (AVEIA_IA=0)."


def provedor() -> str | None:
    """`gemini`, `gateway` (reserva) ou None (sem chave ou IA desligada)."""
    if not ligada():
        return None
    if _env("GEMINI_API_KEY"):
        return "gemini"
    if _env("AI_GATEWAY_API_KEY") or _env("VERCEL_OIDC_TOKEN"):
        return "gateway"
    return None


def chave() -> str | None:
    p = provedor()
    if p == "gemini":
        return _env("GEMINI_API_KEY")
    if p == "gateway":
        return _env("AI_GATEWAY_API_KEY") or _env("VERCEL_OIDC_TOKEN")
    return None


def configurada() -> bool:
    return provedor() is not None


def url() -> str:
    return _env("AVEIA_IA_URL") or (GATEWAY_URL if provedor() == "gateway" else GEMINI_URL)


def modelo(variavel: str | None = None, padrao_gateway: str = "") -> str:
    """Modelo do serviço: a variável do serviço (se definida) vale nos dois provedores; senão, o do provedor em uso."""
    if variavel and _env(variavel):
        return _env(variavel)
    if provedor() == "gateway":
        return padrao_gateway
    return _env("AVEIA_MODELO_GEMINI") or GEMINI_MODELO


def raciocinio() -> str:
    v = _env("AVEIA_IA_RACIOCINIO").lower()
    return v if v in RACIOCINIOS else "minimal"


def parametros() -> dict:
    """Parâmetros de raciocínio do provedor em uso (resumir e nomear não precisam de raciocínio longo)."""
    if provedor() == "gemini":
        return {"reasoning_effort": raciocinio()}
    return {"reasoning": {"enabled": False}}


def sem_opcionais(corpo: dict) -> bool:
    """Tira de `corpo` os parâmetros opcionais; devolve True se havia algum (então vale refazer a chamada)."""
    achou = False
    for k in PARAMETROS_OPCIONAIS:
        if k in corpo:
            del corpo[k]
            achou = True
    return achou


def destino() -> str:
    return {"gemini": "Google (Gemini)", "gateway": "Vercel AI Gateway"}.get(provedor() or "", "nenhuma IA configurada")


def plano() -> str:
    """Plano da chave do Gemini, como o usuário informou: `pago` (padrão; ele usa o plano pago) ou `gratuito`
    (`AVEIA_GEMINI_PLANO=gratuito`). Só muda o aviso de privacidade; a IA funciona igual."""
    return "gratuito" if _env("AVEIA_GEMINI_PLANO").lower() in ("gratuito", "gratis", "free") else "pago"


def aviso_privacidade() -> str:
    if provedor() == "gemini":
        if plano() == "pago":
            return ("Com a IA ligada, o texto da gravação vai ao Google (Gemini). Plano pago: pela política do Google, esse texto "
                    "não é usado para melhorar os produtos dele.")
        return ("Com a IA ligada, o texto da gravação vai ao Google (Gemini). Na camada gratuita o Google pode usar esse texto "
                "para melhorar os produtos dele; para conteúdo privado, use uma chave de projeto com faturamento (camada paga).")
    if provedor() == "gateway":
        return "Com a IA ligada, o texto da gravação vai ao Vercel AI Gateway e ao modelo escolhido por ele."
    return "A IA está desligada: nenhum texto sai deste computador." if not ligada() else "Sem chave de IA: nenhum texto sai deste computador."


# ---- chamada genérica (usada pelos recursos novos; resumo, título e locutores têm a própria, com regras específicas) --------
class Falha(RuntimeError):
    """A chamada não saiu ou não deu certo. `motivo`: sem_chave, teto ou falha."""

    def __init__(self, mensagem: str, motivo: str = "falha"):
        super().__init__(mensagem)
        self.motivo = motivo


def chamar(servico: str, sistema: str, usuario: str, max_tokens: int, temperatura: float = 0.0, timeout: float = 45,
           tentativas: int = 3) -> str:
    """Uma chamada de chat ao provedor em uso. Conta no teto mensal ANTES de sair (`servico` aparece no uso por serviço);
    refaz em 429 e 5xx e, se o modelo recusar um parâmetro opcional (400), sem ele."""
    c = chave()
    if not c:
        raise Falha(sem_ia(), "sem_chave" if ligada() else "desligada")
    try:
        metricas.contar_ia(servico)
    except metricas.TetoIA as e:
        raise Falha(str(e), "teto") from None
    corpo = {"model": modelo(), "temperature": temperatura, "max_tokens": max_tokens, **parametros(),
             "messages": [{"role": "system", "content": sistema}, {"role": "user", "content": usuario}]}
    for tentativa in range(tentativas):
        req = urllib.request.Request(url(), data=json.dumps(corpo).encode(),
                                     headers={"Authorization": f"Bearer {c}", "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                resposta = json.load(r)
            break
        except urllib.error.HTTPError as e:
            if e.code == 400 and sem_opcionais(corpo):
                continue
            if e.code in (429, 500, 502, 503, 504) and tentativa + 1 < tentativas:
                try:
                    espera = float(e.headers.get("Retry-After") or 0)
                except ValueError:
                    espera = 0
                time.sleep(min(espera or 2 * (tentativa + 1), 10))
                continue
            raise Falha(f"A IA respondeu {e.code}.") from None
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as e:
            if tentativa + 1 < tentativas:
                time.sleep(1.5)
                continue
            raise Falha(f"Não foi possível falar com a IA: {e}") from None
    else:
        raise Falha("A IA não respondeu.")
    try:
        return resposta["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError):
        raise Falha("Resposta inesperada da IA.") from None


CHAVES_EXTRAS = {"GEMINI_API_KEY", "AI_GATEWAY_API_KEY", "GROQ_API_KEY"}


def carregar_chaves_extras() -> None:
    """Se `AVEIA_ENV_EXTRA` apontar para um arquivo `.env`, lê dele SÓ as chaves de IA e de transcrição na nuvem (nunca as demais).
    Serve para guardar a chave fora da pasta do app. Não sobrescreve nada já definido e nunca imprime valor."""
    caminho = os.environ.get("AVEIA_ENV_EXTRA")
    if not caminho or not Path(caminho).is_file():
        return
    for linha in Path(caminho).read_text(encoding="utf-8").splitlines():
        linha = linha.strip()
        if linha and not linha.startswith("#") and "=" in linha:
            chave, valor = linha.split("=", 1)
            if chave.strip() in CHAVES_EXTRAS and valor.strip():
                os.environ.setdefault(chave.strip(), valor.strip().strip("\"'"))
