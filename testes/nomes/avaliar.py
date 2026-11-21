"""Avaliação do passo de identificação de nomes (chamadas reais à IA configurada: Gemini por padrão).

    .venv\\Scripts\\python testes\\nomes\\avaliar.py                 # todos os casos
    .venv\\Scripts\\python testes\\nomes\\avaliar.py --repeticoes 3   # repete (mede estabilidade)
    .venv\\Scripts\\python testes\\nomes\\avaliar.py --longo          # gravação de ~3 h sintética (janelas de contexto)
    .venv\\Scripts\\python testes\\nomes\\avaliar.py --saida r.json   # guarda o detalhe em JSON

Precisão = acertos / (acertos + nomes errados ou inventados); revocação = acertos / nomes esperados.
Nome dado a quem deveria ficar sem nome conta como erro (falso positivo). Casos: testes/nomes/casos.json.
Nunca imprime a chave de API.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ))


def carregar_env() -> None:
    env = RAIZ / ".env"
    if env.exists():
        for linha in env.read_text(encoding="utf-8").splitlines():
            if "=" in linha and not linha.strip().startswith("#"):
                k, v = linha.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip("\"'"))


carregar_env()
from transcritor import ia as _ia  # noqa: E402
_ia.carregar_chaves_extras()
from transcritor import locutores as L  # noqa: E402


def expandir(caso: dict) -> list[dict]:
    t, falas = 0.0, []
    for rot, texto in caso["falas"]:
        dur = max(1.5, len(texto.split()) * 0.4)
        falas.append({"inicio": round(t, 2), "fim": round(t + dur, 2), "locutor": f"Locutor {rot}", "texto": texto})
        t += dur + 0.6
    return falas


def rodar(caso: dict, falas: list[dict] | None = None) -> dict:
    falas = falas or expandir(caso)
    fixos = {f"Locutor {k}": v for k, v in caso.get("fixos", {}).items()}
    inicio = time.time()
    try:
        r = L.identificar(caso.get("titulo", ""), caso.get("notas", []), falas, fixos)
        erro = None
    except Exception as e:   # noqa: BLE001
        r, erro = {"sugestoes": {}, "descartados": [], "janelas": {}}, f"{type(e).__name__}: {e}"
    return {"id": caso["id"], "r": r, "erro": erro, "seg": round(time.time() - inicio, 1)}


def avaliar(caso: dict, saida: dict) -> list[dict]:
    linhas = []
    for k, esperado in caso["esperado"].items():
        rotulo = f"Locutor {k}"
        pred = (saida["r"]["sugestoes"].get(rotulo) or {}).get("nome")
        aceitos = {L._norm(x) for x in [esperado] + caso.get("aceitos", {}).get(k, []) if x}
        ok = (pred is None and esperado is None) or (pred is not None and L._norm(pred) in aceitos)
        linhas.append({"caso": caso["id"], "rotulo": rotulo, "esperado": esperado, "previsto": pred, "ok": ok,
                       "tipo": "TP" if ok and pred else "TN" if ok else "FN" if pred is None else "FP"})
    return linhas


def longo() -> tuple[dict, list[dict]]:
    """Cerca de 3 h de conversa de enchimento (sem nomes, com ruído de nomes de terceiros) entre uma abertura com
    apresentações e uma chamada por vocativo bem no final: as janelas precisam achar as três pessoas."""
    base = [["1", "Bom dia a todos. Aqui é o Marcos, eu vou conduzir a reunião de hoje."], ["2", "Bom dia, Marcos. Eu sou a Ana, do financeiro."], ["1", "Perfeito, então vamos começar pelo primeiro ponto da pauta."]]
    enchimento = ["Então, sobre esse ponto específico, eu acho que a gente precisa olhar com calma os números do período antes de tomar qualquer decisão de investimento.",
                  "O João disse na semana passada que o fornecedor aceitaria negociar prazos, mas ainda não recebemos nada formal por escrito sobre isso.",
                  "Concordo. A Maria já levantou essa questão em outras reuniões e a gente acabou deixando para depois por causa das prioridades do trimestre.",
                  "Do ponto de vista operacional, o impacto maior aparece na logística, porque o custo do frete subiu bastante nos últimos meses em todas as regiões."]
    falas = list(base)
    for i in range(800):
        falas.append([str(1 + i % 2), enchimento[i % len(enchimento)] + f" Ponto número {i}."])
    falas += [["1", "Beatriz, você pode resumir os próximos passos para a gente encerrar?"], ["3", "Posso sim. Primeiro, revisar o orçamento; segundo, enviar a proposta até sexta-feira."], ["1", "Perfeito, obrigado a todos."]]
    caso = {"id": "longo-3h", "falas": falas, "esperado": {"1": "Marcos", "2": "Ana", "3": "Beatriz"}}
    return caso, expandir(caso)


def referencia() -> tuple[dict, list[dict]]:
    """testes/audio-referencia/referencia.json (leitura dramática com 2 vozes: Carteiro e Marta). Há nomes ditos, mas todos
    enganosos: "Marta" e "Pedro" aparecem numa CARTA lida por Marta ("Marta, peço desculpa... Esquece este assunto, Pedro") e
    "A letra é de Pedro" é terceira pessoa. O certo é não inventar "Pedro" (quem escreve a carta) nem dar nome ao Carteiro
    (um papel). "Marta" para a voz feminina é a resposta correta, mas sem evidência segura de que é ela: aceitável ou sem nome."""
    dados = json.loads((RAIZ / "testes" / "audio-referencia" / "referencia.json").read_text(encoding="utf-8"))
    rot = {"Carteiro": "Locutor 1", "Marta": "Locutor 2"}
    falas = [{"inicio": d["inicio"], "fim": d["fim"], "locutor": rot[d["locutor"]], "texto": d["texto"]} for d in dados]
    caso = {"id": "referencia-ines-carteiro", "esperado": {"1": None, "2": None}, "aceitos": {"2": ["Marta"]}}
    return caso, falas


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeticoes", type=int, default=1)
    ap.add_argument("--longo", action="store_true")
    ap.add_argument("--referencia", action="store_true", help="o áudio de referência da sessão 0 (carta com nomes enganosos)")
    ap.add_argument("--saida")
    ap.add_argument("--caso", help="só os casos cujo id contém este texto")
    a = ap.parse_args()
    if not L.configurada():
        print("GEMINI_API_KEY não definida: nada a avaliar.")
        return 2

    print(f"modelo: {L.modelo()} · limiar: {L.LIMIAR}")
    if a.longo:
        caso, falas = longo()
        s = rodar(caso, falas)
        j = s["r"].get("janelas", {})
        print(f"falas {j.get('falas')} · enviadas {j.get('enviadas')} · {j.get('caracteres')} caracteres (~{(j.get('caracteres') or 0) // 4} tokens) · completo={j.get('completo')} · {s['seg']} s")
        for l in avaliar(caso, s):
            print(f"  {l['rotulo']}: esperado={l['esperado']} previsto={l['previsto']} {'OK' if l['ok'] else 'ERRO'}")
        if s["erro"]:
            print("erro:", s["erro"])
        print("descartados:", json.dumps(s["r"].get("descartados"), ensure_ascii=False))
        return 0

    if a.referencia:
        caso, falas = referencia()
        for _ in range(a.repeticoes):
            s = rodar(caso, falas)
            print(f"{s['seg']} s · sugestões: {json.dumps(s['r']['sugestoes'], ensure_ascii=False)}")
            print("descartados:", json.dumps(s["r"]["descartados"], ensure_ascii=False))
            for l in avaliar(caso, s):
                print(f"  {l['rotulo']}: previsto={l['previsto']} {'OK' if l['ok'] else 'ERRO'}")
        return 0

    casos = [c for c in json.loads((Path(__file__).parent / "casos.json").read_text(encoding="utf-8"))
             if not a.caso or a.caso in c["id"]]
    todas, detalhe, falhas = [], [], 0
    for rep in range(a.repeticoes):
        for c in casos:
            s = rodar(c)
            falhas += bool(s["erro"])
            ls = avaliar(c, s)
            todas += ls
            detalhe.append({"rep": rep, "caso": c["id"], "sugestoes": s["r"]["sugestoes"], "descartados": s["r"]["descartados"], "erro": s["erro"], "seg": s["seg"]})
            marca = "OK  " if all(l["ok"] for l in ls) else "ERRO"
            extra = "" if not s["erro"] else f"  [{s['erro'][:80]}]"
            prev = {l["rotulo"][-1]: l["previsto"] for l in ls}
            print(f"{marca} {c['id']:<42} {prev} ({s['seg']} s){extra}")
            for l in ls:
                if not l["ok"]:
                    print(f"       {l['rotulo']}: esperado={l['esperado']} previsto={l['previsto']} [{l['tipo']}]")
    tp = sum(l["tipo"] == "TP" for l in todas)
    fp = sum(l["tipo"] == "FP" for l in todas)
    fn = sum(l["tipo"] == "FN" for l in todas)
    tn = sum(l["tipo"] == "TN" for l in todas)
    precisao = tp / (tp + fp) if tp + fp else 1.0
    revocacao = tp / (tp + fn) if tp + fn else 1.0
    print(f"\nrótulos avaliados: {len(todas)} · acertos {tp} · sem nome correto {tn} · nomes errados/inventados (FP) {fp} · nomes perdidos (FN) {fn}")
    print(f"PRECISÃO {precisao:.1%} · REVOCAÇÃO {revocacao:.1%} · casos com falha de chamada: {falhas}")
    if a.saida:
        Path(a.saida).write_text(json.dumps({"precisao": precisao, "revocacao": revocacao, "tp": tp, "fp": fp, "fn": fn, "tn": tn, "casos": detalhe}, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
