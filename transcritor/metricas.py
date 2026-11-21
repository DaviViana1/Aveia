"""Métricas locais: uma linha por gravação e o contador mensal das chamadas de IA na nuvem.

Só números e o nome do motor. Nunca texto, áudio, nome de arquivo, nome de locutor nem chave: `registrar_gravacao` só aceita os
campos de `CAMPOS`. Os arquivos ficam em `dados/` (fora do Git; `AVEIA_DADOS` troca a pasta) e nunca vão para a vault.

- `dados/metricas.jsonl`: uma linha por gravação (tempo até o 1º texto, parar até pronto, fator tempo/áudio...).
- `dados/ia_uso.json`: chamadas de IA do mês. Teto: `AVEIA_IA_TETO_MES` (padrão 300; 0 bloqueia a nuvem). Aviso a 80%.
"""
from __future__ import annotations

import json
import os
import threading
from datetime import datetime
from pathlib import Path

TETO_PADRAO = 300
AVISO_FRACAO = 0.8
CAMPOS = {
    "via", "motor", "duracao_s", "primeiro_texto_s", "parar_ao_pronto_s", "total_s", "fator_tempo", "blocos", "janelas_30s",
    "lacunas", "locutores", "diarizacao", "cpu_threads", "versao", "erro",
}
_trava = threading.Lock()


class TetoIA(RuntimeError):
    """O teto mensal de chamadas de IA na nuvem foi atingido (o app segue só com o que é local)."""


def pasta() -> Path:
    return Path(os.environ.get("AVEIA_DADOS") or Path(__file__).resolve().parent.parent / "dados")


def registrar_gravacao(**campos) -> None:
    """Acrescenta uma linha em `metricas.jsonl`. Campos fora de `CAMPOS` e valores que não são número ou texto curto são
    ignorados. Nunca levanta: métrica não pode derrubar uma gravação."""
    try:
        linha = {"quando": datetime.now().isoformat(timespec="seconds")}
        for k, v in campos.items():
            if k in CAMPOS and (v is None or isinstance(v, (int, float, bool)) or (isinstance(v, str) and len(v) <= 40)):
                linha[k] = round(v, 2) if isinstance(v, float) else v
        with _trava:
            p = pasta()
            p.mkdir(parents=True, exist_ok=True)
            with open(p / "metricas.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(linha, ensure_ascii=False) + "\n")
    except Exception as e:  # noqa: BLE001
        print(f"Métrica não gravada (ignorado): {e}")


# ---- chamadas de IA na nuvem --------------------------------------------------------------------------------------
def teto_mes() -> int:
    try:
        return max(0, int(os.environ.get("AVEIA_IA_TETO_MES", TETO_PADRAO)))
    except ValueError:
        return TETO_PADRAO


def _ler_uso() -> dict:
    mes = datetime.now().strftime("%Y-%m")
    try:
        d = json.loads((pasta() / "ia_uso.json").read_text(encoding="utf-8"))
        if d.get("mes") == mes and isinstance(d.get("chamadas"), int):
            return d
    except (OSError, ValueError):
        pass
    return {"mes": mes, "chamadas": 0, "por_servico": {}}


def uso_ia() -> dict:
    """Uso do mês para a interface: chamadas, teto, se passou de 80% e se está bloqueada."""
    with _trava:
        d = _ler_uso()
    teto = teto_mes()
    return {"mes": d["mes"], "chamadas": d["chamadas"], "teto": teto,
            "aviso": teto > 0 and d["chamadas"] >= AVISO_FRACAO * teto, "bloqueada": d["chamadas"] >= teto,
            "por_servico": d.get("por_servico", {})}


def contar_ia(servico: str) -> None:
    """Conta uma chamada de IA na nuvem ANTES de ela sair. Levanta `TetoIA` no teto (a chamada não deve sair)."""
    with _trava:
        d = _ler_uso()
        if d["chamadas"] >= teto_mes():
            raise TetoIA(f"Teto mensal de {teto_mes()} chamadas de IA atingido; o app segue só com o que é local "
                         "(AVEIA_IA_TETO_MES no .env muda o limite).")
        d["chamadas"] += 1
        d["por_servico"][servico] = d["por_servico"].get(servico, 0) + 1
        try:
            p = pasta()
            p.mkdir(parents=True, exist_ok=True)
            (p / "ia_uso.json").write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
        except OSError as e:
            print(f"Contador de IA não gravado (ignorado): {e}")
