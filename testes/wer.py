"""WER (taxa de erro de palavras) reutilizável: `wer(ref, hip)`.

Normalização (igual para ref e hipótese): minúsculas, hífen vira espaço ("peço-te" -> "peço te"),
pontuação removida, espaços colapsados; números ficam como estão no texto. `sem_acentos=True` também
tira os acentos. Levenshtein de palavras (inserção, remoção e troca valem 1), sem dependências.

Uso:  python testes/wer.py referencia.txt hipotese.txt      (ou)      from testes.wer import wer
"""
from __future__ import annotations

import re
import sys
import unicodedata


def normalizar(texto: str, sem_acentos: bool = False) -> str:
    t = texto.lower().replace("-", " ").replace("’", "").replace("'", "")
    if sem_acentos:
        t = "".join(c for c in unicodedata.normalize("NFD", t) if unicodedata.category(c) != "Mn")
    t = re.sub(r"[^\w\s]", " ", t)
    return " ".join(t.split())


def distancia(ref: list[str], hip: list[str]) -> int:
    anterior = list(range(len(hip) + 1))
    for i, r in enumerate(ref, 1):
        atual = [i]
        for j, h in enumerate(hip, 1):
            atual.append(min(anterior[j] + 1, atual[j - 1] + 1, anterior[j - 1] + (r != h)))
        anterior = atual
    return anterior[-1]


def wer(ref: str, hip: str, sem_acentos: bool = False) -> float:
    """Fração de erros por palavra da referência (0.0 = idêntico; pode passar de 1.0)."""
    r = normalizar(ref, sem_acentos).split()
    h = normalizar(hip, sem_acentos).split()
    if not r:
        return 0.0 if not h else float(len(h))
    return distancia(r, h) / len(r)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    ler = lambda p: open(p, encoding="utf-8").read()
    print(f"WER = {wer(ler(sys.argv[1]), ler(sys.argv[2])):.4f}")
