# Scripts das medições de 12/11/2026

Roteiros usados nas seções 10 e 11 de `../RESULTADOS.md`. Foram copiados do diretório temporário da sessão e só tiveram os caminhos ajustados: a raiz do projeto agora é achada a partir do próprio arquivo (`parents[3]`), então funcionam de onde estiverem dentro do repositório. **Não foram executados de novo depois da cópia.** Todos usam o Whisper local real e `HF_HUB_OFFLINE=1`; nenhum lê o `.env` nem usa IA externa. Rode com `.venv\Scripts\python`.

| Script | O que faz | Seção |
|---|---|---|
| `rodar_gloss.py sem\|com` | `pipeline.transcrever` (caminho clássico) em `testes/audio-ditado/ditado.m4a`, com ou sem glossário; calcula o WER contra `transcricao-referencia.txt` e grava `bruto/gloss__<sem\|com>__ditado.json` | 10 |
| `srv.py` | Sobe o app de teste na **porta 8799** (nunca a 8765), com chaves de IA vazias, sem abrir navegador e com as gravações numa **pasta temporária** (`aveia-gravacoes-teste`, dentro do `TEMP` do sistema), não em `gravacoes/` | 11.1 |
| `cliente.py aovivo\|classico` | Cliente que imita o navegador contra o `srv.py`: no modo `aovivo`, envia o `audio.webm` de referência em 29 pedaços, um a cada 2 s, consulta o texto a cada 1 s e finaliza; no `classico`, `POST /api/transcrever`. Grava `bruto/aovivo_ocioso__<via>__ref58-webm.json` | 11.1 |
| `blocos.py <clipe>...` | Alimenta a `Sessao` de `transcritor/aovivo.py` (blocos de 15 a 30 s) com clipes que truncam, sem esperar tempo real e com a diarização desligada. Clipes: `real-ganho-30db`, `cv`, `cv-webm`, `conv-2loc`, `conv-4loc`. Grava `bruto/blocos_atual__<clipe>.json` e `log_blocos.txt`; precisa dos áudios e do cache de `testes/lacunas_revalidar.py` (que não estão no repositório) | 11.2 |

Observações para reuso:
- Para medir o ao vivo: inicie `srv.py` (deixe-o rodando), espere "modelo carregado" e rode `cliente.py aovivo` e depois `cliente.py classico`. Pare o servidor ao terminar.
- `rodar_gloss.py`, `cliente.py` e `blocos.py` **sobrescrevem** os `bruto/*.json` com o mesmo nome (o `blocos.py` pula um clipe cujo arquivo já existe). Faça cópia antes se quiser guardar os resultados de 12/11/2026.
- `blocos.py` importa `lacunas_revalidar` de `testes/` e lê as saídas `q1b__*`, `q2__*` de `bruto/`; sem elas falha.
- `cliente.py` usa constantes fixas do clipe de referência (57,7 s de duração).
