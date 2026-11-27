# Aveia

App local de notas por voz: grava ou importa áudio, transcreve e separa as pessoas pelo timbre. Roda no seu computador; a nuvem só entra se você ligar.

- **Transcrição:** faster-whisper, modelo `large-v3-turbo`, na CPU (sem GPU).
- **Locutores:** sherpa-onnx (segmentação pyannote 3.0, embeddings CAM++, agrupamento Ward com `scipy`).
- **Interface:** `static/index.html`, servida por FastAPI em `http://127.0.0.1:8765` (só esta máquina).
- **Notas:** ficam no navegador (localStorage). Áudio e transcrições guardados ficam em `gravacoes/`.

## Instalar (Windows 10 22H2 ou 11, 64 bits)

1. **Baixe**: neste repositório, **Code > Download ZIP**, e extraia (ou `git clone https://github.com/DaviViana1/Aveia.git`).
2. **Abra** a pasta e dê dois cliques em **`iniciar.bat`**. Na primeira vez ele confere o computador, instala o Python 3.12 pelo winget se você não tiver (pede confirmação), cria o ambiente e instala as bibliotecas (alguns minutos).
3. O navegador abre em `http://127.0.0.1:8765` (se essa porta estiver com outro programa, o Aveia usa a próxima livre e mostra o endereço na janela; abrir de novo a mesma pasta só reabre o navegador). Na primeira vez o app baixa o modelo de voz (cerca de 1,6 GB; precisa de internet e de uns 6 GB livres).
4. Em **Ajustes > Seu computador**, toque em **Testar este computador**: ele mede a velocidade real e diz se dá para transcrever ao vivo.

Para encerrar: feche a janela preta ou use `parar.bat`. Para atualizar: `atualizar.bat` (com `git clone`) ou baixe o ZIP novo.

Se o Windows perguntar se quer executar o arquivo (comum com ZIP baixado da internet), confirme: o `iniciar.bat` é texto simples, você pode abri-lo no Bloco de Notas antes. Nada é instalado fora desta pasta, exceto o Python (só se faltar e você aceitar).

Pré-requisitos mínimos (conferidos pelo `iniciar.bat`): 4 GB de memória (8 GB recomendados), 6 GB livres em disco, processador com AVX2. O `scipy` vem em `requirements.txt` (sem ele a separação de locutores cai no método antigo e o app avisa). Para conferir: o selo **Local** ao lado do logo.

## Usar

Gravar: botão vermelho. Importar: ícone de upload. Renomear locutor: clique no nome (usar o nome de outro mescla os dois). Exportar: `.txt`, `.srt` e `.vtt`. Ajustes: microfone, idioma, número de locutores, glossário, áudio do computador, motor e IA.

**Modelo da IA (comparação de 2026-11-21, chamadas reais, texto sintético do projeto):** `gemini-3.1-flash-lite` contra `gemini-3.5-flash-lite`, 3 repetições de 33 casos de nomes e 9 de destaques, 1 repetição de títulos e resumos . Nomes: precisão 1,0 nos dois; revocação 0,92 nas 3 rodadas do 3.1 (estável) contra 0,96, 0,96 e 0,84 do 3.5. Destaques: 27 de 27 exatos (3.1) contra 25 de 27 (3.5). Custo das 129 chamadas: US$ 0,072 contra US$ 0,105. Mediana por chamada: 1,3 s contra 1,2 s. Títulos e resumos válidos nos dois. Repetir: `testes/comparar_modelos_gemini.py`.

**Nomes dos locutores (IA):** a IA só sugere nome com evidência: autoapresentação, vocativo respondido, apresentação por terceiro ou **mensagem recebida** (a pessoa recebe uma carta/e-mail na conversa e a lê em voz alta; o nome da saudação é dela, com confiança mínima de 0,85 e a entrega presente na transcrição). Ler em voz alta uma mensagem que não foi entregue a quem lê não vale. Avaliar: `testes/nomes/avaliar.py`; comparar modelos: `testes/comparar_modelos_gemini.py`.

**Renomear locutor atualiza o resumo (sem IA):** ao renomear (no card da transcrição, por sugestão da IA, ao mesclar ou pelo chat), o resumo e o título automático trocam "Locutor N" e o nome anterior pelo nome novo, só como palavra inteira ("Locutor 1" não pega "Locutor 10"). Suas edições no resumo ficam; a nota que você escreve nunca é tocada (`static/nomes.js`).

**Resumo editável como bloco de notas:** cada texto do resumo (parágrafos, pontos-chave, linha do tempo, decisões, próximos passos, destaques) se edita no próprio lugar, sem botão de editar. Texto simples; Enter cria item, Backspace no começo junta, item vazio some. Grava sozinho e fica marcado como editado: a IA automática não o sobrescreve, Refazer pergunta antes e o botão de desfazer volta ao resumo anterior (um nível). Cópia, download e Análises saem com o texto editado.

**Identificar quem fala pelo chat:** durante a gravação, uma frase como "Marina que está falando", "Marina que falou agora", "é a Marina", "agora quem fala é o Carlos" ou "Locutor 2 é a Marina" não vira destaque: dá o nome ao locutor do instante em que você começou a digitar, em todas as falas, quando a gravação termina (durante a gravação o texto ao vivo ainda não tem locutor). Frases conhecidas são entendidas por regra de texto, sem rede; com a IA ligada, a IA (`/api/destaque-comando`, conta no teto como `identificacao`) entende variações e o nome tem de estar escrito na frase. Nome que você definiu à mão pede confirmação antes de trocar; dois locutores no mesmo instante: pergunta qual é.

**Cabeçalho flutuante:** o cabeçalho fica fixo no topo, em vidro fosco (opaco com transparência reduzida ou contraste alto).

**Chat da gravação:** enquanto grava, o chat flutuante mostra a transcrição ao vivo e os seus destaques (até 10 linhas, os mais antigos mais apagados). Um destaque cobre as falas dos **10 s anteriores** ao instante em que você começou a digitar e aparece antes desse trecho, com o trecho realçado. Com a **IA** ligada, depois da transcrição final a IA (Gemini, `/api/destaque-escopo`) escolhe, dentro dessa janela, só as frases do assunto do destaque (uma chamada por destaque, conta no teto mensal). Sem IA (interruptor desligado, sem chave, teto atingido ou erro), vale a janela dos 10 s (o app avisa quando é por erro, chave ou teto). Falas longas são partidas na exibição no ponto do trecho; o horário dentro de uma fala longa é estimado (o navegador não recebe o tempo de cada palavra).

## Gravação automática por controle

Outro programa local pode mandar o Aveia gravar sem nenhum clique: o servidor captura o microfone padrão (canal E) e o áudio que sai do computador (canal D, loopback WASAPI) com o PyAudioWPatch, transcreve ao vivo e, ao parar, grava a pasta `saida/<AAAA-MM-DD>_<HHMM>_<titulo>/` (fora do Git) com `meta.json`, `transcricao.md`, `resumo.md`, `locutores.json`, `falas.json` e, por último, `concluido.flag`.

- Subir sem tela: `AVEIA_SEM_NAVEGADOR=1 .venv\Scripts\python app.py`. Na subida o app grava `%LOCALAPPDATA%\Aveia\controle.json` com `porta`, `token` e `pid`.
- Rotas (cabeçalho `X-Aveia-Token` obrigatório): `GET /api/controle/estado`, `POST /api/controle/iniciar` (`evento_id`, `titulo`, `participantes`, `fim_previsto`), `POST /api/controle/parar`. Gravando de novo, `iniciar` responde 409; com gravação manual da tela em andamento, 409 `tela-gravando`.
- Na tela (sem barra no topo): ao ver uma gravação nova no painel (`GET /api/controle/painel?pagina=<id da aba>`, sem token, a cada 3 s), a página a **assume** (`POST /api/controle/assumir {sessao, pagina}`; só uma página assume, a outra recebe 409; a mesma aba assume de novo ao recarregar, e outra aba só se a dona ficar 20 s sem consultar), cria uma **nota nova** com o título da reunião (participantes da agenda viram a linha "Participantes: ..." da nota, pistas para os nomes; o canal do microfone, Locutor 1, recebe o nome do usuário) e entra no mesmo estado de uma gravação manual: tempo, texto ao vivo (`GET /api/aovivo/{sessao}`), chat e destaques. Sem pausa (a captura é do servidor). O vínculo evento -> nota fica no armazenamento local da tela.
- Parar: o botão da própria nota chama `POST /api/controle/parar-tela` (devolve o `job`), e a parada automática do servidor (silêncio ou limite) é detectada pelo painel como se a pessoa tivesse parado. A página roda o pós-processamento normal (job, nomes, título, resumo, salvar, confirmar, arquivar) e, no fim, manda o resultado final por `POST /api/controle/saida` (`evento_id`, `titulo`, `falas` com rótulos originais, `locutores`, `resumo`, `ia`; ou `desistir: true`). `assumir`, `parar-tela` e `saida` não usam token e exigem `Origin` local.
- Uma saída só: com página dona, o servidor não chama a IA; ele grava a pasta com `concluido.flag` por último a partir da `saida`. Sem página (navegador fechado) ou se ela ficar 60 s sem consultar o painel depois da parada, o servidor processa sozinho (reserva). Um segundo envio do mesmo evento recebe 409. O áudio segue o fluxo normal da tela (`confirmar`: vai para `gravacoes/Áudio` enquanto houver vaga).
- `AVEIA_APOS_SAIDA` (linha de comando): executada uma vez depois do `concluido.flag`, desanexada e sem janela, com `AVEIA_SAIDA_PASTA=<pasta>`; falha só vai ao log.
- `GET /api/controle/estado` devolve também `pagina_aberta` (alguma página consultou o painel nos últimos 10 s), `sessao`, `job` e `assumida`.
- Parada automática: depois de `fim_previsto`, com 10 min seguidos de silêncio no canal do computador, ou no máximo em `fim_previsto` + 30 min.
- IA desligada (`AVEIA_IA=0` ou sem chave): nenhuma chamada, `resumo.md` com as seções vazias e `meta.ia = false`. Ligada: nomes pelos participantes e resumo executivo.
- Sucesso apaga o WAV temporário de `gravacoes/` (`AVEIA_CONTROLE_GUARDAR_AUDIO=1` mantém). `AVEIA_NOME_USUARIO` dá nome ao canal do microfone.
- Nomes pelo Teams: `.venv\Scripts\python -m transcritor.alinhar_teams <pasta_saida> <arquivo.vtt>`.
- Diagnóstico dos dispositivos: `.venv\Scripts\python -m transcritor.captura --testar`.

## Privacidade

| Recurso | Onde roda | O que sai da máquina |
|---|---|---|
| Transcrição e separação de locutores | Local | nada |
| Resumo, título e nomes por IA (chave no `.env`) | Nuvem (Google Gemini) | texto |

- O painel **Privacidade** (Ajustes) mostra essa tabela com o uso real, o uso mensal da IA e os arquivos no disco.
- **IA: um interruptor só** (Ajustes), **ligado por padrão**. Desligado = nenhuma chamada de IA (resumo, título, nomes dos locutores e escolha do trecho do destaque; o destaque usa a janela dos 10 s e o botão Gerar fica inativo). Ligado = tudo ativo, sem outro botão. `AVEIA_IA=0` desliga a IA no servidor inteiro. O aviso de privacidade (para onde o texto vai) fica só nos Ajustes (ícone de configurações), nunca solto na tela.
- **IA = Google Gemini** (`GEMINI_API_KEY`), modelo padrão `gemini-3.1-flash-lite` (o mais barato e o mais estável dos dois em 3 repetições de 33 casos de nomes e 9 de destaques: veja `testes/comparar_modelos_gemini.py`; cerca de US$ 0,0006 por chamada). **Privacidade:** com a IA ligada, o texto da gravação vai ao Google. Com uma chave de **plano pago**, pela política do Google o texto não é usado para melhorar os produtos dele. Na camada gratuita seria diferente; com uma chave gratuita, defina `AVEIA_GEMINI_PLANO=gratuito` para o aviso do app dizer isso. Sem `GEMINI_API_KEY`, o Vercel AI Gateway (`AI_GATEWAY_API_KEY`) serve de reserva. Dentro da vault, o app lê do `.env` dela só as chaves `GEMINI_API_KEY` e `AI_GATEWAY_API_KEY`.
- **Teto de IA:** `AVEIA_IA_TETO_MES` (padrão 300 chamadas por mês; 0 bloqueia). Aviso a 80%.
- **Apagar tudo** e **Exportar tudo** (Ajustes > Privacidade). Retenção: `AVEIA_RETENCAO_DIAS` (padrão 30) só marca o que passou do prazo; nada é apagado sozinho.
- Gravar o áudio do computador pede um aviso na primeira vez: avise os participantes. A regra legal que vale para você: consulte um advogado.
- O servidor só atende `Host` local; pedido que muda algo exige `Origin` igual ao `Host` (mesma porta) e não pode vir de outro site; há teto de tamanho (upload 1 GB, pedaço ao vivo 8 MB, JSON 5 MB), sem `/docs`, e as páginas não abrem dentro de iframe.
- **Limitação conhecida:** a interface carrega Tailwind e ícones de CDN (`cdn.tailwindcss.com`, `unpkg.com`) sem integridade fixada. Se um desses serviços fosse comprometido, o script rodaria no app. Servir esses arquivos localmente é um item em aberto (`PENDENCIAS.md`).
- A interface carrega estilos e fontes de CDN (internet a cada abertura); modelos baixam no primeiro uso.

## Desempenho (medido em 2026-11-20, i5-1335U, 10 núcleos, sem GPU)

- Cada chamada do Whisper custa uma janela de 30 s (cerca de 17 s), mesmo para 8 s de áudio. Um áudio de 58 s custa 35 a 37 s no mínimo (2 janelas).
- Threads (4 a 12), `beam_size=1`, lote e `word_timestamps` não mudaram o tempo de forma demonstrada (ruído de 17 a 64% entre repetições). VAD, int8 e `condition_on_previous_text=False` já estão ligados.
- Ao vivo com o `scipy` instalado (clipe de 58 s): primeiro texto em 26 a 33 s, parar até pronto em 24 s, 2 locutores achados (antes, sem `scipy`: 1 locutor).
- Piso com este modelo e este CPU: cerca de 17 s por janela mais a diarização (3 a 4 s). Passar disso exige modelo menor ou aceleração por hardware (não testados).
- `AVEIA_BLOCO_ADAPTATIVO=1` enche mais os blocos: -8% a -38% de CPU em 5 clipes, WER agregado +0,4 ponto, mas um clipe (`cv-webm`) piora 3,5 pontos. Por isso vem desligado. Medir: `testes/bench_aovivo.py`.
- Métricas por gravação (só números) em `dados/metricas.jsonl`, fora do Git.

## Testes

```
.venv\Scripts\python -m unittest testes.test_melhorias testes.test_canais testes.test_aovivo testes.test_destaques testes.test_pipeline testes.test_titulo testes.test_diarizacao testes.test_lacunas testes.test_glossario testes.nomes.test_validacao testes.test_chat_flutuante testes.test_ia_gemini testes.test_destaque_escopo testes.test_interruptor_ia testes.test_identificacao testes.test_nomes_js testes.test_resumo_editavel testes.test_perfil_pc testes.test_porta
```

377 testes rodam e passam (2026-11-21); 58 pulam sozinhos porque dependem de áudios de referência e de modelos que não vão no repositório (as gravações de teste do autor ficam de fora). Para rodar todos: `.venv\Scripts\python -m unittest discover -s testes -p "test_*.py"`. Uma fala só termina por troca de locutor (timbre), por pausa de 30 s ou mais, ou ao passar de 5 min (`PAUSA_NOVA_FALA`, `FALA_MAX_S` em `transcritor/pipeline.py`). Não há teste automático da interface; a conferência de 2026-11-20 foi feita à mão no navegador (painel, VTT, apagar tudo, aviso do áudio do computador, 375 px sem rolagem horizontal).

## Variáveis (`.env` ou ambiente)

Chaves (só o usuário preenche; ver `.env.example`): `GEMINI_API_KEY` (IA) e, como reserva da IA, `AI_GATEWAY_API_KEY`.

| Variável | Efeito |
|---|---|
| `AVEIA_DIARIZACAO` | `novo` (padrão) ou `antigo` |
| `AVEIA_DIAR_AOVIVO` | `0` desliga a diarização durante a gravação |
| `AVEIA_LACUNAS` | `0` desliga a retranscrição de lacunas |
| `AVEIA_BLOCO_ADAPTATIVO` | `1` liga o corte de blocos que enche a janela |
| `AVEIA_IA_TETO_MES` | teto de chamadas de IA por mês (padrão 300) |
| `AVEIA_RETENCAO_DIAS` | prazo para marcar arquivos antigos (padrão 30) |
| `AVEIA_DADOS` | pasta das métricas (padrão `dados/`) |
| `AVEIA_PREPROC` | pré-processamento de áudio (desliga o ao vivo) |
| `AVEIA_MODELO_GEMINI` | modelo do Gemini para todos os recursos de IA (padrão `gemini-3.1-flash-lite`; alternativa `gemini-3.5-flash-lite`) |
| `AVEIA_IA` | `0` desliga toda a IA no servidor (padrão: ligada) |
| `AVEIA_GEMINI_PLANO` | `pago` (padrão) ou `gratuito`: só muda o aviso de privacidade |
| `AVEIA_IA_RACIOCINIO` | `minimal` (padrão), `low`, `medium` ou `high` (`reasoning_effort` do Gemini) |
| `AVEIA_MODELO_IA`, `AVEIA_MODELO_TITULO`, `AVEIA_MODELO_LOCUTORES`, `AVEIA_IA_URL` | modelo por recurso (valem nos dois provedores) e endereço da IA |
| `AVEIA_ENV_EXTRA` | caminho de um `.env` fora da pasta do app, de onde só `GEMINI_API_KEY` e `AI_GATEWAY_API_KEY` são lidas |
| `AVEIA_MAX_UPLOAD_MB` | teto do arquivo importado (padrão 1024) |
| `AVEIA_SEM_AQUECER`, `AVEIA_SEM_NAVEGADOR` | testes e servidores sem carregar o modelo ou abrir o navegador |

## Licença

MIT (arquivo `LICENSE`). Use, copie e modifique à vontade; sem garantia.
