# Pendências

Atualizado em 2026-11-21.

## Testes que dependem de uma pessoa (não dá para automatizar)

| Teste | Por que não dá para automatizar |
|---|---|
| Gravar com o microfone real e conferir o texto, os locutores e o marcador de trecho incompleto | Sem áudio de reunião real com texto corrigido, o WER medido vale só para leitura e frases soltas |
| Chamada real de Teams ou Meet com o áudio do computador (o aviso agora aparece ao ligar a opção em Ajustes) | Exige permissão de tela e outra pessoa |
| Reunião longa, de preferência com o ao vivo ligado: o primeiro texto, o "parar até pronto" e se a contagem de pessoas bate | É o único teste de fila atrasada e de reunião de verdade |
| Resumo, título e nomes por IA com a chave no `.env`, com a IA automática ligada e desligada | A chave não existe nesta máquina |
| Fechar a gravação com a rede instável (repetição do `/finalizar`) | Só simulado em Node |

## Gravação automática por controle (2026-11-27)

- Testado com microfone e loopback reais (taxas e amostras), subida sem tela e parada manual; **não testado**: reunião real longa, troca de fone Bluetooth no meio da gravação (a sonda de mudança do dispositivo padrão só tem teste simulado) e IA ligada.
- O áudio dos dois canais fica em memória como float32 (cerca de 460 MB por hora); reuniões acima de 2 h pedem atenção.
- O ao vivo disputa a CPU com o Whisper: em reunião longa o texto final pode demorar alguns minutos depois do parar.
- A tela aberta durante uma gravação por controle não participa dela (a sessão é do servidor).

## Decisões abertas

- **Corte adaptativo dos blocos** (`AVEIA_BLOCO_ADAPTATIVO=1`): economiza 8% a 38% de CPU, mas um clipe piorou 3,5 pontos de WER. Ligar só depois do teste de reunião real.
- **Modelo menor ou aceleração por hardware** (small ou medium no ao vivo, OpenVINO): o piso atual é cerca de 17 s por janela de 30 s. Não foi testado porque exige baixar modelos (centenas de MB). Dizer se pode.
- **Prazo de retenção e texto do aviso de gravação**: padrão de 30 dias só marca; o texto legal deve passar por um advogado.
- **Teto da IA**: padrão 300 chamadas por mês. Ajustar com `AVEIA_IA_TETO_MES`.

## Limites conhecidos

- **CDN sem integridade:** Tailwind (`cdn.tailwindcss.com`) e ícones (`unpkg.com`) vêm de fora; servir localmente (e então fechar com uma política de segurança de conteúdo) ainda não foi feito.
- **Downloads de modelos sem hash:** os modelos vêm do GitHub (sherpa-onnx) e do Hugging Face sem conferência de SHA256 nem revisão fixada.
- **Dependências sem arquivo de travamento (lock) com hashes:** só há versões mínimas em `requirements.txt`.
- **Apagar nota remove também o áudio de notas com o mesmo nome e sufixo numérico** (`x`, `x-2`): perda local de dado em colisão de nomes.
- **Sem teste de ponta a ponta com áudio:** os áudios de referência não vão no repositório; gere o seu com `testes/microfone/gerar_tts.ps1`.

- A interface carrega estilos e fontes de CDN: sem internet a tela perde o estilo.
- Notas ficam só no navegador: trocar de computador ou navegador exige Exportar tudo.
- Sem teste automático da interface (a conferência foi manual).
- Medições de tempo variam 17 a 60% entre execuções nesta máquina (CPU compartilhada).
