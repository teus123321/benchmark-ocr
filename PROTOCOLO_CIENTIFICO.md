# Protocolo e análise da meta de 80%

## Escopo e evidência disponível

A auditoria de 24/09/2026 usou o commit `53536ca75aecb0624aa604d2d81e85c70ff44c2d`. Foram conferidos 150 JSONs, o manifesto, as exclusões e os resultados das três condições antigas. O CER foi recalculado e as 18 linhas de comparação conferidas. A inspeção visual foi dirigida a casos suspeitos; não constitui revisão independente de todos os gabaritos.

O conjunto antigo tem 281 imagens elegíveis: 149 chassis e 132 motores. Existem 17 ausências anotadas de motor e dois registros com referência válida sem crop. A interseção de sucessos técnicos das nove combinações antigas tem 275 imagens: 148 chassis e 127 motores. Seis imagens foram retiradas dessa comparação por falhas do Qwen.

| Condição antiga | Qwen: acerto chassi | Qwen: acerto motor |
|---|---:|---:|
| Sem tratamento | 29/148 = 19,59% | 22/127 = 17,32% |
| Nativo | 40/148 = 27,03% | 34/127 = 26,77% |
| CLAHE | 41/148 = 27,70% | 34/127 = 26,77% |

Essas são estatísticas dos gabaritos antigos e não são resultados da v3. O conjunto CLAHE tem 75/281 = 26,69% quando as falhas permanecem no denominador. Nativo versus CLAHE não demonstra superioridade estatística: no Qwen, CLAHE ganhou quatro e perdeu três acertos de chassi; em motor, ganhou três e perdeu três.

## Correções de anotação

`revisao_anotacoes.json` registra quatro correções visuais com valor anterior, valor novo e imagem de evidência. As fotos originais foram consultadas e os recortes reproduzidos para conferir os caracteres. Não se aplicou regra automática de completar identificadores até 17 caracteres.

O arquivo também registra a revisão de uma região que incluía dois campos de uma placa. A foto original identifica **Série Nº** à esquerda e **Modelo** à direita. Somente o campo de série é mantido. A mesma fração horizontal é aplicada à BBox e ao crop antigo antes dos tratamentos. Assim, `sem_tratamento` significa sem melhoria fotométrica; nesse caso seu escopo foi corrigido e não é pixel a pixel igual ao recorte legado.

O preparo aplica as revisões em memória e salva novas cópias, sem alterar `assets`. Recusa correção se o gabarito/BBox anterior não corresponder ao esperado. Salva a revisão utilizada dentro de `dataset_v3`. Os CSVs e gráficos anteriores continuam intactos.

Ainda há uma pendência conhecida: um recorte inclui `REM` depois do número, enquanto o gabarito contém somente o número. Não removemos esse texto apenas de uma predição para beneficiar um modelo. É necessário definir a região alvo com o responsável pelas anotações. A lista de pendências não é exaustiva. Recomenda-se revisão de todas as imagens por dois anotadores, sem usar as predições como fonte da verdade, e adjudicação das divergências.

## Ausências e elegibilidade

1. `Dados do Veículo` deve conter duas posições explícitas: chassi e motor. Não se usa um `strip()` global que desloque uma segunda linha para a primeira.
2. A mesma função reconhece as convenções de ausência no preparo e na limpeza. `NÃO`, zeros isolados ou pontuados e demais sentinelas explícitas não geram inferência ou métrica. Zeros de um identificador real, como `001234`, são mantidos.
3. `ERRO`, `ILEGÍVEL` e marcadores ambíguos exigem revisão. Imagem difícil não equivale a objeto sem identificador.
4. Referência válida sem crop gera `crop_ausente`. É exclusão de cobertura do conjunto, não erro da IA. A regra é igual para todos os tratamentos.
5. Recorte preparado faltante interrompe a execução. Não reduz o denominador silenciosamente.

É responsabilidade da curadoria confirmar que um marcador representa ausência real, não falha de anotação. Nenhuma lista de palavras resolve essa ambiguidade universalmente.

## Tratamentos e partições

| Condição | Operação |
|---|---|
| `sem_tratamento` | Crop legado convertido para RGB/PNG, com a correção de escopo documentada |
| `nativo` | Recorte da foto original pela BBox, sem o limite legado de 280 pixels |
| `clahe` | Nativo com CLAHE em LAB/L, clipLimit=2, grade 8x8 |
| `ampliado` | Nativo ampliado convencionalmente em até 2x, maior dimensão limitada a 2048, margem branca de 16 pixels |

A ampliação não recupera detalhes que a foto não capturou; pode ajudar o processamento interno de algum modelo e precisa ser medida. Não há super-resolução generativa nem reconstrução de caracteres por IA.

Todos os modelos recebem exatamente os mesmos arquivos de cada condição. Os pré-processamentos internos de cada biblioteca permanecem diferentes. A transformação QUAD reutiliza a geometria anotada, sem detectar nem garantir que o anotador selecionou a região correta. A v3 valida pontos finitos, convexidade e limites.

A divisão considera componentes conectados por pasta, mesmo identificador do mesmo tipo ou pixels nativos idênticos. Chassi e motor da mesma pasta permanecem juntos. O hash do representante do grupo determina cerca de 70% desenvolvimento e 30% avaliação; a proporção efetiva depende do número e tamanho dos grupos. O manifesto salva grupo e partição, permitindo auditoria.

Essa divisão não detecta todos os duplicados visuais aproximados ou veículos com gabaritos diferentes. Como o conjunto completo e seus resultados antigos já foram vistos, a avaliação reservada é **exploratória**, não um teste cego retroativo. Para uma afirmação confirmatória, obtenha novas imagens de veículos independentes e congele o protocolo antes de examiná-las.

## Inferência, memória e retomada

- A execução padrão inclui três modelos, cada um em seu processo. O processo termina antes de carregar a próxima OCR. O Qwen deste experimento é descarregado entre etapas.
- OCRs usam CPU, uma thread; PaddleOCR usa MKL-DNN desativado. Qwen usa o dispositivo escolhido pelo Ollama. Essas configurações são diferentes da execução legada e não devem ser misturadas no mesmo resultado.
- O prompt é fixo e pede transcrição, sem completar ou repetir caracteres. Nenhum adaptador recebe gabarito, tamanho esperado do identificador ou resposta de outro modelo.
- Qwen: temperatura 0, seed 0, contexto 4096 e no máximo 64 tokens de saída. Seed/temperatura não garantem determinismo entre versões e dispositivos.
- O limite de saída é uma hipótese operacional fixada antes da nova rodada. Uma saída truncada permanece pontuada com seu texto efetivo e flag de truncamento. Não é recortada depois usando o tamanho do gabarito.
- Há uma chamada de aquecimento com a mesma imagem sintética fora das métricas. A latência medida inclui chamada e transporte. Não inclui carregamento inicial dos pesos, mas pode incluir variações de cache e execução. O relatório não afirma igualdade de hardware nem latência intrínseca do algoritmo.
- Cada chamada grava evento de início e resultado, com flush/fsync. A leitura bruta e os metadados Ollama são preservados. A pontuação normaliza gabarito e predição com a mesma regra A–Z/0–9, sem substituir O/0, I/1 ou completar caracteres.
- Até duas tentativas são permitidas apenas para falhas técnicas. O primeiro sucesso encerra a amostra, mesmo que vazio ou errado. Uma tentativa interrompida consome orçamento. Falha persistente permanece falha; não há repetição até acertar.
- Falhas não obrigam a repetir modelos que já concluíram aquela imagem. Duas execuções simultâneas na mesma pasta são bloqueadas por trava.
- Arquivo incompleto não é silenciosamente descartado. Após falta de energia, pode ser necessária recuperação manual da última linha/trava, preservando uma cópia. Não existe garantia absoluta contra corrupção física do disco.

A configuração congela versões, código, dados, modelo Qwen/digest, servidor e opções. Os hashes usam caminhos POSIX e leitura textual normalizada para evitar diferença artificial por CRLF do Windows. Pesos efetivamente baixados de EasyOCR/PaddleOCR são verificados por hash. Informe CPU/RAM/GPU com `--hardware`; os dispositivos efetivamente empregados pelo Ollama também devem ser conferidos com `ollama ps` durante a execução e relatados no artigo.

## Métricas

**Métrica principal:** número de identificadores integralmente corretos dividido por todas as imagens elegíveis da partição. Falhas técnicas são não acertos, mas não recebem um CER fictício.

**CER por imagem:** distância de Levenshtein / comprimento do gabarito. Leitura vazia bem-sucedida tem CER=1. Inserções podem levar CER acima de 1; esse valor não deve ser limitado a 100% nem removido por parecer extremo.

**CER macro nos sucessos:** média dos CERs definidos. Tem denominador condicionado a sucesso técnico de cada modelo; deve acompanhar cobertura e não ser usado isoladamente para declarar um vencedor.

**CER micro nos sucessos:** soma das distâncias / soma dos comprimentos dos gabaritos. Dá maior peso às sequências longas.

**CER e acerto no conjunto comum:** métricas secundárias pareadas, nas imagens com sucesso em todas as combinações selecionadas. O CSV registra `N_comum`. Essa análise pode excluir casos difíceis, portanto não substitui o acerto principal sobre todas as elegíveis.

**Cobertura:** N elegível, sucessos, falhas, falhas em porcentagem e saídas truncadas. Resultados por imagem e histórico permitem calcular também custos de retentativas. O tempo resumido é da chamada selecionada bem-sucedida, não custo total de serviço com falhas.

As métricas incluem Chassi, Motor e Total. O total é calculado a partir das imagens, sem média não ponderada das porcentagens de categorias. Não se confunde `1-CER` com porcentagem de identificadores totalmente corretos. Não se muda a métrica principal após observar resultados.

Intervalos de confiança e testes de superioridade ainda não são automáticos. Para o artigo, use reamostragem por grupo de veículo e comparações pareadas, relatando tamanho da amostra e incerteza. Fotos repetidas do mesmo veículo não são observações independentes.

## Caminho para investigar 80%

Uma verificação retrospectiva ajuda a dimensionar a meta: nos resultados antigos, apenas **117/281 = 41,64%** das imagens têm alguma resposta integralmente correta entre os três modelos e os três tratamentos. Esse cálculo usa o gabarito para escolher a melhor resposta e, por isso, **não é um algoritmo permitido de ensemble nem um resultado publicável de um modelo**. É somente um diagnóstico: combinar essas saídas antigas não bastaria para 80%. As quatro correções visuais não justificam prever um salto dessa dimensão.

Experimentos propostos, em ordem:

1. **Curadoria:** conferir gabaritos e escopo de cada crop, resolver a pendência conhecida e preservar as decisões antes de medir novamente. Separar casos sem referência confiável, com regra anterior às predições e relatório de cobertura; nunca removê-los porque um modelo errou.
2. **Qwen2.5 controlado:** executar desenvolvimento nas quatro condições. Comparar acerto, CER, falhas e truncamentos. O recorte nativo é a referência técnica inicial; CLAHE e ampliação são hipóteses, não melhorias presumidas.
3. **Qwen3-VL-8B-Instruct:** executar no mesmo desenvolvimento e nas mesmas condições. Manter prompt e orçamento de saída iguais aos do Qwen2.5. Fixar quantização/digest e registrar memória. Se não couber, uma variante menor é outro candidato, não uma alteração silenciosa do modelo.
4. **Decisão no desenvolvimento:** escolher configuração por acerto principal, com falhas no denominador. Ganho pequeno não demonstra superioridade sem incerteza. Avaliar tipos de erro: caracteres ausentes no crop, baixa resolução, contraste, sequência em duas linhas, caracteres confundidos e alucinações.
5. **Se continuar distante da meta:** coletar imagens melhores/mais diversas e considerar ajuste supervisionado de um reconhecedor especializado, como PP-OCRv5. Treinar somente em veículos separados; usar validação para hiperparâmetros e um teste independente. A documentação de reconhecimento do PaddleOCR oferece um ponto de partida, mas não demonstra 80% neste dataset.
6. **Avaliação final:** congelar modelo, pesos, código, tratamentos, prompt e regras. Executar uma vez o protocolo nas novas imagens reservadas e publicar o resultado obtido, mesmo abaixo de 80%. Se houver ajuste após essa avaliação, trata-se de novo desenvolvimento e exige novo teste independente.

Não se deve selecionar o tratamento por imagem com base no gabarito, fornecer a referência ao LLM, trocar O/0 somente quando melhora o acerto, cortar respostas no comprimento correto conhecido ou descartar falhas do denominador principal. Um seletor adaptativo pode ser estudado, mas precisa decidir sem gabarito e ser avaliado como um sistema separado, com seu custo computacional.

## Validação realizada nesta revisão

Os testes automatizados usam imagens sintéticas e leitores simulados: ausência versus ilegibilidade; correções sem alterar a fonte; geometria inválida; grupos sem vazamento; retomada; orçamento de tentativas; resposta bruta; falhas no denominador; leitura vazia; CER acima de 100%; truncamento; logs incompletos; trava; geração de gráficos e totais. A inferência real em Windows e a acurácia das novas configurações permanecem não medidas.

## Referências técnicas

- API de chat e motivo de término: https://docs.ollama.com/api/chat
- Descarregamento: https://docs.ollama.com/api/generate
- Parâmetros de geração: https://docs.ollama.com/modelfile
- Qwen3-VL-8B-Instruct: https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct
- Distribuição Ollama: https://ollama.com/library/qwen3-vl:8b-instruct
- Reconhecimento especializado: https://paddlepaddle.github.io/PaddleOCR/main/en/version3.x/module_usage/text_recognition.html

As métricas publicadas pelos fabricantes se referem aos conjuntos deles e não são estimativas de acerto neste benchmark.
