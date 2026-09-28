# Benchmark de reconhecimento de chassi e motor

Pesquisa experimental sobre reconhecimento de identificadores de chassi
e motor em imagens, comparando ferramentas especializadas em OCR e
modelos multimodais.

O projeto executa cada modelo separadamente, registra suas transcrições
e compara os resultados com referências textuais do dataset.

## Objetivo

Avaliar os modelos em relação a:

- Acerto integral do identificador.
- Taxa de erro por caractere — Character Error Rate (CER).
- Cobertura das imagens processadas.
- Falhas técnicas durante a execução.
- Tempo observado por imagem.

O estudo avalia o desempenho nas condições documentadas do experimento.
Os resultados não representam uma garantia de desempenho em outros
conjuntos de imagens.

## Modelos avaliados

| Modelo | Execução | Interface |
|---|---|---|
| EasyOCR | Local, em CPU | Biblioteca Python |
| PaddleOCR 2.x | Local, em CPU | Biblioteca Python |
| Qwen2.5-VL | Local, pelo Ollama | API HTTP local |
| Nemotron 3 Nano Omni | Remota, pela NVIDIA | API HTTP |

Identificadores configurados nos scripts:

- Qwen: `qwen2.5vl:latest`
- Nemotron: `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning`

A tag `latest` pode mudar. Para reproduzir uma execução, registre também
o identificador do modelo instalado no Ollama e sua versão.

## Dataset

O conjunto desta execução contém **281 imagens**:

| Categoria | Quantidade |
|---|---:|
| Chassi | 149 |
| Motor | 132 |
| Total | 281 |

Os arquivos JSON armazenam as referências utilizadas na avaliação.
No campo `Dados do Veículo`, a primeira linha corresponde ao chassi e
a segunda ao motor.

**Os gabaritos não são enviados aos modelos.** Eles são consultados
somente pelo comparador, após a leitura das imagens.

### Ausências e pendências

A auditoria identificou 17 registros com ausência declarada e dois
registros com referência textual, mas sem BBox e sem recorte.

- **Ausência declarada:** o JSON contém um marcador previsto pelo
  comparador, como `NÃO`, `SEM MOTOR` ou somente zeros. O registro não
  participa do teste de transcrição.
- **Referência sem BBox e sem recorte:** existe um identificador no JSON,
  mas falta a imagem recortada necessária à execução. O caso é registrado
  como pendência, sem contar como erro da IA.
- **BBox preenchida e recorte esperado ausente:** o comparador interrompe
  a análise para revisão.

A BBox representa as coordenadas da região a ser recortada.

A existência de uma referência textual não garante sua correção.
As anotações precisam de revisão humana.

## Organização dos arquivos

| Caminho | Finalidade |
|---|---|
| `assets/` | Imagens de origem e JSONs com referências |
| `recortes_margem10_v2/com_margem/` | Entrada compartilhada pelos scripts |
| `easyOCR/ler_easyOCR.py` | Execução do EasyOCR |
| `paddleOCR/ler_paddleOCR.py` | Execução do PaddleOCR |
| `qwen/ler_qwen.py` | Execução do Qwen pelo Ollama |
| `nemotrom/ler_nemotron.py` | Execução do Nemotron pela NVIDIA |
| `comparar_resultados.py` | Avaliação e geração de gráficos |
| `requirements.txt` | Dependências propostas para instalação |
| `requirements-experimento.txt` | Versões efetivamente instaladas no experimento |

A pasta `nemotrom` mantém a grafia utilizada no projeto.

Cada script de inferência cria um arquivo `resultados_*.csv` na própria
pasta do modelo. O comparador cria uma nova pasta `comparacao_*`.

## Requisitos

- Windows x64.
- Python 3.10 de 64 bits.
- Ambiente virtual Python.
- Ollama instalado para executar o Qwen.
- Acesso à internet e chave da NVIDIA para executar o Nemotron.

EasyOCR e PaddleOCR podem precisar de internet na primeira execução
para baixar os pesos dos modelos.

Os scripts de OCR apresentados utilizam CPU. O dispositivo utilizado
pelo Qwen depende da instalação e dos recursos disponíveis no Ollama.

## Instalação

Execute os comandos no PowerShell, na raiz do projeto.

### Ambiente virtual

Para uma instalação nova:

```powershell
py -3.10 -m venv .venv310
.\.venv310\Scripts\python.exe --version
```

Se o ambiente já existe e foi utilizado nos testes, registre suas
dependências antes de modificá-lo:

```powershell
.\.venv310\Scripts\python.exe -m pip freeze > requirements-experimento.txt
```

O arquivo `requirements.txt` fixa dependências selecionadas, mas não é
um lock completo de todas as dependências transitivas.

### PyTorch CPU

Instale primeiro o par de versões compatíveis:

```powershell
.\.venv310\Scripts\python.exe -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
```

### Demais dependências

```powershell
.\.venv310\Scripts\python.exe -m pip install -r requirements.txt
.\.venv310\Scripts\python.exe -m pip check
```

Uma verificação simples dos imports:

```powershell
.\.venv310\Scripts\python.exe -c "import torch, torchvision, easyocr, paddle, paddleocr, cv2, numpy, requests, matplotlib; print('Imports concluídos')"
```

Essas verificações ajudam a detectar problemas de instalação, mas não
substituem uma execução real com imagens.

### Limitação das dependências OpenCV

As versões utilizadas de EasyOCR e PaddleOCR declaram dependências de
diferentes distribuições OpenCV, que compartilham o módulo `cv2`.

O arquivo mantém essas distribuições na mesma versão do conjunto legado.
Isso não elimina a sobreposição de arquivos entre os pacotes.
`pip check` também não detecta necessariamente esse tipo de conflito.

Não atualize o ambiente durante uma rodada experimental. Mudanças de
dependências devem ser registradas e verificadas em uma nova instalação,
com teste de leitura antes de uma nova rodada.

## Execução dos modelos

Execute **um modelo por vez**, a partir da raiz do projeto.

### EasyOCR

```powershell
.\.venv310\Scripts\python.exe .\easyOCR\ler_easyOCR.py
```

O script inicializa o reconhecimento em inglês, detecta e reconhece
texto na imagem, reúne os trechos retornados e salva os resultados.

### PaddleOCR

```powershell
.\.venv310\Scripts\python.exe .\paddleOCR\ler_paddleOCR.py
```

O script utiliza PaddleOCR 2.x, idioma inglês e classificação de
orientação. Os textos reconhecidos são reunidos e registrados no CSV.

### Qwen

Confira se o modelo está disponível:

```powershell
ollama list
```

Se ainda não estiver instalado:

```powershell
ollama pull qwen2.5vl:latest
```

O Ollama precisa estar em execução. Caso o aplicativo já esteja ativo,
não é necessário iniciar outro servidor. Caso contrário, execute em
outro terminal:

```powershell
ollama serve
```

Depois, execute:

```powershell
.\.venv310\Scripts\python.exe .\qwen\ler_qwen.py
```

O script envia uma imagem por chamada, com o prompt de extração, e
verifica se o Ollama informou a conclusão da resposta.

### Nemotron

Configure a chave no terminal em que executará o script:

```powershell
$env:NVIDIA_API_KEY = "SUA_CHAVE_DA_NVIDIA"
.\.venv310\Scripts\python.exe .\nemotrom\ler_nemotron.py
```

Não salve a chave no código nem a publique no repositório.

O script envia as imagens à API da NVIDIA. Em falhas temporárias
previstas no código, faz até cinco tentativas por imagem, esperando
dez segundos entre tentativas.

Respostas recebidas não são repetidas por parecerem incorretas.
Se as tentativas falharem, o CSV registra `API_ERROR`.

### Configuração de geração

| Configuração | Qwen | Nemotron |
|---|---:|---:|
| Temperatura | 0 | 0 |
| Semente | 0 | 0 |
| Limite de geração | 128 tokens | 4.096 tokens |
| Contexto solicitado | 4.096 tokens | Não definido no script |
| Orçamento de raciocínio | Não definido no script | 1.024 tokens |

Qwen e Nemotron recebem o mesmo prompt, solicitando somente o
identificador e orientando a ignorar símbolos decorativos e informações
alheias, preservar zeros e não completar caracteres ilegíveis.

EasyOCR e PaddleOCR não recebem prompts.

As configurações e políticas de recuperação não são idênticas entre
as ferramentas. As entradas e as regras de comparação são comuns.

## Registro das leituras

| Campo | Conteúdo |
|---|---|
| `File_Name` | Caminho relativo da imagem |
| `Extracted_Characters` | Transcrição normalizada |
| `Raw_Response` | Texto original retornado |
| `Status` | Situação do processamento |
| `Error` | Descrição de eventual falha |
| `Seconds` | Tempo registrado por imagem |
| `Truncated` | Indicação de limite de geração atingido |

O script do Nemotron também registra `Attempts` e `Finish_Reason`.

### Significado dos status

- `OK`: houve uma resposta processada; não significa acerto.
- `NOT_FOUND`: a leitura não produziu caracteres do identificador.
- `TRUNCATED`: a geração atingiu o limite de tokens.
- `API_ERROR`: falha na chamada ou na validação da resposta da API.
- `MODEL_ERROR`: falha do processamento local.
- `INPUT_ERROR`: falha na leitura do arquivo, quando identificada
  separadamente pelo script.

Cada execução cria um CSV novo e começa do início.
Não há retomada automática.

## Comparação dos resultados

Depois das execuções:

```powershell
.\.venv310\Scripts\python.exe .\comparar_resultados.py
```

O comparador exige os CSVs das quatro ferramentas e verifica:

- Correspondência entre imagens e referências.
- Ausência de registros duplicados.
- Presença das mesmas imagens em todos os arquivos.
- Status reconhecidos e tempos válidos.
- Consistência da normalização das respostas recebidas.

Se houver mais de um `resultados_*.csv` numa pasta, preencha o
dicionário `ARQUIVOS` do comparador com o nome exato da execução
escolhida. O script não escolhe automaticamente o melhor resultado.

## Métricas

### Acerto exato

A transcrição precisa coincidir integralmente com o gabarito após
normalização.

```text
Acerto exato (%) = 100 × acertos / imagens elegíveis
```

Falhas técnicas permanecem no denominador principal como não acertos.

### Cobertura

```text
Cobertura (%) = 100 × imagens com resposta / imagens elegíveis
```

Uma resposta pode estar incorreta. Cobertura não mede qualidade
da transcrição.

### CER

```text
CER = distância de Levenshtein / número de caracteres do gabarito
```

A distância considera inserções, remoções e substituições.

- **CER macro:** média do CER por imagem com resposta.
- **CER micro:** soma das distâncias dividida pela soma dos comprimentos
  dos gabaritos avaliados.
- **CER mediano:** valor central da distribuição.

Quanto menor o CER, melhor. O CER pode superar 100% quando há muitas
inserções.

Falhas técnicas têm CER indefinido. Uma leitura vazia com resposta
recebida tem CER de 100% diante de um gabarito válido.

Respostas longas e truncadas permanecem na avaliação. Os alertas não
são usados para excluir os piores resultados.

### Conjunto comum

O comparador calcula métricas também sobre as imagens com resposta
em todos os modelos.

Essa interseção permite avaliar as mesmas imagens, mas pode remover
casos difíceis. Seus resultados devem ser apresentados junto da
cobertura e das métricas do conjunto completo.

Se a interseção estiver vazia, as métricas correspondentes ficam
indefinidas.

### Intervalo de confiança

O intervalo de confiança de 95% do acerto é estimado por bootstrap,
com 2.000 reamostragens e semente 2026.

A unidade de reamostragem é a pasta do veículo, mantendo chassi e motor
agrupados. Se o mesmo veículo estiver em diferentes pastas, o agrupamento
precisa ser corrigido.

Esse intervalo não comprova, isoladamente, diferença estatisticamente
significativa entre modelos.

## Normalização

O comparador aplica ao gabarito a mesma regra dos scripts de leitura:

1. Normalização Unicode NFKC.
2. Conversão para maiúsculas.
3. Retenção somente de `A–Z` e `0–9`.

Espaços e separadores não causam erro por si só. Não há substituição
automática entre `O` e `0`, nem entre `I` e `1`.

A resposta original é preservada. Letras de outros alfabetos são
sinalizadas, pois o filtro pode descartá-las.

As correções propostas para o gabarito precisam de validação humana
e não são aplicadas automaticamente pelo comparador.

## Arquivos da análise

| Arquivo | Finalidade |
|---|---|
| `resumo.csv` | Métricas por modelo, categoria e total |
| `por_imagem.csv` | Gabarito, resposta e métricas individuais |
| `exclusoes_dataset.csv` | Exclusões e pendências, quando existentes |
| `acerto.png` | Acerto exato no conjunto completo |
| `cobertura.png` | Porcentagem de respostas recebidas |
| `cer.png` | CER médio nas respostas de cada modelo |
| `cer_comum.png` | CER médio no conjunto comum |
| `tempo.png` | Tempo médio observado nas respostas |
| `proveniencia.json` | Hashes dos arquivos utilizados |
| `LEIA_METODOLOGIA.txt` | Critérios e limites da avaliação |

## Reprodutibilidade e limites

Para cada execução, preserve:

- Scripts e commit do repositório.
- CSVs originais e parâmetros de geração.
- Gabaritos e imagens utilizados.
- `requirements-experimento.txt`.
- Versão do Python, sistema operacional, CPU, RAM e GPU.
- Versão do Ollama e identificação do modelo local.
- Identificador do modelo remoto e data da execução.

Comandos úteis:

```powershell
.\.venv310\Scripts\python.exe --version
.\.venv310\Scripts\python.exe -m pip freeze > requirements-experimento.txt
ollama --version
ollama list
git rev-parse HEAD
```

Os hashes identificam os arquivos presentes na análise. Eles não
comprovam que as referências estejam corretas nem recuperam versões
anteriores de modelos remotos.

Os tempos incluem comunicação e reenvios quando registrados.
Comparações entre CPU local, Ollama e API remota devem considerar
as diferenças de infraestrutura.

Como o dataset já foi examinado, ajustes escolhidos a partir de seus
resultados precisam ser avaliados em um conjunto independente.

## Referências das ferramentas

- [EasyOCR](https://github.com/JaidedAI/EasyOCR)
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)
- [PyTorch e versões anteriores](https://docs.pytorch.org/get-started/previous-versions/)
- [Ollama](https://docs.ollama.com/)
- [Nemotron na NVIDIA](https://build.nvidia.com/nvidia/nemotron-3-nano-omni-30b-a3b-reasoning)
- [Orientações de instalação do OpenCV](https://pypi.org/project/opencv-python/)
