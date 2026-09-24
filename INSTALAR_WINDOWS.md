# Instalar e executar no Windows

Esta revisão mantém os scripts simples e usa a pasta `assets` na raiz, como nas capturas enviadas. Os comandos abaixo funcionam em PowerShell e CMD, salvo quando indicado. Execute na pasta que contém este documento. Não execute dentro de `benchmark-ocr`.

## 1 Ambiente Python

Instale Python 3.10 de 64 bits caso ainda não exista no computador. Esta configuração preserva a API do PaddleOCR 2.8.1; não use o Python 3.14 para este ambiente. Confira as versões:

```text
py -0p
py -3.10 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m pip check
.venv\Scripts\python.exe -c "import paddle; paddle.utils.run_check(); import easyocr, paddleocr, ollama, cv2, Levenshtein; print('Importacoes OK')"
```

Não é necessário ativar o ambiente nem alterar a política de execução do PowerShell. No VS Code, selecione `.venv\Scripts\python.exe` em Python Select Interpreter. As OCRs usam CPU; o Ollama gerencia CPU/GPU do Qwen. Os pesos das OCRs podem ser baixados na primeira execução. Se alguma instalação falhar, pare e analise o primeiro erro antes de executar o benchmark.

O arquivo fixa dependências diretas, não todas as transitivas. Depois de instalar com sucesso, registre o ambiente:

```text
.venv\Scripts\python.exe -m pip freeze > ambiente_instalado.txt
```

A instalação completa não foi executada em Windows nesta entrega. Foram conferidos os requisitos publicados e testado o fluxo de dados com inferências simuladas. `pip check`, importações e uma execução local são os passos de validação da sua máquina.

## 2 Verificar o Ollama

O pacote Python `ollama` é o cliente; ele não instala o servidor nem os pesos. No terminal:

```text
ollama --version
ollama list
```

Se listar modelos, o servidor já está disponível. Não abra um segundo servidor. Se houver erro de conexão, abra o aplicativo Ollama pelo menu Iniciar ou execute, em um terminal separado:

```text
ollama serve
```

Deixe esse terminal aberto e use outro para o benchmark. Mensagem de porta 11434 ocupada normalmente indica servidor já iniciado; confirme com `ollama list`. Se o comando não for reconhecido, instale o aplicativo em https://ollama.com/download/windows e abra um terminal novo.

## 3 Usar os modelos do HD

A captura mostra `manifests\registry.ollama.ai\library\qwen2.5vl`. A pasta correta para `OLLAMA_MODELS` é a que contém **blobs e manifests**, não a pasta `qwen2.5vl` nem a pasta `library`.

Se `ollama list` já mostra o Qwen desejado, não altere caminhos nem baixe novamente. Caso o servidor não enxergue os modelos do HD, saia do Ollama pelo ícone da bandeja. Descubra o caminho real da pasta que contém blobs e manifests e use uma das alternativas. `D:\CAMINHO_REAL\models` abaixo é um exemplo que precisa ser substituído.

PowerShell, configuração só deste terminal e do servidor iniciado nele:

```powershell
$env:OLLAMA_MODELS = 'D:\CAMINHO_REAL\models'
ollama serve
```

CMD, equivalente:

```bat
set "OLLAMA_MODELS=D:\CAMINHO_REAL\models"
ollama serve
```

Para tornar persistente, crie a variável de usuário OLLAMA_MODELS nas Variáveis de Ambiente do Windows e reinicie o Ollama. A presença de manifests sem os blobs correspondentes não basta para carregar um modelo.

## 4 Escolher a tag exata do Qwen

No segundo terminal, rode `ollama list`. Copie exatamente o nome e tag da coluna NAME para `TAG_QWEN` em `benchmark-ocr/main.py`. O padrão continua:

```python
TAG_QWEN = "qwen2.5vl:latest"
```

Se a lista mostrar `qwen2.5vl:7b`, por exemplo, use essa tag no código. Não escolha qwen2.5 apenas: o benchmark precisa da variante visual. Se não houver o modelo, com o servidor já usando o diretório correto:

```text
ollama pull qwen2.5vl:7b
```

Nesse caso, configure TAG_QWEN como `qwen2.5vl:7b`. O main.py carrega o modelo pela API; não é necessário manter um chat `ollama run` aberto. Memória insuficiente pode causar falha técnica; o tamanho adequado depende da RAM e da GPU da sua máquina.

## 5 Preparar o dataset

A organização esperada é `assets/<pasta da vistoria>/imgs`, com os originais e `cropped_*`. O JSON pode estar na pasta da vistoria ou em imgs. A captura mostra essa organização. É necessário manter as fotos originais e as BBoxes, além dos crops, para testar resolução nativa e CLAHE.

```text
.venv\Scripts\python.exe preprocessar_imagens.py
```

Confira `dataset_tratado/exclusoes.csv`, `amostras_elegiveis.csv` e `preparo_concluido.txt`. A pasta de saída não pode existir antes do preparo: para preservar resultados anteriores, renomeie-a e execute novamente. Um erro de JSON ou BBox deixa uma pasta parcial, sem o marcador final; corrija a anotação e use uma saída nova.

## 6 Rodar e gerar os gráficos

Em `benchmark-ocr/main.py`, configure a tag do Qwen e:

```python
RODAR_INTELIGENCIA_ARTIFICIAL = True
```

Depois:

```text
.venv\Scripts\python.exe benchmark-ocr\main.py
```

O programa faz limpeza e relatórios ao final. Não precisa rodar limpar_dataset.py à parte. Use resultados novos para esta revisão: se a pasta antiga `benchmark-ocr/resultados_ajustados` existir, renomeie-a ou altere PASTA_RESULTADOS. O código deve bloquear retomadas com protocolo diferente.

São nove combinações por imagem elegível, sem contar retentativas. A comparação exige que todas as imagens esperadas tenham uma tentativa registrada em cada condição. Falhas técnicas continuam registradas e a comparação de reconhecimento usa a interseção de sucessos das nove combinações. Leia cobertura.csv junto dos gráficos.

Para refazer somente os relatórios, use False. Preserve o dataset preparado e seu inventário, pois a completude é conferida novamente. Não rode duas instâncias escrevendo na mesma saída.

## Referências de instalação

- PyTorch CPU 2.5.1 e torchvision 0.20.1: https://docs.pytorch.org/get-started/previous-versions/
- PaddlePaddle 2.6.2 com wheel Windows CPython 3.10 x64: https://pypi.org/project/paddlepaddle/2.6.2/
- Dependências PaddleOCR 2.8.1: https://pypi.org/project/paddleocr/2.8.1/
- Ollama no Windows e localização dos modelos: https://docs.ollama.com/windows
