# Executar no Windows com Python 3.10

O projeto foi desenvolvido para Python 3.10 x64. A combinação PyTorch 2.5.1/torchvision 0.20.1 não possui wheel para Python 3.14. Você pode manter os dois Pythons instalados; use explicitamente `.venv310` neste projeto.

## Instalação inicial

No PowerShell, dentro da pasta do projeto:

```powershell
py -3.10 -m venv .venv310
.venv310\Scripts\python.exe -m pip install --upgrade pip
.venv310\Scripts\python.exe -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
.venv310\Scripts\python.exe -m pip install -r requirements.txt
.venv310\Scripts\python.exe -m pip check
.venv310\Scripts\python.exe testar_validacao.py
```

Se `.venv310` já está instalado e funcionando, não é necessário recriá-lo. Os testes usam modelos simulados, não baixam pesos e não demonstram acurácia real. A validação real em Windows deve ser feita na sua máquina; não foi executada no ambiente de desenvolvimento desta revisão.

## Ollama e memória

Confira `ollama list`. O modelo padrão é `qwen2.5vl:latest`. Caso o aplicativo Ollama já esteja ativo, não abra outro servidor na mesma porta.

Para iniciar manualmente com limites de concorrência, primeiro encerre o Ollama da bandeja ou o servidor anterior, com o benchmark parado. No mesmo terminal:

```powershell
$env:OLLAMA_MAX_LOADED_MODELS="1"
$env:OLLAMA_NUM_PARALLEL="1"
ollama serve
```

Esses limites não impõem um teto de RAM. O código v3 executa os modelos em processos separados para liberar a memória das bibliotecas ao trocar de modelo. Qwen continua sujeito à capacidade de RAM/VRAM, contexto e resolução. `gc.collect()` não seria suficiente para impor esse limite.

Para um novo candidato:

```powershell
ollama --version
ollama pull qwen3-vl:8b-instruct
```

A página oficial do modelo informa Ollama 0.12.7 como versão mínima. Registre a versão efetivamente utilizada, que a v3 consulta automaticamente. Não atualize servidor, pesos ou bibliotecas no meio de uma execução já iniciada. Use outra pasta se precisar mudar o ambiente.

## Preparar e executar

```powershell
.venv310\Scripts\python.exe preprocessar_imagens.py
.venv310\Scripts\python.exe benchmark-ocr\main.py --saida benchmark-ocr\resultados_v3_dev
```

O preparo exige fotos originais, BBoxes e crops em `assets`, como no repositório. Gera `dataset_v3`, mantendo o dataset e os resultados anteriores intactos. Os parâmetros adicionais estão no README e no `--help`.

EasyOCR/PaddleOCR podem baixar pesos na primeira execução para `cache_modelos`. O download não faz parte da medição de inferência. Há uma chamada de aquecimento com imagem sintética antes das medições; seus caracteres não entram no benchmark.

## Parar e retomar

- Pressione `Ctrl+C` e aguarde o término do processo Python antes de iniciar outro.
- Repita **o mesmo comando**, com a mesma pasta e parâmetros. Sucessos técnicos já salvos não são refeitos.
- A tentativa iniciada, mas interrompida, consome uma das duas oportunidades. Isso impede repetição ilimitada até obter uma resposta favorável.
- Se o desligamento deixou `execucao.lock`, confirme no Gerenciador de Tarefas que os processos Python desse benchmark terminaram. Só então remova esse arquivo da pasta de resultados e retome. Não apague logs, configuração ou resultados.
- JSONL com linha incompleta gera erro explícito. Preserve uma cópia e revise o final do arquivo. O programa não presume que uma resposta perdida foi sucesso nem descarta a linha silenciosamente.
- CSV parcial não gera gráficos finais. O relatório informa quantas combinações ainda podem ser retomadas. Após esgotar duas tentativas, falhas persistentes entram na cobertura e no denominador do acerto.
- `OK` significa chamada bem-sucedida. Não significa que o identificador está correto.

Se precisar liberar o Qwen após interromper:

```powershell
ollama stop qwen2.5vl:latest
ollama ps
```

Para o candidato Qwen3, use a tag correspondente no `ollama stop`. Não é necessário reiniciar o computador apenas para descarregar um modelo.

## Fontes técnicas

- https://pytorch.org/get-started/previous-versions/
- https://docs.ollama.com/api/chat
- https://docs.ollama.com/api/generate
- https://docs.ollama.com/faq
- https://ollama.com/library/qwen3-vl:8b-instruct
