# Benchmark OCR de chassi e motor

Compara EasyOCR, PaddleOCR e Qwen em imagens recortadas, com as mesmas amostras, tratamentos e regras de pontuação. Esta revisão mantém os quatro scripts do projeto. Não há framework novo, banco de dados ou interface adicional.

**Estado:** protocolo v3 implementado e testado com adaptadores simulados. Não foi realizada nova inferência real nesta revisão. Não há resultado de 80% comprovado. As anotações ainda precisam de revisão independente para publicação.

## O que mudou

- Quatro gabaritos corrigidos por inspeção visual, com valores anteriores e evidências em `revisao_anotacoes.json`. As correções são aplicadas à nova cópia; `assets` não é alterado.
- Um recorte corrigido: a placa que juntava **Série Nº** e **Modelo** passa a conter somente a série, nas quatro condições. As demais BBoxes são reutilizadas, com validação de convexidade e limites.
- Ausências continuam fora da inferência. Referência válida sem crop é registrada separadamente. O marcador `ERRO` agora exige revisão, em vez de ser interpretado como ausência.
- Quatro tratamentos: `sem_tratamento`, `nativo`, `clahe` e `ampliado` (ampliação convencional com margem, sem IA generativa).
- Um modelo por processo. EasyOCR e PaddleOCR não ficam carregados juntos. O Qwen é descarregado entre etapas. Isso reduz a memória simultânea, mas não estabelece um limite rígido de RAM.
- Qwen com `num_predict=64`, `num_ctx=4096`, temperatura 0 e seed 0. Respostas brutas, metadados, truncamento e falhas são preservados. Comunicação pela API HTTP local do Ollama, sem depender da validação do SDK antigo.
- Até duas tentativas por imagem/modelo/tratamento; somente falhas técnicas são repetidas. Acertos e erros de leitura não são refeitos. Reiniciar não reinicia o orçamento.
- Acerto principal sobre **todas as imagens elegíveis da partição**, incluindo falhas como não acertos. CER de falhas fica indefinido; os relatórios identificam o denominador e mostram também CER no conjunto comum.
- Desenvolvimento e avaliação exploratória separados por grupo de veículo/identificador/pixels repetidos. Resultados antigos já foram vistos: essa separação posterior **não transforma o conjunto em teste cego**.
- Configuração, hashes, versões, pesos OCR e histórico de tentativas registrados. Mudanças de configuração bloqueiam a retomada na mesma pasta.

## Executar no Windows

Abra o PowerShell na raiz do projeto. Mantenha `assets` na raiz e use o Python 3.10 já instalado. Não use a antiga `.venv` de Python 3.14.

```powershell
.venv310\Scripts\python.exe -m pip install -r requirements.txt
.venv310\Scripts\python.exe testar_validacao.py
.venv310\Scripts\python.exe preprocessar_imagens.py
```

O preparo cria `dataset_v3` sem sobrescrever `dataset_tratado`. Se `dataset_v3` já existir, o preparo para. Para outra revisão, use `--saida dataset_v3_revisado` e informe esse caminho ao `main.py` com `--dataset`.

Com o Ollama ativo e `qwen2.5vl:latest` instalado:

```powershell
.venv310\Scripts\python.exe benchmark-ocr\main.py --saida benchmark-ocr\resultados_v3_dev
```

Esse comando usa somente **desenvolvimento** e executa os três modelos nos quatro tratamentos. Para registrar a máquina, acrescente `--hardware "CPU: seu modelo; RAM: sua capacidade; GPU: RTX 5060; VRAM: 8 GB"`, preenchendo os dados reais. Repita exatamente o comando para retomar.

Para avaliar o candidato Qwen3-VL, em outra pasta, com as mesmas condições e todos os comparadores:

```powershell
ollama pull qwen3-vl:8b-instruct
.venv310\Scripts\python.exe benchmark-ocr\main.py --qwen qwen3-vl:8b-instruct --saida benchmark-ocr\resultados_v3_qwen3_dev
```

O download e a execução podem exigir memória adicional. É um candidato experimental, sem garantia de melhora. Para um piloto apenas do Qwen, acrescente `--modelos Qwen_VL` e use uma pasta exclusiva; esse piloto não representa a comparação completa com as três IAs.

Depois de decidir a configuração usando desenvolvimento, preserve os parâmetros e execute a avaliação em outra pasta:

```powershell
.venv310\Scripts\python.exe benchmark-ocr\main.py --parte avaliacao --saida benchmark-ocr\resultados_v3_avaliacao
```

O exemplo usa Qwen2.5. Se o candidato escolhido for Qwen3, informe também `--qwen qwen3-vl:8b-instruct`. Não volte a ajustar os parâmetros com base nessa avaliação e depois a chame de teste independente. Para um artigo confirmatório, reserve novas imagens de outros veículos.

## Resultados

Cada pasta de execução contém:

| Arquivo | Conteúdo |
|---|---|
| `configuracao.json` | Dados, código, ambiente, modelo e protocolo congelados |
| `pesos_*.json` | Hashes dos pesos usados pelas OCRs |
| `<tratamento>/tentativas_*.jsonl` | Início e fim de cada tentativa, resposta original e falhas |
| `resultados_por_imagem.csv` | Formato longo: uma linha por imagem/modelo/tratamento |
| `comparacao_tratamentos.csv` | Acerto, CER macro/micro, cobertura, truncamentos e tempo, por categoria e total |
| `acertos_exatos.csv`, `analise_erros.csv` | Acertos e erros, incluindo pasta de origem e contagem de operações |
| `Acerto_pct.png`, `CER_macro_comum_pct.png`, `Tempo_sucessos_s.png` | Comparações visuais dos tratamentos |

Os 12 gráficos antigos permanecem em `benchmark-ocr/resultados_ajustados`. A v3 gera três gráficos comparativos, cada um contendo os modelos e tratamentos selecionados. `N=0` significa ausência de dados, nunca acerto perfeito. Para regenerar relatórios sem IAs:

```powershell
.venv310\Scripts\python.exe benchmark-ocr\main.py --saida benchmark-ocr\resultados_v3_dev --relatorios
```

Use `--parte avaliacao` se a execução salva for dessa partição. Não misture CSVs antigos, novas referências e novas predições.

Leia [PROTOCOLO_CIENTIFICO.md](PROTOCOLO_CIENTIFICO.md) para métricas, limitações e estratégia para buscar 80%, e [INSTALAR_WINDOWS.md](INSTALAR_WINDOWS.md) para instalação e recuperação após interrupções.
