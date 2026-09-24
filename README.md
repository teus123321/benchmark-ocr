# Benchmark OCR ajustado para Windows

Leia **INSTALAR_WINDOWS.md** para instalar e executar. Esta revisão usa `assets` na raiz e mantém main.py, limpar_dataset.py, acertos.py e preprocessar_imagens.py. O pacote não inclui fotos nem pesos de modelos.

## Correções desta revisão

- A mesma função de exclusão é usada no preparo, no main e na limpeza. Acentos, espaços e pontuação são normalizados antes de reconhecer sentinelas. `0.0`, `00-00`, `0 0`, `N/A`, `NÃO`, `SEM NÚMERO`, `SEM NUMERAÇÃO`, `SEM CHASSI`, `SEM MOTOR`, `NÃO POSSUI`, `NÃO TEM`, `NÃO SE APLICA` e `INEXISTENTE` são barrados antes de inferir. Zeros dentro de identificadores reais, como 001234, são mantidos. A lista exata está em SENTINELAS_AUSENCIA. É uma convenção do dataset, não reconhecimento universal de linguagem.
- Exige Dados do Veículo textual, primeira linha chassi e segunda motor. Campo ausente, null, linha faltante ou linhas adicionais com conteúdo interrompem o preparo com indicação do JSON. Ausência real deve ser anotada explicitamente nas posições corretas, por exemplo `ABC123\nNÃO`.
- Rótulos com dois-pontos, caracteres inesperados e marcadores ambíguos como ILEGÍVEL interrompem o preparo. Imagem difícil não deve ser excluída automaticamente como se o objeto não tivesse número. A revisão humana continua necessária para gabaritos plausíveis, linhas trocadas e marcações incorretas.
- O preparo grava amostras_elegiveis.csv. Os gráficos finais verificam se todas as imagens esperadas foram registradas nas três condições. Um crop preparado apagado causa erro explícito; o sistema não reduz silenciosamente o conjunto em execução.
- Hash dos quatro scripts e versões adicionais de dependências são registrados junto com hash do dataset, tag/digest do Qwen e prompt. Não se deve retomar resultados da revisão anterior nesta versão.
- requirements.txt separado para o benchmark, sem empacotadores de aplicativo; guia de instalação Python 3.10 x64 e PyTorch CPU.

## Proteções mantidas

Referência inválida não gera chamada, acerto, erro nem tempo de inferência. Referência válida sem crop original é excluída das três condições com motivo crop_ausente. Ambos aparecem em exclusoes.csv. Ausência legítima e imagem faltante são situações diferentes e precisam ser reportadas separadamente.

As três condições continuam: crop antigo sem tratamento, recorte da foto original sem limite de 280 pixels e recorte nativo com CLAHE (LAB, clipLimit 2, grade 8x8). Não modifica os assets nem inventa caracteres. Recorte automático reutiliza a BBox anotada, não detecta a região nem corrige marcações erradas.

Falha técnica gera status e CER vazio; leitura vazia com execução bem-sucedida gera CER 1. A normalização de gabarito e predição retém A–Z/0–9 sem trocar O/0 ou I/1. CER usa distância de Levenshtein sobre comprimento do gabarito; os gráficos mostram média por imagem. Acerto exato exige sequência integral igual.

Retomada por imagem e tipo; última tentativa vale para as estatísticas. Se um modelo falha tecnicamente, os três são refeitos nessa imagem na próxima execução. A comparação usa as mesmas imagens com sucesso nas nove combinações. Essa interseção pode excluir casos difíceis; cobertura.csv deve acompanhar os 12 gráficos. Não confundir ausência de falhas atuais com ausência de falhas no histórico.

## Limites científicos e validação

Não há garantia de ausência de erros ou melhora de acerto. O protocolo depende de conferir todas as convenções reais de ausência e os gabaritos. Cabeçalhos, tipos e posições incorretos conhecidos são barrados, mas uma sequência plausível na posição errada ainda pode passar. A lista de sentinelas precisa ser revisada com o responsável pelas anotações.

As OCRs rodam em CPU; Qwen usa o dispositivo escolhido pelo Ollama. As entradas são comuns, os pré-processamentos internos são diferentes. Tempos incluem primeira chamada e comunicação; comparações de latência exigem controle adicional de hardware, aquecimento e ordem. Não há teste estatístico de significância ou intervalo de confiança automático.

As verificações desta entrega usam dados sintéticos e adaptadores simulados, não inferência real dos modelos. A instalação completa em Windows precisa ser validada pelos comandos do guia. O documento Word anterior descreve a versão anterior; as mudanças acima o complementam e corrigem as limitações das sentinelas e da completude apontadas nele. Nenhuma alteração foi enviada ao GitHub.

Para repetir os testes sem pesos nem servidor: `.venv\Scripts\python.exe testar_validacao.py`. Eles criam apenas dados temporários e não alteram seus assets.
