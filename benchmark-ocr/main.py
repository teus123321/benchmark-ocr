"""
BENCHMARK CIENTÍFICO DE OCR E VISÃO COMPUTACIONAL
Arquitetura com Checkpointing: Permite retomar o processamento de onde parou 
em caso de esgotamento de memória (OOM), sem perder dados.
"""

import csv
import gc
import hashlib
import io
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
import Levenshtein
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
from limpar_dataset import (CABECALHO, MODELOS, normalizar, referencia_invalida,
                            limpar_dados_injustos, ler_resultados, ler_referencias)
from preprocessar_imagens import TRATAMENTOS
from acertos import gerar_relatorio_acertos

# Mesma organização do projeto: configurações aqui, execução no final do arquivo.
PASTA_DATASET = RAIZ / "dataset_tratado"
PASTA_RESULTADOS = RAIZ / "benchmark-ocr/resultados_ajustados"
RODAR_INTELIGENCIA_ARTIFICIAL = True  # True executa; False refaz somente relatórios.
PROMPT = "You are an OCR system. Extract ONLY the alphanumeric characters from this image. Do not include spaces, special characters, or conversational text. Output only the raw string."

reader_easyocr = None
reader_paddle = None
TAG_QWEN = "qwen2.5vl:latest"

def inicializar_modelos():
    """Carrega os pesos das IAs para a memória apenas se for estritamente necessário."""
    import tempfile
    pasta_cache = RAIZ / "cache_temporario"
    pasta_cache.mkdir(exist_ok=True)
    os.environ["TMP"] = os.environ["TEMP"] = str(pasta_cache)
    os.environ["EASYOCR_MODULE_PATH"] = str(pasta_cache)
    tempfile.tempdir = str(pasta_cache)
    import easyocr
    from paddleocr import PaddleOCR
    global reader_easyocr, reader_paddle
    print("\nCarregando modelos tradicionais na CPU. Isso pode demorar alguns segundos, aguarde...")
    reader_easyocr = easyocr.Reader(['en'], gpu=False) 
    reader_paddle = PaddleOCR(lang='en', use_angle_cls=True, use_gpu=False, show_log=False)
    print("Modelos carregados com sucesso!")

# =============================================================================
# 2. FUNÇÕES DE INFERÊNCIA
# =============================================================================
def ler_com_easyocr(caminho_imagem):
    resultados = reader_easyocr.readtext(caminho_imagem, detail=0)
    return "".join(resultados).replace(" ", "").upper()

def ler_com_paddle(caminho_imagem):
    resultados = reader_paddle.ocr(caminho_imagem, cls=True)
    if not resultados or not resultados[0]: return ""
    textos = [linha[1][0] for linha in resultados[0]]
    return "".join(textos).replace(" ", "").upper()

def ler_com_ollama(caminho_imagem, modelo):
    import ollama
    resposta = ollama.chat(model=modelo, options={"temperature": 0, "seed": 0},
        messages=[{"role": "user", "content": PROMPT, "images": [str(caminho_imagem)]}])
    return resposta["message"]["content"]  # Exceções são registradas pelo laço principal.

# =============================================================================
# 3. ANÁLISE METODOLÓGICA DE ERROS
# =============================================================================
def mapear_detalhes_erro(texto_real, texto_predito):
    if not texto_real:
        raise ValueError("CER exige gabarito válido não vazio")
    if texto_real == texto_predito: 
        return "Exato", 0.0
    
    cer = Levenshtein.distance(texto_real, texto_predito) / max(len(texto_real), 1)
    operacoes = Levenshtein.editops(texto_real, texto_predito)
    detalhes = []
    
    for op, i, j in operacoes:
        if op == 'replace': detalhes.append(f"Trocou '{texto_real[i]}' por '{texto_predito[j]}'")
        elif op == 'insert': detalhes.append(f"Inventou '{texto_predito[j]}'")
        elif op == 'delete': detalhes.append(f"Omitiu '{texto_real[i]}'")
            
    return " | ".join(detalhes), cer

# =============================================================================
# 4. GERAÇÃO DE ESTATÍSTICAS E GRÁFICOS (PÓS-PROCESSAMENTO)
# =============================================================================
def gerar_relatorio_final(arq_geral, chaves_comuns):
    print(f"\nCalculando estatísticas finais a partir de {arq_geral}...")
    modelos_ativos = ["EasyOCR", "PaddleOCR", "Qwen_VL"]
    estatisticas = {
        "Chassi": {m: {"acertos": 0, "cer_total": 0.0, "tempo_total": 0.0, "qtd": 0} for m in modelos_ativos},
        "Motor": {m: {"acertos": 0, "cer_total": 0.0, "tempo_total": 0.0, "qtd": 0} for m in modelos_ativos}
    }

    if not os.path.exists(arq_geral):
        print("Arquivo CSV não encontrado. Impossível gerar gráficos.")
        return

    # Mesmas imagens nos três modelos e nos três tratamentos.
    linhas = list(ler_resultados(arq_geral).values())
    with open(Path(arq_geral).with_name("cobertura.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Tipo", "Modelo", "Imagens_no_CSV", "Sucessos_tecnicos", "Falhas_tecnicas", "Imagens_na_comparacao"])
        for tipo in ("Chassi", "Motor"):
            grupo = [r for r in linhas if r[1] == tipo]
            for i, modelo in enumerate(MODELOS):
                sucessos = sum(r[13+i] == "OK" for r in grupo)
                comuns = sum(tuple(r[:4]) in chaves_comuns for r in grupo)
                w.writerow([tipo, modelo, len(grupo), sucessos, len(grupo)-sucessos, comuns])
    for linha in linhas:
        if tuple(linha[:4]) not in chaves_comuns:
            continue
        tipo, texto_real = linha[1], linha[3]
        for i, modelo in enumerate(MODELOS):
            coluna = 4 + 3*i
            predicao, cer, tempo = linha[coluna], float(linha[coluna+1]), float(linha[coluna+2])
            stats = estatisticas[tipo][modelo]
            stats["qtd"] += 1
            stats["acertos"] += predicao == texto_real
            stats["cer_total"] += cer
            stats["tempo_total"] += tempo

    print("\n" + "="*70)
    print(" MATRIZ DE RESULTADOS (MESMAS IMAGENS NAS NOVE COMBINAÇÕES)")
    print("="*70)
    for tipo in ["Chassi", "Motor"]:
        print(f"\nCategoria Analisada: {tipo}")
        print(f"{'Modelo':<15} | {'Acerto Exato (EM)':<18} | {'CER Médio':<12} | {'Tempo Médio/Img'}")
        print("-" * 70)
        for modelo, stats in estatisticas[tipo].items():
            if not stats["qtd"]:
                print(f"{modelo:<15} | Sem imagens comuns com execução bem-sucedida")
                continue
            qtd = stats["qtd"]
            em_pct = (stats["acertos"] / qtd) * 100
            cer_med = (stats["cer_total"] / qtd) * 100
            tmp_med = stats["tempo_total"] / qtd
            print(f"{modelo:<15} | {em_pct:>6.2f}% ({stats['acertos']}/{stats['qtd']})   | {cer_med:>6.2f}%     | {tmp_med:>5.2f}s")

    print("\nGerando gráficos científicos (.png)...")
    x = np.arange(len(modelos_ativos))
    largura = 0.35

    em_chassi = [(estatisticas["Chassi"][m]["acertos"] / (estatisticas["Chassi"][m]["qtd"] or float("nan")))*100 for m in modelos_ativos]
    em_motor = [(estatisticas["Motor"][m]["acertos"] / (estatisticas["Motor"][m]["qtd"] or float("nan")))*100 for m in modelos_ativos]
    
    cer_chassi = [(estatisticas["Chassi"][m]["cer_total"] / (estatisticas["Chassi"][m]["qtd"] or float("nan")))*100 for m in modelos_ativos]
    cer_motor = [(estatisticas["Motor"][m]["cer_total"] / (estatisticas["Motor"][m]["qtd"] or float("nan")))*100 for m in modelos_ativos]

    tempo_chassi = [estatisticas["Chassi"][m]["tempo_total"] / (estatisticas["Chassi"][m]["qtd"] or float("nan")) for m in modelos_ativos]
    tempo_motor = [estatisticas["Motor"][m]["tempo_total"] / (estatisticas["Motor"][m]["qtd"] or float("nan")) for m in modelos_ativos]

    def plotar_barras(dados_chassi, dados_motor, titulo, ylabel, nome_arquivo):
        fig, ax = plt.subplots(figsize=(10, 6))
        barras_chassi = ax.bar(x - largura/2, dados_chassi, largura, label='Chassi', color='#1f77b4')
        barras_motor = ax.bar(x + largura/2, dados_motor, largura, label='Motor', color='#ff7f0e')

        ax.set_ylim(bottom=0)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(titulo, fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(modelos_ativos, fontsize=11)
        ax.legend()
        ax.grid(axis='y', linestyle='--', alpha=0.7)

        ax.text(0, -0.13, f"Chassi n={estatisticas['Chassi'][modelos_ativos[0]]['qtd']} | Motor n={estatisticas['Motor'][modelos_ativos[0]]['qtd']} | n=0: sem dados", transform=ax.transAxes, fontsize=9)
        ax.bar_label(barras_chassi, fmt='%.1f', padding=3)
        ax.bar_label(barras_motor, fmt='%.1f', padding=3)

        plt.tight_layout()
        caminho_salvar = Path(arq_geral).parent / nome_arquivo
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", dpi=300)
        caminho_salvar.write_bytes(buffer.getvalue())
        plt.close()

    plotar_barras(em_chassi, em_motor, 'Taxa de Acerto Exato (EM) por IA', 'Acerto (%) - Maior é Melhor', 'grafico_1_acerto_exato.png')
    plotar_barras(cer_chassi, cer_motor, 'Taxa Média de Erro de Caractere (CER) por IA', 'Erro (%) - Menor é Melhor', 'grafico_2_taxa_erro_cer.png')
    plotar_barras(tempo_chassi, tempo_motor, 'Tempo Médio de Inferência por Imagem', 'Segundos (s) - Menor é Melhor', 'grafico_3_tempo_inferencia.png')
    return estatisticas

# =============================================================================
# 5. EXECUÇÃO COM RETOMADA POR IMAGEM (NÃO MAIS POR PASTA)
# =============================================================================
def executar_benchmark(pasta_assets, arq_geral):
    anteriores = ler_resultados(arq_geral)
    novo = not Path(arq_geral).exists()
    with open(arq_geral, 'a', encoding='utf-8', newline='') as f:
        escritor = csv.writer(f)
        if novo:
            escritor.writerow(CABECALHO)
        for pasta in sorted(p for p in Path(pasta_assets).iterdir() if p.is_dir()):
            jp = pasta / 'dados_vistoria_Chassi-Motor-LABELED.json'
            if not jp.exists():
                jp = pasta / 'imgs' / jp.name
            dados = json.loads(jp.read_text(encoding='utf-8-sig'))
            for item in dados if isinstance(dados, list) else [dados]:
                partes = ler_referencias(item)
                for i, (tipo, campo) in enumerate((('Chassi', 'URL CHASSI'), ('Motor', 'URL Motor'))):
                    real = partes[i]
                    if referencia_invalida(real) or not normalizar(real):
                        continue
                    nome = 'cropped_' + (item.get(campo) or '')
                    imagem = pasta / 'imgs' / nome
                    if not imagem.is_file():
                        if not item.get(campo):
                            continue  # Exclusão registrada durante o preparo.
                        raise FileNotFoundError(f'Crop preparado ausente: {imagem}. Não continuar com dataset incompleto.')
                    chave = (pasta.name, tipo, nome)
                    if chave in anteriores and all(s == 'OK' for s in anteriores[chave][13:16]):
                        continue
                    linha, status = [*chave, normalizar(real)], []
                    funcoes = (ler_com_easyocr, ler_com_paddle, lambda p: ler_com_ollama(p, TAG_QWEN))
                    for funcao in funcoes:
                        inicio = time.perf_counter()
                        try:
                            predicao = normalizar(funcao(str(imagem)))
                            tempo = time.perf_counter() - inicio
                            _, cer = mapear_detalhes_erro(linha[3], predicao)
                            estado = 'OK'
                        except Exception as erro:
                            tempo = time.perf_counter() - inicio
                            predicao, cer = '', ''
                            estado = f'{type(erro).__name__}: {erro}'
                        linha.extend([predicao, cer, tempo])
                        status.append(estado)
                    escritor.writerow(linha + status)
                    f.flush()
                    print(f'{tipo}: {pasta.name} | {status}')
                    gc.collect()


def gerar_detalhes(arq_limpo):
    """Recria erros e acertos usando a última tentativa, sem relatórios antigos acumulados."""
    saida = Path(arq_limpo).parent
    with (saida / '2_analise_cientifica_erros.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['Modelo', 'Tipo', 'Arquivo', 'Texto_Real', 'Texto_Predito', 'CER', 'Detalhe_Cientifico_do_Erro', 'Pasta'])
        for linha in ler_resultados(arq_limpo).values():
            for i, modelo in enumerate(MODELOS):
                if linha[13+i] != 'OK':
                    continue  # O motivo técnico continua no CSV geral e na cobertura.
                predicao = linha[4+3*i]
                detalhe, cer = mapear_detalhes_erro(linha[3], predicao)
                if detalhe != 'Exato':
                    w.writerow([modelo, linha[1], linha[2], linha[3], predicao, cer, detalhe, linha[0]])
    gerar_relatorio_acertos(arq_limpo, saida / '3_acertos_exatos.csv')


def comparar_tratamentos(pasta_resultados):
    """Mantém os gráficos antigos e acrescenta tratamento × modelo sobre os mesmos IDs."""
    pasta_resultados = Path(pasta_resultados)
    arquivos = [pasta_resultados / t / '1_benchmark_geral_limpo.csv' for t in TRATAMENTOS]
    with (PASTA_DATASET / 'amostras_elegiveis.csv').open(encoding='utf-8', newline='') as f:
        leitor = csv.reader(f)
        next(leitor)
        esperadas = {tuple(r) for r in leitor}
    for arquivo in arquivos:
        encontradas = {tuple(r[:4]) for r in ler_resultados(arquivo).values()}
        if encontradas != esperadas:
            raise RuntimeError(f'Resultados incompletos ou incompatíveis em {arquivo}: '
                               f'{len(esperadas-encontradas)} ausentes, {len(encontradas-esperadas)} inesperadas. '
                               'Retome a execução antes de comparar.')
    conjuntos = [{tuple(r[:4]) for r in ler_resultados(p).values() if all(s == 'OK' for s in r[13:16])} for p in arquivos]
    comuns = set.intersection(*conjuntos)
    estatisticas = {t: gerar_relatorio_final(p, comuns) for t, p in zip(TRATAMENTOS, arquivos)}
    # n=0 vira NaN (sem barra); nunca exibir ausência como zero erros ou zero tempo.
    with (pasta_resultados / 'comparacao_tratamentos.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['Tratamento', 'Tipo', 'Modelo', 'N_comum', 'Acerto_pct', 'CER_pct', 'Tempo_s'])
        for t in TRATAMENTOS:
            for tipo in ('Chassi', 'Motor'):
                for modelo in MODELOS:
                    s = estatisticas[t][tipo][modelo]
                    n = s['qtd']
                    w.writerow([t, tipo, modelo, n, *([100*s['acertos']/n, 100*s['cer_total']/n, s['tempo_total']/n] if n else ['', '', ''])])
    for metrica, rotulo, fator in [('acertos', 'Acerto exato (%) — maior é melhor', 100),
                                  ('cer_total', 'CER médio (%) — menor é melhor', 100),
                                  ('tempo_total', 'Tempo médio (s)', 1)]:
        fig, eixos = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
        for ax, tipo in zip(eixos, ('Chassi', 'Motor')):
            for j, tratamento in enumerate(TRATAMENTOS):
                valores = [fator*estatisticas[tratamento][tipo][m][metrica]/(estatisticas[tratamento][tipo][m]['qtd'] or float('nan')) for m in MODELOS]
                barras = ax.bar(np.arange(3)+(j-1)*0.25, valores, 0.25, label=tratamento)
                ax.bar_label(barras, fmt='%.1f', fontsize=8, padding=2)
            n = estatisticas[TRATAMENTOS[0]][tipo][MODELOS[0]]['qtd']
            ax.set_title(f'{tipo} — n={n}' + (' (sem dados)' if n == 0 else ''))
            ax.set_xticks(np.arange(3), MODELOS)
            ax.set_ylim(bottom=0)
            ax.set_ylabel(rotulo)
            ax.grid(axis='y', alpha=.2)
            ax.set_axisbelow(True)
        eixos[0].legend(fontsize=8)
        fig.suptitle('Comparação dos tratamentos nas mesmas imagens')
        fig.text(.5, .02, 'Resultados descritivos; consulte cobertura.csv. Tempos refletem CPU/GPU e incluem primeira chamada.', ha='center', fontsize=8)
        fig.tight_layout(rect=(0, .06, 1, .94))
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png', dpi=180)
        (pasta_resultados / f'comparacao_{metrica}.png').write_bytes(buffer.getvalue())
        plt.close(fig)
    print(f'Gráficos comparativos: {pasta_resultados}')


if __name__ == '__main__':
    if RODAR_INTELIGENCIA_ARTIFICIAL:
        if not (PASTA_DATASET / 'preparo_concluido.txt').exists():
            raise RuntimeError('Execute preprocessar_imagens.py primeiro; o preparo está ausente/incompleto.')
        # Guardar uma configuração simples evita retomar CSV com imagens/modelo diferentes.
        import ollama
        disponiveis = ollama.list()
        disponiveis = disponiveis.model_dump() if hasattr(disponiveis, 'model_dump') else disponiveis
        digest = next((m.get('digest') for m in disponiveis['models'] if m.get('model', m.get('name')) == TAG_QWEN), None)
        if not digest:
            raise RuntimeError('Modelo não encontrado em ollama list.')
        inicializar_modelos()
    for tratamento in TRATAMENTOS:
        saida = PASTA_RESULTADOS / tratamento
        saida.mkdir(parents=True, exist_ok=True)
        bruto, limpo = saida / '1_benchmark_geral.csv', saida / '1_benchmark_geral_limpo.csv'
        if RODAR_INTELIGENCIA_ARTIFICIAL:
            pasta = PASTA_DATASET / tratamento
            hash_dados = hashlib.sha256()
            for p in sorted(p for p in pasta.rglob('*') if p.is_file()):
                hash_dados.update(str(p.relative_to(pasta)).encode())
                hash_dados.update(p.read_bytes())
            hash_codigo = hashlib.sha256()
            for codigo in [Path(__file__), RAIZ/'limpar_dataset.py', RAIZ/'preprocessar_imagens.py', Path(__file__).with_name('acertos.py')]:
                # O interruptor de execução não muda a metodologia.
                fonte = codigo.read_text(encoding='utf-8')
                fonte = fonte.replace('RODAR_INTELIGENCIA_ARTIFICIAL = True', 'RODAR_INTELIGENCIA_ARTIFICIAL = False')
                hash_codigo.update(fonte.encode())
            config = dict(protocolo='windows_v2', codigo_sha256=hash_codigo.hexdigest(), dataset_sha256=hash_dados.hexdigest(), qwen=TAG_QWEN, digest=digest,
                          prompt=PROMPT, temperature=0, seed=0, normalizacao='A-Z0-9',
                          versoes={p: importlib.metadata.version(p) for p in ['easyocr', 'paddleocr', 'paddlepaddle', 'ollama', 'Levenshtein', 'torch', 'torchvision', 'numpy', 'Pillow', 'opencv-python', 'matplotlib']})
            arquivo_config = saida / 'configuracao.json'
            if arquivo_config.exists():
                if json.loads(arquivo_config.read_text(encoding='utf-8')) != config:
                    raise RuntimeError('Dados/configuração mudaram. Use outra PASTA_RESULTADOS.')
            elif bruto.exists():
                raise RuntimeError('CSV sem configuração; use outra PASTA_RESULTADOS.')
            else:
                arquivo_config.write_text(json.dumps(config, indent=2), encoding='utf-8')
            executar_benchmark(pasta, bruto)
        if not bruto.exists():
            raise FileNotFoundError('Sem resultados: mude RODAR_INTELIGENCIA_ARTIFICIAL para True e execute novamente.')
        limpar_dados_injustos(bruto, limpo)
        gerar_detalhes(limpo)
    comparar_tratamentos(PASTA_RESULTADOS)
