"""Benchmark v3: processos separados, tentativas auditáveis e denominador fixo.

Execute --help. Nenhum gabarito é enviado aos modelos. Os resultados antigos
permanecem em resultados_ajustados; esta versão grava em resultados_v3.
"""
import argparse
from contextlib import contextmanager
import base64
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from urllib.request import Request, urlopen

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
from limpar_dataset import normalizar, referencia_invalida
from preprocessar_imagens import TRATAMENTOS
from acertos import gerar_relatorio_acertos

MODELOS = ('EasyOCR', 'PaddleOCR', 'Qwen_VL')
PROMPT = ('Transcribe the alphanumeric text visible in this cropped image exactly. '
          'Read left to right, then top to bottom. Do not guess, complete, correct, '
          'or repeat characters. Return only the transcription, without explanations.')
OPCOES_QWEN = dict(temperature=0, seed=0, num_predict=64, num_ctx=4096)
MAX_TENTATIVAS = 2  # Somente falhas técnicas; leitura errada/vazia não é repetida.
SERVIDOR = 'http://127.0.0.1:11434'
COLUNAS = ['Tratamento', 'Modelo', 'Pasta', 'Tipo', 'Arquivo', 'Texto_Real', 'Grupo',
           'Parte', 'Status', 'Predicao', 'CER', 'Tempo_s', 'Tentativas', 'Truncada']


def api_ollama(endpoint, dados=None, timeout=180):
    corpo = json.dumps(dados).encode() if dados is not None else None
    req = Request(SERVIDOR + '/api/' + endpoint, data=corpo,
                  headers={'Content-Type': 'application/json'})
    with urlopen(req, timeout=timeout) as resposta:
        return json.load(resposta)


def chave(amostra):
    return tuple(amostra[k] for k in ('Pasta', 'Tipo', 'Arquivo'))


def gravar_json(caminho, objeto):
    """Substituição atômica para configurações/relatórios, no mesmo filesystem."""
    temp = caminho.with_suffix(caminho.suffix + '.tmp')
    with temp.open('w', encoding='utf-8', newline='\n') as f:
        json.dump(objeto, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    temp.replace(caminho)


@contextmanager
def bloquear_execucao(pasta):
    """Impede dois terminais de alterarem o mesmo histórico simultaneamente."""
    pasta.mkdir(parents=True, exist_ok=True)
    trava = pasta/'execucao.lock'
    try:
        with trava.open('x', encoding='utf-8') as f:
            f.write(f'PID={os.getpid()}\nRemova somente após confirmar que o processo terminou.\n')
    except FileExistsError as erro:
        raise RuntimeError(f'Execução em andamento ou interrompida: {trava}. Confirme que não há outra execução antes de remover a trava.') from erro
    try:
        yield
    finally:
        trava.unlink()


def registrar(caminho, evento):
    """Salva antes e depois de cada chamada, inclusive em caso de exceção."""
    evento = dict(evento, instante=datetime.now(timezone.utc).isoformat())
    with caminho.open('a', encoding='utf-8', newline='\n') as f:
        f.write(json.dumps(evento, ensure_ascii=False) + '\n')
        f.flush()
        os.fsync(f.fileno())


def historico(caminho):
    """Recusa corrupção: não descarta silenciosamente a última linha incompleta."""
    tentativas = {}
    if caminho.exists():
        for numero, linha in enumerate(caminho.read_text(encoding='utf-8').splitlines(), 1):
            try:
                evento = json.loads(linha)
                k = tuple(evento['chave'])
                n = evento['tentativa']
                if evento['evento'] not in ('inicio', 'fim') or not isinstance(n, int) or not 1 <= n <= MAX_TENTATIVAS:
                    raise ValueError('Evento inválido')
                anterior = tentativas.setdefault(k, {}).get(n)
                if evento['evento'] == 'inicio' and anterior is not None:
                    raise ValueError('Tentativa iniciada duas vezes')
                if evento['evento'] == 'fim' and (anterior is None or anterior['evento'] != 'inicio'):
                    raise ValueError('Fim sem início ou resultado duplicado')
                tentativas[k][n] = evento
            except (ValueError, KeyError, TypeError) as erro:
                raise ValueError(f'Log inválido {caminho}:{numero}. Preserve o arquivo para recuperação.') from erro
    return tentativas


def resultado_final(tentativas):
    """Primeiro sucesso; se não há sucesso, última falha. Nunca escolhe por acerto."""
    ordenadas = [tentativas[n] for n in sorted(tentativas)]
    for evento in ordenadas:
        if evento.get('status') == 'OK':
            return evento
    return ordenadas[-1] if ordenadas else None


def carregar_amostras(dataset, parte):
    if not (dataset/'preparo_concluido.txt').is_file():
        raise ValueError('Dataset incompleto. Execute preprocessar_imagens.py.')
    with (dataset/'amostras_elegiveis.csv').open(encoding='utf-8', newline='') as f:
        todas = list(csv.DictReader(f))
    vistos, grupos = set(), {}
    for r in todas:
        if chave(r) in vistos or referencia_invalida(r['Texto_Real']):
            raise ValueError('Manifesto com duplicata ou referência ausente.')
        vistos.add(chave(r))
        if r['Texto_Real'] != normalizar(r['Texto_Real']) or r['Parte'] not in ('desenvolvimento', 'avaliacao'):
            raise ValueError('Manifesto incompatível com v3.')
        if grupos.setdefault(r['Grupo'], r['Parte']) != r['Parte']:
            raise ValueError('Mesmo grupo presente nas duas partições.')
        for campo in ('Pasta', 'Arquivo'):
            if Path(r[campo]).name != r[campo] or '\\' in r[campo] or r[campo] in ('', '.', '..'):
                raise ValueError('Caminho inválido no manifesto.')
    selecionadas = [r for r in todas if r['Parte'] == parte]
    if not selecionadas:
        raise ValueError(f'Nenhuma amostra em {parte}.')
    return selecionadas


def hash_arquivos(raiz, caminhos):
    h = hashlib.sha256()
    for p in sorted(caminhos):
        h.update(p.relative_to(raiz).as_posix().encode() + b'\0')
        dados = p.read_bytes() if p.suffix.lower() == '.png' else p.read_text(encoding='utf-8').encode()
        h.update(hashlib.sha256(dados).digest())
    return h.hexdigest()


def hash_codigo():
    return hash_arquivos(RAIZ, [Path(__file__), Path(__file__).with_name('acertos.py'),
                                RAIZ/'limpar_dataset.py', RAIZ/'preprocessar_imagens.py'])


def configurar(args, amostras):
    """Congela dados, código, ambiente e parâmetros antes da primeira inferência."""
    arquivos = [p for p in args.dataset.rglob('*') if p.is_file()]
    for t in args.tratamentos:
        for r in amostras:
            if not (args.dataset/t/r['Pasta']/'imgs'/r['Arquivo']).is_file():
                raise FileNotFoundError(f'Crop preparado ausente: {t}/{chave(r)}')
    versoes = {}
    for pacote in ['easyocr', 'paddleocr', 'paddlepaddle', 'torch', 'torchvision', 'numpy',
                   'Pillow', 'opencv-python', 'opencv-python-headless', 'Levenshtein', 'matplotlib']:
        try:
            versoes[pacote] = importlib.metadata.version(pacote)
        except importlib.metadata.PackageNotFoundError:
            versoes[pacote] = None
    qwen = None
    if 'Qwen_VL' in args.modelos:
        modelos = api_ollama('tags')['models']
        qwen = next((m for m in modelos if m.get('model', m.get('name')) == args.qwen), None)
        if qwen is None:
            raise ValueError(f'Modelo ausente no Ollama: {args.qwen}. Instale-o antes de iniciar.')
        qwen = dict(tag=args.qwen, digest=qwen['digest'], servidor=api_ollama('version'),
                    detalhes=api_ollama('show', {'model': args.qwen}))
    config = dict(protocolo='v3', parte=args.parte, modelos=args.modelos,
                  tratamentos=args.tratamentos, qwen=qwen, prompt=PROMPT,
                  opcoes=OPCOES_QWEN, max_tentativas=MAX_TENTATIVAS, timeout_s=180,
                  normalizacao='NFKC-uppercase-AZ09-sem-substituicao',
                  dataset_sha256=hash_arquivos(args.dataset, arquivos),
                  codigo_sha256=hash_codigo(), versoes=versoes,
                  python=platform.python_version(), sistema=platform.platform(),
                  hardware=args.hardware, threads=1,
                  tempo='chamada após aquecimento; OCR CPU; Qwen conforme Ollama; inclui transporte')
    args.saida.mkdir(parents=True, exist_ok=True)
    destino = args.saida/'configuracao.json'
    if destino.exists():
        if json.loads(destino.read_text(encoding='utf-8')) != config:
            raise ValueError('Configuração mudou. Use --saida com outra pasta; não misture experimentos.')
    else:
        if list(args.saida.glob('**/tentativas_*.jsonl')):
            raise ValueError('Há resultados sem configuração. Preserve-os e escolha outra pasta.')
        gravar_json(destino, config)
    return config


def criar_leitor(modelo, tag, saida=None):
    """Importa apenas a IA do processo atual. Nenhuma função recebe gabarito."""
    if modelo == 'EasyOCR':
        import torch
        import easyocr
        torch.set_num_threads(1)
        cache = RAIZ/'cache_modelos'/'easyocr'
        leitor = easyocr.Reader(['en'], gpu=False, model_storage_directory=str(cache))
        def ler(p):
            textos = leitor.readtext(str(p), detail=0)
            return {'texto_bruto': '\n'.join(textos), 'metadados': {'textos': textos}}
    elif modelo == 'PaddleOCR':
        from paddleocr import PaddleOCR
        cache = RAIZ/'cache_modelos'/'paddleocr'
        leitor = PaddleOCR(lang='en', use_angle_cls=True, use_gpu=False, show_log=False,
                           cpu_threads=1, enable_mkldnn=False,
                           det_model_dir=str(cache/'det'), rec_model_dir=str(cache/'rec'),
                           cls_model_dir=str(cache/'cls'))
        def ler(p):
            resultados = leitor.ocr(str(p), cls=True)
            linhas = resultados[0] if resultados and resultados[0] else []
            textos = [r[1][0] for r in linhas]
            return {'texto_bruto': '\n'.join(textos),
                    'metadados': {'textos': textos, 'confiancas': [float(r[1][1]) for r in linhas]}}
    else:
        def ler(p):
            resposta = api_ollama('chat', {'model': tag, 'stream': False, 'keep_alive': '2m',
                'options': OPCOES_QWEN, 'messages': [{'role': 'user', 'content': PROMPT,
                'images': [base64.b64encode(p.read_bytes()).decode()]}]})
            mensagem = resposta.get('message', {})
            if resposta.get('done') is not True or mensagem.get('role') != 'assistant' or not isinstance(mensagem.get('content'), str):
                raise ValueError(f'Resposta Ollama inválida: {resposta}')
            return {'texto_bruto': mensagem['content'], 'metadados': resposta,
                    'truncada': resposta.get('done_reason') == 'length'}
    if modelo != 'Qwen_VL' and saida is not None:
        pesos = {p.relative_to(cache).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in sorted(cache.rglob('*')) if p.is_file() and
                 p.suffix in ('.pth', '.pdiparams', '.pdmodel', '.json')}
        if not pesos:
            raise RuntimeError(f'Não foi possível identificar os pesos em {cache}.')
        registro = saida/f'pesos_{modelo}.json'
        if registro.exists() and json.loads(registro.read_text(encoding='utf-8')) != pesos:
            raise ValueError('Pesos OCR mudaram; use outra pasta de resultados.')
        gravar_json(registro, pesos)
    return ler


def executar_amostras(amostras, pasta, log, leitor):
    """Uma tentativa interrompida consome orçamento; sucessos não são repetidos."""
    anteriores = historico(log)
    for r in amostras:
        k = chave(r)
        tentativas = anteriores.get(k, {})
        ultimo = resultado_final(tentativas)
        if ultimo and ultimo.get('status') == 'OK':
            continue
        for n in range(max(tentativas, default=0)+1, MAX_TENTATIVAS+1):
            evento = dict(chave=list(k), tentativa=n, evento='inicio')
            registrar(log, evento)
            inicio = time.perf_counter()
            try:
                resposta = leitor(pasta/r['Pasta']/'imgs'/r['Arquivo'])
                if not isinstance(resposta.get('texto_bruto'), str):
                    raise ValueError('Adaptador retornou texto inválido')
                fim = dict(evento, evento='fim', status='OK', **resposta)
            except Exception as erro:
                fim = dict(evento, evento='fim', status='FALHA_TECNICA',
                           erro=f'{type(erro).__name__}: {erro}')
            fim['tempo_s'] = time.perf_counter()-inicio
            registrar(log, fim)
            print(f'{r["Tipo"]}: {r["Pasta"]} | tentativa {n} | {fim["status"]}', flush=True)
            if fim['status'] == 'OK':
                break


def worker(args, amostras):
    for nome in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ[nome] = '1'
    leitor = criar_leitor(args.worker, args.qwen, args.saida)
    from PIL import Image, ImageDraw
    aquecimento = args.saida/'aquecimento.png'
    im = Image.new('RGB', (320, 64), 'white')
    ImageDraw.Draw(im).text((20, 20), 'AB123456', fill='black')
    im.save(aquecimento)
    try:
        leitor(aquecimento)  # Sem gabarito; excluído das métricas e tentativas.
        for t in args.tratamentos:
            pasta = args.saida/t
            pasta.mkdir(exist_ok=True)
            executar_amostras(amostras, args.dataset/t, pasta/f'tentativas_{args.worker}.jsonl', leitor)
    finally:
        if args.worker == 'Qwen_VL':
            api_ollama('generate', {'model': args.qwen, 'keep_alive': 0}, timeout=30)


def consolidar(args, amostras):
    import Levenshtein
    linhas, pendentes = [], 0
    for t in args.tratamentos:
        for modelo in args.modelos:
            eventos = historico(args.saida/t/f'tentativas_{modelo}.jsonl')
            if set(eventos)-{chave(r) for r in amostras}:
                raise ValueError('Log contém amostras inesperadas.')
            for r in amostras:
                tentativas = eventos.get(chave(r), {})
                e = resultado_final(tentativas)
                status = (e or {}).get('status', 'INTERROMPIDA' if e else 'NAO_EXECUTADA')
                if status != 'OK' and max(tentativas, default=0) < MAX_TENTATIVAS:
                    pendentes += 1
                pred = normalizar(e['texto_bruto']) if status == 'OK' else ''
                cer = Levenshtein.distance(r['Texto_Real'], pred)/len(r['Texto_Real']) if status == 'OK' else ''
                linhas.append(dict(r, Tratamento=t, Modelo=modelo, Status=status, Predicao=pred,
                    CER=cer, Tempo_s=(e or {}).get('tempo_s', ''), Tentativas=len(tentativas),
                    Truncada=bool((e or {}).get('truncada', False))))
    escrever_csv(args.saida/'resultados_por_imagem.csv', COLUNAS, linhas)
    if pendentes:
        # Relatórios antigos não podem continuar aparentando uma execução completa.
        raise RuntimeError(f'{pendentes} combinações pendentes. CSV parcial salvo; retome antes dos gráficos.')
    return linhas


def escrever_csv(caminho, campos, linhas):
    temp = caminho.with_suffix('.csv.tmp')
    with temp.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=campos)
        w.writeheader()
        w.writerows(linhas)
    temp.replace(caminho)


def estatisticas(linhas, comuns):
    """Falhas ficam no denominador do acerto. CER só existe com transcrição."""
    n = len(linhas)
    ok = [r for r in linhas if r['Status'] == 'OK']
    pares = [r for r in ok if chave(r) in comuns]
    exatos = sum(r['Predicao'] == r['Texto_Real'] for r in ok)
    media = lambda xs: sum(xs)/len(xs) if xs else ''
    return dict(N_elegivel=n, Sucessos=len(ok), Falhas=n-len(ok), Acertos=exatos,
        Acerto_pct=100*exatos/n if n else '', Falha_pct=100*(n-len(ok))/n if n else '',
        CER_macro_sucessos_pct=media([100*r['CER'] for r in ok]),
        CER_micro_sucessos_pct=100*sum(r['CER']*len(r['Texto_Real']) for r in ok)/sum(len(r['Texto_Real']) for r in ok) if ok else '',
        N_comum=len(pares), CER_macro_comum_pct=media([100*r['CER'] for r in pares]),
        Acerto_comum_pct=100*sum(r['Predicao']==r['Texto_Real'] for r in pares)/len(pares) if pares else '',
        Tempo_sucessos_s=media([r['Tempo_s'] for r in ok]), Truncadas=sum(r['Truncada'] for r in ok))


def relatorios(args, linhas):
    import Levenshtein
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    conjuntos = [{chave(r) for r in linhas if r['Tratamento']==t and r['Modelo']==m and r['Status']=='OK'}
                 for t in args.tratamentos for m in args.modelos]
    comuns = set.intersection(*conjuntos)
    resumo = []
    for t in args.tratamentos:
        for tipo in ('Chassi', 'Motor', 'Total'):
            for m in args.modelos:
                grupo = [r for r in linhas if r['Tratamento']==t and r['Modelo']==m and (tipo=='Total' or r['Tipo']==tipo)]
                resumo.append(dict(Tratamento=t, Tipo=tipo, Modelo=m, **estatisticas(grupo, comuns)))
    escrever_csv(args.saida/'comparacao_tratamentos.csv', list(resumo[0]), resumo)
    gerar_relatorio_acertos(args.saida/'resultados_por_imagem.csv', args.saida/'acertos_exatos.csv')
    erros = []
    for r in linhas:
        if r['Status'] == 'OK' and r['Predicao'] != r['Texto_Real']:
            ops = [op for op, _, _ in Levenshtein.editops(r['Texto_Real'], r['Predicao'])]
            erros.append(dict(r, Substituicoes=ops.count('replace'), Insercoes=ops.count('insert'), Omissoes=ops.count('delete')))
    escrever_csv(args.saida/'analise_erros.csv', COLUNAS+['Substituicoes', 'Insercoes', 'Omissoes'], erros)
    for metrica, titulo in [('Acerto_pct', 'Acerto exato: todas as amostras elegíveis (%)'),
                           ('CER_macro_comum_pct', 'CER médio: amostras com sucesso em todas as combinações (%)'),
                           ('Tempo_sucessos_s', 'Tempo de chamada após aquecimento: sucessos (s)')]:
        fig, eixos = plt.subplots(1, 2, figsize=(13, 5))
        largura = .8/len(args.tratamentos)
        for ax, tipo in zip(eixos, ('Chassi', 'Motor')):
            for j, t in enumerate(args.tratamentos):
                valores = [next(r[metrica] for r in resumo if (r['Tratamento'],r['Tipo'],r['Modelo'])==(t,tipo,m)) for m in args.modelos]
                barras = ax.bar([i+(j-(len(args.tratamentos)-1)/2)*largura for i in range(len(args.modelos))],
                               [float('nan') if v=='' else v for v in valores], largura, label=t)
                ax.bar_label(barras, fmt='%.1f', fontsize=8, padding=3)
            referencia = next(r for r in resumo if r['Tipo']==tipo)
            ax.set_title(f"{tipo} | n elegível={referencia['N_elegivel']}, n comum={referencia['N_comum']}")
            ax.set_xticks(range(len(args.modelos)), [args.qwen if m=='Qwen_VL' else m for m in args.modelos])
            ax.set_ylim(bottom=0)
            ax.margins(y=.15)
            ax.grid(axis='y', alpha=.2)
            ax.set_axisbelow(True)
        eixos[0].legend(fontsize=8)
        fig.suptitle(titulo)
        fig.text(.5, .02, f'{args.parte}: análise exploratória. Consulte N, cobertura e configuração no CSV. CPU/Ollama: dispositivos distintos.', ha='center', fontsize=8)
        fig.tight_layout(rect=(0,.06,1,.95))
        fig.savefig(args.saida/f'{metrica}.png', dpi=180)
        plt.close(fig)
    print(f'Relatórios completos em {args.saida}. OK indica execução, não acerto.')


def argumentos():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', type=Path, default=RAIZ/'dataset_v3')
    p.add_argument('--saida', type=Path, default=RAIZ/'benchmark-ocr/resultados_v3')
    p.add_argument('--parte', choices=['desenvolvimento', 'avaliacao'], default='desenvolvimento')
    p.add_argument('--modelos', nargs='+', choices=MODELOS, default=list(MODELOS))
    p.add_argument('--tratamentos', nargs='+', choices=TRATAMENTOS, default=list(TRATAMENTOS))
    p.add_argument('--qwen', default='qwen2.5vl:latest')
    p.add_argument('--hardware', default='nao_informado', help='Descrição: CPU, RAM, GPU e VRAM, para reprodução.')
    p.add_argument('--relatorios', action='store_true', help='Recalcula relatórios sem carregar IAs.')
    p.add_argument('--worker', choices=MODELOS, help=argparse.SUPPRESS)
    args = p.parse_args()
    args.dataset = args.dataset.resolve()
    args.saida = args.saida.resolve()
    if args.saida.is_relative_to(args.dataset):
        p.error('A pasta de resultados deve ficar fora do dataset para preservar seu hash.')
    if len(set(args.modelos)) != len(args.modelos) or len(set(args.tratamentos)) != len(args.tratamentos):
        p.error('Não repita modelos ou tratamentos.')
    return args


def main():
    args = argumentos()
    amostras = carregar_amostras(args.dataset, args.parte)
    if args.worker:
        worker(args, amostras)
        return
    with bloquear_execucao(args.saida):
        executar(args, amostras)


def executar(args, amostras):
    if args.relatorios:
        config = json.loads((args.saida/'configuracao.json').read_text(encoding='utf-8'))
        if config['parte'] != args.parte or config['dataset_sha256'] != hash_arquivos(args.dataset, [p for p in args.dataset.rglob('*') if p.is_file()]):
            raise ValueError('Dataset/partição não corresponde à execução salva.')
        if config['codigo_sha256'] != hash_codigo():
            raise ValueError('Código diferente da execução. Use o commit registrado para reproduzir seus relatórios.')
        args.modelos, args.tratamentos = config['modelos'], config['tratamentos']
        if config['qwen']:
            args.qwen = config['qwen']['tag']
    else:
        configurar(args, amostras)
        for modelo in args.modelos:
            # Apenas o Qwen deste experimento; não descarrega outros modelos do usuário.
            if 'Qwen_VL' in args.modelos:
                api_ollama('generate', {'model': args.qwen, 'keep_alive': 0}, timeout=30)
            comando = [sys.executable, str(Path(__file__)), '--worker', modelo,
                '--dataset', str(args.dataset), '--saida', str(args.saida), '--parte', args.parte,
                '--qwen', args.qwen, '--tratamentos', *args.tratamentos]
            subprocess.run(comando, check=True)
    relatorios(args, consolidar(args, amostras))


if __name__ == '__main__':
    main()
