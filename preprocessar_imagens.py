"""Prepara três cópias do dataset mantendo o formato de pastas/JSON do projeto."""
import argparse
import csv
import hashlib
import json
import io
import math
from pathlib import Path
from PIL import Image, ImageOps
from limpar_dataset import referencia_invalida, normalizar, ler_referencias

RAIZ = Path(__file__).resolve().parent
PASTA_ORIGINAL = RAIZ / "assets"  # Organização mostrada nas capturas de tela
PASTA_SAIDA = RAIZ / "dataset_v3"
TRATAMENTOS = ("sem_tratamento", "nativo", "clahe", "ampliado")


def aplicar_ampliacao(imagem):
    """Ampliação convencional 2x, limitada a 2048px, e margem branca de 16px.

    Não reconstrói caracteres por IA. É uma hipótese experimental independente
    do CLAHE e recebe os mesmos parâmetros para todos os modelos.
    """
    escala = min(2.0, 2048 / max(imagem.size))
    tamanho = tuple(max(1, round(v * escala)) for v in imagem.size)
    return ImageOps.expand(imagem.resize(tamanho, Image.Resampling.LANCZOS), 16, 'white')


def aplicar_revisao(pasta, item, revisao, aplicadas):
    partes = ler_referencias(item)
    for correcao in revisao.get('correcoes', []):
        if correcao['Pasta'] != pasta:
            continue
        i = ('Chassi', 'Motor').index(correcao['Tipo'])
        chave = (pasta, correcao['Tipo'])
        if chave in aplicadas or partes[i] != correcao['antes']:
            raise ValueError(f'Correção incompatível ou duplicada: {chave}')
        partes[i] = correcao['depois']
        aplicadas.add(chave)
    item['Dados do Veículo'] = '\n'.join(partes)


def revisar_recorte(pasta, tipo, item, baseline, revisao, aplicados):
    """Retira o campo MODELO vizinho somente onde a foto confirma a separação.

    A mesma fração horizontal é aplicada ao crop antigo e à BBox da foto.
    Não escolhe região usando respostas ou acertos dos modelos.
    """
    for r in revisao.get('recortes', []):
        if (r['Pasta'], r['Tipo']) != (pasta, tipo):
            continue
        chave = (pasta, tipo)
        pontos = item[tipo+'_BBox']
        if chave in aplicados or pontos != r['bbox_antes']:
            raise ValueError(f'Revisão de BBox incompatível: {chave}')
        fracao = r['fracao_horizontal']
        if not 0 < fracao < 1:
            raise ValueError('Fração de recorte inválida')
        tl, tr, br, bl = pontos
        interp = lambda a,b: [a[i]+fracao*(b[i]-a[i]) for i in range(2)]
        item[tipo+'_BBox'] = [tl, interp(tl,tr), interp(bl,br), bl]
        baseline = baseline.crop((0,0,max(1,round(baseline.width*fracao)),baseline.height))
        aplicados.add(chave)
    return baseline


def separar_grupos(linhas, pixels):
    """Agrupa por pasta, identificador repetido e pixels idênticos antes do split.

    Como os resultados antigos já foram inspecionados, a avaliação reservada
    continua exploratória. Um teste cego exige novos veículos independentes.
    """
    pais = {r[0]: r[0] for r in linhas}
    def raiz(x):
        while pais[x] != x:
            pais[x] = pais[pais[x]]
            x = pais[x]
        return x
    vistos = {}
    for r in linhas:
        for identidade in [('referencia', r[1], r[3]), ('pixels', pixels[tuple(r[:3])])]:
            if identidade in vistos:
                a, b = raiz(r[0]), raiz(vistos[identidade])
                pais[max(a, b)] = min(a, b)
            vistos[identidade] = r[0]
    for r in linhas:
        grupo = hashlib.sha256(raiz(r[0]).encode()).hexdigest()[:16]
        parte = 'avaliacao' if int(grupo[:8], 16) % 100 < 30 else 'desenvolvimento'
        r.extend([grupo, parte])
    return linhas


def recortar_original(imagem, pontos):
    """Reutiliza os quatro pontos anotados, removendo somente o limite de 280px."""
    if (not isinstance(pontos, list) or len(pontos) != 4 or
        any(not isinstance(p, (list, tuple)) or len(p) != 2 or
            any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in p) for p in pontos)):
        raise ValueError("BBox sem quatro pontos; revise a anotação.")
    if any(not (0 <= x <= imagem.width and 0 <= y <= imagem.height) for x, y in pontos):
        raise ValueError("BBox fora da foto original.")
    tl, tr, br, bl = pontos
    cruzamentos = []
    for i in range(4):
        a, b, c = pontos[i], pontos[(i+1) % 4], pontos[(i+2) % 4]
        cruzamentos.append((b[0]-a[0])*(c[1]-b[1]) - (b[1]-a[1])*(c[0]-b[0]))
    if not (all(v > 0 for v in cruzamentos) or all(v < 0 for v in cruzamentos)):
        raise ValueError('BBox não convexa ou com pontos cruzados; revise a anotação.')
    largura, altura = int(math.dist(tl, tr)), int(math.dist(tl, bl))
    if min(largura, altura) < 2:
        raise ValueError("BBox degenerada.")
    return imagem.convert("RGB").transform((largura, altura), Image.Transform.QUAD,
        [*tl, *bl, *br, *tr], resample=Image.Resampling.BILINEAR)


def aplicar_clahe(imagem):
    """Contraste local na luminosidade, com parâmetros iguais para todas as imagens."""
    import cv2
    import numpy as np
    lab = cv2.cvtColor(np.asarray(imagem), cv2.COLOR_RGB2LAB)
    lab[:, :, 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[:, :, 0])
    return Image.fromarray(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB))


def preparar_imagens(pasta_original=PASTA_ORIGINAL, pasta_saida=PASTA_SAIDA, arquivo_revisao=None):
    pasta_original, pasta_saida = Path(pasta_original), Path(pasta_saida)
    if not pasta_original.is_dir():
        raise FileNotFoundError(pasta_original)
    revisao = json.loads(Path(arquivo_revisao).read_text(encoding='utf-8')) if arquivo_revisao else {}
    aplicadas, recortes_aplicados, pixels = set(), set(), {}
    pasta_saida.mkdir(parents=True, exist_ok=False)  # Não sobrescrever um experimento.
    for tratamento in TRATAMENTOS:
        (pasta_saida / tratamento).mkdir()
    quantidade = 0
    elegiveis = []
    with (pasta_saida / "exclusoes.csv").open("w", encoding="utf-8", newline="") as log:
        escritor = csv.writer(log)
        escritor.writerow(["Pasta", "Tipo", "Referencia", "Motivo"])
        for pasta in sorted(p for p in pasta_original.iterdir() if p.is_dir()):
            arquivo_json = pasta / "dados_vistoria_Chassi-Motor-LABELED.json"
            if not arquivo_json.exists():
                arquivo_json = pasta / "imgs" / arquivo_json.name
            dados = json.loads(arquivo_json.read_text(encoding="utf-8-sig"))
            itens = dados if isinstance(dados, list) else [dados]
            for item in itens:
                try:
                    aplicar_revisao(pasta.name, item, revisao, aplicadas)
                    partes = ler_referencias(item)
                except ValueError as erro:
                    raise ValueError(f"{arquivo_json}: {erro}") from erro
                for i, (tipo, campo) in enumerate((("Chassi", "URL CHASSI"), ("Motor", "URL Motor"))):
                    referencia = partes[i]
                    nome = item.get(campo) or ""
                    if Path(nome).name != nome or "\\" in nome:
                        raise ValueError(f"Esperado nome local de imagem: {nome}")
                    crop = pasta / "imgs" / ("cropped_" + nome)
                    motivo = ("referencia_invalida" if referencia_invalida(referencia) or not normalizar(referencia)
                              else "crop_ausente" if not crop.is_file() else "")
                    if motivo:
                        escritor.writerow([pasta.name, tipo, referencia, motivo])
                        item[campo] = ""  # Campo não será processado nas três variantes.
                        continue
                    with Image.open(crop) as antiga:
                        baseline = antiga.convert("RGB")
                    baseline = revisar_recorte(pasta.name, tipo, item, baseline, revisao, recortes_aplicados)
                    with Image.open(pasta / "imgs" / nome) as original:
                        nativo = recortar_original(original, item.get(tipo + "_BBox"))
                    item[campo] = nome + ".png"
                    for tratamento, imagem in zip(TRATAMENTOS, (baseline, nativo, aplicar_clahe(nativo), aplicar_ampliacao(nativo))):
                        destino = pasta_saida / tratamento / pasta.name / "imgs" / ("cropped_" + item[campo])
                        destino.parent.mkdir(parents=True, exist_ok=True)
                        if destino.exists():
                            raise ValueError(f"Imagem repetida no JSON: {destino}")
                        buffer = io.BytesIO()
                        imagem.save(buffer, format="PNG")
                        destino.write_bytes(buffer.getvalue())
                    elegiveis.append([pasta.name, tipo, "cropped_" + item[campo], normalizar(referencia)])
                    pixels[tuple(elegiveis[-1][:3])] = hashlib.sha256(str(nativo.size).encode() + nativo.tobytes()).hexdigest()
                    quantidade += 1
            for tratamento in TRATAMENTOS:
                destino = pasta_saida / tratamento / pasta.name / arquivo_json.name
                destino.parent.mkdir(parents=True, exist_ok=True)
                destino.write_text(json.dumps(dados, ensure_ascii=False, indent=2), encoding="utf-8")
    if not quantidade:
        raise ValueError("Nenhuma imagem elegível. Revise exclusoes.csv e os caminhos antes de executar as IAs.")
    esperadas = {(c['Pasta'], c['Tipo']) for c in revisao.get('correcoes', [])}
    if aplicadas != esperadas:
        raise ValueError(f'Correções sem amostra correspondente: {esperadas-aplicadas}')
    if recortes_aplicados != {(r['Pasta'],r['Tipo']) for r in revisao.get('recortes',[])}:
        raise ValueError('Revisão de recorte sem imagem correspondente.')
    separar_grupos(elegiveis, pixels)
    (pasta_saida / 'revisao_anotacoes.json').write_text(json.dumps(revisao, ensure_ascii=False, indent=2), encoding='utf-8')
    with (pasta_saida / "amostras_elegiveis.csv").open("w", encoding="utf-8", newline="") as f:
        escritor = csv.writer(f)
        escritor.writerow(["Pasta", "Tipo", "Arquivo", "Texto_Real", "Grupo", "Parte"])
        escritor.writerows(elegiveis)
    (pasta_saida / "preparo_concluido.txt").write_text(f"{quantidade} imagens por tratamento\n", encoding="utf-8")
    print(f"Pronto: {quantidade} imagens por tratamento em {pasta_saida}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets', type=Path, default=PASTA_ORIGINAL)
    parser.add_argument('--saida', type=Path, default=PASTA_SAIDA)
    parser.add_argument('--revisao', type=Path, default=RAIZ/'revisao_anotacoes.json')
    args = parser.parse_args()
    preparar_imagens(args.assets, args.saida, args.revisao)
