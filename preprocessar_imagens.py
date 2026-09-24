"""Prepara três cópias do dataset mantendo o formato de pastas/JSON do projeto."""
import csv
import json
import io
import math
from pathlib import Path
from PIL import Image
from limpar_dataset import referencia_invalida, normalizar, ler_referencias

RAIZ = Path(__file__).resolve().parent
PASTA_ORIGINAL = RAIZ / "assets"  # Organização mostrada nas capturas de tela
PASTA_SAIDA = RAIZ / "dataset_tratado"
TRATAMENTOS = ("sem_tratamento", "nativo", "clahe")


def recortar_original(imagem, pontos):
    """Reutiliza os quatro pontos anotados, removendo somente o limite de 280px."""
    if not isinstance(pontos, list) or len(pontos) != 4 or any(len(p) != 2 for p in pontos):
        raise ValueError("BBox sem quatro pontos; revise a anotação.")
    if any(not (0 <= x <= imagem.width and 0 <= y <= imagem.height) for x, y in pontos):
        raise ValueError("BBox fora da foto original.")
    tl, tr, br, bl = pontos
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


def preparar_imagens(pasta_original=PASTA_ORIGINAL, pasta_saida=PASTA_SAIDA):
    pasta_original, pasta_saida = Path(pasta_original), Path(pasta_saida)
    if not pasta_original.is_dir():
        raise FileNotFoundError(pasta_original)
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
                    with Image.open(pasta / "imgs" / nome) as original:
                        nativo = recortar_original(original, item.get(tipo + "_BBox"))
                    with Image.open(crop) as antiga:
                        baseline = antiga.convert("RGB")
                    item[campo] = nome + ".png"
                    for tratamento, imagem in zip(TRATAMENTOS, (baseline, nativo, aplicar_clahe(nativo))):
                        destino = pasta_saida / tratamento / pasta.name / "imgs" / ("cropped_" + item[campo])
                        destino.parent.mkdir(parents=True, exist_ok=True)
                        if destino.exists():
                            raise ValueError(f"Imagem repetida no JSON: {destino}")
                        buffer = io.BytesIO()
                        imagem.save(buffer, format="PNG")
                        destino.write_bytes(buffer.getvalue())
                    elegiveis.append([pasta.name, tipo, "cropped_" + item[campo], normalizar(referencia)])
                    quantidade += 1
            for tratamento in TRATAMENTOS:
                destino = pasta_saida / tratamento / pasta.name / arquivo_json.name
                destino.parent.mkdir(parents=True, exist_ok=True)
                destino.write_text(json.dumps(dados, ensure_ascii=False, indent=2), encoding="utf-8")
    if not quantidade:
        raise ValueError("Nenhuma imagem elegível. Revise exclusoes.csv e os caminhos antes de executar as IAs.")
    with (pasta_saida / "amostras_elegiveis.csv").open("w", encoding="utf-8", newline="") as f:
        escritor = csv.writer(f)
        escritor.writerow(["Pasta", "Tipo", "Arquivo", "Texto_Real"])
        escritor.writerows(elegiveis)
    (pasta_saida / "preparo_concluido.txt").write_text(f"{quantidade} imagens por tratamento\n", encoding="utf-8")
    print(f"Pronto: {quantidade} imagens por tratamento em {pasta_saida}")


if __name__ == "__main__":
    preparar_imagens()
