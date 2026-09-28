import csv
import hashlib
import json
import re
import unicodedata
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ================= CONFIGURAÇÕES =================

RAIZ = Path(__file__).resolve().parent
ASSETS = RAIZ / "assets"
IMAGENS = RAIZ / "recortes_margem10_v2" / "com_margem"

PASTAS = {
    "EasyOCR": "easyOCR",
    "PaddleOCR": "paddleOCR",
    "Qwen": "qwen",
    "Nemotron": "nemotrom",
}

# Deixe vazio se existir apenas um resultados_*.csv na pasta.
# Se houver vários, informe o nome do CSV que será comparado.
ARQUIVOS = {
    "EasyOCR": "",
    "PaddleOCR": "",
    "Qwen": "",
    "Nemotron": "",
}

RESPOSTAS = {"OK", "NOT_FOUND", "TRUNCATED"}
FALHAS = {"API_ERROR", "MODEL_ERROR", "INPUT_ERROR"}


# ================= FUNÇÕES =================

def normalizar(texto):
    texto = unicodedata.normalize("NFKC", texto).upper()
    return re.sub(r"[^A-Z0-9]", "", texto)


def ausente(texto):
    texto = unicodedata.normalize("NFKD", texto)
    texto = normalizar(
        "".join(c for c in texto if not unicodedata.combining(c))
    )
    sentinelas = {
        "NAO", "NA", "SEMNUMERO", "SEMNUMERACAO",
        "SEMCHASSI", "SEMMOTOR", "NAOPOSSUI",
        "NAOTEM", "NAOSEAPLICA", "INEXISTENTE",
    }
    return (
        texto in sentinelas
        or (bool(texto) and set(texto) == {"0"})
    )


def distancia(a, b):
    """Distância de Levenshtein: inserções, remoções e substituições."""
    if len(a) < len(b):
        a, b = b, a

    anterior = list(range(len(b) + 1))

    for i, x in enumerate(a, 1):
        atual = [i]

        for j, y in enumerate(b, 1):
            atual.append(min(
                atual[-1] + 1,
                anterior[j] + 1,
                anterior[j - 1] + (x != y),
            ))

        anterior = atual

    return anterior[-1]


def sha(arquivo):
    return hashlib.sha256(arquivo.read_bytes()).hexdigest()


def salvar_csv(caminho, linhas):
    if not linhas:
        return

    with caminho.open("x", newline="", encoding="utf-8-sig") as arquivo:
        writer = csv.DictWriter(arquivo, fieldnames=list(linhas[0]))
        writer.writeheader()
        writer.writerows(linhas)


def inventario():
    gabarito = {}
    auditoria = []
    hashes = {}

    encontrados = {
        p.relative_to(IMAGENS).as_posix(): p
        for p in IMAGENS.rglob("*")
        if p.is_file()
        and p.suffix.lower() in {".png", ".jpg", ".jpeg"}
    }

    for arquivo in sorted(
        ASSETS.rglob("dados_vistoria_Chassi-Motor-LABELED.json")
    ):
        hashes[str(arquivo.relative_to(ASSETS))] = sha(arquivo)

        pasta = arquivo.parent
        if pasta.name == "imgs":
            pasta = pasta.parent
        pasta = pasta.relative_to(ASSETS).as_posix()

        dados = json.loads(arquivo.read_text(encoding="utf-8-sig"))
        itens = dados if isinstance(dados, list) else [dados]

        for indice, item in enumerate(itens):
            linhas = (
                item["Dados do Veículo"]
                .replace("\r\n", "\n")
                .replace("\r", "\n")
                .split("\n")
            )

            if len(linhas) < 2 or any(t.strip() for t in linhas[2:]):
                raise ValueError(
                    f"Revisar as duas linhas do gabarito: {arquivo}"
                )

            for pos, tipo in enumerate(["Chassi", "Motor"]):
                original = linhas[pos].strip()

                if not original:
                    raise ValueError(
                        f"Referência vazia/ambígua: {arquivo}, {tipo}"
                    )

                motivo = ""
                referencia = normalizar(original)
                campo = "URL CHASSI" if tipo == "Chassi" else "URL Motor"
                nome = item.get(campo, "")

                chave = (
                    f"{pasta}/{indice:03d}_{tipo}_{Path(nome).stem}.png"
                    if nome else ""
                )

                if ausente(original):
                    motivo = "AUSENCIA_DECLARADA"

                elif not referencia:
                    raise ValueError(
                        f"Referência inválida: {arquivo}, {tipo}"
                    )

                elif chave not in encontrados:
                    if not item.get(f"{tipo}_BBox"):
                        motivo = "REFERENCIA_VALIDA_SEM_BBOX_E_RECORTE"
                    else:
                        raise ValueError(
                            f"Recorte esperado está faltando: {chave}"
                        )

                if motivo:
                    auditoria.append({
                        "Pasta": pasta,
                        "Tipo": tipo,
                        "Referencia_original": original,
                        "Motivo": motivo,
                    })
                    continue

                if chave in gabarito:
                    raise ValueError(f"Anotação duplicada: {chave}")

                gabarito[chave] = {
                    "Pasta": pasta,
                    "Tipo": tipo,
                    "Referencia": referencia,
                    "Referencia_original": original,
                }

    extras = set(encontrados) - set(gabarito)

    if extras:
        raise ValueError(
            f"Imagens sem gabarito elegível: {sorted(extras)[:5]}"
        )

    if not gabarito:
        raise ValueError(
            "Nenhuma imagem elegível. Confira assets e com_margem."
        )

    return gabarito, auditoria, hashes, encontrados


def carregar(modelo, chaves):
    pasta = RAIZ / PASTAS[modelo]

    candidatos = (
        [pasta / ARQUIVOS[modelo]]
        if ARQUIVOS[modelo]
        else sorted(pasta.glob("resultados_*.csv"))
    )

    if len(candidatos) != 1:
        raise ValueError(
            f"{modelo}: informe em ARQUIVOS um único CSV. "
            f"Encontrados: {[p.name for p in candidatos]}"
        )

    arquivo = candidatos[0]

    with arquivo.open(encoding="utf-8-sig", newline="") as f:
        leitor = csv.DictReader(f)

        obrigatorios = {
            "File_Name", "Extracted_Characters", "Raw_Response",
            "Status", "Error", "Seconds", "Truncated",
        }

        if not obrigatorios <= set(leitor.fieldnames or []):
            raise ValueError(f"Cabeçalho incompatível: {arquivo}")

        registros = list(leitor)

    dados = {}

    for registro in registros:
        if None in registro or any(
            valor is None for valor in registro.values()
        ):
            raise ValueError(f"Linha CSV incompleta: {arquivo}")

        chave = registro["File_Name"].replace("\\", "/")

        if chave in dados:
            raise ValueError(f"Imagem duplicada em {modelo}: {chave}")

        if registro["Status"] not in RESPOSTAS | FALHAS:
            raise ValueError(
                f"Status desconhecido: {registro['Status']}"
            )

        dados[chave] = registro

    if set(dados) != set(chaves):
        raise ValueError(
            f"{modelo}: faltam {len(set(chaves) - set(dados))} imagens; "
            f"sobram {len(set(dados) - set(chaves))}. "
            "Não misture datasets."
        )

    return dados, arquivo


def ic_acerto(linhas):
    """Bootstrap por veículo/pasta, mantendo chassi e motor juntos."""
    grupos = {}

    for registro in linhas:
        grupo = grupos.setdefault(registro["Pasta"], [0, 0])
        grupo[0] += registro["Acerto"]
        grupo[1] += 1

    if len(grupos) < 2:
        return np.nan, np.nan

    valores = np.array(list(grupos.values()), dtype=float)
    rng = np.random.default_rng(2026)

    amostras = rng.integers(
        0, len(valores), size=(2000, len(valores))
    )
    somas = valores[amostras].sum(axis=1)
    percentuais = 100 * somas[:, 0] / somas[:, 1]

    return np.percentile(percentuais, [2.5, 97.5])


def main():
    csv.field_size_limit(10_000_000)

    gabarito, auditoria, hashes, imagens = inventario()
    modelos = {}
    arquivos = {}

    for modelo in PASTAS:
        modelos[modelo], arquivos[modelo] = carregar(
            modelo, gabarito
        )

    # Mesmas imagens com resposta em TODOS os modelos.
    comum = {
        chave for chave in gabarito
        if all(
            modelos[modelo][chave]["Status"] in RESPOSTAS
            for modelo in modelos
        )
    }

    detalhes = []

    for modelo, dados in modelos.items():
        for chave, gt in gabarito.items():
            registro = dados[chave]
            respondeu = registro["Status"] in RESPOSTAS
            predicao = registro["Extracted_Characters"]
            bruto = registro["Raw_Response"]

            esperado = (
                ""
                if bruto.strip().upper() == "NOT_FOUND"
                else normalizar(bruto)
            )

            if respondeu and predicao != esperado:
                raise ValueError(
                    f"Normalização diferente em {modelo}: {chave}"
                )

            segundos = float(registro["Seconds"])

            if not np.isfinite(segundos) or segundos < 0:
                raise ValueError(
                    f"Tempo inválido: {modelo}, {chave}"
                )

            d = (
                distancia(gt["Referencia"], predicao)
                if respondeu else None
            )

            detalhes.append({
                "Modelo": modelo,
                "File_Name": chave,
                **gt,
                "Predicao": predicao,
                "Raw_Response": bruto,
                "Status": registro["Status"],
                "Error": registro["Error"],
                "Resposta_recebida": int(respondeu),
                "Acerto": int(
                    respondeu and predicao == gt["Referencia"]
                ),
                "Distancia": d,
                "CER": (
                    d / len(gt["Referencia"])
                    if respondeu else None
                ),
                "Seconds": segundos,
                "Attempts": registro.get("Attempts", ""),
                "Truncated": registro["Truncated"],
                "Conjunto_comum": int(chave in comum),
                "Saida_maior_100": int(len(predicao) > 100),
                "Contem_UNK": int("<unk>" in bruto.lower()),
                "Letra_nao_ASCII": int(any(
                    c.isalpha() and ord(c) > 127 for c in bruto
                )),
            })

    resumo = []

    for modelo in modelos:
        for tipo in ["Total", "Chassi", "Motor"]:
            linhas = [
                r for r in detalhes
                if r["Modelo"] == modelo
                and (tipo == "Total" or r["Tipo"] == tipo)
            ]

            if not linhas:
                continue

            validos = [
                r for r in linhas if r["Resposta_recebida"]
            ]
            pareados = [
                r for r in validos if r["Conjunto_comum"]
            ]

            def media(valores):
                return float(np.mean(valores)) if valores else np.nan

            def micro(registros):
                if not registros:
                    return np.nan
                return (
                    100 * sum(r["Distancia"] for r in registros)
                    / sum(len(r["Referencia"]) for r in registros)
                )

            baixo, alto = ic_acerto(linhas)

            resumo.append({
                "Modelo": modelo,
                "Tipo": tipo,
                "N": len(linhas),
                "Respostas": len(validos),
                "Falhas_tecnicas": len(linhas) - len(validos),
                "Cobertura_pct": 100 * len(validos) / len(linhas),
                "Acertos": sum(r["Acerto"] for r in linhas),
                "Acerto_pct": (
                    100 * sum(r["Acerto"] for r in linhas) / len(linhas)
                ),
                "IC95_inferior": baixo,
                "IC95_superior": alto,
                "CER_macro_pct": (
                    100 * media([r["CER"] for r in validos])
                ),
                "CER_micro_pct": micro(validos),
                "CER_mediana_pct": (
                    100 * float(np.median([r["CER"] for r in validos]))
                    if validos else np.nan
                ),
                "N_comum": len(pareados),
                "Acerto_comum_pct": (
                    100 * media([r["Acerto"] for r in pareados])
                ),
                "CER_macro_comum_pct": (
                    100 * media([r["CER"] for r in pareados])
                ),
                "CER_micro_comum_pct": micro(pareados),
                "Tempo_medio_respostas_s": media(
                    [r["Seconds"] for r in validos]
                ),
                "Tempo_total_registrado_s": sum(
                    r["Seconds"] for r in linhas
                ),
                "Saidas_maior_100": sum(
                    r["Saida_maior_100"] for r in linhas
                ),
                "Com_UNK": sum(r["Contem_UNK"] for r in linhas),
                "Letras_nao_ASCII": sum(
                    r["Letra_nao_ASCII"] for r in linhas
                ),
                "Truncadas": sum(
                    r["Truncated"].lower() == "true" for r in linhas
                ),
            })

    saida = RAIZ / (
        "comparacao_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    )
    saida.mkdir()

    salvar_csv(saida / "resumo.csv", resumo)
    salvar_csv(saida / "por_imagem.csv", detalhes)
    salvar_csv(saida / "exclusoes_dataset.csv", auditoria)

    proveniencia = {
        "condicao": "recortes_margem10_v2/com_margem",
        "script_sha256": sha(Path(__file__)),
        "csvs": {
            modelo: {
                "arquivo": str(arquivo),
                "sha256": sha(arquivo),
            }
            for modelo, arquivo in arquivos.items()
        },
        "jsons_sha256": hashes,
        "imagens_sha256": {
            chave: sha(arquivo)
            for chave, arquivo in imagens.items()
        },
        "bootstrap": {
            "unidade": "pasta/veiculo",
            "repeticoes": 2000,
            "seed": 2026,
        },
    }

    (saida / "proveniencia.json").write_text(
        json.dumps(proveniencia, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    graficos = [
        (
            "Acerto_pct",
            "Acerto exato (%) — todas as imagens",
            "acerto.png",
        ),
        (
            "Cobertura_pct",
            "Respostas recebidas (%)",
            "cobertura.png",
        ),
        (
            "CER_macro_pct",
            "CER médio (%) — respostas de cada modelo",
            "cer.png",
        ),
        (
            "CER_macro_comum_pct",
            "CER médio (%) — mesmas imagens com resposta",
            "cer_comum.png",
        ),
        (
            "Tempo_medio_respostas_s",
            "Tempo observado por resposta (s)",
            "tempo.png",
        ),
    ]

    for coluna, titulo, nome in graficos:
        fig, eixos = plt.subplots(1, 3, figsize=(15, 5))

        for ax, tipo in zip(eixos, ["Total", "Chassi", "Motor"]):
            grupo = [r for r in resumo if r["Tipo"] == tipo]

            for i, registro in enumerate(grupo):
                valor = registro[coluna]

                if np.isfinite(valor):
                    ax.bar(i, valor, color="#2878B5")
                    ax.annotate(
                        f"{valor:.2f}",
                        (i, valor),
                        xytext=(0, 4),
                        textcoords="offset points",
                        ha="center",
                        fontsize=8,
                    )
                else:
                    ax.text(i, 0, "N/D", ha="center")

            ax.set_xticks(
                range(len(grupo)),
                [r["Modelo"] for r in grupo],
                rotation=25,
            )
            ax.set_title(tipo)
            ax.set_ylim(bottom=0)

            if coluna in {"Acerto_pct", "Cobertura_pct"}:
                ax.set_ylim(0, 110)

            ax.margins(y=0.2)

        fig.suptitle(titulo)
        fig.tight_layout()
        fig.savefig(saida / nome, dpi=200)
        plt.close(fig)

    notas = """PROTOCOLO DA COMPARAÇÃO

Condição: recortes com margem de 10%. Não compara outros tratamentos.

Acerto exato: transcrição normalizada integralmente igual ao gabarito.
Denominador principal: todas as imagens elegíveis, incluindo falhas técnicas.
Cobertura: respostas recebidas / imagens elegíveis; não significa acerto.

CER: distância de Levenshtein / comprimento do gabarito.
Macro: média do CER por imagem.
Micro: soma das distâncias / soma dos comprimentos dos gabaritos.
Falhas técnicas têm CER indefinido, não zero.
Leitura vazia com resposta recebida tem CER 100%.
Respostas longas e truncadas permanecem na avaliação.
CER pode superar 100%.

Conjunto comum: imagens com resposta em todos os modelos.
Essa interseção pode sofrer viés de seleção e deve acompanhar a cobertura.
N/D ou nan: medida indefinida, não desempenho perfeito.

Normalização: NFKC, maiúsculas e A-Z/0-9, igual à dos scripts executados.
Letras não ASCII e saídas >100 são sinalizadas, sem correção automática.
O limite de 100 é somente um alerta diagnóstico, não uma exclusão.

Gabaritos originais: correções propostas anteriormente NÃO são aplicadas
automaticamente. A validade dos rótulos exige revisão humana.
Ausências declaradas não são exemplos OCR.
Referências válidas sem BBox/recorte são pendências do dataset.

IC95: bootstrap por pasta/veículo, 2000 repetições, seed 2026.
Assume pastas independentes; se um veículo aparece em várias pastas,
é necessário agrupá-las pelo mesmo veículo.
IC não certifica representatividade do dataset, nem diferença
estatisticamente significativa entre modelos.

Tempo inclui comunicação e reenvios quando registrados.
Exclui carga inicial dos modelos e pausas externas.
CPU local, Ollama e API remota não medem somente velocidade do modelo.

O CSV não comprova modelo exato, pesos ou prompt:
guarde também os scripts e as versões utilizadas.

Como este dataset já foi inspecionado, otimizações futuras devem ser
avaliadas em um conjunto de teste independente.
"""

    (saida / "LEIA_METODOLOGIA.txt").write_text(
        notas, encoding="utf-8"
    )

    for registro in resumo:
        if registro["Tipo"] == "Total":
            print(
                f"{registro['Modelo']}: "
                f"{registro['Acertos']}/{registro['N']} = "
                f"{registro['Acerto_pct']:.2f}% | "
                f"falhas: {registro['Falhas_tecnicas']} | "
                f"CER: {registro['CER_macro_pct']:.2f}%"
            )

    print(f"\nResultados em: {saida}")


if __name__ == "__main__":
    main()