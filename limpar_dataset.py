"""Limpa referências inválidas e recalcula CER com a mesma regra para todas as IAs."""
import csv
import re
import unicodedata
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
MODELOS = ("EasyOCR", "PaddleOCR", "Qwen_VL")
CABECALHO = ["Pasta", "Tipo", "Arquivo", "Texto_Real", "Pred_EasyOCR", "CER_Easy", "Tempo_Easy",
             "Pred_PaddleOCR", "CER_Paddle", "Tempo_Paddle", "Pred_Qwen", "CER_Qwen", "Tempo_Qwen",
             "Status_Easy", "Status_Paddle", "Status_Qwen"]


def normalizar(texto):
    """Retém A–Z e 0–9 em gabarito e predição; não troca O/0 nem I/1."""
    return re.sub(r"[^A-Z0-9]", "", unicodedata.normalize("NFKC", str(texto or "")).upper())


# Convenções explícitas de ausência, aplicadas somente ao gabarito.
SENTINELAS_AUSENCIA = {"", "NAO", "NA", "NENHUM", "NENHUMA", "NULL", "NONE",
    "SEMNUMERO", "SEMNUMERACAO", "SEMCHASSI", "SEMMOTOR", "NAOPOSSUI", "NAOTEM",
    "NAOSEAPLICA", "NA", "INEXISTENTE"}


def referencia_invalida(texto):
    if texto is None:
        return True
    valor = unicodedata.normalize("NFKD", str(texto)).encode("ascii", "ignore").decode()
    valor = re.sub(r"[^A-Z0-9]", "", valor.upper())
    return valor in SENTINELAS_AUSENCIA or bool(re.fullmatch("0+", valor))


def ler_referencias(item):
    """Exige duas posições explícitas: chassi na primeira, motor na segunda.

    Ausência pode ser representada por linha vazia ou sentinela. Ilegibilidade
    não é ausência: exige revisão do gabarito antes do experimento.
    """
    if not isinstance(item, dict) or not isinstance(item.get("Dados do Veículo"), str):
        raise ValueError("JSON: Dados do Veículo deve ser texto com duas linhas (chassi e motor).")
    partes = item["Dados do Veículo"].replace("\r\n", "\n").replace("\r", "\n").split("\n")
    if len(partes) < 2 or any(p.strip() for p in partes[2:]):
        raise ValueError("JSON: informe chassi na primeira linha e motor na segunda; use NÃO para ausência.")
    for valor in partes[:2]:
        if referencia_invalida(valor):
            continue
        if ":" in valor or not re.fullmatch(r"[A-Za-z0-9\s./*_-]+", valor) or normalizar(valor) in {
            "ERRO", "ILEGIVEL", "NAOLEGIVEL", "DESCONHECIDO", "NAOINFORMADO", "NAOIDENTIFICADO", "SEMIDENTIFICACAO"}:
            raise ValueError(f"Referência ambígua: {valor!r}. Revise o JSON; não excluir imagem difícil como ausência.")
    return partes[:2]


def ler_resultados(arquivo):
    """A última linha de cada imagem vale; retentativas não duplicam estatísticas."""
    resultados = {}
    if not Path(arquivo).exists():
        return resultados
    with open(arquivo, encoding="utf-8", newline="") as entrada:
        leitor = csv.reader(entrada)
        cabecalho = next(leitor, [])
        for linha in leitor:
            if len(linha) != len(cabecalho) or len(linha) not in (13, 16):
                raise ValueError(f"Linha incompleta em {arquivo}. Preserve o CSV e revise a última linha antes de retomar.")
            if len(linha) == 13:
                # No código antigo, Qwen retornava ERRO quando uma exceção ocorria.
                linha += ["OK", "OK", "ERRO_LEGADO" if linha[10] == "ERRO" else "OK"]
            resultados[tuple(linha[:3])] = linha
    return resultados


def limpar_dados_injustos(arquivo_entrada=None, arquivo_saida=None):
    import Levenshtein
    arquivo_entrada = Path(arquivo_entrada or RAIZ / "benchmark-ocr/1_benchmark_geral.csv")
    arquivo_saida = Path(arquivo_saida or arquivo_entrada.with_name("1_benchmark_geral_limpo.csv"))
    if not arquivo_entrada.exists():
        raise FileNotFoundError(arquivo_entrada)
    mantidas = 0
    with arquivo_saida.open("w", encoding="utf-8", newline="") as saida:
        escritor = csv.writer(saida)
        escritor.writerow(CABECALHO)
        for linha in ler_resultados(arquivo_entrada).values():
            if referencia_invalida(linha[3]) or not normalizar(linha[3]):
                continue
            linha[3] = normalizar(linha[3])
            for i in range(3):
                coluna = 4 + 3*i
                if linha[13+i] == "OK":
                    linha[coluna] = normalizar(linha[coluna])
                    linha[coluna+1] = Levenshtein.distance(linha[3], linha[coluna]) / len(linha[3])
                else:
                    linha[coluna+1] = ""  # Falha técnica não é uma transcrição errada.
            escritor.writerow(linha)
            mantidas += 1
    print(f"Limpeza: {mantidas} imagens em {arquivo_saida}")


if __name__ == "__main__":
    limpar_dados_injustos()
