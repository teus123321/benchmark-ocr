"""Exporta acertos do formato longo v3, sempre com a pasta de origem."""
import csv
from pathlib import Path


def gerar_relatorio_acertos(entrada, saida):
    with Path(entrada).open(encoding='utf-8', newline='') as f:
        leitor = csv.DictReader(f)
        campos = leitor.fieldnames
        linhas = [r for r in leitor if r['Status']=='OK' and r['Predicao']==r['Texto_Real']]
    with Path(saida).open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=campos)
        w.writeheader()
        w.writerows(linhas)
