import csv
import os

# Aponta para o ficheiro limpo e garante a pasta correta
arquivo_entrada = 'benchmark-ocr/1_benchmark_geral_limpo.csv'
arquivo_saida = 'benchmark-ocr/3_acertos_exatos.csv'

def gerar_relatorio_acertos(entrada=None, saida=None):
    entrada = entrada or arquivo_entrada
    saida = saida or arquivo_saida
    if not os.path.exists(entrada):
        print(f"Erro: O arquivo '{entrada}' não foi encontrado.")
        return

    with open(entrada, 'r', encoding='utf-8') as f_in, \
         open(saida, 'w', newline='', encoding='utf-8') as f_out:
        
        leitor = csv.reader(f_in)
        escritor = csv.writer(f_out)
        
        # Pula o cabeçalho do arquivo original
        next(leitor, None)
        
        # NOVO CABEÇALHO: Exibe o texto real e o predito lado a lado para auditoria acadêmica
        escritor.writerow(["Modelo", "Tipo", "Arquivo", "Texto_Real", "Texto_Predito", "Tempo_Gasto_(s)"])
        
        qtd_acertos = 0
        for linha in leitor:
            if not linha or len(linha) < 13:
                continue
            
            tipo, arquivo, texto_real = linha[1], linha[2], linha[3]
            
            pred_easy, tempo_easy = linha[4], linha[6]
            pred_paddle, tempo_paddle = linha[7], linha[9]
            pred_qwen, tempo_qwen = linha[10], linha[12]
            
            if pred_easy == texto_real and (len(linha) < 16 or linha[13] == "OK"):
                escritor.writerow(["EasyOCR", tipo, arquivo, texto_real, pred_easy, tempo_easy])
                qtd_acertos += 1
                
            if pred_paddle == texto_real and (len(linha) < 16 or linha[14] == "OK"):
                escritor.writerow(["PaddleOCR", tipo, arquivo, texto_real, pred_paddle, tempo_paddle])
                qtd_acertos += 1
                
            if pred_qwen == texto_real and (len(linha) < 16 or linha[15] == "OK"):
                escritor.writerow(["Qwen_VL", tipo, arquivo, texto_real, pred_qwen, tempo_qwen])
                qtd_acertos += 1

    print(f"[OK] Sucesso! Arquivo '{saida}' gerado com {qtd_acertos} acertos exatos registrados (base de dados limpa).")

if __name__ == "__main__":
    gerar_relatorio_acertos()
