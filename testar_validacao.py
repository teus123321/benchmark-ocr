"""Testes com dados sintéticos; não baixa pesos nem chama modelos reais."""
import csv
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from PIL import Image
from limpar_dataset import referencia_invalida, ler_referencias, normalizar, limpar_dados_injustos, ler_resultados
import preprocessar_imagens as preparo


class Validacao(unittest.TestCase):
    def test_sentinelas_e_estrutura(self):
        for x in ['NÃO', '00', '0 0', '0.0', '00-00', 'N/A', 'SEM NÚMERO', 'SEM MOTOR', '', None]:
            self.assertTrue(referencia_invalida(x), repr(x))
        for x in ['001234', 'AB00CD', 'ABC123']:
            self.assertFalse(referencia_invalida(x))
        for item in [{}, {'Dados do Veículo': None}, {'Dados do Veículo': 'ABC123'},
                     {'Dados do Veículo': 'ABC123\nDEF456\nEXTRA'},
                     {'Dados do Veículo': 'Chassi: ABC123\nNÃO'},
                     {'Dados do Veículo': 'ILEGÍVEL\nNÃO'}]:
            with self.assertRaises(ValueError):
                ler_referencias(item)
        self.assertEqual(ler_referencias({'Dados do Veículo': 'ABC123\r\nNÃO'}), ['ABC123', 'NÃO'])

    def test_fluxo_sem_ia_real(self):
        sys.path.insert(0, str(Path(__file__).parent/'benchmark-ocr'))
        spec = importlib.util.spec_from_file_location('benchmark_teste', Path(__file__).parent/'benchmark-ocr/main.py')
        main = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(main)
        with tempfile.TemporaryDirectory() as tmp:
            raiz = Path(tmp); assets = raiz/'assets'; saida = raiz/'dataset'
            # 3 chassi válidos, 1 motor válido, 1 motor ausente e 1 crop de motor faltante.
            for pasta, motor in [('a', 'DEF456'), ('b', 'SEM NÚMERO'), ('c', 'GHI789')]:
                imgs = assets/pasta/'imgs'; imgs.mkdir(parents=True)
                for nome in ['chassi.jpg', 'motor.jpg']:
                    Image.new('RGB', (100, 30), 'gray').save(imgs/nome)
                    if nome == 'chassi.jpg' or pasta == 'a':
                        Image.new('RGB', (50, 15), 'gray').save(imgs/('cropped_'+nome))
                dados = {'Dados do Veículo': 'ABC123\n'+motor, 'URL CHASSI': 'chassi.jpg', 'URL Motor': 'motor.jpg',
                         'Chassi_BBox': [[0,0],[100,0],[100,30],[0,30]], 'Motor_BBox': [[0,0],[100,0],[100,30],[0,30]]}
                (assets/pasta/'dados_vistoria_Chassi-Motor-LABELED.json').write_text(json.dumps(dados), encoding='utf-8')
            preparo.preparar_imagens(assets, saida)
            with (saida/'exclusoes.csv').open(encoding='utf-8') as f:
                self.assertEqual({r['Motivo'] for r in csv.DictReader(f)}, {'referencia_invalida', 'crop_ausente'})
            main.PASTA_DATASET = saida
            chamadas=[]
            def leitura(caminho):
                chamadas.append(caminho)
                return 'DEF456' if 'motor' in Path(caminho).name else 'ABC123'
            main.ler_com_easyocr = leitura
            main.ler_com_paddle = leitura
            main.ler_com_ollama = lambda caminho, modelo: leitura(caminho)
            resultados = raiz/'resultados'
            for tratamento in preparo.TRATAMENTOS:
                d=resultados/tratamento; d.mkdir(parents=True)
                bruto=d/'1_benchmark_geral.csv'; limpo=d/'1_benchmark_geral_limpo.csv'
                main.executar_benchmark(saida/tratamento, bruto)
                self.assertEqual(len(ler_resultados(bruto)), 4)
                main.executar_benchmark(saida/tratamento, bruto)
                limpar_dados_injustos(bruto, limpo)
            self.assertEqual(len(chamadas), 36) # Ausências e retomadas não chamam IA.
            self.assertEqual(main.mapear_detalhes_erro('ABC123', '')[1], 1)
            main.comparar_tratamentos(resultados)
            self.assertEqual(len(list(resultados.rglob('*.png'))), 12)
            alvo=resultados/'clahe/1_benchmark_geral_limpo.csv'
            with alvo.open(encoding='utf-8', newline='') as f: linhas=list(csv.reader(f))
            with alvo.open('w', encoding='utf-8', newline='') as f: csv.writer(f).writerows(linhas[:-1])
            with self.assertRaisesRegex(RuntimeError, 'incompletos'):
                main.comparar_tratamentos(resultados)
            # Amostra completa com falha técnica sai da comparação em todas as condições.
            linhas[-1][15]='RuntimeError: simulado'; linhas[-1][11]=''
            with alvo.open('w', encoding='utf-8', newline='') as f: csv.writer(f).writerows(linhas)
            main.comparar_tratamentos(resultados)
            with (resultados/'comparacao_tratamentos.csv').open(encoding='utf-8') as f:
                resumo=list(csv.DictReader(f))
            self.assertEqual({r['N_comum'] for r in resumo if r['Tipo']=='Chassi'}, {'2'})
            self.assertEqual({r['N_comum'] for r in resumo if r['Tipo']=='Motor'}, {'1'})


if __name__ == '__main__':
    unittest.main()
