"""Regressões metodológicas sem pesos de IA nem servidor Ollama.

Os adaptadores são simulados. Estes testes não medem a acurácia dos modelos.
"""
import csv
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import patch
from PIL import Image
from limpar_dataset import referencia_invalida, ler_referencias, normalizar
import preprocessar_imagens as preparo

sys.path.insert(0, str(Path(__file__).parent/'benchmark-ocr'))
spec = importlib.util.spec_from_file_location('benchmark', Path(__file__).parent/'benchmark-ocr/main.py')
main = importlib.util.module_from_spec(spec)
spec.loader.exec_module(main)


def amostra(pasta='a', real='ABC123', tipo='Chassi'):
    return dict(Pasta=pasta, Tipo=tipo, Arquivo='crop.png', Texto_Real=real,
                Grupo=pasta, Parte='desenvolvimento')


class Validacao(unittest.TestCase):
    def test_ausencia_nao_e_ilegibilidade(self):
        for valor in ['NÃO', '00', '0 0', '0.0', '00-00', 'N/A', 'SEM NÚMERO', '', None]:
            self.assertTrue(referencia_invalida(valor))
        for valor in ['001234', 'AB00CD', 'ABC123', 'ERRO']:
            self.assertFalse(referencia_invalida(valor))
        for dado in [None, 'ABC123', 'ABC123\nXYZ\nEXTRA', 'ERRO\nNÃO', 'ILEGÍVEL\nNÃO']:
            with self.assertRaises(ValueError):
                ler_referencias({'Dados do Veículo': dado})
        self.assertEqual(normalizar(' a-O01_i '), 'AO01I')

    def test_bbox_invalida(self):
        im = Image.new('RGB', (100, 30))
        for pontos in [None, [1,2,3,4], [[0,0],[100,30],[100,0],[0,30]],
                       [[0,0],[101,0],[100,30],[0,30]]]:
            with self.assertRaises(ValueError):
                preparo.recortar_original(im, pontos)

    def test_recorte_remove_campo_vizinho_para_todas_condicoes(self):
        im = Image.new('RGB', (100,30), 'white')
        for x in range(50,100):
            for y in range(30): im.putpixel((x,y),(0,0,0))
        pontos = [[0,0],[100,0],[100,30],[0,30]]
        item = {'Chassi_BBox': pontos}
        revisao = {'recortes':[dict(Pasta='a',Tipo='Chassi',bbox_antes=pontos,fracao_horizontal=.5)]}
        baseline = preparo.revisar_recorte('a','Chassi',item,im,revisao,set())
        nativo = preparo.recortar_original(im,item['Chassi_BBox'])
        self.assertEqual(baseline.size,(50,30))
        self.assertEqual(nativo.size,(50,30))
        self.assertEqual(baseline.tobytes(),nativo.tobytes())
        self.assertEqual(nativo.getpixel((25,15)),(255,255,255))

    def test_congelamento_configuracao(self):
        with tempfile.TemporaryDirectory() as tmp:
            raiz=Path(tmp);dataset=raiz/'dataset';img=dataset/'nativo/a/imgs/crop.png'
            img.parent.mkdir(parents=True);Image.new('RGB',(10,10)).save(img)
            args=SimpleNamespace(dataset=dataset,saida=raiz/'resultados',parte='desenvolvimento',
                modelos=['EasyOCR'],tratamentos=['nativo'],qwen='nao_usado',hardware='teste')
            original=main.configurar(args,[amostra()])
            self.assertEqual(main.configurar(args,[amostra()]),original)
            Image.new('RGB',(10,10),'red').save(img)
            with self.assertRaisesRegex(ValueError,'Configuração mudou'):
                main.configurar(args,[amostra()])
            self.assertEqual(json.loads((args.saida/'configuracao.json').read_text()),original)

    def test_grupos_transitivos_sem_vazamento(self):
        linhas = [['a','Chassi','a.png','CH1'], ['a','Motor','b.png','MO1'],
                  ['b','Motor','c.png','MO1'], ['b','Chassi','d.png','CH2'],
                  ['c','Chassi','e.png','CH2'], ['d','Chassi','f.png','CH3']]
        pixels = {tuple(r[:3]):str(i) for i,r in enumerate(linhas)}
        pixels[tuple(linhas[-1][:3])] = pixels[tuple(linhas[0][:3])]
        saida = preparo.separar_grupos([r[:] for r in linhas], pixels)
        inversa = preparo.separar_grupos([r[:] for r in reversed(linhas)], pixels)
        self.assertEqual(len({tuple(r[4:]) for r in saida}), 1)
        self.assertEqual(sorted(saida), sorted(inversa))

    def test_preparo_corrige_sem_alterar_fonte(self):
        with tempfile.TemporaryDirectory() as tmp:
            raiz=Path(tmp); assets=raiz/'assets'; saida=raiz/'dataset'
            for nome,motor in [('a','DEF456'),('b','SEM NÚMERO'),('c','GHI789')]:
                pasta=assets/nome; imgs=pasta/'imgs'; imgs.mkdir(parents=True)
                for arq in ['chassi.jpg','motor.jpg']:
                    Image.new('RGB',(100,30),'gray').save(imgs/arq)
                    if arq=='chassi.jpg' or nome=='a':
                        Image.new('RGB',(50,15),'gray').save(imgs/('cropped_'+arq))
                dados={'Dados do Veículo':'ABC123\n'+motor,'URL CHASSI':'chassi.jpg','URL Motor':'motor.jpg',
                       'Chassi_BBox':[[0,0],[100,0],[100,30],[0,30]],'Motor_BBox':[[0,0],[100,0],[100,30],[0,30]]}
                (pasta/'dados_vistoria_Chassi-Motor-LABELED.json').write_text(json.dumps(dados),encoding='utf-8')
            fonte=assets/'a/dados_vistoria_Chassi-Motor-LABELED.json'; antes=fonte.read_bytes()
            revisao=raiz/'revisao.json'; revisao.write_text(json.dumps({'correcoes':[dict(Pasta='a',Tipo='Chassi',antes='ABC123',depois='ABC0123')]}))
            preparo.preparar_imagens(assets,saida,revisao)
            self.assertEqual(fonte.read_bytes(),antes)
            with (saida/'amostras_elegiveis.csv').open() as f:
                rows=list(csv.DictReader(f))
            self.assertEqual(len(rows),4)
            self.assertEqual(next(r['Texto_Real'] for r in rows if r['Pasta']=='a' and r['Tipo']=='Chassi'),'ABC0123')
            for t in preparo.TRATAMENTOS:
                self.assertEqual(len(list((saida/t).rglob('*.png'))),4)
            with (saida/'exclusoes.csv').open() as f:
                motivos={r['Motivo'] for r in csv.DictReader(f)}
            self.assertEqual(motivos,{'referencia_invalida','crop_ausente'})
            with Image.open(saida/'sem_tratamento/a/imgs/cropped_chassi.jpg.png') as im:
                self.assertEqual(im.size,(50,15))
            with Image.open(saida/'ampliado/a/imgs/cropped_chassi.jpg.png') as im:
                self.assertEqual(im.size,(232,92))
            with self.assertRaises(FileExistsError):preparo.preparar_imagens(assets,saida)

    def test_retoma_por_modelo_limita_falhas_e_preserva_resposta(self):
        with tempfile.TemporaryDirectory() as tmp:
            pasta=Path(tmp);log=pasta/'tentativas.jsonl';r=amostra();calls=[]
            def ler(p):
                calls.append(str(p))
                if len(calls)==1:raise RuntimeError('OOM simulado')
                return dict(texto_bruto='a b c 123\n',metadados={'raw':'preservado'})
            main.executar_amostras([r],pasta,log,ler)
            main.executar_amostras([r],pasta,log,ler)
            self.assertEqual(len(calls),2)
            history=main.historico(log)[main.chave(r)]
            self.assertEqual(main.resultado_final(history)['texto_bruto'],'a b c 123\n')
            self.assertEqual(len(history),2)
            outro=pasta/'falha.jsonl'
            def falhar(p):raise RuntimeError('sempre')
            main.executar_amostras([r],pasta,outro,falhar)
            main.executar_amostras([r],pasta,outro,falhar)
            self.assertEqual(len(main.historico(outro)[main.chave(r)]),2)

    def test_interrupcao_consume_tentativa(self):
        with tempfile.TemporaryDirectory() as tmp:
            pasta=Path(tmp);log=pasta/'t.jsonl';r=amostra()
            main.registrar(log,dict(chave=list(main.chave(r)),tentativa=1,evento='inicio'))
            calls=[]
            def ler(p):calls.append(p);return {'texto_bruto':'ERRADO'}
            main.executar_amostras([r],pasta,log,ler)
            main.executar_amostras([r],pasta,log,ler)
            self.assertEqual(len(calls),1)  # Leitura errada é sucesso técnico e não se repete.
            with log.open('a') as f:f.write('{"incompleta":')
            with self.assertRaises(ValueError):main.historico(log)

    def test_metricas_denominador_fixo_vazio_e_repeticao(self):
        with tempfile.TemporaryDirectory() as tmp:
            saida=Path(tmp);(saida/'nativo').mkdir();amostras=[amostra('a'),amostra('b'),amostra('c')]
            args=SimpleNamespace(saida=saida,tratamentos=['nativo'],modelos=['Qwen_VL'])
            log=saida/'nativo/tentativas_Qwen_VL.jsonl'
            def ler(p):
                if p.parent.parent.name=='c':raise RuntimeError('falha')
                return {'texto_bruto':'ABC123' if p.parent.parent.name=='a' else ''}
            main.executar_amostras(amostras,saida,log,ler)
            rows=main.consolidar(args,amostras)
            self.assertEqual([r['CER'] for r in rows],[0,1,''])
            s=main.estatisticas(rows,{main.chave(r) for r in amostras[:2]})
            self.assertEqual((s['N_elegivel'],s['Sucessos'],s['Falhas'],s['Acertos']),(3,2,1,1))
            self.assertAlmostEqual(s['Acerto_pct'],100/3)
            self.assertEqual(s['CER_macro_sucessos_pct'],50)
            self.assertEqual(s['N_comum'],2)
            # CER>1 e saída truncada permanecem pontuados, sem recorte artificial.
            outro=saida/'outro';(outro/'nativo').mkdir(parents=True);args.saida=outro
            main.executar_amostras([amostras[0]],saida,outro/'nativo/tentativas_Qwen_VL.jsonl',lambda p:dict(texto_bruto='X'*100,truncada=True))
            row=main.consolidar(args,[amostras[0]])[0]
            self.assertGreater(row['CER'],1)
            self.assertTrue(row['Truncada'])
            self.assertEqual(row['Tentativas'],1)

    def test_relatorio_bloqueia_incompleto(self):
        with tempfile.TemporaryDirectory() as tmp:
            args=SimpleNamespace(saida=Path(tmp),tratamentos=['nativo'],modelos=['EasyOCR'])
            with self.assertRaisesRegex(RuntimeError,'pendentes'):main.consolidar(args,[amostra()])

    def test_qwen_nao_recebe_gabarito_e_limita_saida(self):
        with tempfile.TemporaryDirectory() as tmp:
            im=Path(tmp)/'im.png';Image.new('RGB',(100,30),'white').save(im)
            with patch.object(main,'api_ollama',return_value={'done':True,'done_reason':'length','message':{'role':'assistant','content':'XYZ'}}) as api:
                r=main.criar_leitor('Qwen_VL','qwen-teste')(im)
                corpo=api.call_args.args[1]
                self.assertEqual(corpo['options']['num_predict'],64)
                self.assertNotIn('Texto_Real',json.dumps(corpo))
                self.assertEqual(r['texto_bruto'],'XYZ')
                self.assertTrue(r['truncada'])

    def test_trava_de_execucao(self):
        with tempfile.TemporaryDirectory() as tmp:
            pasta=Path(tmp)
            with main.bloquear_execucao(pasta):
                with self.assertRaises(RuntimeError):
                    with main.bloquear_execucao(pasta):pass
            self.assertFalse((pasta/'execucao.lock').exists())

    def test_fluxo_completo_graficos_sem_ia(self):
        with tempfile.TemporaryDirectory() as tmp:
            saida=Path(tmp);rows=[amostra(),amostra('b','DEF456','Motor')]
            args=SimpleNamespace(saida=saida,tratamentos=list(preparo.TRATAMENTOS),modelos=list(main.MODELOS),parte='desenvolvimento',qwen='qwen-simulado')
            for t in args.tratamentos:
                (saida/t).mkdir()
                for m in args.modelos:
                    main.executar_amostras(rows,saida,saida/t/f'tentativas_{m}.jsonl',lambda p:dict(texto_bruto='ABC123'))
            resultados=main.consolidar(args,rows)
            main.relatorios(args,resultados)
            self.assertEqual(len(list(saida.glob('*.png'))),3)
            with (saida/'comparacao_tratamentos.csv').open() as f:
                resumo=list(csv.DictReader(f))
            self.assertEqual(len(resumo),36)
            for r in resumo:
                if r['Tipo']=='Chassi':self.assertEqual(float(r['Acerto_pct']),100)
                if r['Tipo']=='Motor':self.assertEqual(float(r['Acerto_pct']),0)
                if r['Tipo']=='Total':self.assertEqual(float(r['Acerto_pct']),50)


if __name__=='__main__':
    unittest.main()
