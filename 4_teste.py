# -*- coding: utf-8 -*-
'''
Induz uma máquina de aprendizado utilizando os atributos gerados por 1_extracao_atributos.py
'''
from sklearn.metrics import precision_score, accuracy_score, recall_score
from sklearn.preprocessing import normalize
import joblib
import pickle
import wfdb
from wfdb import processing
import os
import numpy as np
import config

'''
Cria uma string com os registros disponíveis para seleção
'''
registros_disponiveis = []
with os.scandir(config.DIR_DB_TESTE) as arquivos:
    for arquivo in arquivos:
        if arquivo.is_file() and os.path.splitext(arquivo)[1] == '.dat':
            registros_disponiveis.append(arquivo.name.replace('.dat',''))

registros_disponiveis_txt = ', '.join(map(str, registros_disponiveis))

'''
Carrega um registro ECG informado pelo usuário dentre aqueles disponíveis
'''
registro_escolhido = input('\nInforme um registro ECG dentre as opções (' + registros_disponiveis_txt + '): ')

if registro_escolhido not in registros_disponiveis:
	print('O registro escolhido não está disponível!')
	import sys
	sys.exit()

'''
Implementa a detecção dos complexos QRS, pois neste caso estamos supondo que não temos
essa informação
'''
data, info = wfdb.rdsamp(config.DIR_DB_TESTE + '/' + registro_escolhido, channels=[0])
ecg        = data[:, 0]
xqrs       = processing.XQRS(sig=ecg, fs=info['fs'])
xqrs.detect()

'''
Interage pelos picos (ondas R) detectados e extrai os atributos de cada complexo QRS
'''

#Se a taxa de amostragem é diferente de 360Hz, muda o tamanho da janela QRS
AMOSTRAS_ANTES_R = int(config.ANTES_R * info['fs'])
AMOSTRAS_APOS_R  = int(config.APOS_R * info['fs'])
JANELA_ECG       = AMOSTRAS_ANTES_R + AMOSTRAS_APOS_R

#Para armazenar os QRS e o rótulo respectivo
total_QRS     = len(xqrs.qrs_inds)
segmentos_QRS = np.zeros((total_QRS, JANELA_ECG))


for linha, onda_R in enumerate(xqrs.qrs_inds):

    #Segmento do ciclo cardíaco
    ecg_seg = ecg[onda_R - AMOSTRAS_ANTES_R:onda_R + AMOSTRAS_APOS_R]
    
    # garante que somente os atributos do tamanho correto sejam inseridos
    # pode ocorrer que os atributos do último ciclo sejam menores, devido ao corte no sinal
    if len(ecg_seg) == JANELA_ECG:

        #Adiciona o segmento (clico cardico)
        segmentos_QRS[linha, :] = ecg_seg


'''
Carrega as máquinas de redução da dimensionalidade e de reconhecimento de arritmias
apenas do canal 0
'''
pca = joblib.load(config.DIR_MODELOS + 'pca_0.jl')
clf = joblib.load(config.DIR_MODELOS + 'clf_0.jl')

'''
Redução da dimensionalidade utilizando a máquina induzida em 2_inducao.py
'''
segmentos_QRS_D = pca.transform(segmentos_QRS)

'''
Faz as predições utilizando a Máquina de Aprendizado induzida em 2_inducao.py
'''
predicoes      = clf.predict(segmentos_QRS_D)
probabilidades = clf.predict_proba(segmentos_QRS_D)

'''
Cria arquivos html com os resultados
'''
todos_codigos_html = []
todos_codigos_js   = []

#cria uma lista para indexar as amostras do sinal de ECG
x = np.arange(1, JANELA_ECG + 1, 1)
x = ', '.join(map(str, x))

#para armazenar os códigos javascript e html
codigos_html = ''
codigos_js   = ''

#Para controlar a quantidade de segmentos exibidos por página html
#pois se todos forem inseridos em um única página ocorre overflow de memória
pg = 1

for i, segmento in enumerate(segmentos_QRS):

	tipo_batimento = 'NORMAL'
	probabilidade = np.round(probabilidades[i][0] * 100, 4)
	if predicoes[i] == 1:
		tipo_batimento = 'ARRITIMIA'
		probabilidade = np.round(probabilidades[i][1] * 100, 4)

	legenda = tipo_batimento + ' (' + str(probabilidade) + ' %)'

	#cria uma lista das amplitudes do sinal de ECG
	y = ', '.join(map(str, segmento))

	id_div = 'seg '+ str(i)
	codigos_html += '<div id="'+ id_div + '" class="div"></div>'
	codigos_js += 'var data = [{x:['+ x +'], y:['+ y +']}];var layout = {title:{text:"' + legenda + '"}};Plotly.newPlot("'+ id_div + '", data, layout);'

	pg += 1

	if pg > 50 or i == len(segmentos_QRS) - 1:
		todos_codigos_html.append(codigos_html)
		todos_codigos_js.append(codigos_js)
		
		pg           = 1
		codigos_html = ''
		codigos_js   = ''


#Lê o template html e faz as substituições
html_template       = open('html/template.html', 'r')
pg_codigos_original = html_template.read()
html_template.close()

#Cria a paginação
paginacao = '<ul>'
for i in range(0, len(todos_codigos_html)):
	c = ' class="lim"'
	if i == len(todos_codigos_html) - 1:
		c = ''

	paginacao += '<li' + c + '><a href="saida' + str(i) + '.html"> ' + str(i + 1) + ' </a></li>'

paginacao += '</ul>'

#Informação sobre as proporções de batimentos NORMAIS e ARRÍTMICOS
proporcao_NORMAL   = str(np.round((len(np.where(predicoes == 0)[0])/total_QRS) * 100, 4)) + ' %'
proporcao_ARRITMIA = str(np.round((len(np.where(predicoes == 1)[0])/total_QRS) * 100, 4)) + ' %'

#Cria várias páginas html
for i, codigo_html in enumerate(todos_codigos_html):

	codigo_js = todos_codigos_js[i]

	pg_codigos_subs = pg_codigos_original
	pg_codigos_subs = pg_codigos_subs.replace('#CODIGOS_HTML#', codigo_html)
	pg_codigos_subs = pg_codigos_subs.replace('#CODIGOS_JS#', codigo_js)
	pg_codigos_subs = pg_codigos_subs.replace('#CODIGOS_PG#', paginacao)
	pg_codigos_subs = pg_codigos_subs.replace('#REGISTRO#', registro_escolhido)
	pg_codigos_subs = pg_codigos_subs.replace('#TOTAL_NORMAL#', proporcao_NORMAL)
	pg_codigos_subs = pg_codigos_subs.replace('#TOTAL_ARRITMIA#', proporcao_ARRITMIA)
	pg_codigos_subs = pg_codigos_subs.replace('#TOTAL_BATIMENTOS#', str(total_QRS))

	saida_html = open('html/saidas/saida' + str(i) +'.html', 'w+')
	saida_html.write(pg_codigos_subs)
	saida_html.close()
