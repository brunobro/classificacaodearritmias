# -*- coding: utf-8 -*-
'''
Induz uma máquina de aprendizado utilizando os atributos gerados por 1_extracao_atributos.py
'''
from sklearn.metrics import precision_score, accuracy_score, recall_score
import numpy as np
import joblib
import pickle
import config

'''
Lê os atributos salvos e os rótulos respectivos
'''
atributos_0 = pickle.load(open(config.DIR_ATRIBUTOS + 'atributos_validacao_0.pkl', 'rb'))
atributos_1 = pickle.load(open(config.DIR_ATRIBUTOS + 'atributos_validacao_1.pkl', 'rb'))
rotulos     = pickle.load(open(config.DIR_ATRIBUTOS + 'rotulos_validacao.pkl', 'rb'))

'''
Lê os pesos atribuídos a cada máquina induzida
'''
pesos = pickle.load(open(config.DIR_MODELOS + 'pesos.pkl', 'rb'))

'''
Carrega as máquinas de redução da dimensionalidade e de reconhecimento de arritmias
'''
pca_0 = joblib.load(config.DIR_MODELOS + 'pca_0.jl')
pca_1 = joblib.load(config.DIR_MODELOS + 'pca_1.jl')
clf_0 = joblib.load(config.DIR_MODELOS + 'clf_0.jl')
clf_1 = joblib.load(config.DIR_MODELOS + 'clf_1.jl')

'''
Redução da dimensionalidade utilizando a máquina induzida em 2_inducao.py
'''
atributos_reduzidos_0 = pca_0.transform(atributos_0)
atributos_reduzidos_1 = pca_1.transform(atributos_1)

del atributos_0, atributos_1

'''
Faz as predições utilizando a Máquina de Aprendizado induzida em 2_inducao.py
'''
predicoes_0 = clf_0.predict(atributos_reduzidos_0)
predicoes_1 = clf_1.predict(atributos_reduzidos_1)

predicoes_0_proba = clf_0.predict_proba(atributos_reduzidos_0)
predicoes_1_proba = clf_1.predict_proba(atributos_reduzidos_1)

'''
Implementa um comitê de voto suave
Calcula a soma, ponderada pelos pesos obtidos em 2_inducao.py, das probabilidades de predição
'''
soma_probabilidade_classe_negativa = pesos[0] * predicoes_0_proba[:,0] + pesos[1] * predicoes_1_proba[:,0]

soma_probabilidade_classe_positiva = pesos[0] * predicoes_0_proba[:,1] + pesos[1] * predicoes_1_proba[:,1]

#Para armazenar as predições do comitê
predicoes = np.zeros(soma_probabilidade_classe_negativa.shape[0])

for i in range(0, soma_probabilidade_classe_negativa.shape[0]):
	
	if soma_probabilidade_classe_negativa[i] < soma_probabilidade_classe_positiva[i]:
		predicoes[i] = 1

'''
Exibe a performance sobre os dados de teste
'''
print('\nResultados sobre o conjunto de teste - Canal 0')
print(' > Acurácia', accuracy_score(rotulos, predicoes_0))
print(' > Precisão', precision_score(rotulos, predicoes_0))
print(' > Recobrimento', recall_score(rotulos, predicoes_0))

print('\nResultados sobre o conjunto de teste - Canal 1')
print(' > Acurácia', accuracy_score(rotulos, predicoes_1))
print(' > Precisão', precision_score(rotulos, predicoes_1))
print(' > Recobrimento', recall_score(rotulos, predicoes_1))

print('\nResultados sobre o conjunto de teste - Comitê')
print(' > Acurácia', accuracy_score(rotulos, predicoes))
print(' > Precisão', precision_score(rotulos, predicoes))
print(' > Recobrimento', recall_score(rotulos, predicoes))