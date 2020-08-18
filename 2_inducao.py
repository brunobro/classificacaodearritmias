# -*- coding: utf-8 -*-
'''
Induz uma máquina de aprendizado utilizando os atributos gerados por 1_extracao_atributos.py
'''
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier #KNN
from sklearn.metrics import precision_score, accuracy_score, recall_score, confusion_matrix
import numpy as np
import joblib
import pickle
import config

'''
Lê os atributos salvos
'''
atributos_0 = pickle.load(open(config.DIR_ATRIBUTOS + 'atributos_inducao_0.pkl', 'rb'))
atributos_1 = pickle.load(open(config.DIR_ATRIBUTOS + 'atributos_inducao_1.pkl', 'rb'))
rotulos     = pickle.load(open(config.DIR_ATRIBUTOS + 'rotulos_inducao.pkl', 'rb'))

'''
Treina uma máquina para redução da dimensionalidade
'''
pca_0 = PCA(n_components=config.N_ATRIBUTOS).fit(atributos_0, rotulos)
atributos_reduzidos_0 = pca_0.transform(atributos_0)

pca_1 = PCA(n_components=config.N_ATRIBUTOS).fit(atributos_1, rotulos)
atributos_reduzidos_1 = pca_1.transform(atributos_1)

del atributos_0, atributos_1

'''
Treinamento da Máquina de Aprendizado e predições sobre o mesmo conjunto
'''
clf_0 = KNeighborsClassifier(n_neighbors=20).fit(atributos_reduzidos_0, rotulos)
predicoes_0 = clf_0.predict(atributos_reduzidos_0)

clf_1 = KNeighborsClassifier(n_neighbors=20).fit(atributos_reduzidos_1, rotulos)
predicoes_1 = clf_1.predict(atributos_reduzidos_1)

'''
Com base na quantidade de falsos positivos (FP) e falsos negativos (FN) atribui pesos as máquinas de inferências
'''
tn, FP_0, FN_0, tp = confusion_matrix(rotulos, predicoes_0).ravel()
tn, FP_1, FN_1, tp = confusion_matrix(rotulos, predicoes_1).ravel()

print('\n\nFalsos positivos e negativos - Canal 0')
print(' > ', FP_0, ' e ', FN_0)
print('Falsos positivos e negativos - Canal 1')
print(' > ', FP_1, ' e ', FN_1)

#calcula os totais para obter a proporção de cada canal
#em ralação ao total de falsas detecções
total_0  = FP_0 + FN_0
total_1  = FP_1 + FN_1
total_01 = total_0 + total_1

#subtrai 1, pois pretende-se dar mais peso aquele que errar menos
peso_clf_0 = 1 - total_0/total_01
peso_clf_1 = 1 - total_1/total_01

print('\nPesos')
print(' > Clf 0: ', peso_clf_0)
print(' > Clf 1: ', peso_clf_1)

# Armazena os pesos para uso posterior
pickle.dump(np.array([peso_clf_0, peso_clf_1]), open(config.DIR_MODELOS + 'pesos.pkl', 'wb'))

'''
Exibe a performance sobre os dados de treinamento
'''
print('\nResultados sobre o conjunto de treinamento - Canal 0')
print(' > Acurácia', accuracy_score(rotulos, predicoes_0))
print(' > Precisão', precision_score(rotulos, predicoes_0))
print(' > Recobrimento', recall_score(rotulos, predicoes_0))

print('\nResultados sobre o conjunto de treinamento - Canal 1')
print(' > Acurácia', accuracy_score(rotulos, predicoes_1))
print(' > Precisão', precision_score(rotulos, predicoes_1))
print(' > Recobrimento', recall_score(rotulos, predicoes_1))

'''
Salva as máquinas de redução da dimensionalidade e de reconhecimento de arritmias
'''
joblib.dump(pca_0, config.DIR_MODELOS + 'pca_0.jl')
joblib.dump(pca_1, config.DIR_MODELOS + 'pca_1.jl')
joblib.dump(clf_0, config.DIR_MODELOS + 'clf_0.jl')
joblib.dump(clf_1, config.DIR_MODELOS + 'clf_1.jl')