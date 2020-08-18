# -*- coding: utf-8 -*-

'''
Diretórios
'''
DIR_DB        = 'db/'
DIR_DB_TESTE  = 'db/'
DIR_ATRIBUTOS = 'atributos/'
DIR_MODELOS   = 'modelos/'

'''
MIT/BIH
Registros utilizados para treinamento e validação
'''
REG_TREINO = [101, 106, 108, 109, 112, 114, 115, 116,
              118, 119, 122, 124, 201, 203, 205, 207,
              208, 209, 215, 220, 223, 230]

REG_TESTE  = [100, 103, 105, 111, 113, 117, 121, 123,
              200, 202, 210, 212, 213, 214, 219, 221,
              222, 228, 231, 232, 233, 234]

'''
Quantidade de amostras a serem consideradas antes e após a onda R
para a taxa de amostragem de 360 Hz
A soma de ANTES_R e APOS_R
'''
ANTES_R = 0.2
APOS_R  = 0.3

'''
Rótulos physionet que são considerados como da classe "Normal"
'''
rotulos_NORMAL = ['N', 'L', 'R', 'e', 'j']

'''
Rótulos physionet desconsiderados
'''
rotulos_EXCLUIDOS = ['/', 'f', 'Q', 'F']

'''
Número de atributos efetivamente utilizados
Após reduçã da dimensão do espaço de atributos
Máximo é 180
'''
N_ATRIBUTOS = 10