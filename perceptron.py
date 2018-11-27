import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD

import pandas as pd

from sklearn.preprocessing import LabelEncoder
import sklearn.model_selection
from sklearn.metrics import confusion_matrix

import numpy as np
from scipy import interp
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.metrics import roc_curve, auc


def print_confusion_matrix(x_test,y_test,number_of_classes):

    # Confusion Matrix and Classification Report
    y_pred = model.predict(x_test)

    conf_matrix = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))

    class_amounts = np.zeros(number_of_classes)

    for i in range(len(conf_matrix[:, 0])):
        for j in range(len(conf_matrix[0, :])):
            class_amounts[i] = conf_matrix[i, j] + class_amounts[i]

    norm_conf_matrix = np.zeros(np.shape(conf_matrix))

    for i in range(len(conf_matrix[:, 0])):
        for j in range(len(conf_matrix[0, :])):
            norm_conf_matrix[i, j] = conf_matrix[i, j] / class_amounts[i]

    print('Normalized Confusion Matrix')
    print(norm_conf_matrix)


iris = pd.read_csv('iris.csv')

# Convertendo 'class' de string para int
var_mod = ['class']
le = LabelEncoder()
for i in var_mod:
    iris[i] = le.fit_transform(iris[i])

# numero de classes
n_classes = 3
# numero de amostras
num_of_test_samples = 24
# batch size
batch_size = 5


# Inserindo as features em x_all
x_all = iris.drop(['class'], axis = 1).values
# Inserindo as classes  em y_all
y_all = keras.utils.np_utils.to_categorical(iris['class'].values)

# Separando o dataset em: 80% treino e 20% teste
x_train, x_test, y_train, y_test= sklearn.model_selection.train_test_split(x_all, y_all, test_size=0.2, random_state=0)


# Cria o modelo
model = keras.models.Sequential()

# A entrada é especificada na primeira camada, contendo espaço para as 4 features do dataset e 32 neurônios ocultos e completamente conectados
model.add(Dense(32, activation='relu', input_dim=4))
# Adicionada uma segunda camada de neurônios ocultos e completamente conectados
model.add(Dense(32, activation='relu'))
# Saída da Rede com 3 neurônios softmax, equivalente ao número de classes do problema
model.add(Dense(n_classes, activation='softmax'))

# Otimizador: Gradiente Descendente Estocástico
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9)

# Configura o modelo para treinamento
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

# Treina o modelo
model.fit(x_train, y_train,
          epochs=40,
          batch_size= 10)

score = model.evaluate(x_test, y_test, batch_size= batch_size)

print("\nThe final score is %.2f"% score[1])

print_confusion_matrix(x_test,y_test,n_classes)
