#!/usr/bin/env python3
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import sys

# Carregar heart_failure.csv
data = np.loadtxt('data/baseline/heart_failure.csv', delimiter=';', skiprows=1)
X = data[:, :-1]  # todas as colunas menos a última
Y = data[:, -1]   # última coluna é DEATH_EVENT

# Selecionar um subset (solução): 162/299 selecionadas (como no teste C++)
# Para teste: selecionar os primeiros 162 como "selecionadas"
solution = np.array([1]*162 + [0]*(299-162), dtype=int)

# Extrair treino e teste
X_train = X[solution == 1]
Y_train = Y[solution == 1]
X_test = X
Y_test = Y

# Treinar 1-NN
clf = KNeighborsClassifier(n_neighbors=1)
clf.fit(X_train, Y_train)
Y_pred = clf.predict(X_test)

# Calcular métricas (binário)
acc = accuracy_score(Y_test, Y_pred)
prec = precision_score(Y_test, Y_pred)
rec = recall_score(Y_test, Y_pred)
f1 = f1_score(Y_test, Y_pred)
reduction = 1.0 - (len(X_train) / len(X))

print(f"Python (scikit-learn) 1-NN:")
print(f"Acurácia: {acc:.6f}")
print(f"Precisão: {prec:.6f}")
print(f"Recall: {rec:.6f}")
print(f"F1-Score: {f1:.6f}")
print(f"Redução: {reduction:.4f}")
print()
print(f"C++ resultados para comparar:")
print(f"Acurácia: 0.819398")
print(f"Precisão: 0.714286")
print(f"Recall: 0.729167")
print(f"F1-Score: 0.721649")
print(f"Redução: 0.4582 (162/299)")
