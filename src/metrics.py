"""Weighted accuracy da PhysioNet 2022 e matriz de confusão ao nível do paciente."""
import numpy as np
import pandas as pd

CLASSES = ["Present", "Unknown", "Absent"]
WEIGHTS = {"Present": 5, "Unknown": 3, "Absent": 1}


def confusion_matrix(y_true, y_pred):
    """Matriz 3x3 como na Tabela 1 do guião: linhas = classe prevista, colunas = classe verdadeira.

    Atenção: é a transposta da convenção do sklearn (que usa linhas = verdadeira).
    """
    matrix = pd.DataFrame(0, index=CLASSES, columns=CLASSES)
    matrix.index.name, matrix.columns.name = "Previsto", "Verdadeiro"
    for true, pred in zip(y_true, y_pred):
        matrix.loc[pred, true] += 1
    return matrix


def weighted_accuracy(y_true, y_pred):
    """WA_murmur: cada paciente Present conta 5 vezes, Unknown 3 vezes e Absent 1 vez.

    Recebe um rótulo por PACIENTE ("Present", "Unknown" ou "Absent").
    """
    weights = np.array([WEIGHTS[t] for t in y_true])
    correct = np.array([t == p for t, p in zip(y_true, y_pred)])
    return (weights * correct).sum() / weights.sum()
