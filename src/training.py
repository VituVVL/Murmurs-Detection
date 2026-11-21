"""Treino e avaliação comuns aos modelos (CNN e RNN): pesos de classe, ciclo de treino e previsão por paciente.

Fica em src/ porque a agregação janela -> gravação -> paciente e o critério de paragem têm de ser iguais nos
dois modelos, senão as WA não são comparáveis.

Uso típico num notebook de modelo:

    from src.training import set_seed, class_weights, fit, predict_proba, patient_predictions
    set_seed()
    weights = class_weights(meta_train)
    history = fit(model, X_train, y_train, X_val, meta_val, weights)
    patients = patient_predictions(meta_val, predict_proba(model, X_val))
"""
import copy
import random
import time

import numpy as np
import pandas as pd
import torch
from torch import nn

from src.data import SEED
from src.metrics import CLASSES, weighted_accuracy

CLASS_INDEX = {c: i for i, c in enumerate(CLASSES)}   # Present = 0, Unknown = 1, Absent = 2


def set_seed(seed=SEED):
    """Fixa as seeds do Python, numpy e torch para o treino ser reprodutível."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def encode_labels(meta):
    """Rótulo de cada janela como inteiro (índice em CLASSES)."""
    return meta["murmur"].map(CLASS_INDEX).to_numpy()


def class_weights(meta_train):
    """Pesos de classe para a loss: inverso da frequência de cada classe nas JANELAS do treino.

    Normalizados para a média ponderada ser 1 (a escala da loss fica parecida com a de sem pesos).
    """
    counts = meta_train["murmur"].value_counts().reindex(CLASSES).to_numpy()
    weights = counts.sum() / (len(CLASSES) * counts)
    return torch.tensor(weights, dtype=torch.float32)


def predict_proba(model, X, batch_size=256):
    """Probabilidades (n_janelas, 3) na ordem de CLASSES."""
    model.eval()
    probs = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch = torch.as_tensor(X[i:i + batch_size])
            probs.append(torch.softmax(model(batch), dim=1).numpy())
    return np.concatenate(probs)


def patient_predictions(meta, probs):
    """Junta as previsões janela -> gravação -> paciente.

    - Gravação: média das probabilidades das suas janelas; classe = a de maior probabilidade.
    - Paciente: Present se alguma gravação for Present (o sopro pode ser audível só num local);
      senão Unknown se alguma for Unknown; senão Absent.
    Devolve uma linha por paciente: patient_id, murmur (verdadeiro), pred.
    """
    df = meta[["patient_id", "recording", "murmur"]].copy()
    df[CLASSES] = probs
    recordings = df.groupby(["patient_id", "recording", "murmur"], as_index=False)[CLASSES].mean()
    recordings["pred"] = recordings[CLASSES].idxmax(axis=1)

    def decide(preds):
        for c in ["Present", "Unknown"]:
            if (preds == c).any():
                return c
        return "Absent"

    patients = recordings.groupby(["patient_id", "murmur"])["pred"].agg(decide).reset_index()
    return patients


def evaluate(model, X, meta):
    """WA ao nível do paciente (a métrica oficial)."""
    patients = patient_predictions(meta, predict_proba(model, X))
    return weighted_accuracy(patients["murmur"], patients["pred"])


def log_experiment(name, description, history, patients, path=None):
    """Regista uma experiência: grava o histórico em results/<name>_history.csv e acrescenta (ou substitui)
    uma linha em results/experiencias.csv com o resumo, para comparar todas as experiências num só sítio.

    `patients` é o resultado de patient_predictions na VALIDAÇÃO, com o modelo final (melhor época).
    """
    from src.data import ROOT
    results = ROOT / "results"
    history.to_csv(results / f"{name}_history.csv", index=False)
    best = history.loc[history["val_wa"].idxmax()]
    recall = {c: (patients.loc[patients["murmur"] == c, "pred"] == c).mean() for c in CLASSES}
    row = {"experiencia": name, "data": time.strftime("%Y-%m-%d"), "descricao": description,
           "val_wa": round(weighted_accuracy(patients["murmur"], patients["pred"]), 3),
           "val_accuracy": round((patients["murmur"] == patients["pred"]).mean(), 3),
           **{f"recall_{c}": round(r, 3) for c, r in recall.items()},
           "melhor_epoca": int(best["epoch"]), "epocas": len(history),
           "wa_media_ultimas_5": round(history["val_wa"].tail(5).mean(), 3),
           "minutos": round(history["seconds"].sum() / 60, 1)}
    path = path or results / "experiencias.csv"
    log = pd.read_csv(path) if path.exists() else pd.DataFrame()
    if len(log):
        log = log[log["experiencia"] != name]
    log = pd.concat([log, pd.DataFrame([row])], ignore_index=True)
    log.to_csv(path, index=False)
    return log


def fit(model, X_train, y_train, X_val, meta_val, weights, epochs=40, patience=8, batch_size=64, lr=1e-3,
        weight_decay=1e-4, lr_plateau=None, verbose=True):
    """Treina com Adam e CrossEntropy com pesos de classe.

    Early stopping pela WA ao nível do paciente na VALIDAÇÃO (nunca o teste): no fim o modelo fica com os pesos
    da melhor época. Devolve o histórico (uma linha por época, com o lr usado nessa época).

    lr_plateau: None (lr fixo, como na baseline) ou um dict com os argumentos do ReduceLROnPlateau,
    ex. {"factor": 0.3, "patience": 3}. O scheduler olha para a LOSS da validação (mais suave do que a WA,
    porque usa todas as janelas e não só 142 pacientes) e multiplica o lr por `factor` quando ela não desce
    durante `patience` épocas. A sua paciência deve ser menor do que a do early stopping.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = None
    if lr_plateau is not None:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", **lr_plateau)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    X_train_t, y_train_t = torch.as_tensor(X_train), torch.as_tensor(y_train)
    y_val_t = torch.as_tensor(encode_labels(meta_val))

    best_wa, best_state, best_epoch, history = -1.0, None, 0, []
    for epoch in range(1, epochs + 1):
        start = time.time()
        model.train()
        order = torch.randperm(len(X_train_t))
        train_loss = 0.0
        for i in range(0, len(order), batch_size):
            idx = order[i:i + batch_size]
            optimizer.zero_grad()
            loss = loss_fn(model(X_train_t[idx]), y_train_t[idx])
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(idx)

        probs = predict_proba(model, X_val)
        val_loss = loss_fn(torch.log(torch.as_tensor(probs) + 1e-8), y_val_t).item()
        patients = patient_predictions(meta_val, probs)
        val_wa = weighted_accuracy(patients["murmur"], patients["pred"])
        current_lr = optimizer.param_groups[0]["lr"]
        history.append({"epoch": epoch, "lr": current_lr, "train_loss": train_loss / len(order),
                        "val_loss": val_loss, "val_wa": val_wa, "seconds": time.time() - start})
        if verbose:
            h = history[-1]
            print(f"época {epoch:2d}  lr {current_lr:.1e}  loss treino {h['train_loss']:.3f}  "
                  f"loss val {h['val_loss']:.3f}  WA val {val_wa:.3f}  ({h['seconds']:.0f} s)")
        if scheduler is not None:
            scheduler.step(val_loss)

        if val_wa > best_wa:
            best_wa, best_state, best_epoch = val_wa, copy.deepcopy(model.state_dict()), epoch
        elif epoch - best_epoch >= patience:
            if verbose:
                print(f"early stopping: sem melhoria há {patience} épocas")
            break

    model.load_state_dict(best_state)
    if verbose:
        print(f"melhor época: {best_epoch} (WA val {best_wa:.3f})")
    return pd.DataFrame(history)
