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
from src.metrics import CLASSES, WEIGHTS, weighted_accuracy

CLASS_INDEX = {c: i for i, c in enumerate(CLASSES)}   # Present = 0, Unknown = 1, Absent = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")   # GPU se existir (torch com CUDA)


def set_seed(seed=SEED):
    """Fixa as seeds do Python, numpy e torch para o treino ser reprodutível."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def encode_labels(meta):
    """Rótulo de cada janela como inteiro (índice em CLASSES)."""
    return meta["murmur"].map(CLASS_INDEX).to_numpy(dtype=np.int64, copy=True)   # cópia: o torch não aceita arrays só de leitura


def class_weights(meta_train, scheme="balanced"):
    """Pesos de classe para a loss, calculados nas JANELAS do treino.

    - "balanced": inverso da frequência de cada classe (cada classe conta o mesmo no total da loss).
      Present 1.74, Unknown 6.94, Absent 0.44: o Unknown pesa mais, por ser a classe com menos janelas.
    - "metric": os pesos da WA (Present 5, Unknown 3, Absent 1), para a loss imitar a métrica.
      Errar uma janela Present custa 5x errar uma Absent, como errar um paciente na WA.
    Normalizados para a média ponderada (pelas janelas) ser 1: a escala da loss fica parecida com a de sem pesos.
    """
    counts = meta_train["murmur"].value_counts().reindex(CLASSES).to_numpy()
    if scheme == "balanced":
        weights = counts.sum() / (len(CLASSES) * counts)
    elif scheme == "metric":
        w = np.array([WEIGHTS[c] for c in CLASSES], dtype=float)
        weights = w * counts.sum() / (w * counts).sum()
    else:
        raise ValueError(scheme)
    return torch.tensor(weights, dtype=torch.float32)


def predict_proba(model, X, batch_size=256):
    """Probabilidades (n_janelas, 3) na ordem de CLASSES. Corre no device onde o modelo estiver."""
    device = next(model.parameters()).device
    model.eval()
    probs = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch = torch.as_tensor(X[i:i + batch_size]).to(device)
            probs.append(torch.softmax(model(batch), dim=1).cpu().numpy())
    return np.concatenate(probs)


RULES = ["any", "half", "mean"]


def patient_predictions(meta, probs, rule="any"):
    """Junta as previsões janela -> gravação -> paciente.

    - Gravação: média das probabilidades das suas janelas; classe = a de maior probabilidade.
    - Paciente, conforme `rule`:
      "any"  (a original): Present se alguma gravação for Present (o sopro pode ser audível só num local);
             senão Unknown se alguma for Unknown; senão Absent.
      "half": Present se alguma gravação for Present; senão Unknown se PELO MENOS METADE das gravações for Unknown;
             senão Absent. (Unknown = "o médico não conseguiu ouvir bem": deve afetar o paciente todo, não um local.)
      "mean": média das probabilidades das gravações do paciente; classe = a de maior probabilidade.
    Devolve uma linha por paciente: patient_id, murmur (verdadeiro), pred.
    """
    df = meta[["patient_id", "recording", "murmur"]].copy()
    df[CLASSES] = probs
    recordings = df.groupby(["patient_id", "recording", "murmur"], as_index=False)[CLASSES].mean()
    if rule == "mean":
        patients = recordings.groupby(["patient_id", "murmur"], as_index=False)[CLASSES].mean()
        patients["pred"] = patients[CLASSES].idxmax(axis=1)
        return patients[["patient_id", "murmur", "pred"]]
    recordings["pred"] = recordings[CLASSES].idxmax(axis=1)

    def decide(preds):
        if (preds == "Present").any():
            return "Present"
        unknown = (preds == "Unknown").any() if rule == "any" else (preds == "Unknown").mean() >= 0.5
        return "Unknown" if unknown else "Absent"

    patients = recordings.groupby(["patient_id", "murmur"])["pred"].agg(decide).reset_index()
    return patients


def evaluate(model, X, meta):
    """WA ao nível do paciente (a métrica oficial)."""
    patients = patient_predictions(meta, predict_proba(model, X))
    return weighted_accuracy(patients["murmur"], patients["pred"])


def spec_augment(batch, n_freq_masks=2, max_freq=8, n_time_masks=2, max_time=40):
    """SpecAugment simplificado: em cada exemplo, põe a 0 algumas faixas de frequência e de tempo ao acaso.

    Só no TREINO. Funciona com (n, 1, bandas, tempo) ou (n, bandas, tempo): as duas últimas dimensões são
    frequência e tempo. Como as features estão normalizadas (média 0 por banda), 0 é o valor "neutro".
    Por omissão: 2 faixas de até 8 das 64 bandas mel e 2 faixas de até 40 dos 313 instantes (~0.64 s).
    Obriga a rede a não depender de uma única banda ou de um único instante (ex.: um só batimento).
    """
    batch = batch.clone()
    n, n_freq, n_time = batch.shape[0], batch.shape[-2], batch.shape[-1]
    for i in range(n):
        for _ in range(n_freq_masks):
            width = np.random.randint(0, max_freq + 1)
            start = np.random.randint(0, n_freq - width + 1)
            batch[i, ..., start:start + width, :] = 0
        for _ in range(n_time_masks):
            width = np.random.randint(0, max_time + 1)
            start = np.random.randint(0, n_time - width + 1)
            batch[i, ..., start:start + width] = 0
    return batch


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
        weight_decay=1e-4, lr_plateau=None, augment=None, verbose=True, on_epoch=None):
    """Treina com Adam e CrossEntropy com pesos de classe.

    Early stopping pela WA ao nível do paciente na VALIDAÇÃO (nunca o teste): no fim o modelo fica com os pesos
    da melhor época. Devolve o histórico (uma linha por época, com o lr usado nessa época).

    lr_plateau: None (lr fixo, como na baseline) ou um dict com os argumentos do ReduceLROnPlateau,
    ex. {"factor": 0.3, "patience": 3}. O scheduler olha para a LOSS da validação (mais suave do que a WA,
    porque usa todas as janelas e não só 142 pacientes) e multiplica o lr por `factor` quando ela não desce
    durante `patience` épocas. A sua paciência deve ser menor do que a do early stopping.

    augment: None ou uma função aplicada a cada batch de TREINO (ex.: spec_augment). A validação nunca é alterada.

    patience=None desliga o early stopping (treina sempre `epochs` épocas). on_epoch: função opcional chamada no fim
    de cada época como on_epoch(epoch, model, probs_val), para guardar pesos/probabilidades de todas as épocas.

    O modelo é movido para DEVICE (GPU se existir); os dados ficam na RAM e vão para o device batch a batch.
    """
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = None
    if lr_plateau is not None:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", **lr_plateau)
    loss_fn = nn.CrossEntropyLoss(weight=weights.to(DEVICE))
    X_train_t, y_train_t = torch.as_tensor(X_train), torch.as_tensor(y_train)
    y_val_t = torch.as_tensor(encode_labels(meta_val)).to(DEVICE)

    best_wa, best_state, best_epoch, history = -1.0, None, 0, []
    for epoch in range(1, epochs + 1):
        start = time.time()
        model.train()
        order = torch.randperm(len(X_train_t))
        train_loss = 0.0
        for i in range(0, len(order), batch_size):
            idx = order[i:i + batch_size]
            optimizer.zero_grad()
            batch = X_train_t[idx] if augment is None else augment(X_train_t[idx])
            loss = loss_fn(model(batch.to(DEVICE)), y_train_t[idx].to(DEVICE))
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(idx)

        probs = predict_proba(model, X_val)
        val_loss = loss_fn(torch.log(torch.as_tensor(probs, device=DEVICE) + 1e-8), y_val_t).item()
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
        if on_epoch is not None:
            on_epoch(epoch, model, probs)

        if val_wa > best_wa:
            best_wa, best_state, best_epoch = val_wa, copy.deepcopy(model.state_dict()), epoch
        elif patience is not None and epoch - best_epoch >= patience:
            if verbose:
                print(f"early stopping: sem melhoria há {patience} épocas")
            break

    model.load_state_dict(best_state)
    if verbose:
        print(f"melhor época: {best_epoch} (WA val {best_wa:.3f})")
    return pd.DataFrame(history)
