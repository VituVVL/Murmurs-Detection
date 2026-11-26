"""Compara as regras de agregação por paciente (any / half / mean) com a MESMA época, só na validação.

Porquê: no notebook 05 a regra e o critério da época foram escolhidos juntos. Aqui fixa-se o critério da
configuração final (época pela menor loss da validação) e muda-se só a regra. A loss é calculada por janela, logo a
época escolhida é a mesma nas três regras e a diferença vem só da regra.

A pergunta concreta: a regra `any` ("basta uma gravação Present para o paciente ser Present") imita a forma como o
rótulo foi dado. A `mean` pode diluir um sopro que só se ouve num local. Quantos Present a `mean` perde, e quantos
falsos Present a `any` acrescenta?

Uso (na raiz do projeto, no desktop, onde existe runs/):
    uv run python -m src.compare_rules
    uv run python -m src.compare_rules --exp rnn_wmetric --splits 50

Saída: imprime as tabelas e grava results/regras_agregacao.csv (resumo por regra). O teste NÃO é usado aqui.
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from src.data import ROOT
from src.metrics import CLASSES, WEIGHTS, confusion_matrix
from src.training import CLASS_INDEX, RULES, patient_predictions

RUNS = ROOT / "runs"
SEEDS = [42, 1, 2, 3, 4]
PATIENCE = 8   # o mesmo early stopping simulado do notebook 05


def es_select(scores, patience=PATIENCE):
    """Época (1, 2, ...) escolhida por early stopping sobre `scores` (quanto maior, melhor)."""
    best, best_epoch = -np.inf, None
    for epoch, s in enumerate(scores, start=1):
        if s > best:
            best, best_epoch = s, epoch
        elif epoch - best_epoch >= patience:
            break
    return best_epoch


def load_run(exp, seed, meta_val):
    """Previsões por paciente (todas as épocas e regras) e loss ponderada por paciente (todas as épocas)."""
    probs = np.load(RUNS / exp / f"seed{seed}" / "val_probs.npy").astype(np.float32)
    y = meta_val["murmur"].map(CLASS_INDEX).to_numpy()
    # Pesos da métrica (5/3/1). A normalização de class_weights cancela-se em sum(w * nll) / sum(w).
    w = meta_val["murmur"].map(WEIGHTS).to_numpy(dtype=float)
    preds, losses = [], []
    for epoch, p in enumerate(probs, start=1):
        for rule in RULES:
            preds.append(patient_predictions(meta_val, p, rule).assign(epoch=epoch, rule=rule))
        nll = -np.log(p[np.arange(len(p)), y] + 1e-8)
        losses.append(pd.DataFrame({"patient_id": meta_val["patient_id"], "wl": w * nll, "w": w})
                      .groupby("patient_id", as_index=False).sum().assign(epoch=epoch))
    preds = pd.concat(preds, ignore_index=True)
    preds["weight"] = preds["murmur"].map(WEIGHTS)
    preds["hit"] = (preds["murmur"] == preds["pred"]) * preds["weight"]
    return preds, pd.concat(losses, ignore_index=True)


def loss_epoch(losses, subset=None):
    """Época pela menor loss da validação (em `subset` de pacientes, ou em toda a validação)."""
    d = losses if subset is None else losses[losses["patient_id"].isin(subset)]
    g = d.groupby("epoch")[["wl", "w"]].sum()
    return es_select(-(g["wl"] / g["w"]).to_numpy())


def wa(preds, rule, epoch, subset=None):
    d = preds[(preds["rule"] == rule) & (preds["epoch"] == epoch)]
    if subset is not None:
        d = d[d["patient_id"].isin(subset)]
    return d["hit"].sum() / d["weight"].sum()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", nargs="+", default=["cnn_wmetric", "rnn_wmetric"])
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--splits", type=int, default=20, help="divisões da validação em duas metades (WA justa)")
    args = parser.parse_args()

    meta_val = pd.read_csv(RUNS / "meta_val.csv")
    patients = meta_val[["patient_id", "murmur"]].drop_duplicates().reset_index(drop=True)
    halves = []
    for rep in range(args.splits):
        skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=rep)
        halves += [(patients.loc[a, "patient_id"], patients.loc[b, "patient_id"])
                   for a, b in skf.split(patients, patients["murmur"])]

    summary = []
    for exp in args.exp:
        rows, cms, flips = [], {r: 0 for r in RULES}, []
        for seed in args.seeds:
            preds, losses = load_run(exp, seed, meta_val)
            epoch = loss_epoch(losses)   # a época da configuração final (toda a validação)
            # WA justa: época escolhida numa metade, WA medida na outra (a mesma época para as três regras)
            fair = {r: [] for r in RULES}
            for sel, ev in halves:
                e = loss_epoch(losses, sel)
                for r in RULES:
                    fair[r].append(wa(preds, r, e, ev))
            for r in RULES:
                d = preds[(preds["rule"] == r) & (preds["epoch"] == epoch)]
                recall = {c: (d.loc[d["murmur"] == c, "pred"] == c).mean() for c in CLASSES}
                rows.append({"exp": exp, "seed": seed, "rule": r, "epoch": epoch,
                             "wa_justa": np.mean(fair[r]), "wa_val": wa(preds, r, epoch),
                             **{f"recall_{c}": v for c, v in recall.items()}})
                cms[r] = cms[r] + confusion_matrix(d["murmur"], d["pred"])
            # Onde é que any e mean discordam? (mesma época, mesmos pacientes)
            at = preds[preds["epoch"] == epoch].pivot(index=["patient_id", "murmur"], columns="rule", values="pred")
            flips.append(at[at["any"] != at["mean"]].reset_index().assign(seed=seed))

        df = pd.DataFrame(rows)
        summary.append(df)
        print(f"\n===== {exp} (época pela menor loss da validação; a mesma nas três regras) =====")
        print("Épocas por seed:", df.drop_duplicates("seed").set_index("seed")["epoch"].to_dict())
        g = df.groupby("rule")
        table = pd.DataFrame({
            "WA justa": g["wa_justa"].mean().round(3).astype(str) + " ± " + g["wa_justa"].std().round(3).astype(str),
            "WA val (toda)": g["wa_val"].mean().round(3),
            **{f"recall {c}": g[f"recall_{c}"].mean().round(3) for c in CLASSES}}).reindex(RULES)
        print(table.to_string())
        for r in RULES:
            print(f"\nMatriz de confusão por paciente, regra {r}, somada nas {len(args.seeds)} seeds "
                  "(linhas = previsto, colunas = verdadeiro):")
            print(cms[r].to_string())

        flips = pd.concat(flips, ignore_index=True)
        print(f"\nPacientes em que `any` e `mean` discordam (somados nas {len(args.seeds)} seeds): {len(flips)}")
        if len(flips):
            counts = flips.groupby(["murmur", "any", "mean"]).size().rename("n").reset_index()
            print(counts.rename(columns={"murmur": "verdadeiro"}).to_string(index=False))
            gain = (flips["murmur"] == flips["any"]) * flips["murmur"].map(WEIGHTS)
            loss = (flips["murmur"] == flips["mean"]) * flips["murmur"].map(WEIGHTS)
            print(f"Pontos da WA (soma dos pesos) ganhos pela any nesses pacientes: {gain.sum()}; "
                  f"ganhos pela mean: {loss.sum()}")

    out = pd.concat(summary, ignore_index=True)
    out.to_csv(ROOT / "results" / "regras_agregacao.csv", index=False)
    print("\nGravado results/regras_agregacao.csv")


if __name__ == "__main__":
    main()
