"""Avaliação final: validação cruzada com 5 folds nos 942 pacientes, com a configuração já congelada.

Porquê: o dataset é pequeno e o professor prefere validação cruzada a uma só divisão 70/15/15 (aula de 27/11).
Assim cada um dos 942 pacientes é previsto uma vez, por um modelo que não o viu no treino.

Não é usado para decisões: a arquitetura, os pesos de classe (5/3/1), o critério da época (menor loss da validação)
e a regra de agregação foram escolhidos antes, só com a validação da divisão antiga (notebooks 05 e 06,
src/compare_rules.py). Aqui só se mede; nada muda depois de ver estes resultados.

Em cada fold (estratificado por paciente, pela classe Murmur):
    teste             o fold de fora (~188 pacientes)
    validação interna 15% dos restantes, só para escolher a época (menor loss, early stopping simulado, paciência 8)
    treino            o resto (~640 pacientes)
A normalização das features usa as estatísticas do treino de cada fold.

Uso (na raiz do projeto, no desktop com GPU):
    uv run python -m src.run_cv                                  # treina o que faltar e mostra o relatório
    uv run python -m src.run_cv --report                         # só o relatório (sem treinar)
    uv run python -m src.run_cv --exp rnn_wmetric --seeds 42 1 2

Saída:
    data/cv_folds.csv                               fold de cada paciente (vai para o git, como o split.csv)
    runs/cv/<exp>/fold<k>/seed<s>/history.csv       uma linha por época (val_loss = validação interna)
    runs/cv/<exp>/fold<k>/seed<s>/test_probs.npy    (épocas, janelas do fold de teste, 3), float16
    runs/cv/<exp>/fold<k>/test_meta.csv             janelas do fold de teste (a ordem do test_probs)
    results/cv_resultados.csv, results/cv_matriz_<exp>.csv
"""
import argparse
import time

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split

from src.data import ROOT, SEED
from src.features import fit_standardizer, load_features, standardize
from src.metrics import CLASSES, confusion_matrix, weighted_accuracy
from src.run_seeds import EXPERIMENTS, build_model, features_of
from src.training import DEVICE, class_weights, encode_labels, fit, patient_predictions, predict_proba, set_seed

CV_DIR = ROOT / "runs" / "cv"
FOLDS_FILE = ROOT / "data" / "cv_folds.csv"
N_FOLDS = 5
INNER_VAL = 0.15
EPOCHS = 40
PATIENCE = 8
SEEDS = [42, 1, 2]
RULE = "mean"   # congelada antes da validação cruzada (src/compare_rules.py); não mudar depois de ver resultados


def load_all(kind):
    """Features e meta dos 942 pacientes (treino + validação + teste da divisão antiga), sem normalizar."""
    parts = [load_features(subset, kind) for subset in ("train", "val", "test")]
    return np.concatenate([X for X, _ in parts]), pd.concat([m for _, m in parts], ignore_index=True)


def make_folds(meta):
    """Fold de cada paciente (0 a 4), estratificado pela classe. Grava em data/cv_folds.csv e reutiliza-o."""
    if FOLDS_FILE.exists():
        return pd.read_csv(FOLDS_FILE)
    patients = meta[["patient_id", "murmur"]].drop_duplicates().reset_index(drop=True)
    patients["fold"] = -1
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for k, (_, test_idx) in enumerate(skf.split(patients, patients["murmur"])):
        patients.loc[test_idx, "fold"] = k
    patients = patients[["patient_id", "fold"]].sort_values("patient_id")
    patients.to_csv(FOLDS_FILE, index=False)
    return patients


def fold_subsets(folds, meta, k):
    """Pacientes de treino, validação interna e teste do fold k."""
    labels = meta[["patient_id", "murmur"]].drop_duplicates().merge(folds, on="patient_id")
    test = labels.loc[labels["fold"] == k, "patient_id"]
    rest = labels[labels["fold"] != k]
    train, val = train_test_split(rest["patient_id"], test_size=INNER_VAL, stratify=rest["murmur"],
                                  random_state=SEED + k)
    assert not (set(train) & set(val) or set(train) & set(test) or set(val) & set(test)), "paciente repetido"
    return train, val, test


def fold_data(X, meta, folds, k, kind):
    """(X, meta) de treino, validação interna e teste do fold k, normalizados com as estatísticas do treino do fold."""
    out = []
    for patients in fold_subsets(folds, meta, k):
        mask = meta["patient_id"].isin(patients).to_numpy()
        out.append([X[mask], meta[mask].reset_index(drop=True)])   # X[mask] é uma cópia
    stats = fit_standardizer(out[0][0], kind)
    for part in out:
        part[0] = standardize(part[0], stats, inplace=True)
    return out


def train_fold(name, config, k, seed, data):
    out = CV_DIR / name / f"fold{k}" / f"seed{seed}"
    if (out / "history.csv").exists():
        print(f"{name} fold {k} seed {seed}: já existe, a saltar", flush=True)
        return
    out.mkdir(parents=True, exist_ok=True)
    (X_tr, meta_tr), (X_va, meta_va), (X_te, meta_te) = data
    meta_te.to_csv(out.parent / "test_meta.csv", index=False)
    set_seed(seed)
    model = build_model(config)
    test_probs = []

    def on_epoch(epoch, m, probs_val):
        # O fold de teste é previsto em todas as épocas; a época usada é escolhida depois, só pela validação interna.
        test_probs.append(predict_proba(m, X_te).astype(np.float16))

    start = time.time()
    history = fit(model, X_tr, encode_labels(meta_tr), X_va, meta_va, class_weights(meta_tr, config["weights"]),
                  epochs=EPOCHS, patience=None, verbose=False, on_epoch=on_epoch, **config.get("fit_kwargs", {}))
    np.save(out / "test_probs.npy", np.stack(test_probs))
    history.to_csv(out / "history.csv", index=False)
    print(f"{name} fold {k} seed {seed}: {len(meta_tr)} / {len(meta_va)} / {len(meta_te)} janelas, "
          f"{(time.time() - start) / 60:.1f} min, época {select_epoch(history)}", flush=True)


def select_epoch(history, patience=PATIENCE):
    """Época pela menor loss da validação interna, com early stopping simulado (igual ao notebook 05)."""
    best, best_epoch = np.inf, None
    for epoch, loss in zip(history["epoch"], history["val_loss"]):
        if loss < best:
            best, best_epoch = loss, int(epoch)
        elif epoch - best_epoch >= patience:
            break
    return best_epoch


def report(exps, seeds, rule):
    """WA por fold, WA das previsões fora do fold (942 pacientes) e matriz de confusão por paciente."""
    rows, summary = [], []
    for exp in exps:
        oof = {s: [] for s in seeds + ["ensemble"]}
        for k in range(N_FOLDS):
            fold_dir = CV_DIR / exp / f"fold{k}"
            meta_te = pd.read_csv(fold_dir / "test_meta.csv")
            chosen = []
            for seed in seeds:
                d = fold_dir / f"seed{seed}"
                history = pd.read_csv(d / "history.csv")
                epoch = select_epoch(history)
                probs = np.load(d / "test_probs.npy")[epoch - 1].astype(np.float32)
                chosen.append(probs)
                patients = patient_predictions(meta_te, probs, rule)
                oof[seed].append(patients)
                rows.append({"exp": exp, "fold": k, "seed": seed, "epoch": epoch,
                             "wa": weighted_accuracy(patients["murmur"], patients["pred"])})
            patients = patient_predictions(meta_te, np.mean(chosen, axis=0), rule)   # média das seeds
            oof["ensemble"].append(patients)
            rows.append({"exp": exp, "fold": k, "seed": "ensemble", "epoch": None,
                         "wa": weighted_accuracy(patients["murmur"], patients["pred"])})

        df = pd.DataFrame([r for r in rows if r["exp"] == exp])
        per_fold = df[df["seed"] != "ensemble"].groupby("fold")["wa"].mean()   # média das seeds em cada fold
        ens = df[df["seed"] == "ensemble"].set_index("fold")["wa"]
        print(f"\n===== {exp} (regra {rule}, {len(seeds)} seeds, {N_FOLDS} folds) =====")
        print(pd.DataFrame({"WA (média das seeds)": per_fold, "WA (ensemble das seeds)": ens}).round(3).to_string())
        print(f"WA entre folds: {per_fold.mean():.3f} ± {per_fold.std():.3f} "
              f"(ensemble: {ens.mean():.3f} ± {ens.std():.3f})")

        cm, pooled = 0, []
        for seed in seeds:
            p = pd.concat(oof[seed], ignore_index=True)
            pooled.append(weighted_accuracy(p["murmur"], p["pred"]))
            cm = cm + confusion_matrix(p["murmur"], p["pred"])
        p = pd.concat(oof["ensemble"], ignore_index=True)
        cm_ens = confusion_matrix(p["murmur"], p["pred"])
        recall = {c: float(cm_ens.loc[c, c] / cm_ens[c].sum()) for c in CLASSES}
        print(f"WA dos {len(p)} pacientes fora do fold: {np.mean(pooled):.3f} ± {np.std(pooled, ddof=1):.3f} "
              f"entre seeds; ensemble {weighted_accuracy(p['murmur'], p['pred']):.3f}")
        print("Recall (ensemble):", {c: round(r, 3) for c, r in recall.items()})
        print(f"Matriz de confusão por paciente, ensemble das seeds (linhas = previsto, colunas = verdadeiro):")
        print(cm_ens.to_string())
        print(f"Matriz somada nas {len(seeds)} seeds:")
        print(cm.to_string())
        cm_ens.to_csv(ROOT / "results" / f"cv_matriz_{exp}.csv")
        summary.append({"exp": exp, "rule": rule, "seeds": len(seeds), "wa_folds_media": per_fold.mean(),
                        "wa_folds_desvio": per_fold.std(), "wa_ensemble_folds_media": ens.mean(),
                        "wa_ensemble_folds_desvio": ens.std(), "wa_oof_ensemble":
                        weighted_accuracy(p["murmur"], p["pred"]), **{f"recall_{c}": r for c, r in recall.items()}})
    pd.DataFrame(rows).to_csv(ROOT / "results" / "cv_por_fold.csv", index=False)
    pd.DataFrame(summary).round(3).to_csv(ROOT / "results" / "cv_resultados.csv", index=False)
    print("\nGravado results/cv_resultados.csv, results/cv_por_fold.csv e results/cv_matriz_<exp>.csv")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", nargs="+", default=["cnn_wmetric", "rnn_wmetric"])
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--report", action="store_true", help="só o relatório, sem treinar")
    args = parser.parse_args()
    if not args.report:
        print("device:", DEVICE, flush=True)
        for kind in ("mel", "mfcc"):
            exps = [e for e in args.exp if features_of(EXPERIMENTS[e]) == kind]
            if not exps:
                continue
            X, meta = load_all(kind)
            folds = make_folds(meta)
            for k in range(N_FOLDS):
                data = fold_data(X, meta, folds, k, kind)
                if kind == "mel":
                    for part in data:
                        part[0] = part[0][:, None]   # canal da CNN
                for exp in exps:
                    for seed in args.seeds:
                        train_fold(exp, EXPERIMENTS[exp], k, seed, data)
                del data
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    report(args.exp, args.seeds, RULE)


if __name__ == "__main__":
    main()
