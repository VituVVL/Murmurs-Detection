"""Bónus: previsões para os 100 pacientes sem rótulo, com um ensemble da RNN treinado por validação cruzada.

Porquê assim (e não uma só divisão 85/15):
    Com uma só divisão, os 15% de fora nunca entram no treino e a época depende de uma só fatia de pacientes.
    Aqui usam-se os mesmos 5 folds da avaliação (data/cv_folds.csv): o modelo k treina com os outros 4 folds (~80%
    dos 942 pacientes) e usa o fold k só para escolher a época (menor loss, como na configuração final). Assim cada
    paciente entra no treino de 4 dos 5 modelos, nenhuma fatia decide sozinha, e os 5 modelos viram dados diferentes
    (um ensemble mais variado do que só mudar a seed). A previsão final é a média das probabilidades de todos os
    modelos (5 folds x seeds), com a regra de agregação congelada (`mean`).

    O número que se reporta como estimativa do desempenho é o de src/run_cv.py, não o daqui: aqui a época é
    escolhida no próprio fold de fora, por isso a WA nesse fold é otimista (só serve para verificar que nada partiu).

Uso (na raiz do projeto, no desktop com GPU; depois de src.run_cv, para não partilhar a GPU):
    uv run python -m src.run_bonus
    uv run python -m src.run_bonus --exp rnn_wmetric --seeds 42 1 2

Saída:
    PL2_G05_test_results.csv                          colunas ID,Murmur (ID = patient_id), 100 linhas
    runs/bonus/<exp>/fold<k>/seed<s>/history.csv      uma linha por época (val_loss = fold k)
    runs/bonus/<exp>/fold<k>/seed<s>/bonus_probs.npy  (épocas, janelas do bónus, 3), float16
"""
import argparse
import time

import numpy as np
import pandas as pd

from src.data import ROOT
from src.features import fit_standardizer, load_features, standardize
from src.metrics import CLASSES, weighted_accuracy
from src.run_cv import EPOCHS, N_FOLDS, RULE, SEEDS, load_all, make_folds, select_epoch
from src.run_seeds import EXPERIMENTS, build_model, features_of
from src.training import class_weights, encode_labels, fit, patient_predictions, predict_proba, set_seed

BONUS_RUNS = ROOT / "runs" / "bonus"
OUT_CSV = ROOT / "PL2_G05_test_results.csv"
N_BONUS_PATIENTS = 100


def train_model(exp, config, k, seed, X_tr, meta_tr, X_va, meta_va, X_bonus):
    out = BONUS_RUNS / exp / f"fold{k}" / f"seed{seed}"
    if (out / "history.csv").exists():
        print(f"{exp} fold {k} seed {seed}: já existe, a saltar", flush=True)
        return
    out.mkdir(parents=True, exist_ok=True)
    set_seed(seed)
    model = build_model(config)
    bonus_probs, wa_mean = [], []

    def on_epoch(epoch, m, probs_val):
        # O bónus é previsto em todas as épocas; a época usada escolhe-se depois, só pela loss do fold k.
        bonus_probs.append(predict_proba(m, X_bonus).astype(np.float16))
        # O val_wa do fit usa a regra "any"; aqui guarda-se a WA com a regra congelada (só para verificação).
        p = patient_predictions(meta_va, probs_val, RULE)
        wa_mean.append(weighted_accuracy(p["murmur"], p["pred"]))

    start = time.time()
    history = fit(model, X_tr, encode_labels(meta_tr), X_va, meta_va, class_weights(meta_tr, config["weights"]),
                  epochs=EPOCHS, patience=None, verbose=False, on_epoch=on_epoch, **config.get("fit_kwargs", {}))
    history[f"val_wa_{RULE}"] = wa_mean
    np.save(out / "bonus_probs.npy", np.stack(bonus_probs))
    history.to_csv(out / "history.csv", index=False)
    epoch = select_epoch(history)
    print(f"{exp} fold {k} seed {seed}: {len(meta_tr)} / {len(meta_va)} janelas, {(time.time() - start) / 60:.1f} min,"
          f" época {epoch} (WA {RULE} no fold {k}: {wa_mean[epoch - 1]:.3f}, otimista)", flush=True)


def predict(exp, seeds, meta_bonus):
    """Média das probabilidades (na época escolhida) de todos os modelos; regra RULE por paciente."""
    chosen, epochs = [], []
    for k in range(N_FOLDS):
        for seed in seeds:
            d = BONUS_RUNS / exp / f"fold{k}" / f"seed{seed}"
            epoch = select_epoch(pd.read_csv(d / "history.csv"))
            chosen.append(np.load(d / "bonus_probs.npy")[epoch - 1].astype(np.float32))
            epochs.append(epoch)
    print(f"Épocas escolhidas ({len(chosen)} modelos): {epochs}")
    # Sem rótulo: `murmur` vem vazio e o groupby de patient_predictions descartaria esses pacientes.
    meta = meta_bonus.assign(murmur="?")
    patients = patient_predictions(meta, np.mean(chosen, axis=0), RULE)
    assert len(patients) == N_BONUS_PATIENTS, f"esperava {N_BONUS_PATIENTS} pacientes, tenho {len(patients)}"
    return patients


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", default="rnn_wmetric")
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--predict-only", action="store_true", help="só junta as previsões (sem treinar)")
    args = parser.parse_args()
    config = EXPERIMENTS[args.exp]
    kind = features_of(config)

    X_bonus_raw, meta_bonus = load_features("bonus", kind)
    if not args.predict_only:
        X, meta = load_all(kind)
        folds = make_folds(meta)   # os mesmos folds de src/run_cv.py
        fold_of = meta["patient_id"].map(folds.set_index("patient_id")["fold"]).to_numpy()
        for k in range(N_FOLDS):
            tr, va = fold_of != k, fold_of == k
            X_tr, X_va = X[tr], X[va]
            stats = fit_standardizer(X_tr, kind)   # estatísticas do treino deste modelo
            X_tr, X_va = standardize(X_tr, stats, inplace=True), standardize(X_va, stats, inplace=True)
            X_bonus = standardize(X_bonus_raw.copy(), stats, inplace=True)
            if kind == "mel":
                X_tr, X_va, X_bonus = X_tr[:, None], X_va[:, None], X_bonus[:, None]
            for seed in args.seeds:
                train_model(args.exp, config, k, seed, X_tr, meta[tr].reset_index(drop=True),
                            X_va, meta[va].reset_index(drop=True), X_bonus)

    patients = predict(args.exp, args.seeds, meta_bonus)
    out = patients.rename(columns={"patient_id": "ID", "pred": "Murmur"})[["ID", "Murmur"]].sort_values("ID")
    assert out["Murmur"].isin(CLASSES).all()
    out.to_csv(OUT_CSV, index=False)
    print("\nDistribuição prevista:", out["Murmur"].value_counts().to_dict())
    print(f"Gravado {OUT_CSV.name} ({len(out)} linhas)")


if __name__ == "__main__":
    main()
