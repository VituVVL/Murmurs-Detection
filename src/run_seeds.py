"""Treina as configurações em EXPERIMENTS com várias seeds e guarda TUDO de cada época, para análise posterior.

Porquê: com uma só seed e 142 pacientes na validação, as diferenças entre variantes (1 a 4 pacientes Present) são do
tamanho do ruído. Com 5 seeds compara-se a média e o desvio. E como se guardam as probabilidades da validação e os
pesos de TODAS as épocas, dá para comparar depois (sem treinar de novo) regras de agregação por paciente e critérios
de escolha da época. O early stopping (paciência 8) é simulado a partir do histórico: dá o mesmo que parar no treino,
porque a paragem não muda o que acontece nas épocas anteriores.

Uso (na raiz do projeto):
    uv run python -m src.run_seeds                         # todas as experiências, todas as seeds
    uv run python -m src.run_seeds --exp cnn_base rnn_base --seeds 42 1

Saída em runs/<experiência>/seed<k>/ (fora do git):
    history.csv        uma linha por época: train_loss, val_loss, val_wa (regra atual), segundos
    val_probs.npy      (épocas, janelas da validação, 3) probabilidades, float16
    epoch<NN>.pt       pesos do modelo no fim de cada época
O teste NÃO é usado aqui.
"""
import argparse
import time

import librosa
import numpy as np
import torch

from src.data import ROOT
from src.features import fit_standardizer, load_features, standardize
from src.models import RNNClassifier, SmallCNN, count_parameters
from src.training import DEVICE, class_weights, encode_labels, fit, set_seed, spec_augment

RUNS_DIR = ROOT / "runs"
SEEDS = [42, 1, 2, 3, 4]
EPOCHS = 40

# model: "cnn" ou "rnn"; weights: esquema de class_weights.
# Opcionais: features (por omissão "mel" na CNN e "mfcc" na RNN, ver FEATURES), model_kwargs (argumentos do modelo),
# fit_kwargs (argumentos do fit, ex. augment), train_filter ("audible": tira do treino as janelas de pacientes Present
# em que o sopro não é audível naquele local; a validação não muda).
EXPERIMENTS = {
    # 26/11: baselines e pesos de classe (notebook 05)
    "cnn_base":    {"model": "cnn", "weights": "balanced"},
    "cnn_wmetric": {"model": "cnn", "weights": "metric"},
    "rnn_base":    {"model": "rnn", "weights": "balanced"},
    "rnn_wmetric": {"model": "rnn", "weights": "metric"},
    # 27/11: variantes, todas com os pesos da métrica (notebook 06)
    "cnn_specaug": {"model": "cnn", "weights": "metric", "fit_kwargs": {"augment": spec_augment}},
    "cnn_audible": {"model": "cnn", "weights": "metric", "train_filter": "audible"},
    "rnn_lstm":    {"model": "rnn", "weights": "metric", "model_kwargs": {"cell": "lstm"}},
    "rnn_delta":   {"model": "rnn", "weights": "metric", "features": "mfcc_delta", "model_kwargs": {"n_features": 39}},
    "rnn_mel":     {"model": "rnn", "weights": "metric", "features": "mel_seq", "model_kwargs": {"n_features": 64}},
    "rnn_h128":    {"model": "rnn", "weights": "metric", "model_kwargs": {"hidden": 128}},
    "rnn_audible": {"model": "rnn", "weights": "metric", "train_filter": "audible"},
}

# Entradas possíveis:
#   mel         log-mel (n, 1, 64, 313), para a CNN
#   mfcc        MFCC (n, 313, 13), para a RNN
#   mfcc_delta  MFCC + deltas de 1.ª e 2.ª ordem (n, 313, 39): a variação de cada coeficiente no tempo
#   mel_seq     o log-mel como sequência (n, 313, 64), para a RNN: as 64 bandas em vez dos 13 coeficientes
FEATURES = ["mel", "mfcc", "mfcc_delta", "mel_seq"]


def features_of(config):
    return config.get("features", {"cnn": "mel", "rnn": "mfcc"}[config["model"]])


def add_deltas(X):
    """(n, tempo, 13) -> (n, tempo, 39): junta os deltas de 1.ª e 2.ª ordem, calculados no eixo do tempo de cada janela."""
    return np.concatenate([X, librosa.feature.delta(X, axis=1, order=1),
                           librosa.feature.delta(X, axis=1, order=2)], axis=2).astype(np.float32)


def load_data(features):
    """Features de treino e validação, normalizadas com as estatísticas do treino."""
    kind = "mel" if features in ("mel", "mel_seq") else "mfcc"
    X_train, meta_train = load_features("train", kind)
    X_val, meta_val = load_features("val", kind)
    if features == "mfcc_delta":
        X_train, X_val = add_deltas(X_train), add_deltas(X_val)
    stats = fit_standardizer(X_train, kind)
    X_train, X_val = standardize(X_train, stats, inplace=True), standardize(X_val, stats, inplace=True)
    if features == "mel":
        X_train, X_val = X_train[:, None], X_val[:, None]   # canal
    elif features == "mel_seq":
        X_train = np.ascontiguousarray(X_train.transpose(0, 2, 1))   # (n, tempo, bandas)
        X_val = np.ascontiguousarray(X_val.transpose(0, 2, 1))
    return X_train, meta_train, X_val, meta_val


def build_model(config):
    if config["model"] == "cnn":
        return SmallCNN(**config.get("model_kwargs", {}))
    return RNNClassifier(**config.get("model_kwargs", {}))


def run(name, config, seed, data):
    out = RUNS_DIR / name / f"seed{seed}"
    if (out / "history.csv").exists():
        print(f"{name} seed {seed}: já existe, a saltar")
        return
    out.mkdir(parents=True, exist_ok=True)
    X_train, meta_train, X_val, meta_val = data
    if config.get("train_filter") == "audible":
        keep = ~((meta_train["murmur"] == "Present") & ~meta_train["murmur_audible"].astype(bool)).to_numpy()
        X_train, meta_train = X_train[keep], meta_train[keep].reset_index(drop=True)
    set_seed(seed)
    model = build_model(config)
    weights = class_weights(meta_train, config["weights"])
    probs_per_epoch = []

    def on_epoch(epoch, m, probs):
        probs_per_epoch.append(probs.astype(np.float16))
        torch.save(m.state_dict(), out / f"epoch{epoch:02d}.pt")

    start = time.time()
    history = fit(model, X_train, encode_labels(meta_train), X_val, meta_val, weights, epochs=EPOCHS,
                  patience=None, verbose=False, on_epoch=on_epoch, **config.get("fit_kwargs", {}))
    np.save(out / "val_probs.npy", np.stack(probs_per_epoch))
    history.to_csv(out / "history.csv", index=False)
    print(f"{name} seed {seed}: {count_parameters(model):,} parâmetros, {EPOCHS} épocas em "
          f"{(time.time() - start) / 60:.1f} min, WA val máx {history['val_wa'].max():.3f} "
          f"(época {history['val_wa'].idxmax() + 1})", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", nargs="+", default=list(EXPERIMENTS))
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    args = parser.parse_args()
    print("device:", DEVICE, flush=True)
    meta_val_saved = False
    for features in FEATURES:   # carrega cada tipo de entrada uma só vez
        names = [n for n in args.exp if features_of(EXPERIMENTS[n]) == features]
        if not names:
            continue
        data = load_data(features)
        if not meta_val_saved:   # a mesma ordem de janelas da validação para todos (para a análise)
            RUNS_DIR.mkdir(exist_ok=True)
            data[3].to_csv(RUNS_DIR / "meta_val.csv", index=False)
            meta_val_saved = True
        for name in names:
            for seed in args.seeds:
                run(name, EXPERIMENTS[name], seed, data)


if __name__ == "__main__":
    main()
