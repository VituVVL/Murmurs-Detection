"""Leitura dos dados, divisão treino/validação/teste por paciente e pré-processamento do áudio."""
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "circor"
RECORDINGS_DIR = DATA_DIR / "training_data"
SPLIT_FILE = ROOT / "data" / "split.csv"

SAMPLE_RATE = 4000  # todas as gravações do CirCor estão a 4000 Hz
SEED = 42


def load_patients():
    """Uma linha por paciente, com os campos do training_data.csv."""
    df = pd.read_csv(DATA_DIR / "training_data.csv")
    return df.rename(columns={"Patient ID": "patient_id", "Murmur": "murmur"})


def load_recordings():
    """Uma linha por gravação: paciente, local de auscultação, caminhos dos ficheiros e rótulo.

    `murmur` é o rótulo do PACIENTE. `murmur_audible` diz se o sopro foi ouvido
    neste local (só pode ser True em pacientes Present).
    """
    patients = load_patients().set_index("patient_id")
    rows = []
    for txt in sorted(RECORDINGS_DIR.glob("*.txt")):
        lines = txt.read_text().splitlines()
        patient_id, n_recordings = int(lines[0].split()[0]), int(lines[0].split()[1])
        patient = patients.loc[patient_id]
        audible = str(patient["Murmur locations"]).split("+")
        for line in lines[1:n_recordings + 1]:
            location, _, wav, tsv = line.split()
            rows.append({
                "patient_id": patient_id,
                "location": location,
                "wav": RECORDINGS_DIR / wav,
                "tsv": RECORDINGS_DIR / tsv,
                "murmur": patient["murmur"],
                "murmur_audible": patient["murmur"] == "Present" and location in audible,
            })
    return pd.DataFrame(rows)


def make_split(val_size=0.15, test_size=0.15, seed=SEED):
    """Divide os PACIENTES em treino/validação/teste, estratificado pela classe Murmur.

    Guarda o resultado em data/split.csv, que vai para o git para todos usarem a mesma divisão.
    """
    patients = load_patients()
    train_val, test = train_test_split(
        patients, test_size=test_size, stratify=patients["murmur"], random_state=seed)
    train, val = train_test_split(
        train_val, test_size=val_size / (1 - test_size), stratify=train_val["murmur"], random_state=seed)

    split = pd.concat([
        pd.DataFrame({"patient_id": train["patient_id"], "subset": "train"}),
        pd.DataFrame({"patient_id": val["patient_id"], "subset": "val"}),
        pd.DataFrame({"patient_id": test["patient_id"], "subset": "test"}),
    ]).sort_values("patient_id")
    split.to_csv(SPLIT_FILE, index=False)
    return split


def load_split():
    """Lê data/split.csv: colunas patient_id e subset (train / val / test)."""
    return pd.read_csv(SPLIT_FILE)


def load_audio(path):
    """Lê um .wav e devolve o sinal em float32, entre -1 e 1."""
    signal, sr = sf.read(path, dtype="float32")
    assert sr == SAMPLE_RATE, f"{path}: esperado {SAMPLE_RATE} Hz, encontrado {sr} Hz"
    return signal


def load_segmentation(path):
    """Lê o .tsv com a segmentação do ciclo cardíaco.

    Colunas: start, end (segundos) e state: 0 = não anotado, 1 = S1, 2 = sístole, 3 = S2, 4 = diástole.
    """
    return pd.read_csv(path, sep="\t", header=None, names=["start", "end", "state"])


def make_windows(signal, window_s, hop_s=None, sr=SAMPLE_RATE):
    """Corta o sinal em janelas de `window_s` segundos, com passo `hop_s` (por omissão sem sobreposição).

    A última janela é preenchida com zeros se ficar incompleta. Devolve um array (n_janelas, amostras).
    """
    window = int(window_s * sr)
    hop = int((hop_s or window_s) * sr)
    n_windows = max(1, int(np.ceil((len(signal) - window) / hop)) + 1)
    padded = np.zeros((n_windows - 1) * hop + window, dtype=signal.dtype)
    padded[:len(signal)] = signal
    return np.stack([padded[i * hop:i * hop + window] for i in range(n_windows)])
