"""Pré-processamento comum aos modelos: normalização do áudio, janelas, mel-espectrograma, MFCC e normalização das features.

Uso típico num notebook de modelo:

    from src.features import build_features, fit_standardizer, standardize
    X_train, meta_train = build_features("train", kind="mel")
    X_val, meta_val = build_features("val", kind="mel")
    stats = fit_standardizer(X_train, "mel")   # médias e desvios SÓ do treino
    X_train, X_val = standardize(X_train, stats), standardize(X_val, stats)
"""
import librosa
import numpy as np
import pandas as pd

from src.data import SAMPLE_RATE, load_audio, load_recordings, load_split, make_windows

# Parâmetros escolhidos no notebook 02_preprocessamento (justificação lá).
WINDOW_S = 5.0     # duração de cada janela (s); todas as gravações têm >= 5.2 s
HOP_S = None       # passo entre janelas (s); None = sem sobreposição
N_FFT = 512        # 128 ms a 4000 Hz
HOP_LENGTH = 64    # 16 ms -> 313 colunas por janela de 5 s
N_MELS = 64
FMAX = 1000        # provisório: comparar com 2000 nas experiências
N_MFCC = 13


def normalize_audio(signal):
    """Média 0 e desvio-padrão 1 por gravação.

    Corrige as diferenças de volume entre gravações (o desvio varia ~16x). Não se usa o pico porque
    a maioria das gravações tem picos saturados de poucas amostras (batidas no estetoscópio).
    """
    signal = signal - signal.mean()
    return signal / (signal.std() + 1e-8)


def mel_spectrogram(windows, fmax=FMAX):
    """Log-mel-espectrograma de um lote de janelas: (n, amostras) -> (n, N_MELS, colunas), em dB.

    top_db=None: o corte por omissão do librosa é relativo ao máximo do LOTE inteiro, o que faria o resultado
    de uma janela depender das outras janelas do lote.
    """
    mel = librosa.feature.melspectrogram(y=windows, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH,
                                         n_mels=N_MELS, fmax=fmax)
    return librosa.power_to_db(mel, ref=1.0, top_db=None).astype(np.float32)


def mfcc(windows, fmax=FMAX, n_mfcc=N_MFCC):
    """MFCC de um lote de janelas: (n, amostras) -> (n, n_mfcc, colunas)."""
    return librosa.feature.mfcc(S=mel_spectrogram(windows, fmax), n_mfcc=n_mfcc).astype(np.float32)


def subset_recordings(subset):
    """Gravações de um subconjunto ("train", "val" ou "test") da divisão em data/split.csv."""
    recordings = load_recordings().merge(load_split(), on="patient_id")
    return recordings[recordings["subset"] == subset].reset_index(drop=True)


def build_windows(recordings, window_s=WINDOW_S, hop_s=HOP_S):
    """Normaliza cada gravação e corta-a em janelas (a última recua até ao fim do sinal, sem zeros).

    Devolve (janelas, meta): janelas é (n, amostras); meta tem uma linha por janela com o paciente, o local,
    a gravação de origem e os rótulos, para depois juntar as previsões por gravação e por paciente.
    """
    windows, meta = [], []
    for rec in recordings.itertuples():
        w = make_windows(normalize_audio(load_audio(rec.wav)), window_s, hop_s, pad_last=False)
        windows.append(w)
        meta += [{"patient_id": rec.patient_id, "location": rec.location, "recording": rec.wav.stem,
                  "window": i, "murmur": rec.murmur, "murmur_audible": rec.murmur_audible} for i in range(len(w))]
    return np.concatenate(windows), pd.DataFrame(meta)


def build_features(subset, kind="mel", window_s=WINDOW_S, hop_s=HOP_S, fmax=FMAX, batch_size=512):
    """Features de todas as janelas de um subconjunto.

    kind="mel":  X tem forma (n, N_MELS, colunas)  -> entrada da CNN (acrescentar o canal: X[:, None])
    kind="mfcc": X tem forma (n, colunas, N_MFCC)  -> entrada da RNN (sequência de vetores, tempo primeiro)
    O rótulo de cada janela (meta["murmur"]) é o do paciente (opção (a) do notebook 01).
    """
    windows, meta = build_windows(subset_recordings(subset), window_s, hop_s)
    extract = {"mel": mel_spectrogram, "mfcc": mfcc}[kind]
    X = np.concatenate([extract(windows[i:i + batch_size], fmax) for i in range(0, len(windows), batch_size)])
    if kind == "mfcc":
        X = X.transpose(0, 2, 1)
    return X, meta


def fit_standardizer(X_train, kind):
    """Média e desvio de cada banda mel / coeficiente MFCC, calculados SÓ no treino.

    Para "mel" (n, bandas, tempo) a estatística é por banda; para "mfcc" (n, tempo, coeficientes) é por coeficiente.
    """
    axis = {"mel": (0, 2), "mfcc": (0, 1)}[kind]
    mean = X_train.mean(axis=axis, keepdims=True)
    std = X_train.std(axis=axis, keepdims=True)
    return mean.astype(np.float32), std.astype(np.float32)


def standardize(X, stats):
    """Aplica a normalização calculada no treino (a mesma em treino, validação e teste)."""
    mean, std = stats
    return (X - mean) / (std + 1e-8)
