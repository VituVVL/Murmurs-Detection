# Murmurs Detection

Projeto de Machine Learning II (FCUP, 2026/2027): deteção de sopros cardíacos (*Present* / *Unknown* / *Absent*)
no dataset [CirCor DigiScope](https://physionet.org/content/circor-heart-sound/1.0.3/).

**Grupo:** Victor Vargas, Rodrigo, Marvin

O que já foi feito, as decisões tomadas e os próximos passos estão em [`PROGRESSO.md`](PROGRESSO.md).

## Estrutura

```
Arquivos/    guião do projeto e anotações da aula
data/        dataset CirCor em data/circor/ (não vai para o git) e split.csv (vai)
notebooks/   01 exploração, 02 pré-processamento, 03 CNN, 04 RNN, 05 seeds e agregação, 06 variantes
src/         código partilhado (dados, features, modelos, treino, métrica) e os scripts finais
results/     figuras, tabelas e matrizes de confusão
runs/        pesos/probabilidades das corridas (gerado pelos scripts, não vai para o git)
```

## Reproduzir os resultados finais

Precisa do dataset em `data/circor/` e, para o bónus, dos dados sem rótulo em `data/CirCor_test_students/test_data/`.
Corre em CPU, mas é muito mais rápido com uma GPU NVIDIA (`uv sync` instala o torch com CUDA). Tempos numa RTX 5060.
As features são calculadas na primeira corrida e guardadas em `data/cache/`.

```bash
uv sync
# 1. Configurações candidatas, 5 seeds na divisão 70/15/15 (análise nos notebooks 05 e 06)
uv run python -m src.run_seeds --exp cnn_wmetric rnn_wmetric --seeds 42 1 2 3 4
# 2. Regra de agregação por paciente com a mesma época (só validação) -> results/regras_agregacao.csv
uv run python -m src.compare_rules
# 3. Avaliação final: validação cruzada 5 folds nos 942 pacientes, CNN e RNN, 3 seeds (~45 min)
#    -> results/cv_resultados.csv, cv_por_fold.csv, cv_matriz_<modelo>.csv
uv run python -m src.run_cv
# 4. Bónus: ensemble de 15 RNN (5 folds x 3 seeds) -> PL2_G05_test_results.csv (~12 min)
uv run python -m src.run_bonus
```

Configuração final: RNN (GRU bidirecional sobre MFCC de janelas de 5 s), pesos de classe 5/3/1 (os da métrica),
época pela menor loss da validação, previsão por paciente pela média das probabilidades das suas gravações (`mean`).
Os scripts saltam os treinos que já existem em `runs/`; `--report` (em `run_cv`) e `--predict-only` (em `run_bonus`)
só refazem as tabelas/o CSV.

## Ambiente

O ambiente é gerido com o [uv](https://docs.astral.sh/uv/getting-started/installation/).
As bibliotecas estão no `pyproject.toml` e as versões exatas no `uv.lock`.

```bash
uv sync                 # cria/atualiza a .venv; correr depois de cada git pull
uv add <biblioteca>     # adicionar uma biblioteca (fazer commit do pyproject.toml e uv.lock)
uv run python script.py # correr um script com o ambiente do projeto
```

No PyCharm, escolher `.venv` como interpretador (normalmente é detetado sozinho).

## Dados

Colocar o dataset em `data/circor/` (a pasta com `training_data/` e `training_data.csv`).
A divisão treino/validação/teste por paciente já está em `data/split.csv`. **Não a gerar de novo.**

Nos notebooks, para usar o código de `src/`:

```python
import sys; sys.path.append("..")
from src.data import load_recordings, load_split, load_audio
from src.metrics import weighted_accuracy, confusion_matrix
from src.features import build_features, fit_standardizer, standardize   # pré-processamento (notebook 02)
from src.training import fit, predict_proba, patient_predictions, log_experiment   # treino e registo (notebook 03)
```
