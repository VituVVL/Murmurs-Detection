# Murmurs Detection

Projeto de Machine Learning II (FCUP, 2026/2027): deteção de sopros cardíacos (*Present* / *Unknown* / *Absent*)
no dataset [CirCor DigiScope](https://physionet.org/content/circor-heart-sound/1.0.3/).

**Grupo:** Victor Vargas, Rodrigo, Marvin

O que já foi feito, as decisões tomadas e os próximos passos estão em [`PROGRESSO.md`](PROGRESSO.md).

## Estrutura

```
Arquivos/    guião do projeto e anotações da aula
data/        dataset CirCor em data/circor/ (não vai para o git) e split.csv (vai)
notebooks/   exploração dos dados e um notebook por modelo
src/         código usado por todos os notebooks (dados e métrica)
results/     figuras, matrizes de confusão e CSV do bónus
```

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
```
