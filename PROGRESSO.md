# Progresso do projeto

Resumo do que foi feito em **02/11/2026**, das decisões tomadas e do que falta.
Datas: **checkpoint 05/11** · entrega 30/11 · apresentações 03/12 e 10/12.

---

## 1. Estado do git

- Tudo o que está descrito neste ficheiro já tem commit e push (commits "Atualizações iniciais.").
- O ramo principal chama-se **`main`**. Antes chamava-se `master`, o nome que o Git para Windows põe por omissão;
  foi renomeado antes de entrarem o Rodrigo e o Marvin.
- O dataset (`data/circor/`) **não** vai para o git: está no `.gitignore`.

---

## 2. Estrutura do projeto

Primeiro criei uma estrutura grande (configs, scripts, tests, docs...). Ficou confusa e foi substituída por uma simples:

```
Arquivos/    guião do projeto + anotações da aula
data/        circor/ = dataset (fora do git) · split.csv = divisão dos dados (no git)
notebooks/   exploração e um notebook por modelo
src/         código usado por todos os notebooks: data.py e metrics.py
results/     figuras, matrizes de confusão, CSV do bónus (vazia por agora)
```

**Porquê assim:** as anotações da aula dizem que o trabalho é experimentar muito e no fim escolher o que apresentar.
Isso faz-se melhor em notebooks. Em `src/` fica só o que **tem de ser igual para todos**, senão os resultados
dos dois modelos não são comparáveis: a leitura dos dados, a divisão por paciente e a métrica.

---

## 3. Ambiente Python (uv)

Pediste um env conda. Em vez disso usei o **uv**, que é o que o Aly e a Amanda usaram no `aarv_01` e que resolve
exatamente o "manter atualizado o que o ambiente precisa".

- `pyproject.toml` tem a lista de bibliotecas; `uv.lock` tem as versões exatas de tudo. Os dois vão para o git.
- O ambiente fica em `.venv/` (fora do git). Python 3.12.

| Situação | Comando |
|---|---|
| Depois de cada `git pull` (ou na primeira vez) | `uv sync` |
| Preciso de uma biblioteca nova | `uv add <nome>` e depois commit do `pyproject.toml` + `uv.lock` |
| Correr um script | `uv run python script.py` |

O Rodrigo e o Marvin só precisam de [instalar o uv](https://docs.astral.sh/uv/getting-started/installation/) e correr `uv sync`.

**Bibliotecas instaladas:** numpy, pandas, scipy, scikit-learn, librosa, soundfile, torch, matplotlib, seaborn, ipykernel, jupyter.

**Notas:**
- O torch instalado é **só CPU** (2.15.0+cpu): o teu PC não tem GPU NVIDIA. Treinos pesados vão ser lentos.
  As anotações dizem para falar com o professor se algo demorar demasiado.
- O PyCharm detetou a `.venv` e passou a usá-la como interpretador. Antes disso o projeto não tinha nenhum interpretador
  configurado. O Python do sistema era só o atalho da Microsoft Store, que não funciona.

---

## 4. Dataset

- Copiado de `E:\[VITU]\...\circor-heart-sound-1.0.3\the-circor-digiscope-phonocardiogram-dataset-1.0.3` para `data/circor/`.
- 10 435 ficheiros, 577 MB. **Todos os checksums (`SHA256SUMS.txt`) foram verificados e estão corretos.**

**O que há em `data/circor/training_data/`:**

| Ficheiro | Quantos | Conteúdo |
|---|---|---|
| `<paciente>.txt` | 942 | metadados do paciente + lista das suas gravações |
| `<paciente>_<local>.wav` | 3163 | áudio, mono, 4000 Hz |
| `<paciente>_<local>.tsv` | 3163 | segmentação do ciclo cardíaco: início, fim, estado (0 = não anotado, 1 = S1, 2 = sístole, 3 = S2, 4 = diástole) |
| `<paciente>_<local>.hea` | 3163 | cabeçalho técnico da gravação (não usamos) |

Os locais são AV, PV, TV e MV (as 4 válvulas) e, raramente, Phc (4 gravações). Alguns pacientes têm mais do que uma gravação
no mesmo local: `_1`, `_2` (21 casos).

`training_data.csv` tem uma linha por paciente com todos os campos clínicos. Só usamos **`Murmur`** (o rótulo) e
**`Murmur locations`** (secção 5).

**Ficheiro estragado:** `50782_MV_1.tsv` só tem uma linha inválida (`0 0 28`). **Não é culpa nossa:** o checksum
é igual ao publicado pelo PhysioNet e ao original no disco E:. O `.wav` dessa gravação está bom; só a segmentação não serve.

---

## 5. `src/data.py`: leitura dos dados

Para usar num notebook:

```python
import sys; sys.path.append("..")
from src.data import load_patients, load_recordings, load_split, load_audio, load_segmentation, make_windows
```

| Função | O que faz |
|---|---|
| `load_patients()` | `training_data.csv` como tabela, uma linha por paciente. Colunas renomeadas: `patient_id`, `murmur`. |
| `load_recordings()` | Uma linha por **gravação**: `patient_id`, `location`, `wav` (caminho), `tsv` (caminho), `murmur`, `murmur_audible`. |
| `make_split()` | Cria a divisão treino/validação/teste e grava em `data/split.csv`. **Já foi corrida, não voltar a correr.** |
| `load_split()` | Lê `data/split.csv`: colunas `patient_id` e `subset` (`train` / `val` / `test`). |
| `load_audio(path)` | Lê um `.wav` e devolve o sinal (float32, entre -1 e 1). Confirma que está a 4000 Hz. |
| `load_segmentation(path)` | Lê um `.tsv`: colunas `start`, `end` (segundos) e `state` (0–4). |
| `make_windows(sinal, window_s, hop_s)` | Corta o sinal em janelas de `window_s` segundos com passo `hop_s` (sem passo = sem sobreposição). A última janela é completada com zeros. Devolve um array `(n_janelas, amostras)`. Ex.: `make_windows(x, 5, 2.5)` dá janelas de 5 s com 50% de sobreposição. |

### `murmur` vs. `murmur_audible` (importante)

- **`murmur`** é o rótulo do **paciente** (*Present* / *Unknown* / *Absent*), copiado para todas as gravações dele.
- **`murmur_audible`** diz se o sopro foi ouvido **naquele local**. **Não é calculado por nós:** vem do campo
  `Murmur locations` do dataset, onde o médico anotou em que locais ouviu o sopro.

Exemplo: o paciente 13918 é *Present*, tem gravações em AV, PV, TV e MV, e `Murmur locations = TV`. Então:

| Gravação | `murmur` | `murmur_audible` |
|---|---|---|
| 13918_AV | Present | False |
| 13918_PV | Present | False |
| 13918_TV | Present | **True** |
| 13918_MV | Present | False |

Pacientes *Absent* e *Unknown* não têm locais anotados, por isso `murmur_audible` é sempre `False`.

**Porque importa:** 19% das gravações de pacientes *Present* não têm sopro audível. Se treinarmos cada gravação com o
rótulo do paciente, o modelo vê exemplos "*Present*" sem sopro nenhum.

---

## 6. Divisão dos dados e métrica

### Divisão (`data/split.csv`)

| | Pacientes | Present | Unknown | Absent | Gravações |
|---|---|---|---|---|---|
| treino | 658 | 125 | 48 | 485 | 2210 |
| validação | 142 | 27 | 10 | 105 | 474 |
| teste | 142 | 27 | 10 | 105 | 479 |

- 70 / 15 / 15 **por paciente**: todas as gravações de um paciente ficam no mesmo conjunto (o notebook verifica isto).
- **Estratificada:** as três partes têm 19% / 7% / 74%, as mesmas proporções do dataset completo.
- Seed fixa (42) e gravada em CSV no git: os três usam exatamente a mesma divisão.
- **Regra do guião:** o teste **não** serve para decidir nada (arquitetura, hiperparâmetros, early stopping, limiares).
  Tudo isso usa a validação. O teste só se usa no fim, para os resultados finais.

### Métrica (`src/metrics.py`)

```python
from src.metrics import weighted_accuracy, confusion_matrix
weighted_accuracy(y_true, y_pred)   # listas com um rótulo por PACIENTE
confusion_matrix(y_true, y_pred)    # tabela 3x3
```

- **`weighted_accuracy`:** a métrica oficial da PhysioNet 2022 usada no guião. Cada paciente *Present* conta 5 vezes,
  *Unknown* 3 vezes e *Absent* 1 vez. Dá o mesmo resultado que a fórmula da Secção 3.4 do guião (verificado).
- **`confusion_matrix`:** igual à Tabela 1 do guião: **linhas = previsto, colunas = verdadeiro**.
  Atenção: é a **transposta** da `confusion_matrix` do sklearn, que usa linhas = verdadeiro.

**Referências mínimas (classificadores triviais):**

| Prever sempre | Accuracy | Weighted accuracy |
|---|---|---|
| Absent | 0.738 | 0.387 |
| Present | 0.190 | **0.499** |
| Unknown | 0.072 | 0.114 |

Prever sempre *Absent* acerta 74% dos pacientes, mas a WA é baixa. **Um modelo só vale a pena se passar de 0.499.**

---

## 7. Notebook de exploração (`notebooks/01_exploracao_dados.ipynb`)

Já está executado: abre e vê os gráficos sem correr nada. Para correr de novo, escolhe a `.venv` como kernel
(demora ~1–2 min porque lê os 3163 áudios).

| Secção | O que mostra |
|---|---|
| 1. Pacientes e classes | desequilíbrio, classificadores triviais, idade/sexo/campanha por classe |
| 2. Gravações e locais | gravações por paciente (a maioria tem 4), gravações por local e classe |
| 3. Duração | 5.2 a 64.5 s, mediana ~21 s; nenhuma tem menos de 5 s |
| 4. Onde o sopro é audível | 81% das gravações *Present* têm sopro audível; por local e local mais audível |
| 5. Segmentação | fração não anotada e frequência cardíaca por classe |
| 6. Sinais | sinal com o ciclo cardíaco colorido, espetro médio, mel-espectrograma e MFCC por grupo |
| 7. Divisão | confirma proporções e que nenhum paciente está em dois conjuntos |
| 8. Conclusões | resumo abaixo |

### Principais conclusões

1. **Desequilíbrio:** 74% / 19% / 7%. É preciso tratá-lo no treino (pesos de classe na loss ou reamostragem),
   senão o modelo aprende a dizer sempre *Absent*.
2. **Rótulo do paciente ≠ rótulo da gravação** (secção 5 acima). Opções a testar:
   (a) rótulo do paciente em todas as gravações; (b) usar `murmur_audible` e tratar as gravações não audíveis como
   *Absent* ou retirá-las do treino.
3. ***Unknown* parece ser "gravação de má qualidade", não um tipo de sopro:**
   - mediana de 76% da gravação não anotada, contra 39–48% nas outras classes;
   - mais energia em frequências altas (ruído);
   - mais comum em recém-nascidos e bebés (~15–20% dos pacientes, contra ~5% nas crianças e adolescentes).
4. **Janelas de 4–5 s** cabem em todas as gravações e dão várias janelas por gravação (mais exemplos de treino).
5. O **espetro médio** de *Present* e *Absent* é parecido. A diferença está no **padrão ao longo do tempo**: no exemplo
   *Present* vê-se energia contínua durante a sístole, no *Absent* só os "cliques" S1 e S2. Isto justifica usar uma
   representação tempo-frequência (mel-espectrograma → CNN) ou uma sequência (MFCC → RNN).
6. A **segmentação** pode ajudar a alinhar janelas com os batimentos, mas em média ~45% de cada gravação não está anotada.

---

## 8. Decisões (provisórias) para levar ao checkpoint

| Tema | Proposta | Justificação |
|---|---|---|
| Modelos | **CNN 2D** sobre mel-espectrogramas + **RNN** (GRU/LSTM) sobre sequências de MFCC | O professor disse que o MLP não parece adequado; o sopro é um padrão no tempo |
| Divisão | 70/15/15 por paciente, estratificada | Regras da Secção 3.4 do guião |
| Entrada | janelas de 4–5 s | Todas as gravações têm ≥ 5 s |
| Desequilíbrio | pesos de classe na loss | Classes 74/19/7 e métrica com pesos 5/3/1 |
| Uma previsão por paciente | janela → gravação → paciente; *Present* se alguma gravação for *Present* | O sopro pode ser audível só num local |
| Referência | WA de 0.499 (sempre *Present*) | É o valor a bater |

---

## 9. Próximos passos

1. Mostrar o notebook ao Rodrigo e ao Marvin; todos têm de saber explicar tudo no checkpoint (05/11).
2. **Perguntar ao professor no checkpoint:**
   - Os dados escondidos do bónus vão ter os ficheiros `.tsv` de segmentação? Se não, não podemos depender deles na previsão.
   - Usar `murmur_audible` (de `Murmur locations`) para rotular as gravações no treino é permitido? O guião diz para usar
     "só o rótulo Murmur", mas também refere o campo `Murmur locations`.
3. Começar os modelos: um notebook para a CNN (`02_cnn.ipynb`) e outro para a RNN (`03_rnn.ipynb`), os dois a usar
   `src/data.py` e `src/metrics.py`. Ter um primeiro modelo, mesmo simples, até ao checkpoint seria um bónus.
