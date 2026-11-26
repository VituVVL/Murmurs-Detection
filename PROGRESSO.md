# Progresso do projeto

Resumo do que foi feito (início a 02/11/2026; modelos desde 21/11, secção 9), das decisões tomadas e do que falta.
Última atualização: 24/11/2026.
Datas: checkpoint 05/11 (feito, secção 10) · **entrega 30/11** · apresentações 03/12 e 10/12.

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
- **torch com CUDA para todos (desde 25/11):** o `pyproject.toml` manda o uv buscar o torch ao índice CUDA 13.2 do
  PyTorch (`2.15.0+cu132`), em Windows e Linux. Basta `uv sync` / `uv run` normais, **sem** `--no-sync`.
  - Com GPU NVIDIA (desktop do Victor, RTX 5060; driver com CUDA ≥ 13.2): o `src/training.py` usa-a sozinho (`DEVICE`).
    Uma época demora ~3 s na CNN e ~1 s na GRU (contra ~95 s e ~120 s em CPU).
  - Sem GPU NVIDIA (portáteis): o mesmo torch corre no CPU. Só o download é maior (~2 GB, uma vez).
  - Confirmar: `uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"`.
  - Para abrir num PC sem GPU um `.pt` treinado na GPU: `torch.load(caminho, map_location="cpu")`.
  - Os treinos de 26/11 em diante foram feitos na GPU. Os números não são exatamente iguais aos do CPU (operações
    da GPU não são bit-a-bit determinísticas), por isso as tabelas finais usam só corridas da GPU.
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
| `make_windows(sinal, window_s, hop_s, pad_last)` | Corta o sinal em janelas de `window_s` segundos com passo `hop_s` (sem passo = sem sobreposição). Se a última janela ficar incompleta: com `pad_last=True` (por omissão) é completada com zeros; com `pad_last=False` recua para terminar no fim do sinal (sem zeros; é o que o pré-processamento usa). Devolve um array `(n_janelas, amostras)`. Ex.: `make_windows(x, 5, 2.5)` dá janelas de 5 s com 50% de sobreposição. |

### `src/features.py`: pré-processamento comum aos modelos

Justificação de cada escolha em `notebooks/02_preprocessamento.ipynb`.

```python
from src.features import build_features, fit_standardizer, standardize
X_train, meta_train = build_features("train", kind="mel")    # "mel" (CNN) ou "mfcc" (RNN)
X_val, meta_val = build_features("val", kind="mel")
stats = fit_standardizer(X_train, "mel")                     # médias e desvios SÓ do treino
X_train, X_val = standardize(X_train, stats), standardize(X_val, stats)
```

| Passo | Decisão |
|---|---|
| Amplitude | média 0, desvio 1 por gravação (`normalize_audio`); não pelo pico, porque a maioria das gravações tem picos saturados |
| Janelas | 5 s, sem sobreposição, última recuada (sem zeros): 11 225 / 2 431 / 2 426 janelas (treino / validação / teste) |
| CNN | log-mel-espectrograma `(n, 64, 313)`; acrescentar o canal com `X[:, None]` |
| RNN | MFCC `(n, 313, 13)`: sequência de 313 instantes com 13 coeficientes |
| Rótulo da janela | o do paciente (opção (a)); `meta` guarda paciente, local, gravação e `murmur_audible` |
| Normalização das features | por banda mel / coeficiente MFCC, com médias e desvios só do treino |

Calcular tudo demora ~1.5 min, por isso as features não são guardadas em disco.

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

## 9. Modelos e experiências (desde 21/11)

### `src/training.py`: treino comum à CNN e à RNN

Fica em `src/` porque a agregação e o critério de paragem têm de ser iguais nos dois modelos (WA comparáveis).

| Função | O que faz |
|---|---|
| `set_seed()` | fixa as seeds do Python, numpy e torch (seed 42) |
| `class_weights(meta_train)` | pesos da loss: `total / (3 × janelas da classe)` → Present 1.74, Unknown 6.94, Absent 0.44. *Peso × nº de janelas* é igual nas 3 classes |
| `fit(...)` | Adam + CrossEntropy com pesos; early stopping pela **WA do paciente na validação** (paciência 8), fica a melhor época. `lr_plateau={...}` liga o `ReduceLROnPlateau` (sobre a loss da validação). Usa a **GPU** se existir (`DEVICE`, ver secção 3) |
| `predict_proba(model, X)` | probabilidades por janela |
| `patient_predictions(meta, probs)` | janela → gravação (média das probabilidades, `argmax`) → paciente (*Present* se alguma gravação for *Present*; senão *Unknown* se alguma for; senão *Absent*) |
| `log_experiment(nome, descrição, history, patients)` | grava `results/<nome>_history.csv` e acrescenta uma linha a `results/experiencias.csv` |

**Registo:** todas as experiências ficam em `results/experiencias.csv` (WA, recall por classe, melhor época,
média da WA nas últimas 5 épocas, minutos). Os pesos (`*.pt`) não vão para o git; os CSV e PNG vão.

### CNN (`notebooks/03_cnn.ipynb`)

CNN 2D pequena (98 mil parâmetros): 4 blocos `Conv 3×3 → BatchNorm → ReLU → MaxPool` (16 → 32 → 64 → 128 filtros),
*global average pooling*, dropout 0.3, 3 saídas. Entrada: log-mel `(1, 64, 313)`. ~95 s por época em CPU.

| Experiência | O que muda | WA val (melhor época) | WA val (média últimas 5) | Present / Unknown / Absent | Épocas | Tempo |
|---|---|---|---|---|---|---|
| `cnn_baseline` | Adam lr 1e-3 fixo, wd 1e-4, batch 64 | **0.759** (ép. 25) | **0.686** | 22/27 · 4/10 · 83/105 | 33 | 54 min |
| `cnn_plateau` | + `ReduceLROnPlateau` (×0.3 se a loss val não descer em 3 épocas) | 0.719 (ép. 11) | 0.644 | 22/27 · 6/10 · 66/105 | 19 | 29 min |
| `cnn_audible` | sem as janelas de treino *Present* com sopro não audível no local | 0.730 (ép. 10) | 0.615 | 23/27 · 6/10 · 64/105 | 18 | 21 min |
| `cnn_specaug` | + SpecAugment no treino (2 faixas freq ≤ 8 bandas, 2 faixas tempo ≤ 0.64 s) | 0.726 (ép. 16) | 0.647 | 21/27 · 5/10 · 76/105 | 24 | 36 min |
| `cnn_overlap` | janelas de treino com 50% de sobreposição (19 087 em vez de 11 225) | 0.696 (ép. 5) | 0.619 | 20/27 · 3/10 · 79/105 | 13 | 31 min |

**Baseline:** passa bem a referência de 0.499, mas o treino é **muito instável** (WA da validação entre 0.34 e 0.76 de
época para época) e há *overfitting* a partir da época ~13 (loss de treino desce, a da validação sobe). O 0.759 é o pico
de uma curva ruidosa, logo otimista. Nas janelas de pacientes *Present*, 69% são previstas *Present* quando o sopro é
audível no local e só 18% quando não é.

**`cnn_plateau`:** mais estável (WA 0.58–0.67 no fim, sem overfitting), mas **pior**. O lr foi reduzido cedo demais
(época 10) porque a loss da validação quase não desce desde a época 1; chegou a 2.7e-5 e a rede deixou de aprender.
Muitos mais *Absent* dados como *Unknown* (22 contra 7). As 9 primeiras épocas são iguais às da baseline (treino
reprodutível). Diferenças de ~0.04 na WA são ~3 pacientes *Present*: com uma seed só, não são conclusivas.

**Experiências 2–4 (generalização), uma mudança de cada vez em relação à baseline:** nenhuma bateu a baseline.
`cnn_audible` apanha mais *Present* mas erra muitos mais *Absent*; `cnn_specaug` é a mais próxima; `cnn_overlap`
parou cedo (pico na época 5). A 4 foi repetida: a primeira tentativa foi parada por falta de memória, e por isso
`build_features`, `fit_standardizer` e `standardize(..., inplace=True)` passaram a poupar memória (mesmo resultado).

**Conclusão até aqui:** todas as variantes ficam entre ~0.70 e ~0.76, com diferenças de 1 a 4 pacientes *Present*.
Com uma só seed e 142 pacientes na validação, **o ruído da avaliação é do tamanho dos efeitos**: não dá para dizer
qual é melhor. Antes de mais experiências: repetir as melhores com 2–3 seeds e/ou escolher a época por um critério
mais suave (loss da validação, WA média de 3 épocas). **CNN de referência: `cnn_baseline`.**

Mais uma prova da instabilidade: uma corrida da mesma CNN feita à parte (19/11, notebook próprio, não guardado no git)
com **paciência 6** parou na época 13 com WA val **0.604**; a `cnn_baseline`, com paciência 8, só chegou ao pico de
0.759 na época 25. O critério de paragem pesa tanto como as mudanças testadas.

### RNN (`notebooks/04_rnn.ipynb`)

GRU bidirecional, 2 camadas × 64 unidades, dropout 0.3, *pooling* média + máximo no tempo (~106 mil parâmetros,
como a CNN). Entrada: MFCC `(313, 13)` normalizado por coeficiente. Treino igual à `cnn_baseline`. ~2 min por época.

| Experiência | O que muda | WA val (melhor época) | WA val (média últimas 5) | Present / Unknown / Absent | Épocas | Tempo |
|---|---|---|---|---|---|---|
| `rnn_baseline` | — | 0.693 (ép. 4) | 0.637 | 20/27 · 7/10 · 66/105 | 12 | 25 min |

**RNN baseline:** passa a referência mas fica abaixo da CNN, exceto no *Unknown* (7/10, o melhor até agora). Treino
mais calmo (loss da validação estável, pouco overfitting). O erro principal são os **28 *Absent* dados como *Unknown***:
por janela só 14% das janelas *Absent* saem *Unknown*, mas a regra "*Unknown* se **alguma** gravação for *Unknown*"
basta uma em ~4 gravações. **Próximo passo:** rever a regra de agregação com as probabilidades dos modelos já
guardados (sem treinar de novo), igual para a CNN e a RNN.

### Várias seeds na GPU (`notebooks/05_seeds_agregacao.ipynb`, 26/11)

Com a GPU, cada configuração foi treinada com **5 seeds** (`src/run_seeds.py`, 40 épocas fixas, probabilidades e pesos
de todas as épocas guardados em `runs/`, fora do git). Isto permite comparar, sem treinar de novo, pesos de classe,
regras de agregação (`patient_predictions(..., rule="any" | "half" | "mean")`) e critérios de escolha da época.
Para não ser otimista, a época é escolhida numa metade dos pacientes da validação e a WA é medida na outra (20 divisões).

| | procedimento original (`any`, melhor WA) | configuração comum escolhida (`mean`, menor loss val) |
|---|---|---|
| CNN, pesos equilibrados | 0.657 ± 0.014 | 0.655 ± 0.025 |
| CNN, **pesos da métrica (5/3/1)** | 0.707 ± 0.013 | **0.744 ± 0.027** |
| RNN, pesos equilibrados | 0.687 ± 0.030 | 0.689 ± 0.014 |
| RNN, **pesos da métrica (5/3/1)** | 0.768 ± 0.044 | **0.760 ± 0.023** |

- Os valores de uma seed (0.759 da CNN) eram **otimistas**: escolher e medir nos mesmos pacientes dá +0.03 a +0.05.
- **Pesos de classe iguais aos da métrica** são a melhoria mais clara (+0.05 a +0.08, acima do desvio entre seeds):
  acabam com os *Absent* dados como *Unknown* (159 → 18 em 525 na CNN). Passam a ser o padrão.
- Com esses pesos, regra e critério da época mudam ≤ 0.02 (ruído). Escolheu-se uma configuração **comum** aos dois
  modelos: regra `mean` e época pela loss da validação.
- **A RNN fica à frente da CNN** e é muito mais estável entre épocas. (Em CPU tinha parado cedo, na época 12.)

---

## 10. Próximos passos

1. Mostrar os notebooks ao Rodrigo e ao Marvin; todos têm de saber explicar tudo (checkpoint 05/11, apresentações).
2. **Checkpoint (05/11):** o professor achou ok o que foi apresentado (decisões da secção 8). Respostas às perguntas:
   - **Usar `murmur_audible` no treino:** **pode**, mas cabe-nos avaliar se é o melhor. A experiência `cnn_audible`
     (secção 9) é essa avaliação; a escolha tem de ser justificada na apresentação.
   - **Os dados do bónus trazem os `.tsv` de segmentação?** A resposta não ficou registada. Por segurança, a previsão
     **não depende da segmentação** (o pipeline atual já não a usa).
3. Modelos: CNN em `03_cnn.ipynb` e RNN em `04_rnn.ipynb` (baselines feitas, secção 9), ambos com
   `src/training.py`.
4. **Rever a regra de agregação** (o maior erro são *Absent* dados como *Unknown*) com as probabilidades dos modelos
   já treinados. Os `.pt` não estão no git: pedir ao Rodrigo ou treinar de novo.
5. **Fecho (30/11):** resultados no teste só no fim; slides em PDF (10 min); zip do código; CSV do bónus.

### Ideias para as experiências

- **Deltas do MFCC (RNN):** comparar só MFCC (13 por instante) com MFCC + Δ + ΔΔ (39 por instante).
  Os deltas são a 1.ª e a 2.ª derivada de cada coeficiente no tempo: mostram se o espectro está a mudar
  (ex.: início e fim de S1/S2, sopro a começar na sístole).
  ```python
  d1 = librosa.feature.delta(mfcc)            # (13, n_instantes); width=9 colunas ≈ 144 ms com hop_length=64
  d2 = librosa.feature.delta(mfcc, order=2)
  features = np.vstack([mfcc, d1, d2])        # (39, n_instantes)
  ```
  Uma RNN pode aprender a dinâmica sozinha, mas com poucos dados os deltas costumam ajudar. Escolher pela WA na validação.
- **Normalizar cada coeficiente** (média 0, desvio 1, calculados só no treino) antes da RNN. As escalas são muito
  diferentes: numa gravação, o coeficiente 0 do MFCC tem desvio ~43, os últimos ~5; o delta ~9 e o delta-delta ~5.
- **Tamanho da janela do espectrograma** (`n_fft`): testar 256 (64 ms, 15,6 Hz), 512 (atual, 128 ms, 7,8 Hz) e
  1024 (256 ms, 3,9 Hz). Janela maior = frequências mais nítidas, mas S1/S2 mais borrados no tempo.
- **`fmax`:** com `fmax=1000` as 64 bandas mel ficam igualmente espaçadas (15,4 Hz cada), ou seja, é um espectrograma
  linear. Testar `fmax=2000` (bandas de 26 a 52 Hz, mais detalhe nos graves).
- **Seeds nos modelos:** fixar `torch.manual_seed(SEED)` e `np.random.seed(SEED)` para o treino ser reprodutível.
- **Sobreposição de 50% nas janelas do treino** (`hop_s=2.5`): 27 335 janelas em vez de 16 082, uma forma simples de
  *data augmentation*.
- **Volume original como entrada extra:** a normalização apaga o volume, mas o *Present* e sobretudo o *Unknown* são,
  em média, mais altos do que o *Absent* (notebook 02, secção 1). Testar dar `log10(desvio da gravação)` ao modelo.
- **Pesos de classe calculados nas janelas:** nas janelas do treino o *Unknown* é só 4.8% (7% nos pacientes), porque os
  pacientes *Unknown* têm menos gravações.
