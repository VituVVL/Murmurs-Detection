# Contexto do projeto para o Claude Code

Projeto de Machine Learning II (FCUP): deteção de sopros cardíacos no dataset CirCor (Present / Unknown / Absent).
Grupo **PL2 G05**: Victor, Rodrigo, Marvin. Registo detalhado de tudo o que foi feito: `PROGRESSO.md` (ler primeiro).

## Prazos e entregáveis

- **Entrega no Moodle: 30/11/2026, 23:59.** Apresentações a 03/12 e 10/12 (todos os membros presentes).
- Entregar: código com instruções (README), slides em PDF para 10 min, CSV do bónus, formulário de autoavaliação
  (está no Moodle).
- CSV do bónus: `PL2_G05_test_results.csv`, colunas `ID,Murmur` (ID = patient_id), 100 linhas. Dados sem rótulo em
  `data/CirCor_test_students/test_data` (fora do git; ler com `load_features("bonus", ...)` de `src/features.py`).
- O guião (`Arquivos/Guião Projeto.pdf`) exige divisão por paciente, que o teste não seja usado para decisões e que
  se reporte a WA e a matriz de confusão por paciente no teste. Quem não souber explicar a solução é penalizado.

## Regras de trabalho

- Responder em **português** (o Victor escreve em português do Brasil; a documentação do repositório está em
  português europeu). Explicar o porquê das escolhas: o Victor tem de saber explicar tudo na apresentação.
- **Não fazer commit nem push sem aprovação do Victor.** Mensagens de commit em português.
- **O teste só se usa uma vez**, no fim, com as configurações já escolhidas na validação. Na validação cruzada
  final (ver plano), cada fold de teste é previsto uma só vez e nenhuma decisão muda depois de ver esses resultados.
- Treinos na GPU do desktop do Victor (RTX 5060). Os pesos e as probabilidades das corridas ficam em `runs/` (fora
  do git, só existem no desktop); a cache das features fica em `data/cache/` (fora do git).

## Ambiente

- `uv sync` instala o torch com CUDA (`2.15.0+cu132`, índice em `pyproject.toml`). Sem GPU NVIDIA corre em CPU,
  muito mais devagar (~95 s por época na CNN contra ~3 s na GPU).
- Treino com várias seeds: `uv run python -m src.run_seeds --exp <nomes> --seeds 42 1 2 3 4`.

## Plano (aprovado a 25/11)

| Dia | Tarefa | Estado |
|---|---|---|
| 26/11 | Cache das features; baselines CNN e RNN com 5 seeds na GPU; pesos de classe e regra de agregação | feito (notebook 05) |
| 27/11 | Variantes CNN (SpecAugment, audible) e RNN (LSTM, deltas, log-mel, 128 unidades, audible) com 5 seeds; escolher a configuração final | feito (notebook 06) |
| 28/11 | (1) `src.compare_rules` no desktop e congelar a regra (`RULE` em `src/run_cv.py`); (2) `src.run_cv`: validação cruzada 5 folds nos 942 pacientes, as duas redes (WA + matriz por paciente); (3) bónus: RNN retreinada com 85/15 dos 942, ensemble de seeds; README | (1) feito: regra `mean`; (2) feito: RNN 0.768; (3) bónus por validação cruzada (`src/run_bonus.py`); README por fazer |
| 29/11 | Slides em PDF (10 min + apêndice); ensaiar a explicação com o grupo | por fazer |
| 30/11 | Folga; zip do código (sem dataset, sem `runs/`, sem este ficheiro); autoavaliação; submissão | por fazer |

## Decisões e resultados até agora

- Divisão 70/15/15 por paciente, estratificada, seed 42 (`data/split.csv`): 658 / 142 / 142 pacientes.
- Métrica: WA por paciente com pesos 5/3/1 (Present/Unknown/Absent). Prever sempre Present dá 0.499.
- Entrada: janelas de 5 s; log-mel 64×313 para a CNN, MFCC 313×13 para a RNN; normalização com estatísticas do treino.
- **Avaliação justa** (notebook 05): a época é escolhida numa metade dos pacientes da validação e a WA é medida na
  outra (20 divisões). Escolher e medir nos mesmos pacientes inflaciona a WA em 0.03 a 0.05.
- **Configuração comum aos dois modelos** (notebook 05): pesos de classe iguais aos da métrica (5/3/1), regra de
  agregação `mean` (média das probabilidades das gravações do paciente), época pela menor loss da validação.
  Com ela: **CNN 0.744 ± 0.027, RNN (GRU) 0.760 ± 0.023** (WA justa na validação, 5 seeds).
- Os pesos de classe da métrica foram a melhoria mais clara (+0.05 na CNN, +0.08 na RNN).
- **Variantes (notebook 06): nenhuma melhorou as referências** (SpecAugment, filtro `murmur_audible`, LSTM, deltas do
  MFCC, RNN sobre log-mel, 128 unidades: todas 0.02 a 0.04 abaixo). Ensemble das 5 seeds quase igual (CNN 0.749,
  RNN 0.761).
- **Configuração final:** `cnn_wmetric` e `rnn_wmetric` em `src/run_seeds.py` (as 5 seeds já treinadas em `runs/`),
  regra `mean`, época de cada seed pela menor loss da validação (toda a validação). A RNN é o melhor modelo.
- **Respostas do professor (aula de 27/11, `Arquivos/duvidas.txt`):** (1) para o bónus, retreinar com mais dados,
  incluindo o teste, guardando uma parte para validação (85/15, 80/20... à nossa escolha); (2) prefere **validação
  cruzada** a uma só divisão, para usar o dataset todo; (3) regra de agregação e época: escolha nossa, mas mostrada
  e justificada; no bónus o objetivo é a maior WA; (4) ensemble no bónus é permitido; (9) pode correr o código, mas
  não conta para a nota.
- **Regra de agregação:** com a mesma época (menor loss), a `mean` ganha à `any` (CNN 0.744 vs 0.719, RNN 0.760 vs
  0.728, `results/seeds_criterios_regras.csv`). `src.compare_rules` (26/11, `results/regras_agregacao.csv`): a `mean`
  perde ~3 *Present* por seed mas a `any` dá ~23 *Absent* como *Present*; nos pacientes em que discordam a `mean` ganha
  mais WA (CNN 123 vs 95, RNN 158 vs 80 pontos). **Regra congelada: `mean`.**
- **Avaliação final (confirmada com o professor, opção B; script `src/run_cv.py`):** validação cruzada estratificada por paciente, 5 folds
  nos 942 pacientes (treino + validação + teste antigos). Em cada fold: o fold de fora é o teste (~188 pacientes);
  dos restantes, 15% servem de validação interna só para escolher a época (menor loss). Configuração congelada
  (`cnn_wmetric` / `rnn_wmetric`, pesos 5/3/1, regra escolhida). Reportar WA média ± desvio entre folds e a matriz
  por paciente das previsões fora do fold (cada paciente previsto uma vez). Seeds: 3 por fold (~30 treinos, ~1-1.5 h
  na GPU). As decisões anteriores (notebooks 05/06) foram tomadas só com a validação antiga, por isso a validação
  cruzada é só medição.
- **Resultado da validação cruzada (26/11, `results/cv_resultados.csv`):** CNN 0.742 ± 0.022 (ensemble 0.757),
  **RNN 0.768 ± 0.023 (ensemble 0.771)**. Recall Unknown muito baixo (0.04-0.07): a fraqueza a mostrar.
- **Bónus (`src/run_bonus.py`):** ensemble por validação cruzada (mesmos 5 folds; modelo k treina nos outros 4 e
  escolhe a época no fold k), RNN, 3 seeds por fold = 15 modelos, regra `mean` → `PL2_G05_test_results.csv`.
