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
- **O teste só se usa uma vez**, no fim, com as configurações já escolhidas na validação.
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
| 27/11 | Variantes CNN (SpecAugment, audible) e RNN (LSTM, deltas, log-mel, 128 unidades, audible) com 5 seeds; escolher a configuração final | em curso (notebook 06) |
| 28/11 | Avaliação única no teste (WA + matriz por paciente, as duas redes); script do bónus; README com instruções | por fazer |
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
- Dúvidas para o professor (aula de 27/11) em `Arquivos/duvidas.txt`: retreinar com treino + validação para o bónus,
  5 seeds vs. validação cruzada, ensemble no bónus, se o professor corre o código.
