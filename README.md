# Murmurs Detection

Projeto de Machine Learning II (FCUP, 2026/2027): deteção de sopros cardíacos (*Present* / *Unknown* / *Absent*)
no dataset [CirCor DigiScope](https://physionet.org/content/circor-heart-sound/1.0.3/).

**Grupo:** Victor Vargas, Rodrigo, Marvin

## Estrutura

```
Arquivos/    guião do projeto e anotações da aula
data/        dataset CirCor (não vai para o git)
notebooks/   exploração dos dados e um notebook por modelo
src/         código usado por todos os notebooks (dados e métrica)
results/     figuras, matrizes de confusão e CSV do bónus
```

## Como correr

1. `pip install -r requirements.txt`
2. Colocar o dataset em `data/circor/` (pasta com `training_data/` e `training_data.csv`).
3. Abrir os notebooks em `notebooks/`.
