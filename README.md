# Aprendizado Supervisionado — MLP para Regressão

Rede neural **Perceptron de Múltiplas Camadas (MLP)** treinada com scikit-learn para
**prever a nota de qualidade de vinhos tintos** a partir de medidas físico-químicas.

## Vídeo da apresentação
[▶️ Assistir à apresentação](video/apresentacao.mp4) (4 min 28 s)

## Base de dados
**Wine Quality (vinho tinto)** — UCI Machine Learning Repository
(<https://archive.ics.uci.edu/dataset/186/wine+quality>), arquivo `dados/winequality-red.csv`.

- 1599 vinhos tintos portugueses ("Vinho Verde"), sem valores faltantes
- 11 atributos: acidez fixa, acidez volátil, ácido cítrico, açúcar residual, cloretos,
  dióxido de enxofre livre e total, densidade, pH, sulfatos e teor alcoólico
- Alvo: nota de qualidade (0 a 10) dada por especialistas. Na prática vai de 3 a 8,
  com a maioria entre 5 e 6 (média 5,64)
- Atributos mais correlacionados com a nota: **álcool** (+0,48) e **acidez volátil** (−0,39)

## Metodologia
1. **Hold-out** 80% treino (1279) / 20% teste (320).
2. **Pipeline** `StandardScaler` → `MLPRegressor` (normalização ajustada só no treino, sem vazamento).
3. **GridSearchCV com Cross-validation de 5 folds** no treino, testando 36 combinações de
   camadas ocultas, função de ativação, regularização L2 (`alpha`) e taxa de aprendizado,
   escolhendo pelo menor RMSE.
4. **Avaliação final** no conjunto de teste: MAE, MSE, RMSE, MAPE e R², comparadas com uma
   referência que sempre prevê a média do treino.

## Como executar
```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python mlp_regressao.py
```
Leva cerca de 3 minutos. Os gráficos e o CSV de métricas são salvos em `resultados_regressao/`.

## Resultados
Melhor modelo: 11 entradas → 1 camada oculta de 16 neurônios (tanh) → 1 saída,
`alpha=1.0`, `learning_rate_init=0.001`, solver Adam.

| Métrica | Cross-validation (5 folds) | Teste (hold-out) | Chute da média |
|---|---|---|---|
| MAE  | 0,493 ± 0,021 | **0,486** | 0,685 |
| MSE  | 0,418 ± 0,025 | **0,366** | 0,657 |
| RMSE | 0,646 ± 0,020 | **0,605** | 0,811 |
| MAPE | 9,06% ± 0,45  | **8,66%** | 12,24% |
| R²   | 0,355 ± 0,027 | **0,440** | −0,006 |

- Em média a rede erra **meio ponto** na nota (MAE 0,49).
- Arredondando a previsão, acerta a nota exata em **60%** dos vinhos e erra no máximo 1 ponto em **99%**.
- O R² de 0,44 é moderado. É um resultado típico para essa base, porque a nota é subjetiva
  (média de avaliações de especialistas) e quase todos os vinhos têm nota 5 ou 6. O gráfico
  real × previsto mostra que o modelo tende a "puxar para o meio": superestima vinhos ruins
  (3–4) e subestima os ótimos (8), que são raros no treino.

## Arquivos
- `mlp_regressao.py` — código-fonte da regressão
- `dados/winequality-red.csv` — base de dados
- `resultados_regressao/real_vs_previsto.png`, `residuos.png`, `curva_aprendizado.png`, `metricas.png`, `metricas.csv`
