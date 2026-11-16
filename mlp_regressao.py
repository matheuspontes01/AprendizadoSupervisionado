"""
Avaliação - Aprendizado Supervisionado
Rede Neural Perceptron de Múltiplas Camadas (MLP) para REGRESSÃO

Base de dados: Wine Quality (vinho tinto) - UCI Machine Learning Repository
  https://archive.ics.uci.edu/dataset/186/wine+quality
  - 1599 amostras de vinho tinto português "Vinho Verde"
  - 11 atributos físico-químicos (acidez, açúcar, pH, sulfatos, álcool, ...)
  - Alvo: qualidade do vinho, nota de 0 a 10 dada por especialistas

Metodologia:
  1. Hold-out: 80% treino / 20% teste
  2. Busca de hiperparâmetros (GridSearchCV) com Cross-validation de 5 folds,
     somente sobre o conjunto de treino
  3. Avaliação final no conjunto de teste (nunca visto no treino):
     MAE, MSE, RMSE, MAPE e R2
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import GridSearchCV, KFold, cross_validate, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
ARQUIVO_DADOS = "dados/winequality-red.csv"
PASTA_SAIDA = "resultados_regressao"
os.makedirs(PASTA_SAIDA, exist_ok=True)


def linha(titulo=""):
    print("\n" + "=" * 70)
    if titulo:
        print(titulo)
        print("=" * 70)


def calcula_metricas(y_real, y_prev):
    mse = mean_squared_error(y_real, y_prev)
    return {
        "MAE": mean_absolute_error(y_real, y_prev),
        "MSE": mse,
        "RMSE": np.sqrt(mse),
        "MAPE": mean_absolute_percentage_error(y_real, y_prev),
        "R2": r2_score(y_real, y_prev),
    }


# ----------------------------------------------------------------------
# 1. Carregamento e análise exploratória
# ----------------------------------------------------------------------
df = pd.read_csv(ARQUIVO_DADOS, sep=";")
X = df.drop(columns="quality")
y = df["quality"].astype(float)

linha("1. BASE DE DADOS")
print(f"Amostras: {X.shape[0]} | Atributos: {X.shape[1]}")
print(f"Valores faltantes: {df.isna().sum().sum()}")
print(f"Atributos: {', '.join(X.columns)}")
print("\nDistribuição da nota de qualidade (alvo):")
print(y.value_counts().sort_index().to_string())
print(f"Média = {y.mean():.2f} | Desvio padrão = {y.std():.2f}")
print("\nCorrelação de cada atributo com a qualidade:")
print(df.corr()["quality"].drop("quality").sort_values(ascending=False).round(3).to_string())

# ----------------------------------------------------------------------
# 2. Hold-out (split) 80/20
# ----------------------------------------------------------------------
X_treino, X_teste, y_treino, y_teste = train_test_split(
    X, y, test_size=0.2, random_state=SEED
)
linha("2. HOLD-OUT")
print(f"Treino: {len(X_treino)} amostras | Teste: {len(X_teste)} amostras")

# ----------------------------------------------------------------------
# 3. Pipeline: normalização + MLP
#    (o StandardScaler fica dentro do pipeline para evitar vazamento de
#     dados: ele é ajustado apenas nos dados de treino de cada fold)
# ----------------------------------------------------------------------
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("mlp", MLPRegressor(
        max_iter=1000,          # máximo de épocas
        n_iter_no_change=20,    # para quando a loss não melhora por 20 épocas
        random_state=SEED,
    )),
])

grade = {
    "mlp__hidden_layer_sizes": [(16,), (32,), (32, 16)],
    "mlp__activation": ["relu", "tanh"],
    "mlp__alpha": [0.1, 1.0, 10.0],          # regularização L2
    "mlp__learning_rate_init": [1e-3, 1e-2],
}

cv = KFold(n_splits=5, shuffle=True, random_state=SEED)

linha("3. BUSCA DE HIPERPARÂMETROS (GridSearchCV, 5 folds)")
busca = GridSearchCV(pipeline, grade, scoring="neg_root_mean_squared_error", cv=cv, n_jobs=-1)
busca.fit(X_treino, y_treino)
print(f"Combinações testadas: {len(busca.cv_results_['params'])}")
print("Melhores hiperparâmetros:")
for k, v in busca.best_params_.items():
    print(f"  {k.replace('mlp__', ''):<20} = {v}")
print(f"RMSE médio na validação cruzada: {-busca.best_score_:.4f}")

modelo = busca.best_estimator_

# ----------------------------------------------------------------------
# 4. Cross-validation do melhor modelo (todas as métricas, no treino)
# ----------------------------------------------------------------------
linha("4. CROSS-VALIDATION DO MELHOR MODELO (5 folds, conjunto de treino)")
metricas_cv = {
    "MAE": "neg_mean_absolute_error",
    "MSE": "neg_mean_squared_error",
    "RMSE": "neg_root_mean_squared_error",
    "MAPE": "neg_mean_absolute_percentage_error",
    "R2": "r2",
}
res_cv = cross_validate(modelo, X_treino, y_treino, cv=cv, scoring=metricas_cv)
cv_media, cv_desvio = {}, {}
for nome in metricas_cv:
    vals = res_cv[f"test_{nome}"]
    if nome != "R2":          # o sklearn devolve erros com sinal negativo
        vals = -vals
    cv_media[nome], cv_desvio[nome] = vals.mean(), vals.std()
    print(f"  {nome:<5}: {vals.mean():.4f} ± {vals.std():.4f}   folds={np.round(vals, 3)}")

# ----------------------------------------------------------------------
# 5. Avaliação final no conjunto de teste (hold-out)
# ----------------------------------------------------------------------
y_pred = modelo.predict(X_teste)
teste = calcula_metricas(y_teste, y_pred)

# Referência: um "modelo" que sempre chuta a média do treino
base = calcula_metricas(y_teste, np.full(len(y_teste), y_treino.mean()))

linha("5. AVALIAÇÃO NO CONJUNTO DE TESTE (hold-out)")
print(f"  {'Métrica':<6} {'MLP':>8} {'Chute da média':>16}")
for nome in teste:
    print(f"  {nome:<6} {teste[nome]:>8.4f} {base[nome]:>16.4f}")
print(f"\n  MAPE = {teste['MAPE']:.2%}: em média a previsão erra "
      f"{teste['MAE']:.2f} ponto na nota de qualidade.")

# Acerto ao arredondar a previsão para a nota inteira mais próxima
acerto = (np.round(y_pred) == y_teste).mean()
acerto_1 = (np.abs(np.round(y_pred) - y_teste) <= 1).mean()
print(f"  Arredondando a previsão: acerta a nota exata em {acerto:.1%} dos vinhos "
      f"e erra por no máximo 1 ponto em {acerto_1:.1%}.")

print("\nExemplos de previsões no teste:")
exemplos = pd.DataFrame({"real": y_teste.values[:10], "previsto": y_pred[:10].round(2)})
print(exemplos.to_string(index=False))

pd.DataFrame({
    "metrica": list(teste),
    "teste_holdout": list(teste.values()),
    "cv_media": [cv_media[m] for m in teste],
    "cv_desvio": [cv_desvio[m] for m in teste],
    "baseline_media": list(base.values()),
}).to_csv(f"{PASTA_SAIDA}/metricas.csv", index=False)

# ----------------------------------------------------------------------
# 6. Gráficos
# ----------------------------------------------------------------------
# 6.1 Real x previsto
fig, ax = plt.subplots(figsize=(6, 5))
jitter = np.random.default_rng(SEED).uniform(-0.15, 0.15, len(y_teste))
ax.scatter(y_teste + jitter, y_pred, alpha=0.5, s=18)
lim = [y.min() - 0.5, y.max() + 0.5]
ax.plot(lim, lim, "k--", lw=1, label="Previsão perfeita")
ax.set_xlim(lim)
ax.set_ylim(lim)
ax.set_xlabel("Qualidade real")
ax.set_ylabel("Qualidade prevista")
ax.set_title(f"Real x previsto (teste) - R² = {teste['R2']:.3f}")
ax.legend()
fig.tight_layout()
fig.savefig(f"{PASTA_SAIDA}/real_vs_previsto.png", dpi=150)
plt.close(fig)

# 6.2 Distribuição dos resíduos
residuos = y_teste - y_pred
fig, ax = plt.subplots(figsize=(6, 4))
ax.hist(residuos, bins=30, color="tab:purple", edgecolor="white")
ax.axvline(0, color="k", ls="--", lw=1)
ax.set_xlabel("Resíduo (real - previsto)")
ax.set_ylabel("Quantidade de vinhos")
ax.set_title("Distribuição dos erros no teste")
fig.tight_layout()
fig.savefig(f"{PASTA_SAIDA}/residuos.png", dpi=150)
plt.close(fig)

# 6.3 Curva de aprendizado (loss de treino por época)
mlp = modelo.named_steps["mlp"]
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(mlp.loss_curve_, color="tab:red")
ax.set_xlabel("Época")
ax.set_ylabel("Loss (erro quadrático / 2)")
ax.set_title("Curva de aprendizado da MLP")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(f"{PASTA_SAIDA}/curva_aprendizado.png", dpi=150)
plt.close(fig)

# 6.4 Métricas: MLP x chute da média
nomes = ["MAE", "RMSE", "R2"]
fig, ax = plt.subplots(figsize=(7, 4))
pos = np.arange(len(nomes))
ax.bar(pos - 0.2, [teste[m] for m in nomes], 0.4, label="MLP")
ax.bar(pos + 0.2, [base[m] for m in nomes], 0.4, label="Chute da média", color="tab:gray")
ax.set_xticks(pos, ["MAE", "RMSE", "R²"])
ax.axhline(0, color="k", lw=0.8)
ax.set_title("Métricas no teste: MLP x referência")
ax.legend()
fig.tight_layout()
fig.savefig(f"{PASTA_SAIDA}/metricas.png", dpi=150)
plt.close(fig)

linha("6. ARQUIVOS GERADOS")
for arq in ["real_vs_previsto.png", "residuos.png", "curva_aprendizado.png",
            "metricas.png", "metricas.csv"]:
    print(f"  {PASTA_SAIDA}/{arq}")
print(f"\nArquitetura final: {X.shape[1]} entradas -> {mlp.hidden_layer_sizes} ocultas -> "
      f"1 saída | épocas treinadas: {mlp.n_iter_}")
