"""
Avaliação - Aprendizado Supervisionado
Rede Neural Perceptron de Múltiplas Camadas (MLP) para CLASSIFICAÇÃO

Base de dados: Breast Cancer Wisconsin (Diagnostic) - UCI Machine Learning Repository
  https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic
  - 569 amostras, 30 atributos numéricos (características dos núcleos celulares
    extraídas de imagens de biópsia por agulha fina)
  - Classe: 0 = maligno, 1 = benigno

Metodologia:
  1. Hold-out estratificado: 80% treino / 20% teste
  2. Busca de hiperparâmetros (GridSearchCV) com Cross-validation estratificada
     de 5 folds, somente sobre o conjunto de treino
  3. Avaliação final no conjunto de teste (nunca visto no treino):
     acurácia, precisão, recall, F1-score e matriz de confusão
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
PASTA_SAIDA = "resultados"
os.makedirs(PASTA_SAIDA, exist_ok=True)


def linha(titulo=""):
    print("\n" + "=" * 70)
    if titulo:
        print(titulo)
        print("=" * 70)


# ----------------------------------------------------------------------
# 1. Carregamento e análise exploratória
# ----------------------------------------------------------------------
dados = load_breast_cancer(as_frame=True)
X = dados.data
y = dados.target
nomes_classes = list(dados.target_names)  # ['malignant', 'benign']

linha("1. BASE DE DADOS")
print(f"Amostras: {X.shape[0]} | Atributos: {X.shape[1]}")
print("Distribuição das classes:")
for i, nome in enumerate(nomes_classes):
    qtd = (y == i).sum()
    print(f"  {i} = {nome:<10} -> {qtd} ({qtd / len(y):.1%})")
print("\nPrimeiras linhas (5 primeiros atributos):")
print(X.iloc[:5, :5])

# ----------------------------------------------------------------------
# 2. Hold-out (split) estratificado 80/20
# ----------------------------------------------------------------------
X_treino, X_teste, y_treino, y_teste = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED
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
    ("mlp", MLPClassifier(
        max_iter=1000,          # máximo de épocas
        n_iter_no_change=20,    # para quando a loss não melhora por 20 épocas
        random_state=SEED,
    )),
])

grade = {
    "mlp__hidden_layer_sizes": [(16,), (32,), (64,), (32, 16), (64, 32)],
    "mlp__activation": ["relu", "tanh"],
    "mlp__alpha": [1e-3, 1e-2, 1e-1, 1.0],   # regularização L2
    "mlp__learning_rate_init": [1e-3, 1e-2],
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

linha("3. BUSCA DE HIPERPARÂMETROS (GridSearchCV, 5 folds)")
busca = GridSearchCV(pipeline, grade, scoring="f1_macro", cv=cv, n_jobs=-1)
busca.fit(X_treino, y_treino)
print(f"Combinações testadas: {len(busca.cv_results_['params'])}")
print("Melhores hiperparâmetros:")
for k, v in busca.best_params_.items():
    print(f"  {k.replace('mlp__', ''):<20} = {v}")
print(f"F1-macro médio na validação cruzada: {busca.best_score_:.4f}")

modelo = busca.best_estimator_

# ----------------------------------------------------------------------
# 4. Cross-validation do melhor modelo (todas as métricas, no treino)
# ----------------------------------------------------------------------
linha("4. CROSS-VALIDATION DO MELHOR MODELO (5 folds, conjunto de treino)")
metricas_cv = {
    "acuracia": "accuracy",
    "precisao": "precision_macro",
    "recall": "recall_macro",
    "f1": "f1_macro",
}
res_cv = cross_validate(modelo, X_treino, y_treino, cv=cv, scoring=metricas_cv)
for nome in metricas_cv:
    vals = res_cv[f"test_{nome}"]
    print(f"  {nome:<9}: {vals.mean():.4f} ± {vals.std():.4f}   folds={np.round(vals, 3)}")

# ----------------------------------------------------------------------
# 5. Avaliação final no conjunto de teste (hold-out)
# ----------------------------------------------------------------------
y_pred = modelo.predict(X_teste)

acc = accuracy_score(y_teste, y_pred)
prec = precision_score(y_teste, y_pred, average="macro")
rec = recall_score(y_teste, y_pred, average="macro")
f1 = f1_score(y_teste, y_pred, average="macro")
cm = confusion_matrix(y_teste, y_pred)

linha("5. AVALIAÇÃO NO CONJUNTO DE TESTE (hold-out)")
print(f"  Acurácia : {acc:.4f}")
print(f"  Precisão : {prec:.4f}  (média macro)")
print(f"  Recall   : {rec:.4f}  (média macro)")
print(f"  F1-score : {f1:.4f}  (média macro)")
print("\nRelatório por classe:")
print(classification_report(y_teste, y_pred, target_names=nomes_classes, digits=4))
print("Matriz de confusão (linhas = real, colunas = previsto):")
print(pd.DataFrame(cm, index=[f"real {c}" for c in nomes_classes],
                   columns=[f"prev {c}" for c in nomes_classes]))

# Salva métricas em CSV
pd.DataFrame({
    "metrica": ["acuracia", "precisao", "recall", "f1"],
    "teste_holdout": [acc, prec, rec, f1],
    "cv_media": [res_cv[f"test_{m}"].mean() for m in metricas_cv],
    "cv_desvio": [res_cv[f"test_{m}"].std() for m in metricas_cv],
}).to_csv(f"{PASTA_SAIDA}/metricas.csv", index=False)

# ----------------------------------------------------------------------
# 6. Gráficos
# ----------------------------------------------------------------------
# 6.1 Matriz de confusão
fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=nomes_classes, yticklabels=nomes_classes, ax=ax)
ax.set_xlabel("Previsto")
ax.set_ylabel("Real")
ax.set_title("Matriz de confusão - conjunto de teste")
fig.tight_layout()
fig.savefig(f"{PASTA_SAIDA}/matriz_confusao.png", dpi=150)
plt.close(fig)

# 6.2 Curva de aprendizado da MLP (loss de treino por época)
mlp = modelo.named_steps["mlp"]
fig, ax1 = plt.subplots(figsize=(7, 4))
ax1.plot(mlp.loss_curve_, color="tab:red")
ax1.set_xlabel("Época")
ax1.set_ylabel("Loss (entropia cruzada)")
ax1.set_title("Curva de aprendizado da MLP")
ax1.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(f"{PASTA_SAIDA}/curva_aprendizado.png", dpi=150)
plt.close(fig)

# 6.3 Métricas: CV x teste
fig, ax = plt.subplots(figsize=(7, 4))
pos = np.arange(len(metricas_cv))
medias = [res_cv[f"test_{m}"].mean() for m in metricas_cv]
desvios = [res_cv[f"test_{m}"].std() for m in metricas_cv]
ax.bar(pos - 0.2, medias, 0.4, yerr=desvios, capsize=4, label="Cross-validation (5 folds)")
ax.bar(pos + 0.2, [acc, prec, rec, f1], 0.4, label="Teste (hold-out)")
ax.set_xticks(pos, ["Acurácia", "Precisão", "Recall", "F1-score"])
ax.set_ylim(0.85, 1.0)
ax.set_title("Métricas de avaliação")
ax.legend(loc="lower right")
fig.tight_layout()
fig.savefig(f"{PASTA_SAIDA}/metricas.png", dpi=150)
plt.close(fig)

linha("6. ARQUIVOS GERADOS")
print(f"  {PASTA_SAIDA}/matriz_confusao.png")
print(f"  {PASTA_SAIDA}/curva_aprendizado.png")
print(f"  {PASTA_SAIDA}/metricas.png")
print(f"  {PASTA_SAIDA}/metricas.csv")
print(f"\nArquitetura final: 30 entradas -> {mlp.hidden_layer_sizes} ocultas -> "
      f"{mlp.n_outputs_} saída | épocas treinadas: {mlp.n_iter_}")
