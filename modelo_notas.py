import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

SEED = 42
rng = np.random.default_rng(SEED)

N = 500
horas_estudo = np.clip(rng.normal(10, 4, N), 0, 25)
frequencia = np.clip(rng.normal(82, 12, N), 40, 100)
atividades = np.clip(rng.normal(80, 15, N), 0, 100)
nota_anterior = np.clip(rng.normal(6.5, 1.8, N), 0, 10)

ruido = rng.normal(0, 0.7, N)
nota_final = (
    0.8
    + 0.12 * horas_estudo
    + 0.025 * frequencia
    + 0.020 * atividades
    + 0.45 * nota_anterior
    + ruido
)
nota_final = np.clip(nota_final, 0, 10)

df = pd.DataFrame({
    "aluno_id": np.arange(1, N + 1),
    "horas_estudo": horas_estudo.round(1),
    "frequencia": frequencia.round(1),
    "atividades_entregues": atividades.round(1),
    "nota_anterior": nota_anterior.round(2),
    "nota_final": nota_final.round(2),
})
df.to_csv("base_alunos.csv", index=False)

desc = df.drop(columns="aluno_id").describe().round(2)
corr = df.drop(columns="aluno_id").corr().round(3)

plt.style.use("seaborn-v0_8-whitegrid")

fig, ax = plt.subplots(figsize=(6.5, 5))
im = ax.imshow(corr, cmap="Blues", vmin=0, vmax=1)
ax.set_xticks(range(len(corr)))
ax.set_yticks(range(len(corr)))
labels = [c.replace("_", "\n") for c in corr.columns]
ax.set_xticklabels(labels, fontsize=8)
ax.set_yticklabels(labels, fontsize=8)
for i in range(len(corr)):
    for j in range(len(corr)):
        v = corr.iloc[i, j]
        ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                color="white" if v > 0.6 else "black", fontsize=9)
ax.set_title("Matriz de correlação")
fig.colorbar(im)
fig.tight_layout()
fig.savefig("graficos/02_correlacao.png", dpi=150)
plt.close(fig)

features = ["horas_estudo", "frequencia", "atividades_entregues", "nota_anterior"]
X = df[features]
y = df["nota_final"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=SEED)

lin = LinearRegression().fit(X_tr, y_tr)
rf = RandomForestRegressor(n_estimators=200, random_state=SEED).fit(X_tr, y_tr)

def metricas(modelo, nome):
    pred = modelo.predict(X_te)
    return {
        "modelo": nome,
        "MAE": mean_absolute_error(y_te, pred),
        "RMSE": np.sqrt(mean_squared_error(y_te, pred)),
        "R2": r2_score(y_te, pred),
    }

res = pd.DataFrame([metricas(lin, "Regressão Linear"),
                    metricas(rf, "Random Forest")]).round(3)
cv = cross_val_score(LinearRegression(), X, y, cv=5, scoring="r2")

coef = pd.Series(lin.coef_, index=features).round(4)

pred = lin.predict(X_te)
pred = np.clip(pred, 0, 10)
erro = y_te.values - pred

media_real = y_te.mean()
media_pred = pred.mean()
dp_real = y_te.std(ddof=1)
dp_pred = pred.std(ddof=1)
media_erro = erro.mean()
dp_erro = erro.std(ddof=1)
n = len(erro)
ep = dp_erro / np.sqrt(n)
t_crit = stats.t.ppf(0.975, n - 1)
ic_media_erro = (media_erro - t_crit * ep, media_erro + t_crit * ep)

ip = (media_erro - 1.96 * dp_erro, media_erro + 1.96 * dp_erro)
dentro_ip = np.mean(np.abs(erro - media_erro) <= 1.96 * dp_erro) * 100
dentro_1 = np.mean(np.abs(erro) <= 1.0) * 100
dentro_05 = np.mean(np.abs(erro) <= 0.5) * 100

shapiro_p = stats.shapiro(erro).pvalue

resumo = {
    "n_teste": int(n),
    "media_real": media_real, "media_prevista": media_pred,
    "dp_real": dp_real, "dp_previsto": dp_pred,
    "media_erro": media_erro, "dp_erro": dp_erro,
    "erro_padrao": ep,
    "ic95_media_erro": ic_media_erro,
    "intervalo_previsao95_erro": ip,
    "pct_dentro_intervalo95": dentro_ip,
    "pct_erro_ate_1": dentro_1, "pct_erro_ate_05": dentro_05,
    "shapiro_p": shapiro_p,
    "cv_r2_media": cv.mean(), "cv_r2_dp": cv.std(ddof=1),
}
resumo = {k: (list(map(float, v)) if isinstance(v, tuple) else float(v))
          for k, v in resumo.items()}

fig, ax = plt.subplots(figsize=(6.5, 6))
ax.scatter(y_te, pred, s=22, alpha=0.7, color="#2a6f97")
lim = [min(y_te.min(), pred.min()) - 0.2, 10.2]
ax.plot(lim, lim, "--", color="#c1121f", label="Previsão perfeita")
ax.fill_between(lim, [l + ip[0] for l in lim], [l + ip[1] for l in lim],
                color="#c1121f", alpha=0.08, label="Faixa de erro 95%")
ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel("Nota final real")
ax.set_ylabel("Nota final prevista")
ax.set_title(f"Real x Previsto (R² = {res.loc[0, 'R2']:.3f})")
ax.legend()
fig.tight_layout()
fig.savefig("graficos/03_real_vs_previsto.png", dpi=150)
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
axes[0].hist(erro, bins=20, color="#468faf", edgecolor="white")
axes[0].axvline(media_erro, color="#c1121f", lw=2, label=f"Média = {media_erro:.2f}")
axes[0].axvline(ip[0], color="gray", ls="--", label="Intervalo 95%")
axes[0].axvline(ip[1], color="gray", ls="--")
axes[0].set_xlabel("Erro (real − previsto)")
axes[0].set_ylabel("Frequência")
axes[0].set_title("Distribuição dos erros")
axes[0].legend()
axes[1].scatter(pred, erro, s=18, alpha=0.7, color="#2a6f97")
axes[1].axhline(0, color="#c1121f", lw=1.5)
axes[1].set_xlabel("Nota prevista")
axes[1].set_ylabel("Resíduo")
axes[1].set_title("Resíduos x Previsão")
fig.tight_layout()
fig.savefig("graficos/04_erros.png", dpi=150)
plt.close(fig)

fig, ax = plt.subplots(figsize=(6.5, 4))
imp = coef.abs().sort_values()

coef_pad = (coef * X.std()).sort_values()
ax.barh([c.replace("_", " ") for c in coef_pad.index], coef_pad.values, color="#2a6f97")
ax.set_xlabel("Efeito de +1 desvio padrão na nota final")
ax.set_title("Importância das variáveis (coeficientes padronizados)")
fig.tight_layout()
fig.savefig("graficos/05_importancia.png", dpi=150)
plt.close(fig)

print("=== Estatística descritiva ===\n", desc, "\n")
print("=== Correlações ===\n", corr, "\n")
print("=== Comparação de modelos ===\n", res, "\n")
print("=== Coeficientes (Regressão Linear) ===\n", coef,
      f"\nIntercepto: {lin.intercept_:.4f}\n")
print("=== Estatísticas das previsões ===")
for k, v in resumo.items():
    print(f"{k}: {v}")

desc.to_csv("estatistica_descritiva.csv")
res.to_csv("comparacao_modelos.csv", index=False)
out = {
    "resumo": resumo,
    "modelos": res.to_dict(orient="records"),
    "coef": coef.to_dict(),
    "coef_padronizado": (coef * X.std()).round(3).to_dict(),
    "intercepto": float(lin.intercept_),
    "corr_com_nota": corr["nota_final"].to_dict(),
    "desc": desc.to_dict(),
}
with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
