# -*- coding: utf-8 -*-
"""
05. Décomposition asymétrique de la convexité et test d'égalité.

Le terme quadratique de la régression traite identiquement une baisse et une
hausse de même ampleur, le carré effaçant le signe. Un coefficient unique ne
distingue donc pas deux mécanismes opposés : l'amplification des baisses, qui
traduit une défaillance du plancher obligataire, et l'écrasement des hausses,
qui traduit une option de conversion devenue sans valeur.

La spécification est dédoublée selon le signe du mouvement du sous-jacent, dans
l'esprit de Henriksson et Merton (1981). L'égalité des deux coefficients est
testée par un test de Wald sur leur différence.
"""

# %% CELLULE 1 - Chargement et segments
import numpy as np
import pandas as pd
from scipy import stats

panel = pd.read_parquet("panel.parquet")
panel = panel[panel["exploitable"]].copy()
panel["annee"] = panel["date"].dt.year

BORNES = [0, 0.20, 0.40, 0.60, 10]
ETIQUETTES = ["0-20 %", "20-40 %", "40-60 %", "> 60 %"]
RETARDS = 5

delta_initial = panel.sort_values("date").groupby(["isin", "annee"]).first()["delta"] / 100
panel = panel.join(delta_initial.rename("delta_initial"), on=["isin", "annee"])
panel["segment"] = pd.cut(panel["delta_initial"], BORNES, labels=ETIQUETTES)


# %% CELLULE 2 - Estimation asymétrique avec test de Wald
# -----------------------------------------------------------------------------
# Modèle estimé :
#   r_CB = a + b·r + c_bas·r²·1{r<0} + c_haut·r²·1{r>0} + b_ret·r(-1) + e
#
# Le test porte sur H0 : c_bas = c_haut. La variance de la différence vaut
# V[c_bas] + V[c_haut] - 2·Cov[c_bas, c_haut], d'où le recours à la matrice
# de covariance complète et non aux seuls écarts-types.
def estimer_asymetrie(rendement_cb, rendement_action, retards=RETARDS):
    donnees = pd.DataFrame({"y": rendement_cb, "x": rendement_action}).dropna()
    donnees["retard"] = donnees["x"].shift(1)
    donnees = donnees.dropna()
    if len(donnees) < 80:
        return None

    x = donnees["x"].values
    y = donnees["y"].values
    baisse = (x < 0).astype(float)
    hausse = (x > 0).astype(float)
    X = np.column_stack([np.ones(len(x)), x, (x ** 2) * baisse, (x ** 2) * hausse,
                         donnees["retard"].values])

    coefficients, *_ = np.linalg.lstsq(X, y, rcond=None)
    residus = y - X @ coefficients

    # Covariance de Newey-West, qui corrige hétéroscédasticité et autocorrélation
    XtX_inverse = np.linalg.inv(X.T @ X)
    somme = (X * residus[:, None]).T @ (X * residus[:, None])
    for decalage in range(1, retards + 1):
        poids = 1 - decalage / (retards + 1)
        produit = (X[decalage:] * residus[decalage:, None]).T @ (X[:-decalage] * residus[:-decalage, None])
        somme += poids * (produit + produit.T)
    covariance = XtX_inverse @ somme @ XtX_inverse

    difference = coefficients[2] - coefficients[3]
    variance_difference = covariance[2, 2] + covariance[3, 3] - 2 * covariance[2, 3]
    if variance_difference <= 0:
        return None
    t_difference = difference / np.sqrt(variance_difference)
    p_value = 2 * (1 - stats.norm.cdf(abs(t_difference)))

    return dict(
        beta=coefficients[1], c_bas=coefficients[2], c_haut=coefficients[3],
        ecart=difference, p_value=p_value, observations=len(donnees),
    )


# %% CELLULE 3 - Résultat agrégé par année
MIN_TITRES_AGREGE = 30

lignes = []
for annee, bloc in panel.groupby("annee"):
    quotidien = bloc.groupby("date").agg(
        cb=("ret_cb", "mean"), action=("ret_propre", "mean"), effectif=("isin", "size"))
    quotidien = quotidien[quotidien["effectif"] >= MIN_TITRES_AGREGE].sort_index()
    sortie = estimer_asymetrie(quotidien["cb"], quotidien["action"])
    if sortie is None:
        continue
    sortie["annee"] = annee
    lignes.append(sortie)

asymetrie = pd.DataFrame(lignes).set_index("annee")
asymetrie["asymetrique"] = np.where(asymetrie["p_value"] < 0.05, "oui",
                                    np.where(asymetrie["p_value"] < 0.10, "limite", "non"))
asymetrie.to_parquet("test_asymetrie.parquet")

print("Décomposition asymétrique de la convexité, ensemble du gisement")
print(asymetrie[["c_bas", "c_haut", "ecart", "p_value", "asymetrique"]].round(3).to_string())
print()
print("Lecture : un c_bas positif signifie que la convertible amortit les reculs.")
print("Un c_haut négatif signifie qu'elle ne suit pas les rebonds. Une p-value")
print("inférieure à 0,05 établit que les deux directions diffèrent réellement.")
print()
print("ATTENTION : les années de faible amplitude de marché produisent des")
print("coefficients mal identifiés, faute de mouvements assez amples pour")
print("contraindre le terme quadratique. Ne pas les interpréter.")


# %% CELLULE 4 - Résultat par segment, années de stress
ANNEES_EXAMINEES = [2020, 2022]
MIN_TITRES_SEGMENT = 5

lignes = []
for annee in ANNEES_EXAMINEES:
    for etiquette in ETIQUETTES:
        bloc = panel[(panel["annee"] == annee) & (panel["segment"] == etiquette)]
        quotidien = bloc.groupby("date").agg(
            cb=("ret_cb", "mean"), action=("ret_propre", "mean"), effectif=("isin", "size"))
        quotidien = quotidien[quotidien["effectif"] >= MIN_TITRES_SEGMENT].sort_index()
        sortie = estimer_asymetrie(quotidien["cb"], quotidien["action"])
        if sortie is None:
            continue
        sortie.update(annee=annee, segment=etiquette, titres=bloc["isin"].nunique())
        lignes.append(sortie)

par_segment = pd.DataFrame(lignes)
print("Décomposition par segment de delta")
print(par_segment[["annee", "segment", "titres", "c_bas", "c_haut", "ecart", "p_value"]]
      .round(3).to_string(index=False))


# %% CELLULE 5 - Pourquoi conserver aussi l'estimateur symétrique
# -----------------------------------------------------------------------------
# L'estimateur asymétrique divise par deux les observations de chaque coefficient
# et perd donc en puissance. Il sert à tester l'asymétrie, non à mesurer le
# niveau. Le contrôle ci-dessous montre qu'un coefficient symétrique peut
# masquer entièrement la structure : sur 2020, il moyenne deux valeurs de signes
# opposés et hautement significatives pour produire un résultat non significatif.
def estimer_symetrique(rendement_cb, rendement_action, retards=RETARDS):
    donnees = pd.DataFrame({"y": rendement_cb, "x": rendement_action}).dropna()
    donnees["retard"] = donnees["x"].shift(1)
    donnees = donnees.dropna()
    x, y = donnees["x"].values, donnees["y"].values
    X = np.column_stack([np.ones(len(x)), x, x ** 2, donnees["retard"].values])
    coefficients, *_ = np.linalg.lstsq(X, y, rcond=None)
    residus = y - X @ coefficients
    XtX_inverse = np.linalg.inv(X.T @ X)
    somme = (X * residus[:, None]).T @ (X * residus[:, None])
    for decalage in range(1, retards + 1):
        poids = 1 - decalage / (retards + 1)
        produit = (X[decalage:] * residus[decalage:, None]).T @ (X[:-decalage] * residus[:-decalage, None])
        somme += poids * (produit + produit.T)
    ecarts_types = np.sqrt(np.diag(XtX_inverse @ somme @ XtX_inverse))
    return coefficients[2], coefficients[2] / ecarts_types[2]


print("Comparaison des deux estimateurs, segment 40-60 %")
for annee in ANNEES_EXAMINEES:
    bloc = panel[(panel["annee"] == annee) & (panel["segment"] == "40-60 %")]
    quotidien = bloc.groupby("date").agg(
        cb=("ret_cb", "mean"), action=("ret_propre", "mean"), effectif=("isin", "size"))
    quotidien = quotidien[quotidien["effectif"] >= MIN_TITRES_SEGMENT].sort_index()
    coefficient, t = estimer_symetrique(quotidien["cb"], quotidien["action"])
    ligne = par_segment[(par_segment["annee"] == annee) & (par_segment["segment"] == "40-60 %")]
    if len(ligne):
        print(f"  {annee} : symétrique {coefficient:+.2f} (t {t:+.2f}) | "
              f"asymétrique {ligne['c_bas'].iloc[0]:+.2f} à la baisse et "
              f"{ligne['c_haut'].iloc[0]:+.2f} à la hausse")
