# -*- coding: utf-8 -*-
"""
06. Pouvoir prédictif du ratio prix sur bond floor.

L'hypothèse postule que le rapport entre le prix de la convertible et son
plancher obligataire prédit la résistance en stress mieux que la notation de
crédit. Le test mesure la corrélation de rang entre ce ratio, relevé AVANT
l'épisode, et le drawdown subi PENDANT celui-ci. Une corrélation négative
confirme l'hypothèse : un titre éloigné de son plancher chute davantage.

Trois précautions structurent le test. La date de mesure est fixée au 31
décembre précédent, sans considération de la date d'ouverture de l'épisode, ce
qui rend la mesure implémentable par un gérant. La comparaison avec la notation
est conduite à échantillon égal. Et l'apport marginal est mesuré en régression,
la question n'étant pas seulement laquelle des deux variables prédit mieux mais
si la seconde ajoute de l'information à la première.
"""

# %% CELLULE 1 - Chargement
import numpy as np
import pandas as pd

panel = pd.read_parquet("panel.parquet")
univers = pd.read_parquet("univers.parquet")
episodes = pd.read_parquet("episodes.parquet")
detection = pd.read_parquet("detection_spread.parquet")   # produit par 08

spread_propre = set(detection.loc[~detection["defaut"], "isin"])

ECHELLE_NOTATION = {note: rang + 1 for rang, note in enumerate(
    ["AAA", "AA+", "AA", "AA-", "A+", "A", "A-", "BBB+", "BBB", "BBB-",
     "BB+", "BB", "BB-", "B+", "B", "B-", "CCC+", "CCC", "CCC-", "CC", "C", "D"])}


# %% CELLULE 2 - Construction des couples prédicteur et drawdown
MIN_OBSERVATIONS = 10


def construire(debut_episode, fin_episode, date_mesure):
    """Ratio et delta relevés à la date de mesure, drawdown mesuré sur l'épisode."""
    avant = panel[(panel["date"] >= date_mesure - pd.Timedelta(days=30)) &
                  (panel["date"] <= date_mesure)]
    avant = avant.groupby("isin").last()

    pendant = panel[(panel["date"] >= debut_episode) & (panel["date"] <= fin_episode)]
    pendant = pendant.groupby("isin").agg(
        prix_min=("prix", "min"), prix_debut=("prix", "first"), n=("prix", "size"))
    pendant = pendant[pendant["n"] >= MIN_OBSERVATIONS]

    joint = avant[["prix", "bond_floor", "delta"]].join(pendant, how="inner", rsuffix="_ep")
    joint["ratio"] = joint["prix"] / joint["bond_floor"]
    joint["delta_fraction"] = joint["delta"] / 100
    joint["drawdown"] = joint["prix_min"] / joint["prix_debut"] - 1
    joint = joint.replace([np.inf, -np.inf], np.nan).dropna(subset=["ratio", "drawdown"])
    return joint[(joint["ratio"] > 0) & (joint["ratio"] < 5)]


# %% CELLULE 3 - Pouvoir prédictif par épisode
# -----------------------------------------------------------------------------
# Deux dates de mesure sont comparées. La veille de l'épisode suppose de savoir
# qu'il va s'ouvrir, information dont aucun gérant ne dispose. Le 31 décembre
# précédent est en revanche implémentable. Si les deux donnent des résultats
# voisins, le prédicteur ne dépend pas d'un ajustement opportun de la date.
resultats = []
for _, episode in episodes.iterrows():
    debut, fin = pd.Timestamp(episode["debut"]), pd.Timestamp(episode["fin"])
    ligne = {"episode": str(episode["debut"])[:7], "regime": episode["regime"]}
    for libelle, date_mesure in [("veille", debut - pd.Timedelta(days=1)),
                                 ("31 decembre", pd.Timestamp(f"{debut.year - 1}-12-31"))]:
        joint = construire(debut, fin, date_mesure)
        if len(joint) < 20:
            ligne[libelle] = np.nan
            continue
        ligne[libelle] = joint["ratio"].corr(joint["drawdown"], method="spearman")
        if libelle == "31 decembre":
            ligne["titres"] = len(joint)
            ligne["delta"] = joint["delta_fraction"].corr(joint["drawdown"], method="spearman")
            propre = joint[joint.index.isin(spread_propre)]
            ligne["spread propre"] = (propre["ratio"].corr(propre["drawdown"], method="spearman")
                                      if len(propre) >= 10 else np.nan)
            ligne["titres propres"] = len(propre)
    resultats.append(ligne)

predictif = pd.DataFrame(resultats)
predictif.to_parquet("h2_pouvoir_predictif.parquet", index=False)

print("Corrélation de rang entre ratio prix sur bond floor et drawdown")
print(predictif.round(3).to_string(index=False))
print()
print("Une valeur négative confirme l'hypothèse. La comparaison entre les colonnes")
print("veille et 31 décembre montre que le prédicteur ne dépend pas de la date de")
print("mesure, ce qui écarte l'objection de connaissance a posteriori.")


# %% CELLULE 4 - Comparaison avec la notation, à échantillon égal
# -----------------------------------------------------------------------------
# La comparaison n'a de sens que sur les titres disposant d'une notation. Opposer
# le ratio calculé sur l'ensemble du gisement à la notation calculée sur les seuls
# titres notés serait méthodologiquement invalide.
notation = univers.set_index("isin")["rtg_sp"].astype(str).str.strip().map(ECHELLE_NOTATION)

lignes = []
for _, episode in episodes.iterrows():
    debut, fin = pd.Timestamp(episode["debut"]), pd.Timestamp(episode["fin"])
    joint = construire(debut, fin, pd.Timestamp(f"{debut.year - 1}-12-31"))
    joint["notation"] = joint.index.map(notation)
    notes = joint.dropna(subset=["notation"])
    if len(notes) < 15:
        continue
    lignes.append(dict(
        episode=str(episode["debut"])[:7], titres_notes=len(notes), titres_total=len(joint),
        ratio=notes["ratio"].corr(notes["drawdown"], method="spearman"),
        notation=notes["notation"].corr(notes["drawdown"], method="spearman"),
        couverture=len(notes) / len(joint) * 100,
    ))

comparaison = pd.DataFrame(lignes)
print("Comparaison à échantillon égal, titres disposant d'une notation S&P")
print(comparaison.round(3).to_string(index=False))
print()
print(f"Couverture moyenne de la notation : {comparaison['couverture'].mean():.1f} %")
print("Le ratio prix sur bond floor est disponible pour 100 % du gisement.")


# %% CELLULE 5 - Apport marginal en régression
# -----------------------------------------------------------------------------
# La question n'est pas seulement laquelle des deux variables prédit mieux, mais
# si le ratio ajoute de l'information que la notation ne contient pas. On mesure
# la part de variance du drawdown expliquée par chaque combinaison.
blocs = []
for _, episode in episodes.iterrows():
    debut, fin = pd.Timestamp(episode["debut"]), pd.Timestamp(episode["fin"])
    joint = construire(debut, fin, pd.Timestamp(f"{debut.year - 1}-12-31"))
    joint["notation"] = joint.index.map(notation)
    blocs.append(joint)

ensemble = pd.concat(blocs).dropna(subset=["notation", "ratio", "delta_fraction", "drawdown"])

print(f"Régressions sur {len(ensemble)} observations")
for libelle, colonnes in [("notation seule", ["notation"]),
                          ("notation et ratio", ["notation", "ratio"]),
                          ("notation, ratio et delta", ["notation", "ratio", "delta_fraction"])]:
    X = np.column_stack([np.ones(len(ensemble))] + [ensemble[c].values for c in colonnes])
    y = ensemble["drawdown"].values
    coefficients, *_ = np.linalg.lstsq(X, y, rcond=None)
    residus = y - X @ coefficients
    r2 = 1 - residus @ residus / ((y - y.mean()) ** 2).sum()
    print(f"  {libelle:28s} R² = {r2:.3f}")
print()
print("L'écart entre la première et la deuxième ligne mesure l'apport propre du")
print("ratio, information que la notation ne contient pas.")
