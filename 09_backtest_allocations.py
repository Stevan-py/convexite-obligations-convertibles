# -*- coding: utf-8 -*-
"""
09. Grille d'allocations et comparaison avec les portefeuilles classiques.

Quatre portefeuilles de segment, équipondérés en leur sein, servent de briques à
onze allocations. Deux points de méthode distinguent ce script des précédents.

L'affectation est recalculée CHAQUE MOIS, alors que les chapitres de mesure
retiennent le delta de début d'année. Ce n'est pas une incohérence : mesurer la
convexité d'un profil suppose que ce profil reste stable pendant la fenêtre
d'estimation, tandis que simuler un portefeuille réel suppose de le rebalancer.
L'écart est important et doit être assumé, le rebalancement évacuant du segment
une partie du phénomène mesuré ailleurs.

Un plancher de prix est appliqué aux convertibles. Une souche en défaut cotant à
quelques dixièmes de pourcent du nominal produit des variations relatives sans
portée économique, capables à elles seules de doubler la performance annuelle du
gisement équipondéré.
"""

# %% CELLULE 1 - Chargement et plancher de prix
import numpy as np
import pandas as pd

panel = pd.read_parquet("panel.parquet")
panel = panel[panel["exploitable"]].copy()
macro = pd.read_parquet("macro.parquet")
episodes = pd.read_parquet("episodes.parquet")

PLANCHER_PRIX = 10             # en % du nominal
MIN_TITRES_SEGMENT = 3
BORNES = [0, 0.20, 0.40, 0.60, 10]
SEGMENTS = ["S1", "S2", "S3", "S4"]

panel = panel.sort_values(["isin", "date"])
panel["prix_precedent"] = panel.groupby("isin")["prix"].shift(1)
avant = len(panel)
panel = panel[(panel["prix"] >= PLANCHER_PRIX) & (panel["prix_precedent"] >= PLANCHER_PRIX)]
print(f"Plancher de prix : {avant - len(panel)} observations écartées sur {avant}")


# %% CELLULE 2 - Portefeuilles de segment
panel["mois"] = panel["date"].dt.to_period("M")
delta_initial = panel.sort_values("date").groupby(["isin", "mois"]).first()["delta"] / 100
panel = panel.join(delta_initial.rename("delta_initial"), on=["isin", "mois"])
panel["segment"] = pd.cut(panel["delta_initial"], BORNES, labels=SEGMENTS)

agrege = panel.dropna(subset=["segment"]).groupby(["date", "segment"], observed=True).agg(
    rendement=("ret_cb", "mean"), effectif=("isin", "size")).reset_index()
agrege = agrege[agrege["effectif"] >= MIN_TITRES_SEGMENT]

briques = agrege.pivot(index="date", columns="segment", values="rendement")
briques = briques.sort_index().reindex(columns=SEGMENTS)
gisement = panel.groupby("date")["ret_cb"].mean().reindex(briques.index)
sous_jacents = panel.groupby("date")["ret_propre"].mean().reindex(briques.index)

stress = pd.Series(False, index=briques.index)
for _, episode in episodes[episodes["regime"] == "Stress conjoint"].iterrows():
    stress.loc[pd.Timestamp(episode["debut"]):pd.Timestamp(episode["fin"])] = True

print(f"{len(briques)} séances, couverture par segment :")
print(briques.notna().mean().round(3).to_string())


# %% CELLULE 3 - Mesures de performance et de risque
def mesurer(rendements):
    serie = rendements.dropna()
    cumul = (1 + serie).prod() - 1
    annualise = (1 + cumul) ** (252 / len(serie)) - 1
    volatilite = serie.std() * np.sqrt(252)
    cumule = (1 + serie).cumprod()
    drawdown = (cumule / cumule.cummax() - 1).min()
    en_stress = serie[stress.reindex(serie.index).fillna(False)]
    action = sous_jacents.reindex(serie.index)
    baisses, hausses = serie[action < 0], serie[action > 0]
    return dict(
        performance=cumul * 100, annualise=annualise * 100, volatilite=volatilite * 100,
        drawdown=drawdown * 100, sharpe=annualise / volatilite if volatilite else np.nan,
        cumul_stress=((1 + en_stress).prod() - 1) * 100,
        capture_baisse=baisses.sum() / action[action < 0].sum() * 100,
        capture_hausse=hausses.sum() / action[action > 0].sum() * 100,
    )


# %% CELLULE 4 - Grille d'allocations
# -----------------------------------------------------------------------------
# Une précision de lecture. L'allocation équipondérée attribue un quart du poids
# à chaque segment quel que soit le nombre de titres qu'il contient. Le gisement
# équipondéré attribue le même poids à chaque titre et constitue la référence de
# marché. Les deux ne coïncident pas.
GRILLE = {
    "Barbell 50/0/0/50": [.5, 0, 0, .5],
    "Barbell 70/0/0/30": [.7, 0, 0, .3],
    "Barbell 30/0/0/70": [.3, 0, 0, .7],
    "Barbell élargi 40/10/10/40": [.4, .1, .1, .4],
    "Équipondérée 25/25/25/25": [.25, .25, .25, .25],
    "Défensive inclinée 60/20/20/0": [.6, .2, .2, 0],
    "Segment obligataire seul": [1, 0, 0, 0],
    "Segment actions seul": [0, 0, 0, 1],
    "Zone équilibrée 40-60 seule": [0, 0, 1, 0],
    "Zone équilibrée élargie 0/50/50/0": [0, .5, .5, 0],
}
resultats = {nom: mesurer((briques[SEGMENTS] * poids).sum(axis=1, min_count=1))
             for nom, poids in GRILLE.items()}
resultats["Gisement équipondéré (référence)"] = mesurer(gisement)

grille = pd.DataFrame(resultats).T.sort_values("sharpe", ascending=False)
grille.to_parquet("grille_allocations.parquet")
print(grille.round(2).to_string())


# %% CELLULE 5 - Le rebalancement dynamique échoue
# -----------------------------------------------------------------------------
# Basculer vers le barbell quand le régime de corrélation devient positif suppose
# que ce régime annonce quelque chose. Le chapitre consacré à la troisième
# hypothèse montre qu'il n'annonce rien, et le résultat ci-dessous en donne la
# conséquence chiffrée.
serie_macro = macro[["Stoxx 600", "Bund 10 ans"]].dropna()
regime = (serie_macro["Stoxx 600"].pct_change()
          .rolling(126).corr(-serie_macro["Bund 10 ans"].diff()))
regime = regime.reindex(briques.index, method="ffill")

statique = (briques[SEGMENTS] * [.25, .25, .25, .25]).sum(axis=1, min_count=1)
barbell = (briques[SEGMENTS] * [.5, 0, 0, .5]).sum(axis=1, min_count=1)
dynamique = statique.where(regime <= 0, barbell)

print("Barbell tenu en permanence : %.1f %%" % mesurer(barbell)["performance"])
print("Bascule déclenchée par le régime : %.1f %%" % mesurer(dynamique)["performance"])
print("Allocation équipondérée : %.1f %%" % mesurer(statique)["performance"])


# %% CELLULE 6 - Comparaison avec les portefeuilles classiques
# -----------------------------------------------------------------------------
# Objection à traiter frontalement : si l'allocation supérieure combine du quasi
# obligataire et du quasi actions, autant détenir directement des obligations et
# des actions.
#
# LIMITE À ÉNONCER. Les rendements convertibles sont calculés pied de coupon et
# l'indice actions hors dividendes. Le coupon moyen de l'univers s'établit à
# environ 1,8 % contre un rendement du dividende européen de l'ordre de 3 %, ce
# qui avantage légèrement les convertibles.
indices = macro[["Stoxx 600", "IG euro (BofA)", "HY euro (BofA)",
                 "Convertibles euro (BofA)"]].dropna()
rendements = indices.pct_change().reindex(briques.index)

comparaison = {
    "Convertibles, barbell 50/50": barbell,
    "Convertibles, barbell 70/30": (briques[SEGMENTS] * [.7, 0, 0, .3]).sum(axis=1, min_count=1),
    "Convertibles, gisement équipondéré": gisement,
    "Convertibles, indice BofA euro": rendements["Convertibles euro (BofA)"],
    "Convertibles, zone équilibrée seule": briques["S3"],
    "Mixte 70 % HY euro / 30 % actions": .7 * rendements["HY euro (BofA)"] + .3 * rendements["Stoxx 600"],
    "Mixte 70 % IG euro / 30 % actions": .7 * rendements["IG euro (BofA)"] + .3 * rendements["Stoxx 600"],
    "Mixte 50 % IG euro / 50 % actions": .5 * rendements["IG euro (BofA)"] + .5 * rendements["Stoxx 600"],
    "IG euro seul": rendements["IG euro (BofA)"],
    "Actions seules, Stoxx 600": rendements["Stoxx 600"],
}
table = pd.DataFrame({nom: mesurer(serie) for nom, serie in comparaison.items()}).T
table = table.sort_values("sharpe", ascending=False)
print(table.round(2).to_string())


# %% CELLULE 7 - Stabilité annuelle
# -----------------------------------------------------------------------------
# Un résultat obtenu sur une période unique peut masquer un effet de phase. La
# comparaison interne à la classe d'actifs se révèle très stable, celle avec les
# actifs classiques beaucoup moins.
def sharpe_annuel(serie, annee):
    bloc = serie[serie.index.year == annee].dropna()
    if len(bloc) < 120:
        return np.nan
    cumul = (1 + bloc).prod() - 1
    annualise = (1 + cumul) ** (252 / len(bloc)) - 1
    return annualise / (bloc.std() * np.sqrt(252))


annees = range(briques.index.min().year, briques.index.max().year)
sharpes = pd.DataFrame({str(a): {nom: sharpe_annuel(s, a) for nom, s in comparaison.items()}
                        for a in annees})
reference = "Convertibles, barbell 70/30"
print("Nombre d'années où le barbell 70/30 domine, sur", len(annees))
for nom in comparaison:
    if nom == reference:
        continue
    victoires = (sharpes.loc[reference] > sharpes.loc[nom]).sum()
    print(f"  contre {nom:38s} {victoires}/{len(annees)}")
