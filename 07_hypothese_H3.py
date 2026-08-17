# -*- coding: utf-8 -*-
"""
07. Le régime de corrélation actions-obligations comme indicateur avancé.

L'hypothèse postule que le régime de corrélation prédit la destruction de
convexité mieux que les mesures usuelles de volatilité ou de spread. Le test
est rétrospectif : à chaque fin de mois on relève l'état du régime, information
disponible à cette date, puis on mesure la convexité réalisée du gisement sur
les six mois SUIVANTS, période future à cette date mais passée aujourd'hui.

Convention de signe retenue dans tout le mémoire. Le rendement obligataire est
approximé par l'opposé de la variation de taux, la duration s'éliminant dans une
corrélation. Une corrélation actions-obligations NÉGATIVE désigne le régime
protecteur, POSITIVE le régime où la protection disparaît.

Trois précautions. Le pas est mensuel, seul compromis viable entre la fenêtre
d'estimation de six mois qu'exige un terme quadratique et le nombre
d'observations disponibles. Un test sur fenêtres disjointes contrôle le biais
de chevauchement documenté par Hansen et Hodrick puis Valkanov. Et un test
directionnel par quadrants lève l'objection selon laquelle une corrélation est
aveugle au sens des mouvements.
"""

# %% CELLULE 1 - Chargement
import numpy as np
import pandas as pd
from scipy import stats

panel = pd.read_parquet("panel.parquet")
panel = panel[panel["exploitable"]]
macro = pd.read_parquet("macro.parquet")

FENETRE_CONVEXITE = 6          # mois
FENETRE_REGIME = 126           # séances, soit environ six mois
MIN_TITRES = 30
MIN_SEANCES = 60

portefeuille = panel.groupby("date").agg(
    cb=("ret_cb", "mean"), action=("ret_propre", "mean"), effectif=("isin", "size"))
portefeuille = portefeuille[portefeuille["effectif"] >= MIN_TITRES].sort_index()
portefeuille["retard"] = portefeuille["action"].shift(1)
portefeuille = portefeuille.dropna()


# %% CELLULE 2 - Convexité sur fenêtre glissante
def convexite(bloc):
    if len(bloc) < MIN_SEANCES:
        return np.nan
    x = bloc["action"].values
    X = np.column_stack([np.ones(len(x)), x, x ** 2, bloc["retard"].values])
    coefficients, *_ = np.linalg.lstsq(X, bloc["cb"].values, rcond=None)
    return coefficients[2]


# Le régime, calculé sur des séries dont les valeurs manquantes sont retirées
# AVANT le calcul glissant, faute de quoi elles propageraient des NaN.
serie = macro[["Stoxx 600", "Bund 10 ans"]].dropna()
rendement_action = serie["Stoxx 600"].pct_change()
rendement_obligataire = -serie["Bund 10 ans"].diff()
regime = rendement_action.rolling(FENETRE_REGIME).corr(rendement_obligataire)

fins_de_mois = pd.date_range(portefeuille.index.min() + pd.DateOffset(months=FENETRE_CONVEXITE),
                             portefeuille.index.max(), freq="ME")
lignes = []
for date in fins_de_mois:
    valeur_regime = regime.asof(date)
    future = convexite(portefeuille.loc[date:date + pd.DateOffset(months=FENETRE_CONVEXITE)])
    if np.isfinite(valeur_regime) and np.isfinite(future):
        lignes.append(dict(mois=date, regime=valeur_regime, convexite_future=future))

mensuel = pd.DataFrame(lignes)
print(f"{len(mensuel)} observations mensuelles")


# %% CELLULE 3 - Pouvoir prédictif comparé
indicateurs = macro[["VSTOXX", "iTraxx Crossover 5 ans", "Volatilité des taux (MOVE)"]]
indicateurs = indicateurs.resample("ME").last()
mensuel = mensuel.set_index("mois").join(indicateurs).dropna(how="all", axis=1)

print("Corrélation de rang avec la convexité des six mois suivants")
for colonne in ["regime"] + [c for c in indicateurs.columns if c in mensuel]:
    bloc = mensuel.dropna(subset=[colonne, "convexite_future"])
    if len(bloc) < 30:
        continue
    rho = bloc[colonne].corr(bloc["convexite_future"], method="spearman")
    print(f"  {colonne:32s} {rho:+.3f}  (n = {len(bloc)})")
print()
print("L'hypothèse prédit une valeur négative pour le régime, un régime dégradé")
print("devant précéder une convexité dégradée. Le VSTOXX sert de comparateur, ")
print("l'hypothèse prétendant le surpasser.")


# %% CELLULE 4 - Comparaison par régime
mensuel["etat"] = np.where(mensuel["regime"] > 0, "Corrélation positive, protection détruite",
                           "Corrélation négative, régime protecteur")
print(mensuel.groupby("etat").agg(
    mois=("convexite_future", "size"),
    convexite_future_mediane=("convexite_future", "median"),
    part_negative=("convexite_future", lambda s: (s < 0).mean()),
).round(3).to_string())


# %% CELLULE 5 - Contrôle sur fenêtres disjointes
# -----------------------------------------------------------------------------
# Au pas mensuel, deux fenêtres consécutives de six mois partagent cinq sixièmes
# de leurs données. Les observations ne sont donc pas indépendantes et la
# corrélation s'en trouve gonflée. Le seul test honnête emploie des fenêtres
# qui ne se recouvrent pas, au prix d'un échantillon bien plus réduit.
debut = portefeuille.index.min()
lignes = []
while debut + pd.DateOffset(months=2 * FENETRE_CONVEXITE) <= portefeuille.index.max():
    date_mesure = debut + pd.DateOffset(months=FENETRE_CONVEXITE)
    fin = date_mesure + pd.DateOffset(months=FENETRE_CONVEXITE)
    valeur_regime = regime.asof(date_mesure)
    future = convexite(portefeuille.loc[date_mesure:fin])
    if np.isfinite(valeur_regime) and np.isfinite(future):
        lignes.append(dict(date=date_mesure, regime=valeur_regime, convexite_future=future))
    debut = debut + pd.DateOffset(months=FENETRE_CONVEXITE)

disjoint = pd.DataFrame(lignes)
rho_disjoint = disjoint["regime"].corr(disjoint["convexite_future"], method="spearman")
print(f"Fenêtres disjointes : {len(disjoint)} observations indépendantes, rho = {rho_disjoint:+.3f}")
print("À comparer au résultat du pas mensuel. L'écart mesure l'ampleur du biais")
print("de chevauchement.")


# %% CELLULE 6 - Test directionnel par quadrants
# -----------------------------------------------------------------------------
# Une corrélation est aveugle au sens du mouvement. Une corrélation positive
# recouvre aussi bien la configuration où les deux classes reculent ensemble, la
# plus défavorable, que celle où elles progressent ensemble, la plus favorable.
# Le test ci-dessous distingue les quatre configurations.
mois_macro = pd.DataFrame({
    "r_action": serie["Stoxx 600"].resample("ME").last().pct_change(),
    "d_taux": serie["Bund 10 ans"].resample("ME").last().diff(),
}).dropna()


def quadrant(ligne):
    if ligne["r_action"] < 0 and ligne["d_taux"] > 0:
        return "D. Actions en baisse, taux en hausse, stress conjoint"
    if ligne["r_action"] < 0:
        return "C. Actions en baisse, taux en baisse, fuite vers la qualité"
    if ligne["d_taux"] > 0:
        return "A. Actions en hausse, taux en hausse, reflation"
    return "B. Actions en hausse, taux en baisse"


mois_macro["quadrant"] = mois_macro.apply(quadrant, axis=1)
joint = mensuel.join(mois_macro["quadrant"]).dropna(subset=["quadrant", "convexite_future"])

print(joint.groupby("quadrant").agg(
    mois=("convexite_future", "size"),
    convexite_future_mediane=("convexite_future", "median"),
).round(3).to_string())
print()
stress = joint[joint["quadrant"].str.startswith("D")]["convexite_future"]
autres = joint[~joint["quadrant"].str.startswith("D")]["convexite_future"]
test = stats.mannwhitneyu(stress, autres)
print(f"Test de Mann-Whitney, quadrant de stress contre les autres : p = {test.pvalue:.3f}")
print("Ce test compare les rangs plutôt que les valeurs, ce qui le rend insensible")
print("aux valeurs extrêmes. Réserves : effectif limité et fenêtres chevauchantes.")


# %% CELLULE 7 - Persistance de la convexité
# -----------------------------------------------------------------------------
# Même sans indicateur externe, la convexité passée pourrait renseigner sur la
# convexité future. Ce n'est pas le cas, ce qui interdit toute extrapolation.
mensuel["convexite_courante"] = [convexite(portefeuille.loc[d - pd.DateOffset(months=FENETRE_CONVEXITE):d])
                                 for d in mensuel.index]
bloc = mensuel.dropna(subset=["convexite_courante", "convexite_future"])
print(f"Autocorrélation de la convexité à six mois : "
      f"{bloc['convexite_courante'].corr(bloc['convexite_future']):+.3f}")
