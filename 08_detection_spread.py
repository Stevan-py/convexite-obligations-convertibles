# -*- coding: utf-8 -*-
"""
08. Détection du spread de crédit par défaut de 400 points de base.

Le modèle de valorisation applique un flat 5y spread propre à l'émetteur lorsque
celui-ci dispose d'un dérivé de crédit à cinq ans coté, et une valeur par défaut
de 400 points de base à défaut. Cette valeur étant figée dans le temps, le
plancher obligataire des titres concernés ne se dégrade jamais pour raison de
crédit, ce qui neutralise le mécanisme central étudié dans ce mémoire.

Le spread d'entrée n'est pas exposé dans les données extraites. Il est reconstruit
par inversion du plancher : on recherche le taux d'actualisation qui, appliqué aux
flux contractuels, restitue le plancher affiché, puis on retranche un taux sans
risque interpolé selon la maturité résiduelle.

Le champ de spread implicite ne peut PAS servir à cette fin. Il est construit pour
égaliser prix modèle et prix de marché, si bien que l'employer comme mesure de
crédit rendrait la cheapness nulle par construction.
"""

# %% CELLULE 1 - Chargement
import numpy as np
import pandas as pd
from scipy.optimize import brentq

micro = pd.read_parquet("micro_long.parquet")
univers = pd.read_parquet("univers.parquet")
macro = pd.read_parquet("macro.parquet")

BORNE_PLANCHER = (20, 200)     # plancher plausible, en % du nominal
BORNE_MATURITE = (0.5, 15)     # maturité résiduelle en années
MIN_OBSERVATIONS = 24          # observations mensuelles minimales


# %% CELLULE 2 - Préparation
statique = univers.copy()
statique["maturite"] = pd.to_datetime(statique["maturity"], format="%d/%m/%Y", errors="coerce")
statique["coupon_num"] = pd.to_numeric(statique["coupon"], errors="coerce")

donnees = micro[["isin", "date", "bond_floor"]].dropna()
donnees = donnees[donnees["bond_floor"].between(*BORNE_PLANCHER)]
donnees["mois"] = donnees["date"].dt.to_period("M")
donnees = donnees.sort_values("date").groupby(["isin", "mois"]).last().reset_index()
donnees = donnees.merge(statique[["isin", "maturite", "coupon_num"]], on="isin", how="left")
donnees = donnees.dropna(subset=["maturite", "coupon_num"])
donnees["maturite_residuelle"] = (donnees["maturite"] - donnees["date"]).dt.days / 365.25
donnees = donnees[donnees["maturite_residuelle"].between(*BORNE_MATURITE)]

print(f"{len(donnees)} observations mensuelles à inverser, {donnees['isin'].nunique()} titres")


# %% CELLULE 3 - Inversion du plancher
def taux_actuariel(plancher, coupon, maturite):
    """Taux qui égalise la valeur actualisée des flux au plancher affiché."""
    periodes = max(int(round(maturite)), 1)

    def ecart(taux):
        valeur_coupons = sum(coupon / (1 + taux) ** t for t in range(1, periodes + 1))
        return valeur_coupons + 100 / (1 + taux) ** periodes - plancher

    try:
        return brentq(ecart, -0.5, 3.0, maxiter=60)
    except (ValueError, RuntimeError):
        return np.nan


donnees["taux"] = [taux_actuariel(p, c, m) for p, c, m in zip(
    donnees["bond_floor"], donnees["coupon_num"], donnees["maturite_residuelle"])]
donnees = donnees.dropna(subset=["taux"])

# Taux sans risque interpolé entre deux et dix ans selon la maturité résiduelle.
# Employer le seul dix ans introduirait un bruit important sur les maturités
# courtes, particulièrement en 2022 où la courbe s'est fortement déplacée.
bund_2 = macro["Bund 2 ans"].dropna()
bund_10 = macro["Bund 10 ans"].dropna()
donnees["b2"] = bund_2.reindex(donnees["date"], method="ffill").values
donnees["b10"] = bund_10.reindex(donnees["date"], method="ffill").values
poids = (donnees["maturite_residuelle"].clip(2, 10) - 2) / 8
donnees["sans_risque"] = (donnees["b2"] * (1 - poids) + donnees["b10"] * poids) / 100
donnees["spread_bp"] = (donnees["taux"] - donnees["sans_risque"]) * 10000
donnees = donnees[donnees["spread_bp"].between(-500, 5000)]

print(donnees["spread_bp"].describe([.05, .25, .5, .75, .95]).round(0).to_string())
print()
print("Une concentration anormale autour de 400 points de base signale la présence")
print("de la valeur par défaut. Un vrai marché du crédit produirait une distribution")
print("étalée selon la qualité de signature.")


# %% CELLULE 4 - Classement
# -----------------------------------------------------------------------------
# Un émetteur est classé en spread par défaut lorsque trois conditions se
# réunissent : le spread reconstruit varie peu, il ne réagit pas à l'indice de
# crédit, et son niveau avoisine 400 points de base. Le test de sensibilité à
# l'indice est le plus discriminant, un spread figé ne pouvant par construction
# suivre le marché.
crossover = macro["iTraxx Crossover 5 ans"].dropna()
donnees["crossover"] = crossover.reindex(donnees["date"], method="ffill").values
donnees = donnees.sort_values(["isin", "date"])

lignes = []
for isin, bloc in donnees.groupby("isin"):
    bloc = bloc.dropna(subset=["spread_bp", "crossover"])
    if len(bloc) < MIN_OBSERVATIONS:
        continue
    variations = pd.DataFrame({
        "spread": bloc["spread_bp"].diff(), "indice": bloc["crossover"].diff()}).dropna()
    if len(variations) < 18 or variations["indice"].std() == 0:
        continue
    pente = np.polyfit(variations["indice"], variations["spread"], 1)[0]
    lignes.append(dict(
        isin=isin, observations=len(bloc), mediane=bloc["spread_bp"].median(),
        ecart_type=bloc["spread_bp"].std(), sensibilite=pente,
    ))

detection = pd.DataFrame(lignes)
detection["defaut"] = (
    (detection["sensibilite"].abs() < 0.15) &
    (detection["ecart_type"] < 60) &
    (detection["mediane"].between(300, 500))
)
detection.to_parquet("detection_spread.parquet", index=False)

print(f"Titres classables : {len(detection)}")
print(f"Classés en spread par défaut : {int(detection['defaut'].sum())}")
print()
print(detection.groupby("defaut").agg(
    titres=("isin", "size"), mediane=("mediane", "median"),
    ecart_type=("ecart_type", "median"), sensibilite=("sensibilite", "median"),
).round(2).to_string())


# %% CELLULE 5 - Validation
# -----------------------------------------------------------------------------
# La méthode doit être validée par vérification directe sur le Terminal. Ouvrir
# OVCV sur un échantillon de titres et relever l'hypothèse de crédit affichée :
# soit un flat 5y spread propre à l'émetteur, soit la valeur de 400 par défaut.
#
# Le script ci-dessous produit la liste à vérifier, dix titres de chaque groupe
# choisis parmi les cas les plus nets.
suspects = detection[detection["defaut"]].copy()
suspects["distance"] = (suspects["mediane"] - 400).abs()
groupe_defaut = suspects.nsmallest(10, "ecart_type")

reels = detection[(~detection["defaut"]) & detection["mediane"].between(50, 800) &
                  (detection["ecart_type"] > 60)].copy()
groupe_reel = reels.nlargest(10, "sensibilite")

emetteurs = micro.groupby("isin")["issuer"].first()
a_verifier = pd.concat([
    groupe_defaut.assign(prediction="Spread par défaut"),
    groupe_reel.assign(prediction="Spread propre à l'émetteur"),
])
a_verifier["emetteur"] = a_verifier["isin"].map(emetteurs)
a_verifier["hypothese_observee"] = ""
a_verifier["validee"] = ""
a_verifier[["prediction", "isin", "emetteur", "mediane", "ecart_type", "sensibilite",
            "hypothese_observee", "validee"]].to_csv(
    "validation_a_remplir.csv", index=False, sep=";", encoding="utf-8-sig")

print("Liste de validation écrite dans validation_a_remplir.csv")
print("Ouvrir OVCV sur chaque ISIN, relever l'hypothèse de crédit et remplir les")
print("deux dernières colonnes. Un taux de succès inférieur à quinze sur vingt")
print("invaliderait la méthode.")
