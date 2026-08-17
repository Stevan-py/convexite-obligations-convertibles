# -*- coding: utf-8 -*-
"""
04. Identification des épisodes de repli et capture à la baisse par segment de delta.

Un épisode s'ouvre au dernier sommet du Stoxx 600 précédant un repli et se ferme
au point le plus bas atteint avant que l'indice ne regagne ce sommet. Seuls les
replis d'au moins 5 % sont retenus. La durée mesure donc l'intervalle du sommet
au creux, non le temps de retour au niveau initial.

Chaque épisode est classé selon la variation concomitante du Bund à dix ans :
taux en hausse et actions en baisse constituent un stress conjoint, taux en
baisse une fuite vers la qualité.
"""

# %% CELLULE 1 - Chargement
import numpy as np
import pandas as pd

PANEL = "panel.parquet"        # produit par 03_panel_et_convexite.py
MACRO = "macro.parquet"

panel = pd.read_parquet(PANEL)
panel = panel[panel["exploitable"]].copy()
macro = pd.read_parquet(MACRO)

SEUIL_REPLI = 0.05             # repli minimal pour retenir un épisode
SEUIL_CLOTURE = 0.005          # retour à moins de 0,5 % du sommet
SEUIL_TAUX = 0.05              # variation de Bund qualifiant le régime, en points

print("Panel :", panel["isin"].nunique(), "convertibles,", len(panel), "observations")


# %% CELLULE 2 - Identification des épisodes
# -----------------------------------------------------------------------------
# On suit le drawdown de l'indice par rapport à son plus haut glissant. Un
# épisode est ouvert dès que ce drawdown dépasse 2 %, suivi jusqu'à son point
# le plus bas, et refermé quand l'indice revient près de son sommet.
serie = macro[["Stoxx 600", "Bund 10 ans", "iTraxx Crossover 5 ans"]].copy()
serie = serie[serie["Stoxx 600"].notna() & serie["Bund 10 ans"].notna()].sort_index()
serie = serie.loc["2015":]

cours = serie["Stoxx 600"]
sommet_glissant = cours.cummax()
drawdown = cours / sommet_glissant - 1

episodes = []
en_cours = False
for date, valeur in drawdown.items():
    if not en_cours and valeur < -0.02:
        en_cours = True
        debut = cours.loc[:date][cours.loc[:date] == sommet_glissant.loc[date]].index[-1]
        creux = date
    elif en_cours:
        if cours.loc[date] < cours.loc[creux]:
            creux = date
        if valeur > -SEUIL_CLOTURE:
            episodes.append((debut, creux))
            en_cours = False
if en_cours:
    episodes.append((debut, creux))

lignes = []
for debut, creux in episodes:
    if (creux - debut).days < 5:
        continue
    repli = cours.loc[creux] / cours.loc[debut] - 1
    if repli > -SEUIL_REPLI:
        continue
    variation_bund = serie["Bund 10 ans"].loc[creux] - serie["Bund 10 ans"].loc[debut]
    xover = serie["iTraxx Crossover 5 ans"]
    variation_xover = (xover.loc[creux] - xover.loc[debut]) if xover.notna().loc[debut:creux].any() else np.nan
    lignes.append(dict(
        debut=debut, fin=creux, jours=(creux - debut).days,
        repli_actions=repli * 100, variation_bund=variation_bund,
        variation_xover=variation_xover,
        regime="Stress conjoint" if variation_bund > SEUIL_TAUX else (
            "Fuite vers la qualité" if variation_bund < -SEUIL_TAUX else "Taux stables"),
    ))

episodes = pd.DataFrame(lignes).sort_values("debut").reset_index(drop=True)
episodes.to_parquet("episodes.parquet", index=False)

print(episodes.round(2).to_string(index=False))
print()
print(episodes["regime"].value_counts().to_string())


# %% CELLULE 3 - Segments de delta
# -----------------------------------------------------------------------------
# Le delta de début d'année est retenu, de manière à ce que le classement
# précède l'épisode plutôt qu'il ne le suive. Réaffecter en cours d'année
# ferait sortir un titre de son segment au moment même où sa convexité se
# dégrade, ce qui diluerait le phénomène étudié.
BORNES = [0, 0.20, 0.40, 0.60, 10]
ETIQUETTES = ["0-20 %", "20-40 %", "40-60 %", "> 60 %"]

panel["annee"] = panel["date"].dt.year
delta_initial = panel.sort_values("date").groupby(["isin", "annee"]).first()["delta"] / 100
panel = panel.join(delta_initial.rename("delta_initial"), on=["isin", "annee"])
panel["segment"] = pd.cut(panel["delta_initial"], BORNES, labels=ETIQUETTES)


# %% CELLULE 4 - Capture à la baisse par segment et par épisode
# -----------------------------------------------------------------------------
# La capture ne retient que les séances où le sous-jacent recule. Elle rapporte
# la somme des rendements de la convertible à celle des rendements de l'action.
MIN_TITRES = 5


def capture(bloc):
    quotidien = bloc.groupby("date").agg(
        cb=("ret_cb", "mean"), action=("ret_propre", "mean"), effectif=("isin", "size"))
    quotidien = quotidien[quotidien["effectif"] >= MIN_TITRES]
    baisses = quotidien[quotidien["action"] < 0]
    if len(baisses) < 5:
        return np.nan, 0
    return baisses["cb"].sum() / baisses["action"].sum() * 100, bloc["isin"].nunique()


resultats = []
for _, episode in episodes.iterrows():
    debut, fin = pd.Timestamp(episode["debut"]), pd.Timestamp(episode["fin"])
    fenetre = panel[(panel["date"] >= debut) & (panel["date"] <= fin)]
    ligne = {"episode": str(episode["debut"])[:7], "regime": episode["regime"]}
    for etiquette in ETIQUETTES:
        valeur, effectif = capture(fenetre[fenetre["segment"] == etiquette])
        ligne[etiquette] = valeur
        ligne[etiquette + " (n)"] = effectif
    resultats.append(ligne)

captures = pd.DataFrame(resultats)
captures.to_parquet("captures_par_segment.parquet", index=False)

print("Capture à la baisse par segment de delta, en pourcentage")
print(captures[["episode", "regime"] + ETIQUETTES].round(0).to_string(index=False))
print()
print("La progression doit être monotone dans chaque épisode. Une valeur négative")
print("signifie que le segment a progressé pendant le recul du sous-jacent.")


# %% CELLULE 5 - Figure de réplication
# -----------------------------------------------------------------------------
import matplotlib.pyplot as plt

stress = captures[captures["regime"] == "Stress conjoint"]
couleurs = ["#9DB4C8", "#2A9D8F", "#8AB17D", "#C0392B", "#E9C46A", "#E07A34", "#1F3864"]

figure, axe = plt.subplots(figsize=(11, 5.6))
positions = np.arange(len(ETIQUETTES))
largeur = 0.8 / max(len(stress), 1)

for rang, (_, ligne) in enumerate(stress.iterrows()):
    decalage = (rang - (len(stress) - 1) / 2) * largeur
    valeurs = [0 if pd.isna(ligne[e]) else ligne[e] for e in ETIQUETTES]
    axe.bar(positions + decalage, valeurs, largeur, label=ligne["episode"],
            color=couleurs[rang % len(couleurs)], edgecolor="white", linewidth=0.5)

axe.axhline(0, color="#555", lw=0.9)
axe.set_xticks(positions)
axe.set_xticklabels(ETIQUETTES, fontsize=11.5)
axe.set_xlabel("Segment de delta en début d'année", fontsize=10.5, labelpad=9)
axe.set_ylabel("Capture à la baisse (%)", fontsize=10.5)
axe.set_title("Capture à la baisse par segment de delta, épisodes de stress conjoint",
              fontsize=12.5, fontweight="bold", color="#1F3864", loc="left", pad=14)
axe.legend(fontsize=8.8, ncol=2, framealpha=0.95, loc="upper left")
axe.grid(axis="y", alpha=0.25, lw=0.7)
for cote in ["top", "right"]:
    axe.spines[cote].set_visible(False)
plt.tight_layout()
plt.show()
