# =============================================================================
# CONVEXITY RATIO - construction pas a pas
# Format notebook : chaque bloc separe par "# %%" est une cellule.
# Copier-coller cellule par cellule dans Jupyter ou VS Code.
# =============================================================================

# %% CELLULE 1 - Imports et localisation des fichiers
# -----------------------------------------------------------------------------
# Cette cellule cherche les deux fichiers de donnees toute seule.
# Elle regarde dans le dossier du notebook, puis dans Telechargements, Downloads,
# Bureau et Desktop. Si elle ne les trouve pas, elle te dit exactement quoi faire.
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

FICHIERS_ATTENDUS = ["micro_long.parquet", "sous_jacents_traite.parquet"]

# Si tu connais deja le dossier, ecris-le ici et la recherche automatique est ignoree.
# Windows  : DOSSIER_FORCE = r"C:\Users\stevan\Downloads"
# Mac      : DOSSIER_FORCE = "/Users/stevan/Downloads"
DOSSIER_FORCE = None


def trouver_dossier():
    """Retourne le premier dossier contenant les deux fichiers, sinon None."""
    if DOSSIER_FORCE:
        return Path(DOSSIER_FORCE)

    maison = Path.home()
    candidats = [
        Path.cwd(),
        maison / "Downloads", maison / "Telechargements", maison / "Téléchargements",
        maison / "Desktop", maison / "Bureau",
        maison / "Documents",
    ]
    for dossier in candidats:
        if dossier.is_dir() and all((dossier / f).exists() for f in FICHIERS_ATTENDUS):
            return dossier
    return None


dossier = trouver_dossier()

if dossier is None:
    print("FICHIERS INTROUVABLES.")
    print()
    print("Le notebook s'execute actuellement dans :")
    print("   ", Path.cwd())
    print()
    print("Deux solutions, au choix.")
    print()
    print("SOLUTION 1, la plus simple : deplacer les deux fichiers")
    print("    micro_long.parquet et sous_jacents_traite.parquet")
    print("    dans le dossier affiche ci-dessus, puis relancer cette cellule.")
    print()
    print("SOLUTION 2 : ouvrir le dossier ou sont les fichiers, copier son chemin,")
    print("    et le coller plus haut dans DOSSIER_FORCE, entre guillemets.")
    print("    Sur Windows, garder le r avant les guillemets : r\"C:\\Users\\...\"")
    print()
    print("Fichiers parquet visibles dans le dossier courant :")
    trouves = list(Path.cwd().glob("*.parquet"))
    print("   ", [f.name for f in trouves] if trouves else "aucun")
    raise SystemExit("Corrige le chemin puis relance cette cellule.")

CHEMIN_MICRO = dossier / "micro_long.parquet"
CHEMIN_ACTIONS = dossier / "sous_jacents_traite.parquet"
print("Fichiers trouves dans :", dossier)

micro = pd.read_parquet(CHEMIN_MICRO)
actions = pd.read_parquet(CHEMIN_ACTIONS)

print("Convertibles :", micro["isin"].nunique(), "titres,", len(micro), "lignes")
print("Actions      :", actions["IDENTIFIER"].nunique(), "titres,", len(actions), "lignes")
print()
print("Si ces deux lignes s'affichent, tout le reste du notebook fonctionnera.")


# %% CELLULE 2 - Appariement convertible / action
# -----------------------------------------------------------------------------
# Jointure sur le couple (ticker du sous-jacent, date).
# On ne garde que les colonnes utiles pour alleger la memoire.
# ret_propre est le rendement action deja nettoye des operations sur titres
# et de la granularite de tick. On ne recalcule PAS le rendement action ici.
colonnes_micro = ["isin", "issuer", "undl", "defaulted", "date",
                  "prix", "delta", "gamma", "implied_vol", "bond_floor"]
colonnes_actions = ["IDENTIFIER", "DATE", "PX_LAST", "ret_propre",
                    "VOLATILITY_30D", "VOLATILITY_90D"]

panel = micro[colonnes_micro].merge(
    actions[colonnes_actions],
    left_on=["undl", "date"], right_on=["IDENTIFIER", "DATE"], how="inner",
)
panel = panel.sort_values(["isin", "date"]).reset_index(drop=True)

# Rendement de la convertible, en pourcentage du nominal.
panel["ret_cb"] = panel.groupby("isin")["prix"].pct_change()

# Une observation n'est exploitable que si les DEUX rendements le sont.
panel["exploitable"] = panel["ret_cb"].notna() & panel["ret_propre"].notna()

print("Panel apparie :", panel["isin"].nunique(), "convertibles")
print("Observations exploitables :", int(panel["exploitable"].sum()))


# %% CELLULE 3 - Verification prealable : les convertibles sont-elles vraiment cotees ?
# -----------------------------------------------------------------------------
# Une obligation matricee par un teneur de marche affiche beaucoup de rendements
# exactement nuls. Cela biaiserait le beta vers le bas et fausserait la convexite.
# Si la part de zeros est proche de celle des actions, la cotation est reelle.
z = panel[panel["exploitable"]]
print("Rendements convertible exactement nuls : %.1f %%" % ((z["ret_cb"] == 0).mean() * 100))
print("Rendements action exactement nuls      : %.1f %%" % ((z["ret_propre"] == 0).mean() * 100))
print()
print("Si les deux chiffres sont proches, la cotation quotidienne est exploitable.")


# %% CELLULE 4 - La regression de convexite realisee
# -----------------------------------------------------------------------------
# Modele estime titre par titre et annee par annee :
#
#     r_CB = alpha + beta * r_action + c * r_action^2 + beta_retarde * r_action(-1)
#
#   beta         = delta realise, la sensibilite lineaire effectivement observee
#   c            = CONVEXITE REALISEE, l'objet central du memoire
#   beta_retarde = correction de Dimson, capte le decalage de cotation obligataire
#
# c > 0 : la convertible amortit les baisses et amplifie les hausses, convexite positive
# c < 0 : elle amplifie les baisses, la convexite a disparu ou s'est inversee

SEUIL_OBS = 100   # nombre minimal d'observations pour estimer une annee


def estimer_convexite(rendement_cb, rendement_action, rendement_action_retarde):
    """Retourne alpha, beta, c, beta_retarde, leurs t de Student et le R2."""
    y = np.asarray(rendement_cb, dtype=float)
    x = np.asarray(rendement_action, dtype=float)
    xl = np.asarray(rendement_action_retarde, dtype=float)

    X = np.column_stack([np.ones(len(x)), x, x ** 2, xl])
    coefficients, *_ = np.linalg.lstsq(X, y, rcond=None)

    residus = y - X @ coefficients
    ddl = len(x) - X.shape[1]
    if ddl <= 0:
        return None
    variance = residus @ residus / ddl
    try:
        covariance = variance * np.linalg.inv(X.T @ X)
    except np.linalg.LinAlgError:
        return None
    ecarts_types = np.sqrt(np.diag(covariance))

    sct = ((y - y.mean()) ** 2).sum()
    r2 = 1 - residus @ residus / sct if sct > 0 else np.nan

    return dict(
        alpha=coefficients[0],
        beta=coefficients[1],
        c=coefficients[2],
        beta_retarde=coefficients[3],
        t_c=coefficients[2] / ecarts_types[2] if ecarts_types[2] > 0 else np.nan,
        r2=r2,
        n=len(x),
    )


# %% CELLULE 5 - Estimation sur tout l'echantillon, titre par titre et annee par annee
# -----------------------------------------------------------------------------
resultats = []

for isin, groupe in panel.groupby("isin"):
    groupe = groupe.set_index("date").sort_index()
    groupe = groupe[groupe["exploitable"]]
    groupe["ret_action_retarde"] = groupe["ret_propre"].shift(1)
    groupe = groupe.dropna(subset=["ret_cb", "ret_propre", "ret_action_retarde"])

    for annee, bloc in groupe.groupby(groupe.index.year):
        if len(bloc) < SEUIL_OBS:
            continue
        sortie = estimer_convexite(bloc["ret_cb"], bloc["ret_propre"],
                                   bloc["ret_action_retarde"])
        if sortie is None:
            continue
        sortie.update(
            isin=isin,
            annee=annee,
            emetteur=bloc["issuer"].iloc[0],
            delta_moyen=bloc["delta"].mean(),
            vol_realisee=bloc["VOLATILITY_90D"].mean(),
            prix_moyen=bloc["prix"].mean(),
            bond_floor_moyen=bloc["bond_floor"].mean(),
        )
        resultats.append(sortie)

convexite = pd.DataFrame(resultats)
# Delta Bloomberg en pourcentage, on le ramene en fraction.
convexite["delta_moyen"] = convexite["delta_moyen"] / 100
# Ratio prix / bond floor, la variable centrale de H2.
convexite["ratio_px_bf"] = convexite["prix_moyen"] / convexite["bond_floor_moyen"]

print(len(convexite), "regressions,", convexite["isin"].nunique(), "titres")
print(convexite[["beta", "c", "r2"]].describe().round(3))


# %% CELLULE 6 - Convexite realisee par annee, avant tout controle
# -----------------------------------------------------------------------------
par_annee = convexite.groupby("annee").agg(
    titres=("c", "size"),
    convexite_mediane=("c", "median"),
    part_negative=("c", lambda serie: (serie < 0).mean()),
    vol_mediane=("vol_realisee", "median"),
)
print(par_annee.round(3).to_string())

fig, ax = plt.subplots(figsize=(10, 5))
couleurs = ["#c0392b" if v < 0 else "#27ae60" for v in par_annee["convexite_mediane"]]
ax.bar(par_annee.index.astype(str), par_annee["convexite_mediane"], color=couleurs)
ax.axhline(0, color="black", linewidth=0.8)
ax.set_ylabel("Convexite realisee mediane")
ax.set_title("Convexite realisee par annee, avant controle de volatilite")
plt.tight_layout()
plt.show()


# %% CELLULE 7 - Neutralisation de l'effet de niveau de volatilite
# -----------------------------------------------------------------------------
# Une annee tres volatile produit mecaniquement une convexite realisee plus
# negative, independamment de tout regime de marche. Pour comparer 2020 et 2022
# il faut donc raisonner a volatilite egale.
#
# Modele : c = somme des effets annee + pente * volatilite realisee
#
# ATTENTION, choix a assumer dans le memoire : ce controle suppose que la
# volatilite est un FACTEUR DE CONFUSION. On peut soutenir qu'elle est au
# contraire un CANAL DE TRANSMISSION du stress, auquel cas la neutraliser
# retire une partie du mecanisme etudie. Les deux versions doivent etre montrees.

donnees = convexite.dropna(subset=["c", "vol_realisee"]).copy()
donnees["vol"] = donnees["vol_realisee"] / 100

indicatrices = pd.get_dummies(donnees["annee"], prefix="annee").astype(float)
X = np.column_stack([indicatrices.values, donnees["vol"].values])
coefficients, *_ = np.linalg.lstsq(X, donnees["c"].values, rcond=None)

effets_annee = pd.Series(coefficients[: indicatrices.shape[1]],
                         index=[int(nom.split("_")[1]) for nom in indicatrices.columns])
pente_volatilite = coefficients[-1]

print("Pente sur la volatilite : %.4f" % pente_volatilite)
print("Une volatilite superieure de 10 points deplace la convexite de %.4f"
      % (pente_volatilite * 0.10))
print()

comparaison = pd.DataFrame({
    "brut": par_annee["convexite_mediane"],
    "net_de_volatilite": effets_annee,
    "volatilite_mediane": par_annee["vol_mediane"],
})
print(comparaison.round(3).to_string())

fig, ax = plt.subplots(figsize=(10, 5))
largeur = 0.4
positions = np.arange(len(comparaison))
ax.bar(positions - largeur / 2, comparaison["brut"], largeur, label="Brut", color="#7f8c8d")
ax.bar(positions + largeur / 2, comparaison["net_de_volatilite"], largeur,
       label="Net de volatilite", color="#2980b9")
ax.set_xticks(positions)
ax.set_xticklabels(comparaison.index.astype(str))
ax.axhline(0, color="black", linewidth=0.8)
ax.set_ylabel("Convexite realisee")
ax.set_title("Effet du controle de volatilite sur le classement des annees")
ax.legend()
plt.tight_layout()
plt.show()


# %% CELLULE 8 - Test de H1 : les deltas intermediaires sont-ils les plus vulnerables ?
# -----------------------------------------------------------------------------
# AVERTISSEMENT METHODOLOGIQUE
# La convexite estimee titre par titre est bruitee : la correlation entre
# l'estimation quotidienne et l'estimation hebdomadaire n'est que de 0,27.
# Il ne faut donc PAS conclure a partir de valeurs individuelles. On raisonne
# par tranche de delta, sur des medianes, avec un effectif suffisant par tranche.

bornes = [0, 0.20, 0.40, 0.60, 0.80, 10]
etiquettes = ["0-20 %", "20-40 %", "40-60 %", "60-80 %", "> 80 %"]
convexite["tranche_delta"] = pd.cut(convexite["delta_moyen"], bornes, labels=etiquettes)

ANNEE_STRESS = 2022
stress = convexite[convexite["annee"] == ANNEE_STRESS]

resume = stress.groupby("tranche_delta", observed=True).agg(
    titres=("c", "size"),
    convexite_mediane=("c", "median"),
    part_negative=("c", lambda serie: (serie < 0).mean()),
)
print("Annee testee :", ANNEE_STRESS)
print(resume.round(3).to_string())

fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(resume.index.astype(str), resume["convexite_mediane"], color="#8e44ad")
ax.axhline(0, color="black", linewidth=0.8)
ax.set_ylabel("Convexite realisee mediane")
ax.set_xlabel("Tranche de delta")
ax.set_title(f"H1 : convexite realisee par tranche de delta en {ANNEE_STRESS}")
plt.tight_layout()
plt.show()

print()
print("Lecture : H1 est soutenue si la tranche 40-60 % est la plus negative.")
print("Verifier l'effectif de chaque tranche avant de conclure.")


# %% CELLULE 9 - Sauvegarde
# -----------------------------------------------------------------------------
convexite.to_parquet("convexite_realisee.parquet", index=False)
print("Ecrit : convexite_realisee.parquet")
print(convexite.columns.tolist())


# %% CELLULE 10 - Renouvellement de la population de convertibles
# -----------------------------------------------------------------------------
# Les convertibles ont des maturites courtes, autour de cinq ans. La population
# se renouvelle donc presque integralement d'un bout a l'autre de la periode.
# Consequence : comparer la convexite mediane de deux annees eloignees ne compare
# pas deux etats du meme marche mais deux populations differentes.
# Cette cellule produit les deux figures qui documentent la limite.

annees = sorted(convexite["annee"].unique())
populations = {a: set(convexite.loc[convexite["annee"] == a, "isin"]) for a in annees}

recouvrement = pd.DataFrame(index=annees, columns=annees, dtype=float)
for a in annees:
    for b in annees:
        union = len(populations[a] | populations[b])
        recouvrement.loc[a, b] = len(populations[a] & populations[b]) / union if union else np.nan

fig, ax = plt.subplots(figsize=(8, 6.5))
image = ax.imshow(recouvrement.values.astype(float), cmap="YlOrRd", vmin=0, vmax=1)
ax.set_xticks(range(len(annees)))
ax.set_xticklabels(annees, rotation=45)
ax.set_yticks(range(len(annees)))
ax.set_yticklabels(annees)
ax.set_title("Recouvrement des populations de convertibles entre annees")
fig.colorbar(image, ax=ax, label="Part de titres communs")
for i in range(len(annees)):
    for j in range(len(annees)):
        valeur = recouvrement.iloc[i, j]
        ax.text(j, i, f"{valeur:.0%}", ha="center", va="center", fontsize=7,
                color="white" if valeur > 0.55 else "black")
plt.tight_layout()
plt.show()

ANNEE_BASE = 2015
cohorte = populations[ANNEE_BASE]
survie = pd.Series({a: len(cohorte & populations[a]) / len(cohorte) for a in annees})

fig, ax = plt.subplots(figsize=(9, 4.5))
ax.plot(survie.index, survie.values * 100, marker="o", color="#c0392b")
ax.axhline(50, color="grey", linestyle="--", linewidth=0.8)
ax.set_ylabel("Part de la cohorte encore presente (%)")
ax.set_title(f"Survie de la population de convertibles de {ANNEE_BASE}")
ax.set_ylim(0, 105)
plt.tight_layout()
plt.show()

print("Recouvrement avec 2022 :")
for a in annees:
    print(f"  {a} : {recouvrement.loc[a, 2022]:5.1%}")
print()
print("Regle proposee : comparaisons entre annees valides a horizon de trois ans")
print("environ, au-dela la population n'est plus la meme.")


# %% CELLULE 11 - Convexite au niveau PORTEFEUILLE, par tranche de delta
# -----------------------------------------------------------------------------
# POURQUOI CETTE ETAPE
# Estimer la convexite titre par titre puis prendre la mediane empile le bruit :
# chaque estimation individuelle est imprecise, et la mediane de vingt estimations
# bruitees reste bruitee. On inverse donc l'ordre des operations. On construit
# d'abord un portefeuille equipondere par tranche de delta, ce qui elimine le
# risque idiosyncratique par diversification, et on estime la convexite ENSUITE,
# sur une seule serie de 250 rendements.
#
# La grandeur estimee n'est pas la meme : c'est la convexite DU PORTEFEUILLE,
# pas la moyenne des convexites individuelles. C'est aussi celle qui interesse
# un gerant, donc le bon objet pour le chapitre III.
#
# Le classement en tranches utilise le delta de DEBUT D'ANNEE. Utiliser le delta
# moyen reviendrait a classer les titres d'apres une derive qui fait partie du
# phenomene etudie.

MIN_TITRES_PAR_JOUR = 5
MIN_JOURS = 100
BORNES = [0, 0.20, 0.40, 0.60, 10]
ETIQUETTES = ["0-20 %", "20-40 %", "40-60 %", "> 60 %"]

base = panel[panel["exploitable"]].copy()
base["annee"] = base["date"].dt.year

delta_initial = (base.sort_values("date").groupby(["isin", "annee"]).first()["delta"] / 100)
base = base.join(delta_initial.rename("delta_initial"), on=["isin", "annee"])
base["tranche"] = pd.cut(base["delta_initial"], BORNES, labels=ETIQUETTES)


def regression_robuste(y, x, x_retarde):
    """MCO avec ecarts-types robustes a l'heteroscedasticite (White)."""
    X = np.column_stack([np.ones(len(x)), x, x ** 2, x_retarde])
    coefficients, *_ = np.linalg.lstsq(X, y, rcond=None)
    residus = y - X @ coefficients
    XtX_inv = np.linalg.inv(X.T @ X)
    V = XtX_inv @ (X.T @ np.diag(residus ** 2) @ X) @ XtX_inv
    ecarts_types = np.sqrt(np.diag(V))
    sct = ((y - y.mean()) ** 2).sum()
    return coefficients, ecarts_types, 1 - residus @ residus / sct


lignes = []
for (annee, tranche), groupe in base.groupby(["annee", "tranche"], observed=True):
    portefeuille = groupe.groupby("date").agg(
        rendement_cb=("ret_cb", "mean"),
        rendement_action=("ret_propre", "mean"),
        effectif=("isin", "size"),
    )
    portefeuille = portefeuille[portefeuille["effectif"] >= MIN_TITRES_PAR_JOUR].sort_index()
    portefeuille["rendement_action_retarde"] = portefeuille["rendement_action"].shift(1)
    portefeuille = portefeuille.dropna()
    if len(portefeuille) < MIN_JOURS:
        continue
    coefficients, ecarts_types, r2 = regression_robuste(
        portefeuille["rendement_cb"].values,
        portefeuille["rendement_action"].values,
        portefeuille["rendement_action_retarde"].values,
    )
    lignes.append(dict(
        annee=annee, tranche=tranche, titres=groupe["isin"].nunique(),
        jours=len(portefeuille), beta=coefficients[1], c=coefficients[2],
        ecart_type_c=ecarts_types[2], t_c=coefficients[2] / ecarts_types[2], r2=r2,
    ))

portefeuilles = pd.DataFrame(lignes)
portefeuilles.to_parquet("convexite_portefeuille.parquet", index=False)

print("Convexite de portefeuille, coefficient c")
print(portefeuilles.pivot(index="annee", columns="tranche", values="c").round(2).to_string())
print()
print("t de Student robustes (au-dela de 2 en valeur absolue, le coefficient est significatif)")
print(portefeuilles.pivot(index="annee", columns="tranche", values="t_c").round(1).to_string())


# %% CELLULE 12 - La figure centrale de II.2 : 2020 contre 2022
# -----------------------------------------------------------------------------
ANNEES_COMPAREES = [2020, 2022]
extrait = portefeuilles[portefeuilles["annee"].isin(ANNEES_COMPAREES)]

fig, ax = plt.subplots(figsize=(10, 5.5))
largeur = 0.38
positions = np.arange(len(ETIQUETTES))
couleurs = {2020: "#2980b9", 2022: "#c0392b"}

for decalage, annee in zip([-largeur / 2, largeur / 2], ANNEES_COMPAREES):
    bloc = extrait[extrait["annee"] == annee].set_index("tranche").reindex(ETIQUETTES)
    barres = ax.bar(positions + decalage, bloc["c"], largeur,
                    yerr=1.96 * bloc["ecart_type_c"], capsize=4,
                    label=str(annee), color=couleurs[annee])
    for position, valeur, t in zip(positions + decalage, bloc["c"], bloc["t_c"]):
        if pd.notna(t) and abs(t) > 2:
            ax.text(position, valeur - 0.08, "*", ha="center", fontsize=16, color="black")

ax.set_xticks(positions)
ax.set_xticklabels(ETIQUETTES)
ax.axhline(0, color="black", linewidth=0.9)
ax.set_ylabel("Convexite realisee du portefeuille")
ax.set_xlabel("Tranche de delta en debut d'annee")
ax.set_title("Convexite realisee par segment de delta : choc actions (2020) contre stress conjoint (2022)")
ax.legend(title="Annee")
plt.figtext(0.01, 0.01, "Barres d'erreur : intervalle a 95 %. Asterisque : coefficient significatif.",
            fontsize=8, color="grey")
plt.tight_layout()
plt.show()
