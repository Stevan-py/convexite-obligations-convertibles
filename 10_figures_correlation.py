# =============================================================================
# FIGURES DE REGIME DE CORRELATION ACTIONS-OBLIGATIONS
# Remplace la cellule existante du notebook, et ajoute une figure nouvelle.
# Format notebook : chaque bloc separe par "# %%" est une cellule.
# =============================================================================

# %% CELLULE A - Correlation actions-obligations, version corrigee
# -----------------------------------------------------------------------------
# DEUX CORRECTIONS PAR RAPPORT A LA VERSION PRECEDENTE
#
# 1. Le code couleur etait inverse. Le vert etait applique a la correlation
#    positive, qui est le regime OU LES OBLIGATIONS NE PROTEGENT PLUS. Les
#    libelles de legende etaient justes, mais un lecteur qui regarde les bandes
#    avant de lire la legende comprenait l'inverse.
#    Desormais : vert = regime protecteur, rouge = protection detruite.
#
# 2. Le nom de fichier de sortie etait ecrit en dur ("perf_2022.png"). La
#    fonction recevait un parametre "sortie" qu'elle n'utilisait jamais, donc la
#    figure euro ecrasait la figure US, et les deux ecrasaient une troisieme
#    figure sans rapport. Le parametre est maintenant utilise.
#
# RAPPEL DE CONVENTION, a garder identique dans tout le memoire :
#   On approxime le rendement obligataire par l'oppose de la variation de taux,
#   car prix ~ -duration x d(taux). La duration s'elimine dans une correlation.
#   correlation NEGATIVE = les obligations montent quand les actions baissent,
#                          elles couvrent le portefeuille (regime protecteur)
#   correlation POSITIVE = actions et obligations baissent ensemble,
#                          la protection est detruite (2022 et apres)

import warnings

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

warnings.filterwarnings("ignore")

NAVY, VERT, ROUGE = "#1F3864", "#2E7D32", "#C0392B"

# -----------------------------------------------------------------------------
# LOCALISATION DES FICHIERS
# Python cherche les fichiers dans le dossier ou tourne le notebook, pas la ou
# ils sont ranges. Ta liste ci-dessous indique ou chercher. Tes fichiers sont
# repartis dans plusieurs dossiers, d'ou la liste plutot qu'un dossier unique.
# Ajoute ou retire des lignes librement. Le "r" avant les guillemets est
# obligatoire sur Windows, il empeche Python d'interpreter les antislashs.
DOSSIERS = [
    r"C:\Users\steva\Desktop\MÉMOIRE\OLD",       # DATA_MEMOIRE_V3.xlsx
    r"C:\Users\steva\Desktop\MÉMOIRE\Data VF",   # DATA_MACRO_VF.xlsb, parquets
]


def trouver(nom_fichier):
    """Retourne le chemin complet du fichier, ou leve une erreur explicite."""
    from pathlib import Path

    dossiers = [Path(d) for d in DOSSIERS]
    maison = Path.home()
    dossiers += [
        Path.cwd(),
        maison / "Downloads", maison / "Telechargements", maison / "Téléchargements",
        maison / "Desktop", maison / "Bureau", maison / "Documents",
    ]
    for dossier in dossiers:
        candidat = dossier / nom_fichier
        if candidat.exists():
            return candidat

    extension = Path(nom_fichier).suffix
    lignes = ["", "", "FICHIER INTROUVABLE : " + nom_fichier, "", "Dossiers explores :"]
    for dossier in dossiers:
        etat = "existe" if dossier.is_dir() else "n'existe pas"
        lignes.append("   " + str(dossier) + "   (" + etat + ")")
        if dossier.is_dir():
            visibles = sorted(f.name for f in dossier.glob("*" + extension))
            if visibles:
                lignes.append("      fichiers " + extension + " presents : " + ", ".join(visibles))
    lignes += ["", "Corrige un chemin dans DOSSIERS, en haut de cette cellule.",
               "Verifie aussi le nom exact du fichier, majuscules comprises.", ""]
    raise FileNotFoundError("\n".join(lignes))


FICHIER = trouver("DATA_MEMOIRE_V3.xlsx")


def serie(df, col_valeur, col_date, start=3):
    """Extrait une serie datee depuis une feuille Excel sans en-tete."""
    valeurs = pd.to_numeric(df.iloc[start:, col_valeur], errors="coerce")
    dates = pd.to_datetime(df.iloc[start:, col_date], errors="coerce")
    s = pd.Series(valeurs.values, index=dates.values).dropna()
    return s[~s.index.duplicated()].sort_index()


def correlation(actions, taux, fenetre):
    """Correlation glissante entre rendement action et rendement obligataire approxime."""
    d = pd.concat([actions.rename("a"), taux.rename("t")], axis=1, sort=True).dropna()
    rendement_action = np.log(d["a"]).diff()
    rendement_obligataire = -d["t"].diff()
    return rendement_action.rolling(fenetre).corr(rendement_obligataire).dropna()


def figure_correlation(actions, taux, titre, source, sortie, regimes):
    correlation_6_mois = correlation(actions, taux, 126)
    correlation_1_an = correlation(actions, taux, 252)

    fig, ax = plt.subplots(figsize=(13, 6.2))

    # CORRECTION 1 : positif (protection detruite) en rouge, negatif (protecteur) en vert.
    ax.fill_between(correlation_1_an.index, 0, 1, where=correlation_1_an > 0,
                    transform=ax.get_xaxis_transform(), color=ROUGE, alpha=0.08)
    ax.fill_between(correlation_1_an.index, 0, 1, where=correlation_1_an <= 0,
                    transform=ax.get_xaxis_transform(), color=VERT, alpha=0.08)

    ax.axhline(0, color="#888", lw=0.9)
    ax.plot(correlation_6_mois.index, correlation_6_mois.values, color=NAVY, lw=0.6,
            alpha=0.30, label="Fenetre glissante 6 mois")
    ax.plot(correlation_1_an.index, correlation_1_an.values, color=NAVY, lw=1.9,
            label="Fenetre glissante 1 an")

    for date, texte in regimes:
        date = pd.Timestamp(date)
        ax.axvline(date, color=ROUGE, lw=0.9, ls="--", alpha=0.6)
        ax.annotate(texte, xy=(date, 0.78), rotation=90, va="top", ha="right",
                    fontsize=8, color=ROUGE, fontstyle="italic")

    ax.set_ylim(-0.88, 0.88)
    ax.set_title(titre, fontsize=13, fontweight="bold", color=NAVY, loc="left", pad=12)
    ax.set_ylabel("Correlation actions / obligations", fontsize=10)

    bandes = [
        Patch(facecolor=ROUGE, alpha=0.25,
              label="Correlation positive, les obligations ne protegent plus"),
        Patch(facecolor=VERT, alpha=0.25,
              label="Correlation negative, les obligations couvrent les actions"),
    ]
    ax.add_artist(ax.legend(handles=bandes, loc="upper left", fontsize=8.3, framealpha=0.9))
    ax.legend(loc="lower left", fontsize=9, framealpha=0.9)

    ax.xaxis.set_major_locator(mdates.YearLocator(5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(axis="y", alpha=0.25)
    for cote in ["top", "right"]:
        ax.spines[cote].set_visible(False)
    fig.text(0.125, 0.02, source, fontsize=8, color="#666", style="italic")
    plt.tight_layout(rect=[0, 0.03, 1, 1])

    plt.savefig(sortie, dpi=150, bbox_inches="tight")   # CORRECTION 2
    plt.show()


data_intro = pd.read_excel(FICHIER, sheet_name="Data Intro", header=None)
macro_us = pd.read_excel(FICHIER, sheet_name="Macro US", header=None)

stoxx = serie(data_intro, 5, 4)
bund = serie(data_intro, 7, 6)

spx_ancien = serie(data_intro, 1, 0)
ust_ancien = serie(data_intro, 3, 2)
spx_recent = serie(macro_us, 6, 5, start=2)
ust_recent = serie(macro_us, 14, 13, start=2)
spx = pd.concat([spx_ancien[spx_ancien.index < "2008-01-01"],
                 spx_recent[spx_recent.index >= "2008-01-01"]]).sort_index()
ust = pd.concat([ust_ancien[ust_ancien.index < "2008-01-01"],
                 ust_recent[ust_recent.index >= "2008-01-01"]]).sort_index()
spx = spx[~spx.index.duplicated()]
ust = ust[~ust.index.duplicated()]

figure_correlation(
    spx, ust,
    "Correlation glissante actions-obligations aux Etats-Unis depuis 1962 (S&P 500 / Treasury 10 ans)",
    "Source : Bloomberg", "corr_us.png",
    [("1970-01-01", "Stagflation"), ("1994-01-01", "Great Bond Massacre"),
     ("2000-06-01", "Bascule vers correlation negative"),
     ("2022-01-01", "Retour correlation positive")],
)

figure_correlation(
    stoxx, bund,
    "Correlation glissante actions-obligations en zone euro depuis 1989 (Stoxx 600 / Bund 10 ans)",
    "Source : Bloomberg", "corr_euro.png",
    [("2000-06-01", "Bascule vers correlation negative"),
     ("2022-01-01", "Retour correlation positive")],
)


# %% CELLULE B - Figure nouvelle : ce qui explique le regime de correlation
# -----------------------------------------------------------------------------
# POURQUOI CETTE FIGURE
# La cellule precedente montre QUE le regime bascule. Elle ne dit pas POURQUOI.
# La litterature (Campbell, Pflueger et Viceira ; Cieslak et Pflueger) attribue
# la bascule a la cyclicite de l'inflation, c'est-a-dire au type de choc :
#
#   inflation PROCYCLIQUE, choc de demande
#       l'inflation monte quand l'activite accelere. En recession l'inflation
#       recule, les taux baissent, les obligations montent pendant que les
#       actions baissent. Les obligations couvrent.
#
#   inflation CONTRACYCLIQUE, choc d'offre
#       l'inflation monte alors que l'activite ralentit. Les taux montent en
#       meme temps que les actions baissent. La protection disparait.
#
# On mesure la cyclicite par la correlation glissante entre les variations du
# point mort d'inflation et le rendement des actions. Positive signifie
# procyclique, donc regime protecteur attendu.
#
# Le point mort d'inflation euro commence en 2004, la figure demarre donc en 2005.

macro = pd.read_parquet(trouver("Data_macro.parquet"))

colonnes = ["Stoxx 600", "Bund 10 ans", "Point mort inflation 10 ans euro"]
base = macro[colonnes].dropna()

rendement_action = np.log(base["Stoxx 600"]).diff()
rendement_obligataire = -base["Bund 10 ans"].diff()
variation_point_mort = base["Point mort inflation 10 ans euro"].diff()

regime = rendement_action.rolling(252).corr(rendement_obligataire)
cyclicite = variation_point_mort.rolling(252).corr(rendement_action)

fig, (haut, bas) = plt.subplots(2, 1, figsize=(13, 8.5), sharex=True,
                                gridspec_kw={"height_ratios": [1, 1], "hspace": 0.12})

# Panneau du haut : le regime observe.
haut.fill_between(regime.index, 0, 1, where=regime > 0,
                  transform=haut.get_xaxis_transform(), color=ROUGE, alpha=0.08)
haut.fill_between(regime.index, 0, 1, where=regime <= 0,
                  transform=haut.get_xaxis_transform(), color=VERT, alpha=0.08)
haut.axhline(0, color="#888", lw=0.9)
haut.plot(regime.index, regime.values, color=NAVY, lw=1.9)
haut.set_ylabel("Correlation actions / obligations", fontsize=10)
haut.set_title(
    "Regime de correlation et cyclicite de l'inflation en zone euro",
    fontsize=13, fontweight="bold", color=NAVY, loc="left", pad=12,
)
haut.legend(handles=[
    Patch(facecolor=ROUGE, alpha=0.25, label="Protection obligataire detruite"),
    Patch(facecolor=VERT, alpha=0.25, label="Regime protecteur"),
], loc="upper left", fontsize=8.3, framealpha=0.9)

# Panneau du bas : le determinant avance par la litterature.
bas.fill_between(cyclicite.index, 0, 1, where=cyclicite <= 0,
                 transform=bas.get_xaxis_transform(), color=ROUGE, alpha=0.08)
bas.fill_between(cyclicite.index, 0, 1, where=cyclicite > 0,
                 transform=bas.get_xaxis_transform(), color=VERT, alpha=0.08)
bas.axhline(0, color="#888", lw=0.9)
bas.plot(cyclicite.index, cyclicite.values, color="#8E44AD", lw=1.9)
bas.set_ylabel("Cyclicite de l'inflation", fontsize=10)
bas.legend(handles=[
    Patch(facecolor=VERT, alpha=0.25, label="Inflation procyclique, choc de demande"),
    Patch(facecolor=ROUGE, alpha=0.25, label="Inflation contracyclique, choc d'offre"),
], loc="upper left", fontsize=8.3, framealpha=0.9)

for axe in (haut, bas):
    axe.grid(axis="y", alpha=0.25)
    for cote in ["top", "right"]:
        axe.spines[cote].set_visible(False)
bas.xaxis.set_major_locator(mdates.YearLocator(2))
bas.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

fig.text(0.125, 0.02,
         "Source : Bloomberg. Cyclicite mesuree par la correlation glissante sur un an entre "
         "variations du point mort d'inflation 10 ans et rendement du Stoxx 600.",
         fontsize=8, color="#666", style="italic")
plt.tight_layout(rect=[0, 0.03, 1, 1])
plt.savefig("regime_et_cyclicite.png", dpi=150, bbox_inches="tight")
plt.show()

lien = pd.DataFrame({"regime": regime, "cyclicite": cyclicite}).dropna()
print("Correlation entre cyclicite de l'inflation et regime : %+.3f"
      % lien["cyclicite"].corr(lien["regime"]))
print("Signe negatif attendu : une inflation contracyclique accompagne la destruction")
print("de la protection obligataire.")
print()
print("Mediane par periode :")
for libelle, debut, fin in [("2005-2007", "2005", "2007"), ("2008-2009", "2008", "2009"),
                            ("2010-2014", "2010", "2014"), ("2015-2019", "2015", "2019"),
                            ("2020-2021", "2020", "2021"), ("2022", "2022", "2022"),
                            ("2023-2026", "2023", "2026")]:
    bloc = lien.loc[debut:fin]
    if bloc.empty:
        continue
    print("  %-10s cyclicite %+.3f   regime %+.3f"
          % (libelle, bloc["cyclicite"].median(), bloc["regime"].median()))
