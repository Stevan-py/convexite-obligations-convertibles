# -*- coding: utf-8 -*-
"""
01. Fusion des extractions xlsb en un fichier unique au format long.

L'extraction Bloomberg produit un format large : les blocs de dix-huit colonnes
sont juxtaposés, un par obligation convertible, soit 5 436 colonnes pour 302
titres. Cette largeur dépasse les capacités de rapatriement en une seule
opération, l'extraction a donc été découpée en cinq fichiers.

Structure de chaque fichier :
  ligne 1 : en-têtes de métadonnées (ISIN BOND, ISSUER NAME, Underlying Name,
            Defaulted ?)
  ligne 2 : valeurs des métadonnées, plus les libellés de modèle Jump Diffusion
            et Black-Scholes en colonnes 12 et 13, qui indiquent le modèle employé
            pour la volatilité implicite et le spread implicite
  ligne 3 : noms des dix-huit champs
  ligne 4 et suivantes : données quotidiennes

Chaque bloc porte sa propre colonne de dates, l'appariement doit donc se faire
sur la date propre à chaque ligne et jamais par position dans le tableau, faute
de quoi des décalages silencieux fausseraient tous les calculs.
"""

import gc
import os

import numpy as np
import pandas as pd
from pyxlsb import open_workbook

FICHIERS = ["memoire_1.xlsb", "memoire_1BIS.xlsb", "memoire_2.xlsb",
            "memoire_3.xlsb", "memoire_4.xlsb"]
DOSSIER_SOURCE = "."
FICHIER_SORTIE = "micro_long.parquet"

CHAMPS = ["date", "prix", "prix_bid", "prix_ask", "delta", "bond_floor",
          "gamma", "vega", "rho", "parite", "prime_conv_pct", "implied_vol",
          "implied_spread", "oas", "spread_sens", "duration_eff", "convexity",
          "cheapness"]
LARGEUR_BLOC = 18


def nettoyer(valeur):
    """Renvoie un flottant ou NaN, en absorbant les sentinelles d'erreur du Terminal."""
    if valeur is None:
        return np.nan
    if isinstance(valeur, (int, float)):
        return float(valeur)
    texte = str(valeur).strip()
    if not texte or texte.startswith("#"):
        return np.nan
    texte = texte.replace("\u00a0", "").replace(" ", "").replace(",", ".")
    try:
        return float(texte)
    except ValueError:
        return np.nan


def lire_feuille(chemin):
    """Retourne la feuille sous forme de liste de lignes, chaque ligne étant une liste."""
    with open_workbook(chemin) as classeur:
        nom_feuille = classeur.sheets[0]
        with classeur.get_sheet(nom_feuille) as feuille:
            lignes = []
            for ligne in feuille.rows():
                valeurs = [None] * LARGEUR_BLOC * 200
                colonne_max = -1
                for cellule in ligne:
                    if cellule.c < len(valeurs):
                        valeurs[cellule.c] = cellule.v
                        colonne_max = max(colonne_max, cellule.c)
                lignes.append(valeurs[: colonne_max + 1])
    return lignes


def traiter(chemin):
    lignes = lire_feuille(chemin)
    largeur_totale = max(len(ligne) for ligne in lignes)
    nombre_blocs = largeur_totale // LARGEUR_BLOC
    metadonnees = lignes[1]

    morceaux, vides = [], []
    for bloc in range(nombre_blocs):
        depart = bloc * LARGEUR_BLOC
        isin = metadonnees[depart] if depart < len(metadonnees) else None
        if isin is None:
            continue
        isin = str(isin).replace(" corp", "").strip()
        emetteur = metadonnees[depart + 1] if depart + 1 < len(metadonnees) else None
        sous_jacent = metadonnees[depart + 2] if depart + 2 < len(metadonnees) else None
        defaut = metadonnees[depart + 3] if depart + 3 < len(metadonnees) else None

        colonnes = {champ: [] for champ in CHAMPS}
        for index in range(3, len(lignes)):
            ligne = lignes[index]
            if depart >= len(ligne):
                continue
            brut = ligne[depart: depart + LARGEUR_BLOC]
            brut += [None] * (LARGEUR_BLOC - len(brut))
            if all(valeur is None for valeur in brut):
                continue
            for rang, champ in enumerate(CHAMPS):
                colonnes[champ].append(nettoyer(brut[rang]))

        if not colonnes["date"]:
            vides.append(isin)
            continue

        table = pd.DataFrame(colonnes)
        # Les dates arrivent en série Excel, d'origine 1899-12-30. Les valeurs
        # hors bornes correspondent à des cellules d'erreur et sont rejetées.
        table = table[table["date"].between(30000, 60000)]
        if table.empty:
            vides.append(isin)
            continue
        table["date"] = pd.to_datetime(
            table["date"].round().astype("int64"), unit="D", origin="1899-12-30")
        table.insert(0, "isin", isin)
        table.insert(1, "issuer", str(emetteur).strip() if emetteur else None)
        table.insert(2, "undl", str(sous_jacent).strip() if sous_jacent else None)
        table.insert(3, "defaulted", str(defaut).strip().upper() if defaut else None)
        morceaux.append(table)

    fusion = pd.concat(morceaux, ignore_index=True) if morceaux else pd.DataFrame()
    return fusion, nombre_blocs, vides


if __name__ == "__main__":
    ensemble, journal = [], []
    for fichier in FICHIERS:
        chemin = os.path.join(DOSSIER_SOURCE, fichier)
        table, blocs, vides = traiter(chemin)
        titres = table["isin"].nunique() if len(table) else 0
        print(f"{fichier:22s} blocs={blocs:4d}  titres avec données={titres:4d}  "
              f"lignes={len(table):8d}  vides={len(vides)}")
        if vides:
            print("       vides :", ", ".join(vides))
        journal.append((fichier, blocs, titres, len(table), vides))
        ensemble.append(table)
        gc.collect()

    micro = pd.concat(ensemble, ignore_index=True)
    micro = micro.sort_values(["isin", "date"]).reset_index(drop=True)
    micro.to_parquet(FICHIER_SORTIE, index=False, compression="zstd", compression_level=19)

    print()
    print("TOTAL blocs         :", sum(entree[1] for entree in journal))
    print("TOTAL lignes        :", len(micro))
    print("ISIN distincts      :", micro["isin"].nunique())
    print("Période             :", micro["date"].min().date(), "->", micro["date"].max().date())
    print("Doublons isin, date :", micro.duplicated(["isin", "date"]).sum())
    print()
    print("Contrôles à effectuer : aucun doublon, tous les ISIN présents dans")
    print("l'univers statique, delta exprimé en pourcentage et non en fraction.")
