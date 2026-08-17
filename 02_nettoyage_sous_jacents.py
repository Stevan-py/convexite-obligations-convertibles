# -*- coding: utf-8 -*-
"""
Traitement des anomalies de prix des sous-jacents.

Principe : on ne supprime AUCUNE ligne. On ajoute des indicateurs et une serie de
rendements nettoyee, ce qui laisse le choix des seuils ouvert au moment des regressions.

Trois anomalies distinctes, a ne pas confondre :

1. OPERATIONS SUR TITRES (splits, regroupements, augmentations de capital)
   Detection par le nombre d'actions implicite nsh = CUR_MKT_CAP / PX_LAST.
   Un saut de nsh signale que le prix du jour n'est pas comparable a celui de la veille.
   Le rendement de cette date est neutralise, le niveau de prix est conserve.
   On ne reconstruit pas de serie ajustee : les evenements observes melangent
   regroupement et dilution, et la dilution est une perte reelle pour le porteur
   qu'il serait faux d'effacer.

2. GRANULARITE DE TICK (penny stocks)
   Sous 0,10 EUR, un tick de 0,001 represente plusieurs pourcents. Les rendements
   deviennent du bruit de quantification, pas de l'information. Exemple : PBY passant
   de 0,002 a 0,050 affiche un rendement de 2400 % pour deux ticks d'ecart.

3. COTATIONS FIGEES (illiquidite)
   Prix strictement identique a la veille : il n'y a pas eu de transaction informative.
   Un rendement nul mesure l'absence de cotation, pas une absence de mouvement.
"""
import numpy as np
import pandas as pd

SRC = "/home/claude/data/sous_jacents.parquet"
DST = "/mnt/user-data/outputs/sous_jacents_traite.parquet"

SEUIL_OP = 0.10      # tolerance sur la continuite de la capitalisation
SEUIL_SPLIT = 0.25   # variation de prix minimale pour qualifier un split
SEUIL_PENNY = 0.10   # prix plancher en devise de cotation

s = pd.read_parquet(SRC).sort_values(["IDENTIFIER", "DATE"]).reset_index(drop=True)

# Prix nuls ou negatifs : aberration pure, mis a NaN.
n_neg = int((s["PX_LAST"] <= 0).sum())
s.loc[s["PX_LAST"] <= 0, "PX_LAST"] = np.nan

g = s.groupby("IDENTIFIER")

# Nombre d'actions implicite et sa variation.
s["nsh"] = s["CUR_MKT_CAP"] / s["PX_LAST"]
ratio_nsh = g["nsh"].pct_change() + 1

s["ret"] = g["PX_LAST"].pct_change()

# 1. Operations sur titres, restreint aux SPLITS ET REGROUPEMENTS.
#
# Signature d'un split pur : le prix et le nombre d'actions varient en sens inverse
# et dans la meme proportion, donc la capitalisation reste continue.
#     ratio_px * ratio_nsh ~ 1  ET  ratio_px eloigne de 1
#
# Une augmentation de capital ne remplit PAS ce critere : le nombre d'actions monte
# sans que la capitalisation reste constante. Le rendement d'un porteur existant y est
# reel, meme s'il est dilue, et il ne doit pas etre efface.
ratio_px = g["PX_LAST"].pct_change() + 1
produit = ratio_px * ratio_nsh

# Le critere de split pur (capitalisation continue) ne detecte RIEN sur cet univers :
# aucune operation observee ne laisse la capitalisation inchangee. Les evenements sont
# des restructurations ou regroupement et dilution surviennent simultanement.
# On retient donc le critere plus large de DISCONTINUITE DU CAPITAL : des que le nombre
# d'actions saute, le prix cote ne porte plus sur la meme unite que la veille.
# Cout assume : sur une restructuration etalee (EMEIS, novembre 2023), plusieurs seances
# consecutives sont neutralisees, y compris des baisses reelles. C'est le prix a payer
# pour ne pas laisser un rendement de +690 % mecanique entrer dans les regressions.
s["flag_operation"] = (
    ratio_nsh.notna()
    & np.isfinite(ratio_nsh)
    & ((ratio_nsh < 1 - SEUIL_OP) | (ratio_nsh > 1 + SEUIL_OP))
)

# 2. Granularite de tick : le rendement compare deux prix dont l'un au moins est sous le plancher.
sous_plancher = s["PX_LAST"] < SEUIL_PENNY
s["flag_penny"] = sous_plancher | g["PX_LAST"].shift(1).lt(SEUIL_PENNY).fillna(False)

# 3. Cotation figee.
s["flag_fige"] = g["PX_LAST"].diff().eq(0)

# Rendement exploitable.
s["ret_propre"] = s["ret"].where(
    ~(s["flag_operation"] | s["flag_penny"]), np.nan
)

s.to_parquet(DST, index=False, compression="zstd", compression_level=19)

# Synthese par sous-jacent.
syn = s.groupby("IDENTIFIER").agg(
    n_obs=("PX_LAST", "size"),
    n_prix=("PX_LAST", "count"),
    px_min=("PX_LAST", "min"),
    px_med=("PX_LAST", "median"),
    n_operations=("flag_operation", "sum"),
    part_penny=("flag_penny", "mean"),
    part_figee=("flag_fige", "mean"),
    n_ret_propre=("ret_propre", "count"),
)
syn["part_exploitable"] = syn["n_ret_propre"] / syn["n_obs"]
syn = syn.sort_values("part_exploitable")
syn.round(4).to_csv(
    "/mnt/user-data/outputs/synthese_sous_jacents.csv", sep=";", encoding="utf-8-sig"
)

print("Prix <= 0 mis a NaN          :", n_neg)
print("Dates d'operation sur titres :", int(s["flag_operation"].sum()),
      "sur", s.loc[s["flag_operation"], "IDENTIFIER"].nunique(), "actions")
print("Lignes sous le plancher      :", int(s["flag_penny"].sum()),
      "sur", s.loc[s["flag_penny"], "IDENTIFIER"].nunique(), "actions")
print("Lignes figees                :", int(s["flag_fige"].sum()),
      "sur", s.loc[s["flag_fige"], "IDENTIFIER"].nunique(), "actions")
print()
print("Rendements bruts   :", int(s["ret"].count()))
print("Rendements propres :", int(s["ret_propre"].count()),
      f"({s['ret_propre'].count()/s['ret'].count():.2%} conserves)")
print()
print("Rendement brut   max/min :", round(s["ret"].max(), 2), "/", round(s["ret"].min(), 2))
print("Rendement propre max/min :", round(s["ret_propre"].max(), 2), "/", round(s["ret_propre"].min(), 2))
print()
print("10 sous-jacents les moins exploitables :")
print(syn.head(10)[["n_obs", "px_med", "part_penny", "part_figee", "part_exploitable"]].round(3).to_string())
