[README.md](https://github.com/user-attachments/files/31136637/README.md)
# Convexité des obligations convertibles en régime de stress conjoint

Code de traitement et d'analyse accompagnant le mémoire de M2 Asset Management,
Université Paris-Dauphine, promotion 222.

## Avertissement sur les données

**Aucune donnée n'est incluse dans ce dépôt.** Les séries employées proviennent
d'un terminal Bloomberg et leur redistribution est interdite par les conditions
d'utilisation du fournisseur. Les scripts attendent en entrée des fichiers dont
le format est décrit ci-dessous ; ils sont reproductibles à partir de toute
extraction équivalente.

## Données attendues en entrée

### `micro_long.parquet`
Historique quotidien des obligations convertibles, format long.

| Colonne | Type | Contenu |
|---|---|---|
| `isin` | texte | Identifiant de la souche |
| `issuer` | texte | Nom de l'émetteur |
| `undl` | texte | Ticker de l'action sous-jacente |
| `defaulted` | texte | YES ou NO |
| `date` | date | Date de cotation |
| `prix`, `prix_bid`, `prix_ask` | flottant | Prix en % du nominal |
| `delta` | flottant | En pourcentage, non en fraction |
| `bond_floor` | flottant | Plancher obligataire, % du nominal |
| `gamma`, `vega`, `rho` | flottant | Grecques |
| `parite` | flottant | Valeur de conversion, % du nominal |
| `prime_conv_pct` | flottant | Prime de conversion |
| `implied_vol`, `implied_spread`, `oas` | flottant | |
| `spread_sens`, `duration_eff`, `convexity` | flottant | |
| `cheapness` | flottant | Écart au prix théorique |

### `sous_jacents.parquet`
Historique quotidien des actions sous-jacentes.

| Colonne | Contenu |
|---|---|
| `IDENTIFIER` | Ticker, doit correspondre à `undl` |
| `DATE` | Date de cotation |
| `PX_LAST` | Cours de clôture |
| `VOLATILITY_30D`, `VOLATILITY_90D` | Volatilité réalisée |
| `PX_VOLUME` | Volume échangé |
| `CUR_MKT_CAP` | Capitalisation boursière |

### `univers.parquet`
Caractéristiques statiques : `isin`, `issuer`, `undl`, `issue_dt`, `maturity`,
`country`, `rtg_sp`, `rtg_mdy`, `rtg_fitch`, `amt_issued`, `coupon`, `cv_ratio`.

### `macro.parquet`
27 séries macroéconomiques en colonnes, date en index. Séries mobilisées :
`Stoxx 600`, `Bund 10 ans`, `Bund 2 ans`, `Point mort inflation 10 ans euro`,
`VSTOXX`, `iTraxx Crossover 5 ans`, indices ICE BofA convertible, IG et HY euro.

## Ordre d'exécution

```
01_fusion_micro.py            fusion des extractions xlsb en format long
02_nettoyage_sous_jacents.py  détection des opérations sur titres et penny stocks
03_panel_et_convexite.py      appariement et estimation de la convexité réalisée
04_episodes_et_captures.py    identification des épisodes, captures par segment
05_test_asymetrie.py          décomposition baisse/hausse et test de Wald
06_hypothese_H2.py            pouvoir prédictif du ratio prix sur bond floor
07_hypothese_H3.py            régime de corrélation comme indicateur avancé
08_detection_spread.py        détection du spread par défaut de 400 bp
09_backtest_allocations.py    grille d'allocations et comparaison vanille
10_figures_correlation.py     figures de régime et de cyclicité de l'inflation
11_figures_chapitre1.ipynb    figures théoriques du chapitre I
```

Les scripts numérotés de 01 à 09 se lisent dans l'ordre, chacun consommant les
fichiers produits par les précédents. Les deux derniers sont indépendants et
produisent les figures du mémoire.

## Fichiers d'entrée supplémentaires pour les figures

`10_figures_correlation.py` et `11_figures_chapitre1.ipynb` lisent un classeur
distinct contenant les séries longues employées en introduction et au chapitre I,
notamment l'historique du S&P 500 depuis 1962 et celui du Stoxx 600 depuis 1989.
Un localisateur en tête de script cherche ce fichier dans les emplacements
habituels ; le chemin peut être forcé si nécessaire.

## Environnement

```
python >= 3.10
pandas, numpy, pyarrow, scipy, matplotlib, pyxlsb, openpyxl
```

## Reproduction

Les sorties des notebooks ont été effacées avant dépôt, pour ne pas diffuser de
séries dérivées de données sous licence et pour alléger l'archive. Exécuter les
cellules dans l'ordre les régénère à partir d'une extraction équivalente.

## Convention

Les scripts sont écrits au format notebook, chaque bloc séparé par `# %%`
constituant une cellule. Les figures se terminent par `plt.show()` sans backend
non interactif, de manière à être exécutées cellule par cellule dans Jupyter ou
VS Code.
