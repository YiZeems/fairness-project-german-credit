# IA Responsable, German Credit Dataset

Projet IA708, Télécom Paris, 2026.

> Version anglaise : [README.md](README.md)

![IA Responsable, German Credit](img/responsiveAI-german-credit.png)

## Équipe

- Julien GIMENEZ <julien.gimenez@telecom-paris.fr>
- Hugo FANCHINI <hugo.fanchini@telecom-paris.fr>
- Paul CINTRA <paul.cintra@telecom-paris.fr>
- Yimou ZHANG <yimou.zhang@telecom-paris.fr>
- Zaher HAMADEH <zaher.hamadeh@telecom-paris.fr>

## Contenu du dossier

| Fichier | Rôle |
|---------|------|
| `responsiveAI-german-credit_rendu.ipynb` | Notebook principal (12 sections, 92 cellules) |
| `responsiveAI-german-credit_rapport.pdf` | Rapport long pédagogique (~30 pages) |
| `responsiveAI-german-credit_rapport.tex` | Source du rapport |
| `responsiveAI-german-credit_rapport_compile.sh` | Script de compilation du rapport (LaTeX, 2 passes) |
| `responsibleAI-4.pdf` | Sujet du projet |
| `requirements.txt` | Dépendances Python |
| `data/` | Données brutes UCI German Credit |

Le rapport PDF est le livrable écrit principal. Le notebook contient l'intégralité du pipeline computationnel et reproduit chacun des chiffres cités dans le rapport.

## Sujet

Audit d'un modèle de scoring crédit (UCI German Credit, 1000 lignes) selon les trois axes de `responsibleAI-4.pdf` :

- **Équité**, audit (DP, EO, Disparate Impact), audit intersectionnel (genre x âge), reweighing en pré-traitement (Kamiran & Calders 2012), seuils par groupe en post-traitement (Hardt et al. 2016), `fairlearn ThresholdOptimizer`.
- **Interprétabilité**, SHAP global et local (`LinearExplainer`, exact pour la régression logistique), permutation importance, détection de proxies (corrélation univariée et régression logistique multivariée), comparaison SHAP avec vs sans attributs sensibles.
- **Quantification d'incertitude**, intervalles de confiance bootstrap (B = 500 sur le baseline, B = 100 sur les trois variantes du modèle).

Sur conseil de l'enseignante, l'axe robustesse adversariale du sujet est remplacé par la quantification d'incertitude par bootstrap.

## Trois variantes du modèle

L'étude compare systématiquement trois versions du modèle :

- **V1**, avec attributs sensibles (`gender`, `age_in_years`).
- **V2**, sans attributs sensibles (baseline).
- **V3**, sans attributs sensibles ET sans proxies univariés (`|r| > 0.15`).

V3 constitue le résultat principal : amélioration simultanée de la performance (AUC 0.785 vers 0.790) et de l'équité par attribut (DI âge 0.69 vers 0.87, au-dessus du seuil EEOC) sans aucune intervention in/post-modèle. Le biais intersectionnel (genre x âge, DI = 0.48) reste non résolu.

## Stack

- [`scikit-learn`](https://scikit-learn.org/) pour les modèles, métriques, pipelines, permutation importance.
- [`fairlearn`](https://fairlearn.org/) pour `MetricFrame`, `demographic_parity_difference`, `equalized_odds_difference`, `ThresholdOptimizer`.
- [`shap`](https://shap.readthedocs.io/) pour `LinearExplainer` (forme exacte pour modèle linéaire).
- `pandas`, `numpy`, `matplotlib`.
- `papermill` pour l'exécution reproductible.

## Installation

```bash
pip install -r requirements.txt
```

Le fichier `german.data` doit être présent dans `data/raw/` (téléchargeable depuis [UCI](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data)).

## Exécution du notebook

### Interactif

Ouvrir `responsiveAI-german-credit_rendu.ipynb` dans Jupyter et exécuter les cellules dans l'ordre.

### Automatisé (papermill)

```bash
papermill responsiveAI-german-credit_rendu.ipynb responsiveAI-german-credit_rendu.ipynb
```

Exécution in-place : les sorties sont sauvegardées dans le notebook source. `random_state = 42` est fixé partout, l'exécution est donc déterministe.

L'exécution génère un dossier `outputs/` contenant toutes les figures (`00_convergence.png` à `19_bootstrap_triple.png`).

## Recompilation du rapport

```bash
bash responsiveAI-german-credit_rapport_compile.sh
```

Le script appelle `pdflatex` deux fois (pour la table des matières et les références). Il détecte automatiquement `pdflatex` ou `xelatex`. Le `.tex` référence les figures du dossier `outputs/`, il faut donc avoir exécuté le notebook au préalable si les figures doivent être régénérées.

## Plan du notebook

1. Chargement et traduction des labels : codes UCI `Axx` mappés vers libellés français.
2. Exploration et données sensibles : distributions, taux de défaut par groupe (seuil âge = 25 ans).
   - 2.1 Détection de proxies (corrélation univariée `|r| > 0.15`, AUC multivariée features vers attribut sensible).
3. Préparation des données : `ColumnTransformer`, split stratifié 60 / 20 / 20.
4. Modèle baseline : régression logistique L2, vérification de convergence, comparaison à des baselines triviaux.
   - 4.1 Visualisations de performance.
   - 4.2 Coût métier (matrice UCI).
5. Métriques de performance et d'équité : AUC, BalAcc, |DP|, |EO|, Disparate Impact (via fairlearn).
   - 5.1 Audit intersectionnel.
6. Atténuation pré-traitement : reweighing $w_i = P(S) \cdot P(Y) / P(S, Y)$.
7. Atténuation post-traitement : seuils par groupe (méthode maison + `fairlearn ThresholdOptimizer`).
8. Comparaison : 4 configurations x 3 variantes, théorème d'impossibilité.
9. Interprétabilité : SHAP global, SHAP local, comparaison avec / sans attributs sensibles, détection de proxies.
10. Quantification d'incertitude : bootstrap B = 500 + B = 100 sur les trois variantes, intervalles de prédiction par individu.
11. Synthèse et conclusion : reproductibilité méthodologique, parcours décisionnel, recommandations.
12. Références.

## Reproductibilité méthodologique

Au-delà des chiffres spécifiques à ce dataset, le projet propose un canevas transférable applicable à toute classification binaire avec attributs sensibles (scoring assurance, recrutement, allocation d'aides sociales, justice prédictive, scoring santé) : identifier les attributs sensibles, auditer le baseline, détecter les proxies, comparer V1/V2/V3, évaluer les méthodes d'atténuation classiques, interpréter en global et en local, quantifier l'incertitude, synthétiser et choisir un parcours décisionnel motivé.

## Résultat principal

Retirer simultanément les attributs sensibles et les 17 proxies univariés (`|r| > 0.15`) produit un modèle à la fois plus performant (AUC +0.005), plus équitable par attribut (|DP_age| divisé par 3, DI âge au-dessus du seuil EEOC 80 %) et plus stable statistiquement (IC bootstrap le plus serré). Sur ce dataset, le retrait amont des proxies est plus efficace que les interventions in/post-modèle ; sur d'autres datasets, V3 pourrait coûter en performance ou ne rien améliorer en équité. La leçon transférable est qu'il faut toujours comparer V1/V2/V3 explicitement plutôt que de supposer a priori l'effet d'un retrait.
