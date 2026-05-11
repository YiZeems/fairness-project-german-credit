# IA Responsable, German Credit Dataset

Projet IA708, Télécom Paris, 2026.

## Équipe

- Julien GIMENEZ <julien.gimenez@telecom-paris.fr>
- Hugo FANCHINI <hugo.fanchini@telecom-paris.fr>
- Paul CINTRA <paul.cintra@telecom-paris.fr>
- Yimou ZHANG <yimou.zhang@telecom-paris.fr>
- Zaher HAMADEH <zaher.hamadeh@telecom-paris.fr>

## Contenu

Le notebook principal **`responsiveAI-german-credit_rendu.ipynb`** étudie un modèle de scoring crédit selon les trois axes du sujet `responsibleAI-4.pdf` :

- **Équité** : audit (DP, EO, Disparate Impact), audit intersectionnel (genre × âge), reweighing (pré-traitement), seuils par groupe (post-traitement maison + fairlearn ThresholdOptimizer)
- **Interprétabilité** : SHAP global et local (LinearExplainer), permutation importance, détection de proxies (corrélation et régression logistique multivariée), comparaison SHAP avec/sans attributs sensibles
- **Quantification d'incertitude** : intervalles de confiance par bootstrap (200 modèles)

Sur conseil de l'enseignante, la robustesse adversariale est remplacée par la quantification d'incertitude par bootstrap.

## Stack

- [`scikit-learn`](https://scikit-learn.org/) : modèles, métriques, pipeline, permutation_importance
- [`fairlearn`](https://fairlearn.org/) : MetricFrame, demographic_parity_difference, equalized_odds_difference, ThresholdOptimizer
- [`shap`](https://shap.readthedocs.io/) : LinearExplainer (formule exacte pour modèle linéaire)
- `pandas`, `numpy`, `matplotlib`
- `papermill` pour l'exécution reproductible

## Installation

```bash
pip install -r requirements.txt
```

Le dataset `german.data` doit être présent dans `data/raw/` (téléchargeable depuis UCI).

## Exécution

### Notebook interactif

Ouvrir `responsiveAI-german-credit_rendu.ipynb` dans Jupyter et exécuter les cellules dans l'ordre.

### Exécution automatisée (papermill)

```bash
papermill responsiveAI-german-credit_rendu.ipynb responsiveAI-german-credit_rendu.ipynb
```

(option in-place : les sorties sont sauvegardées dans le notebook source).

## Sorties générées

Toutes les figures sont dans `outputs/` :

| Fichier | Description |
|---------|-------------|
| `00_convergence.png` | Convergence log-loss et AUC vs `max_iter` |
| `01_exploration.png` | Distributions cible, genre, âge |
| `02_tradeoff.png` | Compromis performance vs équité (4 configs × 2 attributs) |
| `03_shap.png` | Top 10 features par importance SHAP |
| `04_shap_vs_perm.png` | Comparaison SHAP vs permutation importance |
| `05_bootstrap.png` | Intervalles de confiance bootstrap sur AUC, |DP|, |EO| |
| `06_indiv_uncertainty.png` | Intervalles de prédiction par individu |
| `08_shap_compare.png` | SHAP avec/sans attributs sensibles |
| `09_shap_local.png` | Contributions locales pour 2 clients (haut/bas risque) |
| `10_baseline_perf.png` | Matrice de confusion + ROC + PR du baseline |
| `11_threshold_sweep.png` | Paysage des seuils (BalAcc, coût, équité) |
| `12_dashboard.png` | Tableau de bord récapitulatif |
| `13_trivial_baselines.png` | Comparaison à 3 prédicteurs naïfs |
| `14_intersectional.png` | Audit intersectionnel genre × âge |
| `15_fairlearn_postproc.png` | Comparaison PP maison vs fairlearn ThresholdOptimizer |

## Plan du notebook

1. **Chargement et traduction des labels** : codes Axx vers libellés français
2. **Exploration** : distributions, taux de défaut par groupe (seuil âge 25 ans)
3. **Préparation** : ColumnTransformer + split 60/20/20
4. **Modèle baseline** : régression logistique L2, vérification de convergence, comparaison à des baselines triviaux
5. **Métriques** : AUC, BalAcc, |DP|, |EO|, Disparate Impact (via fairlearn) ; audit intersectionnel
6. **Mitigation pré-traitement** : reweighing $w_i = P(S)P(Y) / P(S,Y)$
7. **Mitigation post-traitement** : seuils par groupe maison + fairlearn ThresholdOptimizer (Equal Opportunity)
8. **Comparaison** : 4 configurations × 2 attributs sensibles, théorème d'impossibilité
9. **Interprétabilité** : SHAP global, local, comparaison avec/sans attributs sensibles, détection de proxies, permutation importance
10. **Quantification d'incertitude** : bootstrap B=200, IC sur métriques et prédictions
11. **Tableau de bord récapitulatif** + Conclusion

## Documents complémentaires

- **Sujet** : `responsibleAI-4.pdf`
- **Rapport long pédagogique** : `fairness-beamer/responsiveAI-german-credit_rapport.pdf` (compilation : `bash responsiveAI-german-credit_rapport_compile.sh`)
- **Note d'analyse** (2 pages) : `fairness-beamer/note_analyse.pdf`
- **Présentation Beamer** : `fairness-beamer/explication-parcours-simple_beamer.pdf`
- **Synthèse de cours pour révision** : `cours/cours-beamer/cours-responsible-ai.pdf`
- **TODO équipe** : `todo_team.txt`

## Références cours

- Fairness : `cours/Fairness/fairness-ms-1.pdf`, `cours/Fairness/fairness-mitigation.pdf`
- Interprétabilité : `cours/Introduction - Interpretabilité et explicabilité de l'IA/`
- Incertitude : `cours/Robustness and Uncertainty/_Uncertainty_presentation_MS.pdf`
