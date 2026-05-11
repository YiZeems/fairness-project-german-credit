# German Credit - IA responsable

Ce dépôt regroupe un projet de scoring de crédit sur le dataset UCI German Credit. Le travail est organisé autour de trois questions:

- comment obtenir un modèle utile sur un dataset tabulaire déséquilibré;
- comment mesurer et corriger les écarts d'équité entre groupes;
- comment expliquer les décisions et encadrer l'incertitude.

Le coeur du projet est dans `julien/`. Le dépôt est notebook-first: le notebook et le rapport PDF sont les deux points d'entrée principaux.

## A lire en premier

- `julien/responsiveAI-german-credit_light.ipynb` : notebook principal.
- `julien/responsiveAI-german-credit_rapport.pdf` : rapport compilé.
- `julien/responsiveAI-german-credit_rapport.tex` : source LaTeX du rapport.
- `julien/requirements.txt` : dépendances Python pour rejouer le notebook.

## Arborescence utile

```text
.
├── README.md
├── data/
│   └── raw/german.data
├── doc/
│   ├── german.pdf
│   ├── responsibleAI-4.pdf
│   └── todo_team.txt
└── julien/
    ├── data/raw/german.data
    ├── requirements.txt
    ├── responsiveAI-german-credit_light.ipynb
    ├── responsiveAI-german-credit_rapport.tex
    ├── responsiveAI-german-credit_rapport.pdf
    └── responsiveAI-german-credit_rapport_compile.sh
```

Le notebook lit `data/raw/german.data`. Une copie du même fichier existe aussi dans `julien/data/raw/german.data` si vous préférez travailler depuis ce dossier.

## Données et objectif

Le fichier `german.data` est un fichier texte séparé par des espaces, sans en-tête. Le jeu de données contient 1 000 dossiers de crédit décrits par 20 attributs.

- Cible recodée: `Y = 1` pour un défaut de paiement, `Y = 0` sinon.
- Attribut sensible genre: extrait de `personal_status_sex`.
- Attribut sensible âge: binarisé avec un seuil à 25 ans.

Le traitement du notebook suit une logique simple:

1. charger et nettoyer les données;
2. explorer les biais initiaux sur le genre et l'âge;
3. retirer les attributs sensibles du jeu d'entraînement de base;
4. prétraiter avec `StandardScaler` pour le numérique et `OneHotEncoder(handle_unknown="ignore")` pour le catégoriel;
5. entraîner une régression logistique;
6. choisir un seuil sur la validation;
7. mesurer performance et équité;
8. appliquer le reweighing;
9. appliquer un post-processing par seuils de groupe et `fairlearn`;
10. expliquer le modèle avec SHAP et l'importance par permutation;
11. quantifier l'incertitude avec bootstrap.

## Formules clés

### Notation

| Symbole | Sens |
|---|---|
| `x_i` | vecteur d'attributs de l'exemple `i` |
| `y_i` | label vrai, avec `1 = défaut` et `0 = bon payeur` |
| `S` | attribut sensible |
| `\hat Y` | prédiction binaire après seuillage |
| `p_i` | probabilité prédite de défaut |
| `\tau` | seuil de décision |
| `TP, FP, TN, FN` | éléments de la matrice de confusion |

### Régression logistique

La régression logistique transforme un score linéaire en probabilité:

$$
p_i = \sigma(w^\top x_i + b),
\qquad
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

La perte minimisée dans le notebook est une entropie croisée binaire pondérée avec régularisation L2:

$$
\mathcal{L}
=
-\frac{1}{n}\sum_{i=1}^{n} s_i \Big[y_i \ln p_i + (1-y_i)\ln(1-p_i)\Big]
+
\frac{1}{2C}\|w\|^2
$$

- `s_i` est le poids de l'exemple, égal à `1` sur le baseline et remplacé par les poids de reweighing lors de la mitigation;
- `C` est l'inverse de la régularisation L2;
- `p_i` est la probabilité prédite de défaut.

### Seuil et performance

La décision binaire est obtenue par seuillage:

$$
\hat y_i = \mathbf{1}[p_i \ge \tau]
$$

Le seuil n'est pas fixé à `0.5`. Il est balayé sur la validation et choisi pour maximiser la balanced accuracy:

$$
\text{BalAcc}
=
\frac{1}{2}(\text{TPR}+\text{TNR})
=
\frac{1}{2}\left(\frac{TP}{TP+FN}+\frac{TN}{TN+FP}\right)
$$

Le rapport suit aussi le coût métier:

$$
\text{Coût} = 5\,FN + 1\,FP
$$

Le jeu de données UCI pénalise plus fortement les faux négatifs, car approuver un mauvais crédit est plus coûteux que refuser un bon dossier.

### Métriques d'équité

Les deux métriques centrales du notebook sont:

$$
|\Delta \text{DP}|
=
\left|P(\hat Y=1 \mid S=0) - P(\hat Y=1 \mid S=1)\right|
$$

$$
|\Delta \text{EO}|
=
\left|\text{TPR}_{S=0} - \text{TPR}_{S=1}\right|
$$

On suit aussi le disparate impact comme un ratio de taux de sélection entre un groupe comparé et un groupe de référence:

$$
\text{DI}
=
\frac{P(\hat Y=1 \mid S=\text{groupe comparé})}{P(\hat Y=1 \mid S=\text{groupe de référence})}
$$

Le repère usuel est `0.8` pour le 80% rule. La convention exacte dépend du choix du groupe de référence, donc il faut lire ce ratio avec le sens utilisé dans le rapport.

### Reweighing

Le reweighing corrige la distribution d'entraînement en donnant plus de poids aux combinaisons `(groupe, label)` sous-représentées:

$$
w_i = \frac{P(S=s_i)\,P(Y=y_i)}{P(S=s_i, Y=y_i)}
$$

L'idée est de rapprocher `S` et `Y` d'une indépendance dans la distribution pondérée, sans changer la structure du modèle. Dans le notebook, ces poids sont passés à l'entraînement via `sample_weight`.

### Post-processing par groupe

Le post-processing utilise un seuil spécifique à chaque groupe sensible:

$$
\hat y_i = \mathbf{1}[p_i \ge \tau_{S_i}]
$$

Cette stratégie vise surtout la Demographic Parity. Le notebook compare une heuristique maison et la version officielle de `fairlearn` (`ThresholdOptimizer`) pour DP, EO et equalized odds. Sur un petit jeu de validation, cette approche peut surajuster.

### SHAP et permutation importance

Pour une régression logistique, les SHAP exacts s'écrivent:

$$
\phi_i(x)
=
w_i \cdot \left(x_i - \mathbb{E}_{train}[x_i]\right)
$$

La contribution est positive si la feature pousse vers le défaut, négative sinon. Pour les variables catégorielles encodées en one-hot, les contributions des dummies sont agrégées par feature d'origine.

L'importance par permutation mesure la chute d'AUC quand on mélange une colonne:

$$
\text{Imp}(j)
=
\frac{1}{R}\sum_{r=1}^{R}
\left(AUC_{orig} - AUC_{shuffled(j,r)}\right)
$$

Le notebook répète la permutation 10 fois par feature pour stabiliser l'estimation.

### Bootstrap et incertitude

Le rapport remplace la robustesse adversariale par une quantification d'incertitude par bootstrap. On tire `B = 200` rééchantillonnages avec remise et on calcule des intervalles de confiance par quantiles:

$$
\text{IC}_{95\%} = [q_{0.025}, q_{0.975}]
$$

Cela donne une lecture plus prudente des écarts d'équité observés sur seulement 200 exemples de test.

## Résultats marquants

- L'axe le plus sensible dans le rapport est l'âge, pas le genre.
- Sur la version actuellement documentée, le genre devient peu problématique après retrait de `personal_status_sex` (`|\Delta DP| \approx 0.010`).
- L'âge montre un vrai compromis entre DP et EO: le post-processing réduit `|\Delta DP|` mais augmente `|\Delta EO|`.
- Le rapport illustre ce compromis avec `|\Delta DP| \approx 0.125` avant mitigation sur l'âge, puis `\approx 0.060` après post-processing, tandis que `|\Delta EO|` passe d'environ `0.066` à `0.135`.
- Les SHAP et la permutation importance pointent les mêmes variables dominantes: `checking_status`, `duration_in_month`, `credit_amount`, `credit_history`, `savings_account_bonds`.
- Les intervalles bootstrap restent larges; il faut donc éviter de surinterpréter de petits écarts d'équité.

## Installation et exécution

Le plus simple est de partir du dépôt racine, de créer un environnement virtuel puis d'installer les dépendances du notebook:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install -r julien/requirements.txt
```

Ensuite, lancez Jupyter depuis la racine du dépôt et ouvrez `julien/responsiveAI-german-credit_light.ipynb`. Le notebook s'attend à trouver `data/raw/german.data` par rapport à la racine du dépôt.

Le rapport est déjà disponible en PDF. La source LaTeX et le script de compilation sont conservés dans `julien/` pour régénérer le document si nécessaire.

## Références utiles

- UCI Machine Learning Repository, German Credit Data.
- Kamiran & Calders, reweighing pour la classification sans discrimination.
- Hardt et al., Equal Opportunity et post-processing par seuils.
- Lundberg & Lee, SHAP.
- Chouldechova et Kleinberg, théorème d'impossibilité entre fairness, calibration et contraintes de taux.
