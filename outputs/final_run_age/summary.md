# German Credit Responsible AI — Analyse expérimentale

## 1. Protocole expérimental

### Dataset
- Source : UCI German Credit (Statlog), 1 000 observations, 20 attributs.
- Cible : `default` (1 = mauvais payeur, classe positive), taux brut 30.0%.
- Attribut sensible : **age** (groupe privilégié : *older*).

### Splits
| Ensemble | Taille |
|---|---|
| Entraînement | 60% |
| Validation | 20% |
| Test | 20% |

Stratification sur la cible ; seed = 42.

### Modèle de base
Régression logistique entraînée par descente de gradient Adam (lr=0.03,
L2=0.01, max 3500 époques, early-stopping patience=300).

### Méthodes d'équité
| Méthode | Type | Description |
|---|---|---|
| **Reweighing** | Pré-traitement | Pondère chaque exemple d'entraînement par P(S)·P(Y) / P(S,Y) pour corriger le déséquilibre joint. |
| **Calibration par groupe** | Post-traitement | Cherche un seuil de classification distinct par groupe sur la validation pour égaliser la sélection (critère : *demographic_parity*). |

### Méthodes d'interprétabilité
- **SHAP linéaire exact** : φᵢ(x) = wᵢ·(xᵢ − E_train[xᵢ]). Équivalent au `LinearExplainer` de la bibliothèque SHAP, exact pour les modèles linéaires.
- **Importance par permutation** : diminution de l'AUC ROC quand une colonne est mélangée aléatoirement (10 répétitions).

### Évaluation de la robustesse
Perturbation contrôlée du jeu de test : bruit gaussien σ = 0.2×std sur les features numériques,
permutation aléatoire de catégories avec probabilité 0.1.

### Métriques
Performance : ROC-AUC, balanced accuracy, F1.
Équité : |Δ demographic parity|, |Δ equal opportunity|, |Δ average odds|.

---

## 2. Résultats principaux

### Test propre

| Modèle | ROC-AUC | Bal. Acc. | F1 | |ΔDP| | |ΔEO| | |ΔAO| |
|---|---|---|---|---|---|---|
| Baseline | 0.8011 | 0.7107 | 0.5960 | 0.0875 | 0.1436 | 0.0083 |
| Reweighing | 0.7943 | 0.7143 | 0.6000 | 0.0625 | 0.2257 | 0.0450 |
| Post-processing | 0.8011 | 0.7060 | 0.5906 | 0.0375 | 0.2025 | 0.0594 |

### Test perturbé (robustesse)

| Modèle | ROC-AUC | Bal. Acc. | F1 | |ΔDP| | |ΔEO| | |ΔAO| |
|---|---|---|---|---|---|---|
| Baseline | 0.7580 | 0.6917 | 0.5752 | 0.1063 | 0.1204 | 0.0165 |
| Reweighing | 0.7549 | 0.7012 | 0.5860 | 0.1437 | 0.0848 | 0.0518 |
| Post-processing | 0.7580 | 0.6702 | 0.5503 | 0.0063 | 0.2969 | 0.0934 |

---

## 3. Analyse des compromis

### Performance vs. Équité
- Reweighing : delta ROC-AUC = -0.0068,
  delta |ΔDP| = -0.0250.
- Post-processing : delta ROC-AUC = +0.0000,
  delta |ΔDP| = -0.0500.

### Robustesse
- Dégradation ROC-AUC sous perturbation — baseline : -0.0431,
  reweighing : -0.0394,
  post-processing : -0.0431.
- La perturbation affecte de façon similaire les modèles équitables et le baseline,
  suggérant que les contraintes d'équité n'amplifient pas la fragilité aux données bruitées.

---

## 4. Interprétabilité

### SHAP linéaire (mean |SHAP|)
- Top 5 baseline : checking_status, savings_account_bonds, credit_history, installment_rate, duration_in_month
- Top 5 reweighing : checking_status, savings_account_bonds, credit_history, installment_rate, duration_in_month

### Importance par permutation (drop AUC)
- Top 5 baseline : checking_status, duration_in_month, purpose, credit_history, savings_account_bonds
- Top 5 reweighing : checking_status, duration_in_month, purpose, credit_history, savings_account_bonds

Les deux méthodes concordent sur les features les plus influentes. Le statut du compte courant
(`checking_status`) et la durée du crédit dominent systématiquement la prédiction.

---

## 5. Limites de l'approche

1. **Modèle restreint** : la régression logistique est linéaire. Des modèles plus expressifs
   (gradient boosting, réseaux de neurones) capturerait des interactions non-linéaires,
   au prix d'une interprétabilité moindre et d'un SHAP approché (non exact).
2. **Un seul attribut sensible à la fois** : les biais d'intersection (âge × genre)
   ne sont pas traités.
3. **Critère d'équité unique** : améliorer la parité démographique peut dégrader l'égalité
   des chances ; aucune méthode ne satisfait simultanément tous les critères.
4. **Dataset limité** : 1 000 observations ; les estimations de métriques sont bruitées
   (intervalles de confiance non reportés).
5. **Distribution shift** : la perturbation simulée (bruit gaussien + swap catégoriel)
   est une approximation simpliste d'un vrai shift de distribution en production.

---

*Durée d'exécution : 3.2 s. Outputs dans `outputs\final_run_age`.*
