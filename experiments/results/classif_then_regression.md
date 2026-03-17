# Classification + Regression (global multi-output)

## Approche

1. **Classifieur** : RSSI → zone (voxel 2m³)
2. **Régresseur global** : RSSI (+ zone_id optionnel) → 24 deltas (Δx, Δy, Δz × 8 antennes)
3. **Reconstruction** : position = position_antenne + delta prédit, puis moyenne des 8 estimations

Dataset : `ds_advanced_17Feb26.csv` (3814 samples, 648 features, 23 zones)
Évaluation : CV 5-fold
- **RMSE 3D** : sqrt(mean(||pos_pred - pos_true||²)) — erreur euclidienne 3D
- **RMSE sklearn** : sqrt(mean_squared_error(y_test, y_pred)) — MSE sur les composantes x,y,z aplaties (comparable aux résultats du prof)

## Résultats — ExtraTrees

| Variante       | RMSE 3D (m) | RMSE sklearn (m) | acc@1m | acc@2m | acc@3m |
|----------------|------------|-----------------|--------|--------|--------|
| SANS zone      | **1.282**  | **0.740**       | 46.6%  | **92.5%** | **98.7%** |
| AVEC zone      | 1.446      | 0.835           | 49.6%  | 87.3%  | 97.2%  |

## Résultats — RandomForest

| Variante       | RMSE 3D (m) | RMSE sklearn (m) | acc@1m | acc@2m | acc@3m |
|----------------|------------|-----------------|--------|--------|--------|
| SANS zone      | **1.348**  | **0.778**       | 44.0%  | **91.1%** | **98.3%** |
| AVEC zone      | 1.485      | 0.857           | 47.8%  | 86.7%  | 96.7%  |

## Résultats — MLP

| Variante       | RMSE 3D (m) | RMSE sklearn (m) | acc@1m | acc@2m | acc@3m |
|----------------|------------|-----------------|--------|--------|--------|
| SANS zone      | 1.807      | 1.043           | 28.3%  | 76.8%  | 93.8%  |
| AVEC zone      | **1.662**  | **0.959**       | **36.3%** | **80.5%** | **95.1%** |

## Résultats — SVM RBF

Non relancé (trop lent, ~3min/fold). Résultats précédents (RMSE 3D uniquement) :
- SANS zone : RMSE 3D = 1.978, acc@1m = 30.4%
- AVEC zone : RMSE 3D = 2.163, acc@1m = 40.8%

## Observations

- Pour les tree-based (ExtraTrees, RF), la zone **dégrade** la RMSE (~+0.1m)
- Pour MLP, la zone **améliore** la RMSE (-0.15m) — le réseau utilise l'info de zone
- Les RMSE par antenne sont identiques pour les tree-based : le régresseur multi-output apprend un offset constant entre les 8 groupes de deltas
- Pour MLP, les RMSE par antenne varient légèrement → le réseau différencie un peu les antennes

## Conclusion

ExtraTrees sans zone reste le meilleur modèle (RMSE sklearn **0.740m**). La classification n'aide que pour MLP, pas pour les tree-based qui captent déjà l'info spatiale via les 648 features RSSI.

## Script

```bash
python -m experiments.run_classif_then_regression ExtraTrees
```
