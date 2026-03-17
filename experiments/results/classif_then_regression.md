# Classification + Regression (global multi-output)

## Approche

1. **Classifieur** : RSSI → zone (voxel 2m³)
2. **Régresseur global** : RSSI (+ zone_id optionnel) → 24 deltas (Δx, Δy, Δz × 8 antennes)
3. **Reconstruction** : position = position_antenne + delta prédit, puis moyenne des 8 estimations

Dataset : `ds_advanced_17Feb26.csv` (3814 samples, 648 features, 23 zones)
Évaluation : CV 5-fold, RMSE 3D + accuracy @1m/2m/3m

## Résultats — ExtraTrees

| Variante       | RMSE avg (m) | acc@1m | acc@2m | acc@3m |
|----------------|-------------|--------|--------|--------|
| SANS zone      | **1.282**   | 46.6%  | **92.5%** | **98.7%** |
| AVEC zone      | 1.446       | 49.6%  | 87.3%  | 97.2%  |

## Résultats — MLP

| Variante       | RMSE avg (m) | acc@1m | acc@2m | acc@3m |
|----------------|-------------|--------|--------|--------|
| SANS zone      | **1.807**   | 28.3%  | **76.8%** | **93.8%** |
| AVEC zone      | 1.662       | 36.3%  | 80.5%  | 95.1%  |

## Résultats — SVM RBF

| Variante       | RMSE avg (m) | acc@1m | acc@2m | acc@3m |
|----------------|-------------|--------|--------|--------|
| SANS zone      | **1.978**   | 30.4%  | **70.0%** | **87.2%** |
| AVEC zone      | 2.163       | 40.8%  | 81.8%  | 90.0%  |

## Résultats — RandomForest

| Variante       | RMSE avg (m) | acc@1m | acc@2m | acc@3m |
|----------------|-------------|--------|--------|--------|
| SANS zone      | **1.348**   | 44.0%  | **91.1%** | **98.3%** |
| AVEC zone      | 1.485       | 47.8%  | 86.7%  | 96.7%  |

## Observations

- Ajouter la zone classifiée comme feature **dégrade la RMSE** de +0.16m
- Quand le classifieur se trompe de zone, la feature fausse trompe le régresseur → gros écarts
- La légère amélioration en acc@1m (+3%) ne compense pas la perte globale
- Les RMSE par antenne sont identiques : le régresseur global apprend la corrélation entre les 8 groupes de deltas (offset constant), donc les 8 estimations convergent vers la même position

## Conclusion

Pour un régresseur **global** multi-output, la classification n'apporte pas d'info — les 648 features RSSI contiennent déjà toute l'information spatiale. La classification a plus de sens avec un **régresseur par antenne**, où elle sert à sélectionner l'antenne pertinente.

## Script

```bash
python -m experiments.run_classif_then_regression ExtraTrees
```
