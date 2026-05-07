# Modèles Keras — Régression de position directe

Experimentations rapides de modèles de réseaux neuronaux Keras
## Approche

Trois architectures deep learning entraînées sur les 648 features RSSI pour prédire directement `(x, y, z)` :

1. **KerasLearnedTrilat (non-strict)** : double tête (distances `d_hat` + xyz), la tête xyz voit `[d_hat, hidden]`
2. **KerasLearnedTrilat (strict)** : double tête, la tête xyz ne voit que `d_hat` (bottleneck)
3. **KerasXYZOnly_AggressiveNL** : blocs résiduels avec augmentation trigonométrique (sin/cos), prédit xyz directement

Dataset : `ds_advanced_17Feb26.csv` (3814 samples, 648 features)
Évaluation : CV 5-fold
Entraînement : GPU Metal (Apple M3)

## Résultats

| Modèle | RMSE 3D (m) | RMSE sklearn (m) | acc@1m | acc@2m | acc@3m |
|--------|-------------|------------------|--------|--------|--------|
| KerasLearnedTrilat (non-strict) | **1.406** | **0.812** | **41.5%** | **89.0%** | **97.5%** |
| KerasLearnedTrilat (strict) | 1.429 | 0.825 | 38.8% | 88.3% | 97.4% |
| KerasXYZOnly_AggressiveNL | 1.854 | 1.071 | 39.0% | 85.8% | 96.4% |

## Observations

- Le modèle dual-head **non-strict** est le meilleur des trois : laisser la tête xyz accéder aux features cachées (pas seulement `d_hat`) aide
- Le mode **strict** (bottleneck pur par les distances) est très proche mais légèrement en dessous
- **XYZOnly_AggressiveNL** est instable : le fold 4 a produit un RMSE de 3.1m (écart-type 0.63m), ce qui tire la moyenne vers le haut. Sans ce fold aberrant, le modèle serait autour de ~1.54m
- Les modèles Keras restent en dessous d'ExtraTrees (RMSE 3D 1.247–1.282m) et de XGBoost (1.262m), confirmant que les tree-based dominent sur ce dataset de taille modeste
- Temps d'entraînement : ~110s pour les modèles trilat (5 folds), ~1270s pour XYZOnly_AggressiveNL — nettement plus lent que les tree-based

## Script

```bash
python -m experiments.run_keras_position all
```
