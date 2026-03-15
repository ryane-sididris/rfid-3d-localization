Comparaison des datasets pour extratrees multioutput (approche direct et par multilatération des distances)
---
**Model:** ExtraTrees Regressor  
**Validation:** 5-fold cross-validation  
**Metrics:** RMSE (m), Accuracy within 1 m / 2 m / 3 m (%)
---

## Approche directe (RSSI → xyz)

| Dataset | RMSE mean | RMSE std | acc@1m | acc@2m | acc@3m | n_estimators |
|---|---|---|---|---|---|---|
| ds_elemen.csv | 2.031 | 0.006 | 28.20% | 72.77% | 89.27% | 100 |
| ds_advanced_17Feb26.csv | 1.255 | 0.022 | 48.83% | 92.79% | 98.84% | 200 |
| ds_Tx.csv | 1.249 | 0.019 | 49.13% | 92.81% | 98.89% | 100 |
| **ds_Tx_Rx.csv** | **1.246** | **0.017** | **49.63%** | **93.02%** | **98.91%** | 100 |
 

`ds_elemen.csv` est nettement le moins performant — seulement 4 colonnes RSSI, sans information sur l'émetteur ni sur la puissance d'émission. Les trois jeux de données avancés obtiennent des résultats très proches, `ds_Tx_Rx.csv` se distinguant légèrement sur l'ensemble des métriques.

---

## Approche par trilatération (RSSI → distances → trilatération → xyz)

| Dataset | dist\_mae mean | dist\_mae std | RMSE mean | RMSE std | acc@1m | acc@2m | acc@3m |
|---|---|---|---|---|---|---|---|
| ds_advanced_17Feb26.csv | 0.542 | 0.006 | 1.310 | 0.025 | 45.73% | 91.997% | 98.66% |
| ds_Tx_Rx.csv | 0.535 | 0.004 | 1.287 | 0.021 | 47.13% | 91.97% | 98.84% |
| **ds_Tx.csv** | **0.535** | **0.005** | **1.286** | **0.017** | **47.45%** | **91.68%** | **98.82%** |

`ds_Tx.csv` et `ds_Tx_Rx.csv` sont pratiquement à égalité ; `ds_Tx_Rx.csv` prend un léger avantage sur acc@2m et acc@3m tandis que `ds_Tx.csv` devance légèrement sur acc@1m et l'écart-type du RMSE.

---

## Comparaison des approches (meilleur jeu de données par approche)

| Approach | Dataset | RMSE mean | acc@1m | acc@2m | acc@3m |
|---|---|---|---|---|---|
| Direct | ds_Tx_Rx.csv | **1.246** | **49.63%** | **93.02%** | **98.91%** |
| Trilateration | ds_Tx.csv | 1.286 | 47.45% | 91.68% | 98.82% |
| Δ (direct − trilat.) | | −0.040 | −2.18 pp | −1.34 pp | −0.09 pp |

L'approche directe surpasse systématiquement la trilatération sur toutes les métriques. L'étape intermédiaire de prédiction des distances (erreur moyenne ~0,535 m) introduit un bruit qui se propage jusqu'à l'étape de trilatération. L'écart est faible mais constant.

---



- **Meilleur résultat global :** approche directe sur `ds_Tx_Rx.csv` (RMSE 1,246 m, acc@1m 49,6 %)
- **Tx_Rx vs Tx :** inclure l'identité de l'antenne réceptrice dans les features apporte un gain marginal mais constant dans l'approche directe ; cet effet disparaît dans l'approche par trilatération
- **Coût de la trilatération :** ~0,040 m de RMSE supplémentaire et ~2 pp de perte sur acc@1m par rapport à l'approche directe, en échange d'une représentation intermédiaire géométriquement interprétable (MAE par antenne ~0,535 m)
- **ds_elemen.csv** confirme que la richesse des features (identité de l'émetteur, niveaux de puissance) importe bien davantage que la complexité du modèle


---
(par Claude Sonnet 4.6, en se basant sur les résultats)