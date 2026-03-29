# Zone-Specialized Regression

## Approche

1. **Zones carrées XY** : chaque antenne définit une zone carrée de côté `2 × radius` centrée sur sa projection XY (Z ignoré). Les zones se chevauchent — un sample peut entraîner plusieurs régresseurs.
2. **Régresseur global** : ExtraTrees entraîné sur toutes les données → estimation grossière (x, y, z) pour le routage de zone.
3. **Régresseurs spécialisés** : un ExtraTrees par zone, entraîné uniquement sur les samples de sa zone → prédiction (x, y, z) directe.
4. **Inférence** : le régresseur global produit une estimation grossière, puis :
   - **single** : routage vers le régresseur de la zone la plus proche
   - **top_k** : moyenne pondérée (1/distance XY) des K régresseurs les plus proches

Dataset : `ds_Tx_Rx.csv` (4411 samples, 276 features)
Évaluation : CV 5-fold

## Couverture des zones (radius=3.0m)

| Zones occupées | Samples | % |
|---------------|---------|------|
| 2 zones | 113 | 2.6% |
| 3 zones | 119 | 2.7% |
| 4 zones | 3044 | 69.0% |
| 5 zones | 119 | 2.7% |
| 6 zones | 1016 | 23.0% |
| Non couvert | 0 | 0.0% |

Samples par zone :

| Antenne | Samples |
|---------|---------|
| 041A9F60 | 1660 |
| 041A9F61 | 1660 |
| 0F173B12 | 1912 |
| 0F173B13 | 1972 |
| PORT_1 | 2923 |
| PORT_2 | 3111 |
| PORT_3 | 2982 |
| PORT_4 | 3230 |

## Résultats — ExtraTrees (radius=1.5m)

Couverture : 0% non couvert, 63.6% dans 1 zone, 33.7% dans 2 zones, 2.7% dans 3 zones.

| Mode | RMSE 3D (m) | RMSE xyz (m) | acc@0.5m | acc@1m | acc@2m | acc@3m | Zone routing acc |
|------|------------|-------------|----------|--------|--------|--------|-----------------|
| single | 1.362 ± 0.023 | 0.786 ± 0.013 | 15.1% | 51.1% | 88.1% | 97.8% | 61.3% |

## Résultats — ExtraTrees (radius=3.0m)

| Mode | RMSE 3D (m) | RMSE xyz (m) | acc@0.5m | acc@1m | acc@2m | acc@3m | Zone routing acc |
|------|------------|-------------|----------|--------|--------|--------|-----------------|
| single | 1.250 ± 0.015 | 0.722 ± 0.009 | 10.9% | 49.9% | 93.3% | 98.8% | 61.3% |
| top_k (k=3) | **1.234 ± 0.015** | **0.712 ± 0.009** | 10.7% | **50.5%** | **93.7%** | **99.0%** | 62.0% |

## Observations

- **radius=3.0m** : fort chevauchement (69% dans 4+ zones), chaque régresseur voit la majorité des données → faible spécialisation, résultats proches du baseline global (~0.72m)
- **radius=1.5m** : zones bien plus spécialisées (64% dans 1 seule zone), mais RMSE dégradée (0.786m vs 0.722m) — les régresseurs manquent de données d'entraînement (419-1314 samples/zone vs 4411 global)
- **top_k** légèrement meilleur que **single** à radius=3.0m (~1-2%)
- Zone routing accuracy ~61% dans les deux cas : le régresseur global route souvent vers une zone "incorrecte", mais avec le chevauchement la notion de "zone correcte" est floue
- Curiosité : acc@0.5m est nettement meilleure à radius=1.5m (15.1% vs 10.9%) malgré une RMSE globale plus mauvaise — les zones spécialisées semblent aider pour les prédictions proches

## Script

```bash
uv run -m experiments.run_zone_regression --mode single --radius 3.0
uv run -m experiments.run_zone_regression --mode top_k --radius 3.0
uv run -m experiments.run_zone_regression --mode single --radius 1.5
```
