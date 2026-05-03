# Heuristique `Tx-only` sur `ds_power_tx_rx_freq.csv`

## But

Tester une baseline très simple sur `data/ds_power_tx_rx_freq.csv` :

- on prédit `xy` par barycentre pondéré
- on prédit `z` séparément à partir des mêmes scores agrégés

Le dataset est lu dans une forme sémantique `(sample, power, tx, rx, freq)` via :

- `src/loaders/loader_power_tx_rx_freq_semantic.py`
- `src/loaders/loader_power_tx_rx_freq_tensor.py`

La logique heuristique est portée par :

- `src/models/tx_barycenter_heuristic.py`

Les métriques ajoutées ou réutilisées sont dans :

- `src/evaluation/metrics.py`

Le benchmark est lancé par :

- `experiments/run_tx_barycenter_position.py`

## Logique exacte du calcul

Pour un échantillon donné, chaque case utile du CSV correspond à :

- une puissance d’émission
- un `Tx`
- un `Rx`
- une fréquence

Exemple de colonne :

`max__rssi__27.0__PORT_4__NONE__PORT_2__NONE__926.25`

Cette colonne veut simplement dire :

- puissance = `27.0`
- émetteur = `PORT_4`
- récepteur = `PORT_2`
- fréquence = `926.25`

Dans tout ce qui suit, on part uniquement de ces valeurs-là.

### 1. Score par Tx

On ne fait **que du `Tx-only`**.

Deux variantes ont été testées.

#### Variante `raw_sum`

Pour un `Tx` donné :

1. on prend **toutes** les colonnes du CSV où ce `Tx` apparaît
2. cela inclut donc :
   - les 3 puissances
   - les 4 `Rx`
   - les 50 fréquences
3. on additionne toutes ces valeurs

Le score du `Tx` est donc :

- la somme de tout ce que ce `Tx` "porte" dans le tenseur `(power, tx, rx, freq)`

Interprétation :

- plus un `Tx` accumule de signal sur plusieurs puissances, plusieurs `Rx` et plusieurs fréquences, plus il pèse dans la prédiction

#### Variante `balanced_per_power`

Ici on fait le calcul en 3 temps.

Pour une puissance donnée, par exemple `27.0` :

1. pour chaque `Tx`, on additionne toutes les valeurs de ce `Tx` sur :
   - les 4 `Rx`
   - les 50 fréquences
2. on obtient donc, pour cette puissance, 8 totaux : un total par `Tx`
3. on additionne ensuite ces 8 totaux
4. pour chaque `Tx`, on divise son total par cette somme globale

Autrement dit, à une puissance fixée, chaque `Tx` reçoit une **part relative** comprise entre `0` et `1`, et la somme des 8 parts vaut `1`.

On refait exactement ce calcul pour les 3 puissances :

- `24.0`
- `27.0`
- `31.5`

Puis, pour chaque `Tx`, on prend la moyenne de ses 3 parts.

Le score final d’un `Tx` est donc :

- la moyenne de son importance relative à `24.0`, `27.0` et `31.5`

Interprétation :

- chaque puissance contribue à poids égal
- on évite qu’un niveau de puissance domine mécaniquement le score final

#### Variante `rssi_max`

Cette variante colle davantage à l’idée “prendre le RSSI max par Tx”.

Pour chaque `Tx`, on cherche la plus grande valeur observée pour ce `Tx`, peu importe :

- la puissance
- le `Rx`
- la fréquence

On obtient donc un seul score par `Tx` :

- `score du Tx = meilleur RSSI observé pour ce Tx`

#### Variante `rssi_max_power_corrected`

Cette variante fait la même chose, mais en corrigeant d’abord l’effet de la puissance émise.

On ne divise pas par `24`, `27` ou `31.5` directement, car ces valeurs sont en dBm.

On utilise des facteurs relatifs en prenant `24 dBm` comme référence :

- `24.0 dBm` devient `1.00`
- `27.0 dBm` devient environ `2.00`
- `31.5 dBm` devient environ `5.62`

Pour chaque puissance et chaque `Tx` :

1. on prend le RSSI max sur les `Rx` et les fréquences
2. on divise ce RSSI max par le facteur relatif de la puissance
3. on moyenne ensuite, pour chaque `Tx`, les 3 scores corrigés obtenus à `24.0`, `27.0` et `31.5 dBm`

On obtient donc encore un seul score par `Tx`, mais ramené à une puissance comparable.

### 2. Prédiction de `xy`

À partir des scores `score(t)` :

1. on trie les `Tx` par score décroissant
2. on garde les `top-k` meilleurs `Tx`
3. on prend leurs positions `xy`
4. on calcule un barycentre pondéré par les scores

Concrètement, pour calculer `x` :

1. on multiplie l’abscisse `x` de chaque `Tx` retenu par son score
2. on additionne ces produits
3. on divise par la somme des scores

On fait exactement la même chose pour `y`.

Donc :

- un `Tx` avec un score fort tire davantage la prédiction vers sa position
- un `Tx` avec un score faible compte peu dans le barycentre

Valeurs testées :

- `top_k = 4`
- `top_k = 8`

Pour les variantes `rssi_max` et `rssi_max_power_corrected`, on a aussi testé plusieurs exposants de pondération :

- `poids = score`
- `poids = score^0.5`
- `poids = score^0.333`
- `poids = score^0.25`

Ces exposants correspondent à une pondération plus ou moins agressive par le signal. Plus l’exposant est petit, plus on écrase les différences entre les `Tx`.

### 3. Prédiction de `z`

Le `z` n’est **pas** obtenu par barycentre géométrique.

On construit simplement un petit vecteur de 8 nombres :

- score du `Tx 1`
- score du `Tx 2`
- ...
- score du `Tx 8`

Puis on donne ce vecteur à un modèle très simple pour prédire la hauteur :

- `constant` : moyenne de `z` sur le fold d’entraînement
- `linear` : régression linéaire sur les 8 scores
- `extratrees` : ExtraTrees sur les 8 scores

La prédiction finale est donc :

- `x_hat` vient du barycentre `xy`
- `y_hat` vient du barycentre `xy`
- `z_hat` vient du petit modèle séparé

## Métriques rapportées

- `acc_xy@2m` : pourcentage d’échantillons avec erreur plane `xy` inférieure ou égale à `2 m`
- `rmse_xy` : RMSE de la distance en plan `xy`
- `rmse_z` : RMSE sur l’axe `z`
- `rmse_xyz` : RMSE global au sens `sklearn`, sur les 3 axes `(x,y,z)`

Validation :

- cross-validation 5 folds

## Résultats

### Scores par somme ou part relative

| score_mode | top_k | z_model | acc_xy@2m | rmse_xy | rmse_z | rmse_xyz |
|---|---:|---|---:|---:|---:|---:|
| raw_sum | 4 | constant | 88.96 +/- 0.83 | 1.370 +/- 0.027 | 0.748 +/- 0.012 | 0.901 +/- 0.016 |
| raw_sum | 4 | linear | 88.96 +/- 0.83 | 1.370 +/- 0.027 | 0.730 +/- 0.014 | 0.896 +/- 0.017 |
| raw_sum | 4 | extratrees | 88.96 +/- 0.83 | 1.370 +/- 0.027 | 0.694 +/- 0.013 | 0.887 +/- 0.016 |
| raw_sum | 8 | constant | 89.19 +/- 0.89 | 1.376 +/- 0.028 | 0.748 +/- 0.012 | 0.904 +/- 0.017 |
| raw_sum | 8 | linear | 89.19 +/- 0.89 | 1.376 +/- 0.028 | 0.730 +/- 0.014 | 0.899 +/- 0.017 |
| raw_sum | 8 | extratrees | 89.19 +/- 0.89 | 1.376 +/- 0.028 | 0.694 +/- 0.013 | 0.890 +/- 0.017 |
| balanced_per_power | 4 | constant | 88.96 +/- 0.76 | 1.353 +/- 0.023 | 0.748 +/- 0.012 | 0.892 +/- 0.014 |
| balanced_per_power | 4 | linear | 88.96 +/- 0.76 | 1.353 +/- 0.023 | 0.697 +/- 0.012 | 0.879 +/- 0.012 |
| balanced_per_power | 4 | extratrees | 88.96 +/- 0.76 | 1.353 +/- 0.023 | 0.697 +/- 0.020 | 0.879 +/- 0.015 |
| balanced_per_power | 8 | constant | 89.21 +/- 0.92 | 1.362 +/- 0.023 | 0.748 +/- 0.012 | 0.897 +/- 0.014 |
| balanced_per_power | 8 | linear | 89.21 +/- 0.92 | 1.362 +/- 0.023 | 0.697 +/- 0.012 | 0.883 +/- 0.013 |
| balanced_per_power | 8 | extratrees | 89.21 +/- 0.92 | 1.362 +/- 0.023 | 0.697 +/- 0.020 | 0.883 +/- 0.015 |

### Scores par `RSSImax` Tx

Ici on utilise `top_k = 8` et `z_model = constant`, pour isoler l’effet du barycentre `xy`.

| score_mode | weight_exponent | acc_xy@2m | rmse_xy | rmse_z | rmse_xyz |
|---|---:|---:|---:|---:|---:|
| rssi_max | 1.000 | 79.03 +/- 1.12 | 1.636 +/- 0.029 | 0.748 +/- 0.012 | 1.038 +/- 0.018 |
| rssi_max | 0.500 | 74.90 +/- 1.06 | 1.718 +/- 0.025 | 0.748 +/- 0.012 | 1.082 +/- 0.016 |
| rssi_max | 0.333 | 72.84 +/- 1.31 | 1.759 +/- 0.025 | 0.748 +/- 0.012 | 1.104 +/- 0.016 |
| rssi_max | 0.250 | 71.66 +/- 1.49 | 1.783 +/- 0.025 | 0.748 +/- 0.012 | 1.116 +/- 0.016 |
| rssi_max_power_corrected | 1.000 | 89.05 +/- 0.32 | 1.395 +/- 0.020 | 0.748 +/- 0.012 | 0.914 +/- 0.013 |
| rssi_max_power_corrected | 0.500 | 84.33 +/- 0.95 | 1.507 +/- 0.022 | 0.748 +/- 0.012 | 0.971 +/- 0.014 |
| rssi_max_power_corrected | 0.333 | 80.30 +/- 0.75 | 1.592 +/- 0.023 | 0.748 +/- 0.012 | 1.015 +/- 0.015 |
| rssi_max_power_corrected | 0.250 | 77.49 +/- 1.34 | 1.646 +/- 0.024 | 0.748 +/- 0.012 | 1.044 +/- 0.015 |

## Lecture rapide

- Le meilleur `rmse_xy` est obtenu avec `balanced_per_power + top-4`
- Le meilleur `acc_xy@2m` est obtenu avec `balanced_per_power + top-8`, mais l’écart est très faible
- Le passage de `constant` à `linear` ou `extratrees` améliore un peu `z`, mais l’effet reste modéré
- `linear` et `extratrees` sont pratiquement équivalents sur `balanced_per_power`
- Pour les variantes `RSSImax`, corriger par la puissance émise améliore nettement le résultat
- Utiliser la moyenne des puissances corrigées est meilleur que garder uniquement la meilleure puissance corrigée
- Sur ce dataset, les exposants faibles `0.5`, `0.333` et `0.25` dégradent le barycentre : la meilleure version `RSSImax` reste `poids = score`
- La baseline centrale la plus lisible est :
  `balanced_per_power + top-4` pour `xy`, avec un petit modèle séparé pour `z`
