# Ratio Position Solver: Executive Summary

## Objectif

On teste un **modèle analytique / physique** de localisation basé sur l'approximation :

\[
\mathrm{RSSI} \propto \frac{1}{d^2}
\]

Donc, pour deux antennes \(i\) et \(j\) :

\[
\sqrt{\frac{\mathrm{RSSI}_i}{\mathrm{RSSI}_j}} \approx \frac{d_j}{d_i}
\]

L'idée de l'expérience est de prendre les features
`mean__sqrt_rssi1/rssi2__RX_i|RX_j`
comme des **estimations de ratios de distances**, puis de retrouver la position \((x,y,z)\) qui satisfait au mieux ces contraintes.

---

## Les deux variantes testées

Les deux variantes utilisent :

- le **même modèle physique**
- les **mêmes positions d'antennes**
- le **même solveur numérique**
- les **mêmes bornes spatiales**

La seule différence est le traitement d'un couple de features opposées :

- `i|j`
- `j|i`

### Variante A: avec fusion

Fichier : `experiments/run_ratio_position_solver.py`

Pour une paire \((i,j)\), on lit :

- \(a = \text{feature}(i|j)\)
- \(b = \text{feature}(j|i)\)

et on construit une **contrainte unique** :

\[
\hat r_{ij} =
\begin{cases}
\sqrt{a/b} & \text{si } a \text{ et } b \text{ existent} \\
a & \text{si seul } a \text{ existe} \\
1/b & \text{si seul } b \text{ existe}
\end{cases}
\]

Puis on impose :

\[
\frac{d_j(x,y,z)}{d_i(x,y,z)} \approx \hat r_{ij}
\]

### Variante B: sans fusion

Fichier : `experiments/run_ratio_position_solver_no_fusion.py`

Chaque feature devient sa propre contrainte dirigée.

Si `i|j` existe :

\[
\frac{d_j(x,y,z)}{d_i(x,y,z)} \approx a
\]

Si `j|i` existe aussi, on ajoute séparément :

\[
\frac{d_i(x,y,z)}{d_j(x,y,z)} \approx b
\]

Donc on ne tente plus de réconcilier les deux sens en une seule quantité.

---

## Pourquoi faire une fusion ?

### Intuition courte

La fusion sert à éviter de traiter `i|j` et `j|i` comme **deux observations indépendantes d'informations différentes**, alors qu'en physique elles décrivent en fait **la même grandeur latente**, vue dans les deux sens.

La grandeur latente, c'est :

\[
r_{ij} = \frac{d_j}{d_i}
\]

Dans le monde idéal, on attend :

\[
a \approx r_{ij}
\qquad\text{et}\qquad
b \approx \frac{1}{r_{ij}}
\]

Autrement dit :

- `i|j` estime \(r_{ij}\)
- `j|i` estime son inverse

Ce ne sont donc pas deux phénomènes physiques distincts, mais deux lectures du **même rapport de distances**.

### Problème sans fusion

Sans fusion, on donne au solveur deux contraintes potentiellement contradictoires :

- une venant de `i|j`
- une venant de `j|i`

Si les deux ne sont pas parfaitement réciproques, l'optimiseur essaie de satisfaire simultanément :

\[
\frac{d_j}{d_i} \approx a
\qquad\text{et}\qquad
\frac{d_j}{d_i} \approx \frac{1}{b}
\]

Donc si \(a\) et \(1/b\) ne sont pas proches, on donne au solveur **deux versions concurrentes du même ratio**.

### Ce que fait la fusion

La fusion construit une seule estimation du ratio latent \(r_{ij}\), en utilisant les deux directions.

La formule

\[
\hat r_{ij} = \sqrt{a/b}
\]

est intéressante parce qu'elle est la moyenne naturelle **en espace logarithmique**.

Si on écrit :

\[
\log a = \log r_{ij} + \varepsilon_1
\]

\[
\log b = -\log r_{ij} + \varepsilon_2
\]

alors :

\[
\log \hat r_{ij}
= \frac{\log a - \log b}{2}
\]

et donc :

\[
\hat r_{ij} = \exp\left(\frac{\log a - \log b}{2}\right) = \sqrt{a/b}
\]

### Pourquoi cette fusion est raisonnable

Elle a trois propriétés utiles :

1. Elle traite les deux sens de manière symétrique.
2. Elle évite de compter deux fois la même relation physique.
3. Elle est adaptée à des erreurs multiplicatives, ce qui est cohérent avec des ratios RSSI.

### Limite de la fusion

La fusion n'est utile que si `i|j` et `j|i` sont **à peu près réciproques**.

Si ce n'est pas le cas, alors elle ne corrige pas un simple bruit autour d'une même grandeur :  
elle mélange deux observations déjà incohérentes, et peut donc produire un ratio qui reste physiquement discutable.

En résumé :

- **on fusionne pour estimer une seule grandeur latente à partir de deux vues opposées**
- **on ne fusionne pas** si l'on préfère laisser apparaître explicitement les contradictions des données

---

## Solveur utilisé dans les deux cas

### Optimiseur

Les deux scripts utilisent :

- `scipy.optimize.least_squares`

### Résidus minimisés

On minimise des résidus de type :

\[
\frac{d_j(x,y,z)}{d_i(x,y,z)} - r_{obs}
\]

où \(r_{obs}\) est :

- soit le ratio fusionné
- soit le ratio dirigé brut

### Robustification

Paramètres utilisés :

- `loss="soft_l1"`
- `f_scale=0.2`

Donc l'optimisation reste un problème de moindres carrés, mais avec une pénalisation plus robuste aux contraintes aberrantes qu'un L2 pur.

### Bornes spatiales

Les bornes sont fixes :

- \(x \in [1.5,\ 12.0]\)
- \(y \in [1.0,\ 5.5]\)
- \(z \in [0.0,\ 3.9)\)

Le plafond en \(z\) correspond au plan des antennes `PORT_*`, toutes placées à `z = 3.9`.

### Initialisation

Le point initial est le centre du domaine :

\[
x_0 = \frac{lower + upper}{2}
\]

Donc environ :

- \(x_0 = 6.75\)
- \(y_0 = 3.25\)
- \(z_0 = 1.95\)

### Garde-fous

- `max_nfev = 120`
- `min_constraints = 3`
- si moins de 3 contraintes valides sont disponibles sur une ligne, aucune prédiction n'est produite

---

## Résultats

| Approche | Rows solved | Acc <= 1m | Acc <= 2m | Acc <= 3m | RMSE xyz | RMSE xy | RMSE z |
|---|---:|---:|---:|---:|---:|---:|---:|
| Avec fusion | 4334 | 0.25 % | 8.08 % | 39.06 % | 2.017 m | 2.455 m | 2.486 m |
| Sans fusion | 4356 | 0.37 % | 9.62 % | 38.59 % | 2.025 m | 2.578 m | 2.378 m |

---

## Lecture rapide des résultats

- La variante **sans fusion** améliore légèrement `Acc <= 2m` et `RMSE z`.
- La variante **avec fusion** garde un léger avantage sur `RMSE xy` et `RMSE xyz`.
- Les deux restent très loin des baselines supervisées.

Le point clé à retenir est donc :

> la question n'est pas seulement "faut-il fusionner ou non ?",  
> mais surtout "ces features de ratio sont-elles suffisamment fidèles au modèle physique pour alimenter un solveur géométrique ?"

---

## Recette courte pour reproduire l'expérience

1. Charger `data/ds_analytical26April9h00.csv`.
2. Extraire toutes les colonnes `mean__sqrt_rssi1/rssi2__*`.
3. Utiliser les positions des 4 antennes `PORT_1..PORT_4`.
4. Construire les contraintes :
   - soit **une contrainte fusionnée par paire**
   - soit **une contrainte par ratio dirigé**
5. Résoudre \((x,y,z)\) avec `scipy.optimize.least_squares`.
6. Utiliser :
   - `loss="soft_l1"`
   - `f_scale=0.2`
   - bornes spatiales fixes
   - initialisation au centre
7. Évaluer avec `Acc <= 1m`, `Acc <= 2m`, `Acc <= 3m`, `RMSE xyz`, `RMSE xy`, `RMSE z`.
