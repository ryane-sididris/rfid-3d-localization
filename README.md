
Le notebook  `analyse_rssi_frequence.ipynb` étudie l’évolution du RSSI en fonction de la fréquence en reconstruisant des séquences fréquentielles à partir des données RFID, puis analyse leurs corrélations et autocorrélations pour vérifier l’existence d’une structure fréquentielle exploitable.

Le notebook `comparaison_individuelle.ipynb` contient l'implémentation de plusieurs modèles individuels de machine learning, évalués dans des conditions identiques sur le dataset Tx_Rx. Les paramètres choisit sont 200 estimateurs pour le bagging et 300 pour le boosting.

Le notebook  `metric_learning.ipynb` apprend une représentation latente des signaux RFID en combinant une perte de classification des voxels spatiaux et une perte métrique visant à rapprocher dans l’espace latent les points physiquement proches. Enfin, évalue la qualité de l'espace latent en se basant sur les propriétés d'un espace latent. 

Le notebook `rocket_kalman.ipynb` contient l'implémentation d'une approche orientée séries temporelles utilisant la méthode ROCKET (extraction de caractéristiques par convolutions aléatoires) couplée à des modèles de régression, ainsi qu'une pipeline de prétraitement intégrant un filtre de Kalman 1D pour lisser le signal brut.

Le notebook `stacking_manuel.ipynb` contient l'implémentation d'un stacking de plusieurs modèles (ExtraTrees, MultiOutput ExtraTrees, RegressorChain ExtraTrees, XGBoost, KNN). L'écart type des prédictions des arbres individuels ExtraTree est utilisé comme méta-features comme signal d'incertitude pour le blender Ridge, pour augmenter le gain.

Le notebook `trilateration_et_regressorchains.ipynb` contient l'implémentation manuelle de méthodes de trilatération (3D par Gauss-Newton) et de chaîne de régression (les prédictions des cibles précédentes sont utilisées en variables d'entrée).

Le notebook `Expérimentation NN, barycentre, plus proche voisin et features fréquentielles.ipynb` explore des méthodes de localisation allant des approches géométriques classiques aux réseaux de neurones profonds. Il compare les performances du barycentre pondéré (RSSI-weighted) et du plus proche voisin avec des modèles de régression (Multi-layer Perceptron, ExtraTrees, XGBoost) entraînés sur des signatures fréquentielles pivotées. Le notebook inclut une analyse de corrélation mettant en évidence le phénomène de fading sélectif, prouvant la structure logique du signal RFID face aux interférences multi-trajets.
