
Le notebook `comparaison_individuelle.ipynb` contient l'implémentation de plusieurs modèles individuels de machine learning, évalués dans des conditions identiques sur le dataset Tx_Rx. Les paramètres choisit sont 200 estimateurs pour le bagging et 300 pour le boosting.

Le notebook `trilateration_et_regressorchains.ipynb` contient l'implémentation manuelle de méthodes de trilatération (3D par Gauss-Newton) et de chaîne de régression (les prédictions des cibles précédentes sont utilisées en variables d'entrée).

Le notebook `stacking_manuel.ipynb` contient l'implémentation d'un stacking de plusieurs modèles (ExtraTrees, MultiOutput ExtraTrees, RegressorChain ExtraTrees, XGBoost, KNN). L'écart type des prédictions des arbres individuels ExtraTree est utilisé comme méta-features comme signal d'incertitude pour le blender Ridge, pour augmenter le gain.

Le notebook `rocket_kalman.ipynb` contient l'implémentation d'une approche orientée séries temporelles utilisant la méthode ROCKET (extraction de caractéristiques par convolutions aléatoires) couplée à des modèles de régression, ainsi qu'une pipeline de prétraitement intégrant un filtre de Kalman 1D pour lisser le signal brut.


Le notebook  `analyse_rssi` étudie l’évolution du RSSI en fonction de la fréquence en reconstruisant des séquences fréquentielles à partir des données RFID, puis analyse leurs corrélations et autocorrélations pour vérifier l’existence d’une structure fréquentielle exploitable.
