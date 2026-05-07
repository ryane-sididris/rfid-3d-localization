
Le notebook `comparaison_individuelle.ipynb` contient l'implémentation de plusieurs modèles individuels de machine learning, évalués dans des conditions identiques sur le dataset Tx_Rx. Les paramètres choisit sont 200 estimateurs pour le bagging et 300 pour le boosting.

Le notebook `trilateration_et_regressorchains.ipynb` contient l'implémentation manuelle de méthodes de trilatération (3D par Gauss-Newton) et de chaîne de régression (les prédictions des cibles précédentes sont utilisées en variables d'entrée).

Le notebook `stacking_manuel.ipynb` contient l'implémentation d'un stacking de plusieurs modèles (ExtraTrees, MultiOutput ExtraTrees, RegressorChain ExtraTrees, XGBoost, KNN). L'écart type des prédictions des arbres individuels ExtraTree est utilisé comme méta-features comme signal d'incertitude pour le blender Ridge, pour augmenter le gain.