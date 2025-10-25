# Pipeline ML modulaire avec Snakemake

Ce projet oriente la traçabilité et la reproductibilité du pipeline via **Snakemake**.

## Structure des pipelines

- **prepare_data** : Prépare et split les données, sauvegarde les arrays Numpy pour l'entraînement/test.
- **train_model** : Entraîne le modèle et sauvegarde le modèle/scaler.
- **eval_model** : Évalue le modèle entraîné et sauvegarde un rapport de métriques au format JSON.
- **lint_code**, **test_code**, **security_check** : Vérifient la qualité, les tests, et la sécurité du code Python.
- **all** : Orchestration complète (data, modèle, code).

## Lancer le pipeline complet

```bash
snakemake -j1 --rerun-incomplete
```

## Lancer seulement la partie code

```bash
snakemake lint_code test_code security_check
```

## Ajouter/modifier une étape

- Ajoutez une nouvelle règle dans le `Snakefile`.
- Déclarez ses entrées/sorties pour assurer la traçabilité.

## Bonnes pratiques

- Chaque étape est indépendante et documentée via docstring et README.
- Les dépendances entre étapes sont explicites via les entrées/sorties.
- Les rapports de qualité et sécurité sont archivés dans `reports/`.

## Exemples de fichiers générés

- `data/processed/`: Données traitées (Numpy arrays)
- `models/`: Modèles sauvegardés
- `reports/`: Métriques, résultats de linter, tests, sécurité

