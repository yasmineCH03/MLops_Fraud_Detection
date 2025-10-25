"""
main.py

Script CLI pour exécuter les différentes étapes du pipeline ML de détection de fraude.
Chaque fonction du pipeline est appelée via un argument CLI.

Bonnes pratiques appliquées :
- Vérification des entrées/sorties.
- Test unitaire de chaque fonction possible.
- Documentation claire de chaque fonction.

Auteurs : yasmineCH03
test : github actions
"""

import argparse
import sys
import pandas as pd
from model_pipeline import MLPipeline

def main():
    parser = argparse.ArgumentParser(
        description="CLI du pipeline ML pour la détection de fraude"
    )
    parser.add_argument("--prepare", action="store_true", help="Préparer les données")
    parser.add_argument("--train", action="store_true", help="Entraîner le modèle")
    parser.add_argument("--evaluate", action="store_true", help="Évaluer le modèle")
    parser.add_argument("--predict", action="store_true", help="Prédire sur de nouvelles données")
    parser.add_argument("--compare", action="store_true", help="Comparer plusieurs modèles")
    parser.add_argument("--full-pipeline", action="store_true", help="Exécuter tout le pipeline")
    parser.add_argument("--data", type=str, default="data/raw/creditcard.csv", help="Chemin du CSV de données")
    parser.add_argument("--input", type=str, help="Fichier CSV pour la prédiction")
    parser.add_argument("--output", type=str, help="Fichier de sortie pour les prédictions")
    parser.add_argument("--model-dir", type=str, default="models", help="Dossier des modèles")
    parser.add_argument("--model-name", type=str, default="best_model", help="Nom du modèle à charger/sauver")
    parser.add_argument("--model-type", type=str, default="random_forest", choices=["random_forest", "xgboost", "lightgbm", "catboost"], help="Type de modèle à entraîner")
    args = parser.parse_args()

    pipeline = MLPipeline(
        data_path=args.data,
        model_dir=args.model_dir
    )

    if args.full_pipeline:
        pipeline.pipeline()
        return

    if args.prepare:
        print("Préparation des données...")
        X_train, X_test, y_train, y_test = pipeline.prepare_data()
        print(f"Train : {X_train.shape}, Test : {X_test.shape}")
        return

    if args.train:
        print("Entraînement du modèle...")
        X_train, X_test, y_train, y_test = pipeline.prepare_data()
        pipeline.train_model(X_train, y_train, model_type=args.model_type)
        pipeline.save_model(args.model_name)
        print(f"Modèle '{args.model_name}' entraîné et sauvegardé.")
        return

    if args.evaluate:
        print("Évaluation du modèle...")
        X_train, X_test, y_train, y_test = pipeline.prepare_data()
        pipeline.load_model(args.model_name)
        metrics = pipeline.evaluate_model(X_test, y_test)
        print("Métriques d'évaluation :")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")
        return

    if args.predict:
        if not args.input:
            print("Veuillez fournir --input pour la prédiction.")
            sys.exit(1)
        pipeline.load_model(args.model_name)
        df = pd.read_csv(args.input)
        preds = pipeline.predict(df)
        out_df = df.copy()
        out_df["prediction"] = preds
        output_path = args.output or "predictions.csv"
        out_df.to_csv(output_path, index=False)
        print(f"Prédictions sauvegardées dans {output_path}")
        return

    if args.compare:
        print("Comparaison de plusieurs modèles...")
        X_train, X_test, y_train, y_test = pipeline.prepare_data()
        pipeline.compare_models(X_train, y_train, X_test, y_test)
        return

    parser.print_help()

if __name__ == "__main__":
    main()
