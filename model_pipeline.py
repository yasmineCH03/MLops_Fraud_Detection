"""
model_pipeline.py

Pipeline réutilisable pour la détection de fraude sur les cartes de crédit.
Toutes les étapes du workflow ML sont ici modularisées et documentées pour une utilisation et des tests indépendants.

Fonctions principales :
- prepare_data() : Charger et prétraiter les données.
- train_model() : Entraîner le modèle.
- evaluate_model() : Évaluer les performances du modèle.
- save_model() : Sauvegarder le modèle entraîné avec joblib.
- load_model() : Charger un modèle sauvegardé.
- compare_models() : Comparer plusieurs modèles.
- pipeline() : Exécuter l'ensemble du pipeline.

Auteurs : yasmineCH03
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Tuple, Any, Dict, Optional
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

class MLPipeline:
    """
    Classe pipeline ML réutilisable et modulaire pour la détection de fraude.

    Méthodes principales :
    - prepare_data() : Charger et prétraiter les données.
    - train_model() : Entraîner le modèle.
    - evaluate_model() : Évaluer les performances.
    - save_model() : Sauvegarder le modèle.
    - load_model() : Charger un modèle.
    """

    def __init__(
        self,
        data_path: str = "data/raw/creditcard.csv",
        model_dir: str = "models",
        log_dir: str = "logs",
        random_state: int = 42,
        test_size: float = 0.2
    ):
        """
        Initialisation du pipeline ML.

        Args:
            data_path (str): Chemin vers le fichier CSV contenant les données.
            model_dir (str): Dossier où sauver les modèles.
            log_dir (str): Dossier de logs.
            random_state (int): graine aléatoire.
            test_size (float): Proportion du test set.
        """
        self.data_path = data_path
        self.model_dir = model_dir
        self.log_dir = log_dir
        self.random_state = random_state
        self.test_size = test_size

        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        self.scaler = None
        self.model = None
        self.best_params = None
        self.feature_names = None

    def prepare_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Charger, prétraiter et splitter les données en train/test.
        Applique SMOTE sur le train.

        Returns:
            X_train_scaled (np.ndarray): Features d'entraînement normalisées.
            X_test_scaled (np.ndarray): Features de test normalisées.
            y_train_res (np.ndarray): Labels d'entraînement équilibrés.
            y_test (np.ndarray): Labels de test.
        """
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        df = pd.read_csv(self.data_path)
        if "Class" not in df.columns:
            raise ValueError("Le fichier doit contenir une colonne 'Class' pour la cible.")

        X = df.drop(columns=["Class"])
        y = df["Class"]
        self.feature_names = X.columns.tolist()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )

        # Gestion du déséquilibre avec SMOTE
        smote = SMOTE(random_state=self.random_state)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train_res)
        X_test_scaled = self.scaler.transform(X_test)

        return X_train_scaled, X_test_scaled, y_train_res, y_test

    def train_model(self, X_train: np.ndarray, y_train: np.ndarray, model_type: str = "random_forest") -> Any:
        """
        Entraîner un modèle ML sur les données d'entraînement.

        Args:
            X_train (np.ndarray): Features d'entraînement.
            y_train (np.ndarray): Labels d'entraînement.
            model_type (str): Type de modèle à utiliser ('random_forest', 'xgboost', 'lightgbm', 'catboost').

        Returns:
            model: Modèle entraîné.
        """
        if model_type == "random_forest":
            model = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=self.random_state)
        elif model_type == "xgboost":
            model = xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, use_label_encoder=False, eval_metric="logloss", random_state=self.random_state)
        elif model_type == "lightgbm":
            model = lgb.LGBMClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=self.random_state)
        elif model_type == "catboost":
            model = cb.CatBoostClassifier(iterations=100, depth=6, learning_rate=0.1, verbose=False, random_state=self.random_state)
        else:
            raise ValueError(f"Type de modèle non supporté: {model_type}")

        model.fit(X_train, y_train)
        self.model = model
        return model

    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Évaluer le modèle sur un jeu de test.

        Args:
            X_test (np.ndarray): Features de test.
            y_test (np.ndarray): Labels de test.

        Returns:
            metrics (dict): Dictionnaire des métriques de performances.
        """
        if self.model is None:
            raise ValueError("Aucun modèle entraîné.")
        y_pred = self.model.predict(X_test)
        y_prob = self.model.predict_proba(X_test)[:, 1]
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, y_prob),
            "pr_auc": average_precision_score(y_test, y_prob)
        }
        return metrics

    def save_model(self, model_name: str = "best_model"):
        """
        Sauvegarder le modèle et le scaler avec joblib.

        Args:
            model_name (str): Nom du fichier à créer.
        """
        if self.model is None or self.scaler is None:
            raise ValueError("Modèle ou scaler absent, entraînez d'abord le modèle.")
        model_path = os.path.join(self.model_dir, f"{model_name}.joblib")
        scaler_path = os.path.join(self.model_dir, f"{model_name}_scaler.joblib")
        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)

    def load_model(self, model_name: str = "best_model"):
        """
        Charger un modèle et un scaler sauvegardés.

        Args:
            model_name (str): Nom du modèle à charger.
        """
        model_path = os.path.join(self.model_dir, f"{model_name}.joblib")
        scaler_path = os.path.join(self.model_dir, f"{model_name}_scaler.joblib")
        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            raise FileNotFoundError("Modèle ou scaler introuvable.")
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Prédire à partir de nouvelles données.

        Args:
            X (pd.DataFrame): Features (mêmes colonnes que le train).

        Returns:
            np.ndarray: Prédictions du modèle.
        """
        if self.model is None or self.scaler is None:
            raise ValueError("Modèle ou scaler non chargés.")
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def compare_models(self, X_train, y_train, X_test, y_test, models=["random_forest", "xgboost", "lightgbm", "catboost"]):
        """
        Comparer plusieurs modèles ML sur les mêmes données.

        Args:
            X_train, y_train, X_test, y_test: Données.
            models (list): Liste des modèles à tester.

        Returns:
            results (dict): Résultats des performances par modèle.
            best_model (str): Modèle ayant la meilleure F1.
        """
        results = {}
        for model_type in models:
            print(f"\n==> Entraînement de {model_type}")
            self.train_model(X_train, y_train, model_type=model_type)
            metrics = self.evaluate_model(X_test, y_test)
            results[model_type] = metrics
            print(f"Métriques {model_type}: {metrics}")
        best_model = max(results, key=lambda k: results[k]["f1"])
        print(f"\nMeilleur modèle: {best_model} (F1: {results[best_model]['f1']:.4f})")
        return results, best_model

    def pipeline(self):
        """
        Exécuter tout le pipeline : préparation, comparaison, sauvegarde du meilleur modèle.
        """
        X_train, X_test, y_train, y_test = self.prepare_data()
        results, best_model_type = self.compare_models(X_train, y_train, X_test, y_test)
        self.train_model(X_train, y_train, model_type=best_model_type)
        self.save_model(model_name=best_model_type)
        print("Pipeline terminé.")

