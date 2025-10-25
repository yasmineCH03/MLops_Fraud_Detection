import os
import numpy as np
import pandas as pd
import pytest
from model_pipeline import MLPipeline

# Génère un mini jeu de données synthétique pour le test
@pytest.fixture
def fake_data(tmp_path):
    df = pd.DataFrame({
        'V1': np.random.randn(20),
        'V2': np.random.randn(20),
        'V3': np.random.randn(20),
        'V4': np.random.randn(20),
        'V5': np.random.randn(20),
        'Amount': np.random.uniform(1, 100, 20),
        'Class': [0]*18 + [1]*2,  # 2 fraudes
    })
    data_path = tmp_path / "creditcard.csv"
    df.to_csv(data_path, index=False)
    return str(data_path)

def test_prepare_data(fake_data, tmp_path):
    pipe = MLPipeline(data_path=fake_data, model_dir=tmp_path, log_dir=tmp_path)
    X_train, X_test, y_train, y_test = pipe.prepare_data()
    assert X_train.shape[0] > 0
    assert X_test.shape[0] > 0
    assert set(np.unique(y_train)).issubset({0, 1})

def test_train_and_save_load(fake_data, tmp_path):
    pipe = MLPipeline(data_path=fake_data, model_dir=tmp_path, log_dir=tmp_path)
    X_train, X_test, y_train, y_test = pipe.prepare_data()
    pipe.train_model(X_train, y_train, model_type="random_forest")
    pipe.save_model("unittest_model")

    # Recharge le modèle et teste la prédiction
    pipe2 = MLPipeline(data_path=fake_data, model_dir=tmp_path, log_dir=tmp_path)
    pipe2.load_model("unittest_model")
    preds = pipe2.model.predict(X_test)
    assert len(preds) == len(y_test)

def test_evaluate_model(fake_data, tmp_path):
    pipe = MLPipeline(data_path=fake_data, model_dir=tmp_path, log_dir=tmp_path)
    X_train, X_test, y_train, y_test = pipe.prepare_data()
    pipe.train_model(X_train, y_train, model_type="random_forest")
    metrics = pipe.evaluate_model(X_test, y_test)
    for k in ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]:
        assert k in metrics
        assert 0.0 <= metrics[k] <= 1.0
