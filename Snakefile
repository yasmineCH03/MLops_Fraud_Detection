# Snakefile : Orchestration modulaire du pipeline ML avec Snakemake

# 1. Pipeline de préparation des données
rule prepare_data:
    input:
        raw="data/raw/creditcard.csv"
    output:
        train="data/processed/X_train.npy",
        test="data/processed/X_test.npy",
        ytrain="data/processed/y_train.npy",
        ytest="data/processed/y_test.npy"
    shell:
        """
        python main.py --prepare --data {input.raw} --output_dir data/processed/
        """

# 2. Pipeline d'entraînement du modèle
rule train_model:
    input:
        train="data/processed/X_train.npy",
        ytrain="data/processed/y_train.npy"
    output:
        model="models/best_model.joblib",
        scaler="models/best_model_scaler.joblib"
    shell:
        """
        python main.py --train --model-name best_model --train_data {input.train} --train_labels {input.ytrain}
        """

# 3. Pipeline d'évaluation du modèle
rule eval_model:
    input:
        model="models/best_model.joblib",
        scaler="models/best_model_scaler.joblib",
        test="data/processed/X_test.npy",
        ytest="data/processed/y_test.npy"
    output:
        report="reports/metrics.json"
    shell:
        """
        python main.py --evaluate --model-name best_model --test_data {input.test} --test_labels {input.ytest} --output {output.report}
        """

# 4. Qualité et sécurité du code
rule lint_code:
    input:
        expand("{file}", file=["main.py", "model_pipeline.py"])
    output:
        touch("reports/lint.done")
    shell:
        """
        flake8 . > reports/flake8.txt || true
        pylint main.py model_pipeline.py > reports/pylint.txt || true
        touch {output}
        """

rule test_code:
    input:
        expand("{file}", file=["main.py", "model_pipeline.py"])
    output:
        touch("reports/test.done")
    shell:
        """
        pytest > reports/pytest.txt || true
        touch {output}
        """

rule security_check:
    input:
        expand("{file}", file=["main.py", "model_pipeline.py"])
    output:
        touch("reports/security.done")
    shell:
        """
        bandit -r . > reports/bandit.txt || true
        touch {output}
        """

# 5. Pipeline complet
rule all:
    input:
        "reports/metrics.json",
        "reports/lint.done",
        "reports/test.done",
        "reports/security.done"
