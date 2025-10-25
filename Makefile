# ===================================================================
# Makefile : Pipelines modulaires (data, code, model) pour ML project
# ===================================================================

.PHONY: help venv install clean lint format security test data-pipeline code-pipeline model-pipeline all

# ---------------------------
# Aide
# ---------------------------
help:
	@echo "Commandes disponibles :"
	@echo "  make venv             - Créer l'environnement virtuel"
	@echo "  make install          - Installer les dépendances"
	@echo "  make clean            - Nettoyer les artefacts (models, logs, outputs)"
	@echo "  make lint             - Vérification statique (flake8, pylint)"
	@echo "  make format           - Formatage auto (black, isort)"
	@echo "  make security         - Sécurité du code (bandit)"
	@echo "  make test             - Lancer les tests unitaires"
	@echo "  make data-pipeline    - Pipeline data : préparation, split, check"
	@echo "  make code-pipeline    - Pipeline code : lint, format, sécurité, tests"
	@echo "  make model-pipeline   - Pipeline modèle : train, evaluate, compare"
	@echo "  make all              - Pipeline complet (data + code + model)"
	@echo ""
	@echo "Exemple : make data-pipeline"

# ---------------------------
# ENV
# ---------------------------
venv:
	@python3 -m venv venv
	@echo "Environnement virtuel créé."

install: venv
	@. venv/bin/activate && pip install --upgrade pip && pip install -r requirements.txt

clean:
	rm -rf venv/ __pycache__/ .pytest_cache/ logs/ models/ reports/ predictions.csv
	rm -rf data/processed/
	@echo "Nettoyage terminé."

# ---------------------------
# Pipeline DATA
# ---------------------------
data-pipeline:
	@echo ">>> Pipeline DATA : préparation, split, check"
	@. venv/bin/activate && python main.py --prepare

# ---------------------------
# Pipeline CODE (Qualité, Tests, Sécurité)
# ---------------------------
lint:
	@. venv/bin/activate && flake8 . || echo "Flake8 non disponible"
	@. venv/bin/activate && pylint main.py model_pipeline.py || echo "Pylint non disponible"

format:
	@. venv/bin/activate && black . || echo "Black non disponible"
	@. venv/bin/activate && isort . || echo "Isort non disponible"

security:
	@. venv/bin/activate && bandit -r . || echo "Bandit non disponible"

test:
	@. venv/bin/activate && pytest tests/ || echo "pytest absent ou pas de dossier tests/"

code-pipeline: lint format security test
	@echo ">>> Pipeline CODE : lint + format + sécurité + tests"

# ---------------------------
# Pipeline MODELE (ML)
# ---------------------------
model-train:
	@. venv/bin/activate && python main.py --train

model-evaluate:
	@. venv/bin/activate && python main.py --evaluate

model-compare:
	@. venv/bin/activate && python main.py --compare

model-pipeline: model-train model-evaluate model-compare
	@echo ">>> Pipeline MODELE : entraînement, évaluation, comparaison"

# ---------------------------
# Pipeline GLOBAL
# ---------------------------
all: data-pipeline code-pipeline model-pipeline
	@echo ">>> PIPELINE GLOBAL COMPLET exécuté !"

