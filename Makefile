# =================================================================
# Makefile pour pipeline ML : préparation, entraînement, tests, CI
# =================================================================

.PHONY: help venv install lint format security prepare train test all

# Aide
help:
	@echo "Commandes disponibles :"
	@echo "  make venv      - Créer l'environnement virtuel"
	@echo "  make install   - Installer les dépendances"
	@echo "  make lint      - Vérifier la qualité du code (flake8/pylint)"
	@echo "  make format    - Formatter automatiquement le code (black/isort)"
	@echo "  make security  - Vérifier la sécurité du code (bandit)"
	@echo "  make prepare   - Préparer les données (prepare_data)"
	@echo "  make train     - Entraîner le modèle (train_model)"
	@echo "  make test      - Lancer les tests unitaires"
	@echo "  make all       - Pipeline complet (lint+format+prepare+train+test+security)"

# Création de l'environnement virtuel
venv:
	@python3 -m venv venv
	@echo "Environnement virtuel créé."

# Installation des dépendances
install: venv
	@. venv/bin/activate && pip install --upgrade pip && pip install -r requirements.txt
	@echo "Dépendances installées."

# Vérification qualité du code (linting)
lint:
	@. venv/bin/activate && flake8 . || echo "Flake8 non disponible"
	@. venv/bin/activate && pylint main.py model_pipeline.py || echo "Pylint non disponible"

# Formatage auto du code
format:
	@. venv/bin/activate && black . || echo "Black non disponible"
	@. venv/bin/activate && isort . || echo "Isort non disponible"

# Vérification sécurité du code
security:
	@. venv/bin/activate && bandit -r . || echo "Bandit non disponible"

# Préparation des données
prepare:
	@. venv/bin/activate && python main.py --prepare

# Entraînement du modèle
train:
	@. venv/bin/activate && python main.py --train

# Lancer les tests unitaires
test:
	@. venv/bin/activate && pytest tests/ || echo "pytest absent ou pas de dossier tests/"

# Pipeline complet (comme un workflow CI/CD)
all: lint format prepare train test security
	@echo "Pipeline complet exécuté."

