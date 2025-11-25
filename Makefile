VENV_DIR ?= .venv
VENV_BIN ?= $(VENV_DIR)/bin
PYTHON ?= $(VENV_BIN)/python3
PREPROCESS_CONFIG ?= configs/preprocessamento.json
PATHS_CONFIG ?= configs/global_paths.json
RESULTS_DIR ?= resultados
SCRIPTS_DIR ?= scripts
CLASSIFICAR_CONFIG ?= configs/classificar.json

.PHONY: run preprocess install config venv clean
YOLO_CONFIG ?= configs/yolo_treinamento.json
YOLO_EVAL_CONFIG ?= configs/yolo_avaliacao.json

.PHONY: yolo yolo_eval classificar

run: preprocess yolo classificar

preprocess: venv
	$(PYTHON) $(SCRIPTS_DIR)/run_preprocessamento.py --config $(PREPROCESS_CONFIG) --paths $(PATHS_CONFIG)

yolo: venv
	$(PYTHON) $(SCRIPTS_DIR)/run_yolo_treinamento.py --config $(YOLO_CONFIG)

yolo_eval: venv
	$(PYTHON) $(SCRIPTS_DIR)/run_yolo_avaliacao.py --config $(YOLO_EVAL_CONFIG)

classificar: venv
	$(PYTHON) $(SCRIPTS_DIR)/run_classificar.py --config $(CLASSIFICAR_CONFIG)

venv:
	test -d $(VENV_DIR) || python3 -m venv $(VENV_DIR)
	$(VENV_BIN)/python3 -m pip install --upgrade pip
	$(VENV_BIN)/python3 -m pip install -r requirements.txt

install: venv

config:
	@echo "Pré-processamento: $(PREPROCESS_CONFIG)"
	@echo "Paths: $(PATHS_CONFIG)"
	@echo "YOLO treino: $(YOLO_CONFIG)"
	@echo "YOLO avaliação: $(YOLO_EVAL_CONFIG)"
	@echo "Classificação: $(CLASSIFICAR_CONFIG)"

clean:
	rm -rf .venv \
		hsi_modificado \
		yolo \
		$(RESULTS_DIR)
