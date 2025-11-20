VENV_DIR ?= .venv
VENV_BIN ?= $(VENV_DIR)/bin
PYTHON ?= $(VENV_BIN)/python3
PREPROCESS_CONFIG ?= configs/preprocessamento.json
SELECAO_CONFIG ?= configs/selecao.json
YOLO_CONFIG ?= configs/yolo_treinamento.json
PATHS_CONFIG ?= configs/global_paths.json
RESULTS_DIR ?= resultados
OUT_ROOT ?=
MODELS_ROOT ?=
SCRIPTS_DIR ?= scripts

YOLO_EXTRA_ARGS :=
ifneq ($(strip $(OUT_ROOT)),)
YOLO_EXTRA_ARGS += --output-root $(OUT_ROOT)
endif
ifneq ($(strip $(MODELS_ROOT)),)
YOLO_EXTRA_ARGS += --models-root $(MODELS_ROOT)
endif

.PHONY: run preprocess selecao yolo install config venv clean classificar classificar_infer

run: venv preprocess selecao yolo classificar

preprocess: venv
	$(PYTHON) $(SCRIPTS_DIR)/run_preprocessamento.py --config $(PREPROCESS_CONFIG) --paths $(PATHS_CONFIG)

selecao: venv
	$(PYTHON) $(SCRIPTS_DIR)/run_selecao.py --config $(SELECAO_CONFIG) --paths $(PATHS_CONFIG)

yolo: venv
	mkdir -p train-yolo
	ULTRALYTICS_HOME=$(abspath train-yolo) $(PYTHON) $(SCRIPTS_DIR)/run_yolo_treinamento.py --config $(YOLO_CONFIG) --paths $(PATHS_CONFIG) $(YOLO_EXTRA_ARGS)

classificar: venv classificar_infer

classificar_infer: venv
	mkdir -p train-yolo
	ULTRALYTICS_HOME=$(abspath train-yolo) $(PYTHON) $(SCRIPTS_DIR)/run_classificar.py --config configs/classificar.json --device 0

venv:
	test -d $(VENV_DIR) || python3 -m venv $(VENV_DIR)
	$(VENV_BIN)/python3 -m pip install --upgrade pip
	$(VENV_BIN)/python3 -m pip install -r requirements.txt

install: venv

config:
	@echo "Pré-processamento: $(PREPROCESS_CONFIG)"
	@echo "Seleção: $(SELECAO_CONFIG)"
	@echo "YOLO: $(YOLO_CONFIG)"
	@echo "Paths: $(PATHS_CONFIG)"

clean:
	rm -rf .venv \
		train-yolo \
		modelos \
		*/yolo12_dataset \
		*/yolo12_runs \
		hsi_modificado \
		$(RESULTS_DIR) \
		./*/yolo12_dataset \
		./*/yolo12_runs
