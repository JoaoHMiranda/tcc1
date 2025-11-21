# Pipeline Hiperespectral Completo

Este repositório entrega um pipeline completo para transformar cubos hiperespectrais (HSI) em detectores YOLO12 capazes de localizar *Staphylococcus* spp. em amostras laboratoriais. Apesar do foco biomédico, a arquitetura foi desenhada para ser modular: basta ajustar os JSONs e trocar datasets para que o mesmo fluxo funcione em outros domínios (agro, inspeção industrial, etc.).

Esta documentação longa foi construída para servir tanto como guia técnico quanto referência operacional. Ela está organizada em nove blocos principais:

1. Visão geral do fluxo.
2. Estrutura detalhada do repositório.
3. Passo a passo das quatro etapas (pré-processamento, seleção, treinamento, classificação).
4. Guia de configuração dos arquivos JSON.
5. Dependências, preparação de ambiente e comandos Make.
6. Saídas e relatórios de cada fase.
7. Boas práticas, troubleshooting e dicas de monitoração.
8. Extensões sugeridas e integração com outros sistemas.
9. FAQ e checklist final para rodar o pipeline do zero.

---

## 1. Visão geral do fluxo

```
HSI bruto ──► Pré-processamento (correção + SNV + MSC + pseudo-RGB)
          └─► Seleção de bandas (método único + PCA + classificadores)
                 └─► Treinamento YOLO (dataset train/val/test + Ultralytics)
                         └─► Classificação (execução do modelo em "classificar")
```

Cada etapa é independente, mas compartilham configurações e saídas em `configs/` e `hsi_modificado/`. Você decide se quer rodar o pipeline completo (`make run`) ou etapas isoladas (ex.: repetir seleção sem refazer o pré-processamento).

Principais scripts (todos em `scripts/`):

| Script | Objetivo |
| --- | --- |
| `run_preprocessamento.py` | Aplica filtros, gera pseudo-RGB e relatórios por dataset. |
| `run_selecao.py` | Executa o método único habilitado, grava rankings e PCA. |
| `run_yolo_treinamento.py` | Monta dataset YOLO, treina, avalia e exporta modelos. |
| `run_classificar.py` | Usa o modelo final para prever em `hsi_modificado/classificar`. |

Esses scripts chamam funções em `src/hsi_pipeline/interface/`, o que permite reutilizar a lógica em notebooks ou integrações sem depender dos executáveis.

---

## 2. Estrutura do repositório

```
├── configs/                # JSONs de cada etapa
├── hsi_original/           # Dados brutos (não editados)
├── hsi_modificado/         # Saídas intermediárias (geradas pelo pipeline)
├── modelos/                # Modelos exportados + relatórios copiados
├── train-yolo/             # Datasets YOLO e runs do Ultralytics
├── scripts/                # 4 CLIs
└── src/hsi_pipeline/
    ├── config/             # Dataclasses (PipelineConfig, BandSelectionConfig...)
    ├── data/               # Caminhos globais, leitura ENVI, utilidades de IO
    ├── preprocessing/
    │   ├── resources.py    # Descobre cubos e monta memmaps
    │   ├── bands.py        # Calcula bandas mantidas
    │   └── filters/        # Correção, SNV, MSC (um arquivo por filtro)
    ├── features/
    │   ├── selection_methods/       # dataset.py, summary.py, classification.py
    │   ├── selection_methods/models # ANOVA, RF, VIP-PLS-DA, PCA...
    │   └── selection_helpers/       # Execução dos métodos e pós-processamento
    ├── yolo/
    │   ├── dataset.py, artifacts.py, metrics.py, image_filters.py
    │   └── training_steps/          # dataset_stage, train_stage, eval_stage, report_stage
    ├── classification/    # Config, datasets, inferência, relatórios
    ├── pipeline/          # Orquestrações genéricas e barra de progresso
    └── interface/         # CLIs Python e helpers (parser/env/paths)
```

**Pontos-chave da arquitetura:**  
- Cada filtro do pré-processamento é autocontido (`filters/correcao.py`, `snv.py`, `msc.py`).  
- Métodos de seleção ficam em `selection_methods/models/`. `selection_helpers/` cuida do top‑K, PCA e execuções.  
- O treinamento YOLO foi quebrado em estágios (`training_steps/`), permitindo reaproveitar dataset_stage ou train_stage de forma independente.  
- Os CLIs Python (em `interface/`) separaram parser, paths e manipulações de ambiente, mantendo os scripts Shell enxutos.  
- O diretório `tools/` contém utilitários opcionais (por exemplo, para restaurar labels ou gerar augments adicionais).  

---

## 3. Etapas detalhadas

### 3.1 Pré-processamento
1. **Descoberta de cubos** – `preprocessing/resources.py` usa `data/envi_io.py` para localizar arquivos `.hdr/.raw` (main/dark/white) e monta `DatasetResources` (memmaps + metadados).  
2. **Recorte espectral** – `bands.py` aplica `trimming.left/right` e devolve a lista de índices preservados + comprimentos de onda.  
3. **Correção → SNV → MSC** – cada filtro percorre as bandas mantidas, gera composições RGB (bandas vizinhas) e acumula caches. Pode ativar/desativar via `configs/preprocessamento.json → toggles`.  
4. **Pseudo-RGB** – `features/pseudo_rgb_generation.py` oferece três métodos (manual, PCA, linear combos). Você habilita cada um em `pseudo_rgb.toggles`.  
5. **Relatórios** – `pipeline/pipeline.py` escreve `*_summary.txt`, `band_metadata.csv`, `image_report.csv`.  
6. **Band selection opcional** – se habilitada no JSON, a etapa de seleção embutida usa as bandas SNV+MSC como base. (A seleção standalone é mais completa e recomendada, rodando via `scripts/run_selecao.py`.)  

**Saída**: `hsi_modificado/<dataset>/...` com subpastas para cada variação (`correcao_snv_msc_rgb_bands`, `pseudo_rgb/`, etc.) e relatório detalhado.  

### 3.2 Seleção de bandas
1. **Amostragem** – `selection_methods/dataset.py` cria máscaras circulares (ROI/fundo) e monta `PixelSampleDataset`. Configurado por `BandSelectionConfig` (raio, número de pixels, seed).  
2. **Execução do método** – `selection_methods/models/*.py` implementam ANOVA, t-test, Kruskal, RandomForest, VIP-PLS-DA, PCA. Ative exatamente um método em `configs/selecao.json`.  
3. **Pós-processamento** – `selection_helpers/postprocessing.py` gera `selected_band_indices`, descreve as bandas selecionadas (com índice original e comprimento de onda) e roda PCA apenas nas bandas escolhidas, salvando CSVs dedicados.  
4. **Classificação auxiliar** – `selection_methods/classification.py` oferece SVM linear, RandomForest ou PLS-DA como avaliação adicional, gerando `classification_metrics.csv`.  
5. **Relatórios** – `selection_methods/summary.py` cria `selection_summary.{json,txt}` com motivo do método, métricas e listas de bandas.  

**Saída**: `hsi_modificado/<dataset>/selecao/` contendo `ranking_*.csv`, `pca_components.csv`, `pca_variance.csv`, `ranking_pca_selected.csv`, `selection_summary.*` e, quando habilitado, `classification_metrics.csv`.  

### 3.3 Treinamento YOLO
1. **Dataset YOLO** – `training_steps/dataset_stage.py` percorre `hsi_modificado/`, coleta imagens e labels (respeitando `labels_follow_images` e `annotations_root`), realiza split train/val/test e grava `images/`, `labels/`, `data.yaml`, `dataset_index.csv`, `dataset_split_counts.*`.  
2. **Treino** – `train_stage.py` chama `YOLO.train` com parâmetros do JSON (`augmentations`, `amp`, `epochs`, `batch`, `device`, etc.). Logs padrão do Ultralytics são salvos em `train-yolo/yolo12_runs_*`.  
3. **Avaliação** – `eval_stage.py` (opcional) roda `YOLO.val` com `best.pt` e documenta métricas de validação.  
4. **Exportações e relatórios** – `report_stage.py` escreve `training_metrics.csv`, `training_summary.txt`, copia plots, exporta modelos (`best.pt`, `best.pkl`, ONNX, TorchScript, OpenVINO, CoreML...) e espelha tudo em `modelos/<run>/reports`.  

**Saída**:  
- `train-yolo/yolo12_dataset_*` (dataset final),  
- `train-yolo/yolo12_runs_*` (runs originais),  
- `modelos/<nome_unico>/` com exportações e relatórios completos.  

### 3.4 Classificação
1. **Config** – `configs/classificar.json` define o caminho do modelo `best.pt`, `source_root` (geralmente `hsi_modificado/classificar`) e a pasta de saída (`resultados/`).  
2. **Inferência** – `classification/inference.py` usa `YOLO.predict` para cada amostra (pastas contendo `pseudo_rgb/pca/*.png`).  
3. **Relatórios** – `classification/reporting.py` gera `summary.csv`, `detections.csv` e `summary.txt`, facilitando inspeções manual ou agregação estatística.  

**Saída**: `resultados/<amostra>/summary.csv`, `detections.csv`, `summary.txt`.  

---

## 4. Configurações (JSONs)

### 4.1 `configs/global_paths.json`
- `input_root`: raiz dos cubos brutos (`hsi_original`).  
- `output_root`: onde salvar `hsi_modificado`.  
- `models_root`: destino final dos modelos exportados (ex.: `modelos/`).  
- `dataset_pairs`: lista de pares (nome, input_root, output_root) para processar múltiplos datasets numa única execução.  

### 4.2 `configs/preprocessamento.json`
- `trimming.left/right`: quantas bandas remover no início/final.  
- `delta_bands`: distância para composições tri-banda (left/middle/right).  
- `toggles.correcao/snv/msc`: habilita/disable os filtros e se devem salvar PNGs.  
- `pseudo_rgb`: configura manual/PCA/linear combos (pastas específicas, peso por banda, etc.).  
- `band_selection`: se quiser rodar uma seleção básica no meio do pré-processamento (para a pipeline standalone use `run_selecao.py`).  

### 4.3 `configs/selecao.json`
- `methods.<nome>`: habilite **apenas um** método (ex.: `methods.random_forest.enabled=true`).  
- Parâmetros específicos (alpha, n_estimators, n_jobs, etc.).  
- `classification`: toggle e parâmetros dos classificadores auxiliares (SVM linear, RandomForest, PLS-DA).  
- `top_k_bands`: quantas bandas manter para o PCA final.  

### 4.4 `configs/yolo_treinamento.json`
- `out_root`, `training_root`, `models_root`: caminhos principais.  
- `annotations_root`, `pseudo_root`, `pseudo_method`: localização das labels e pseudo-RGB.  
- `split.train/val/test`: frações do dataset.  
- `patience`: número de épocas sem melhora até o early stopping interromper o treino.  
- `augmentations`: `fliplr`, `flipud`, `degrees`, etc.  
- `export_formats`: onnx, torchscript, openvino, coreml, engine (TensorRT).  
- `missing_label_policy`: `skip` ou `raise`.  
- `run_validation`: controla se `YOLO.val` será executado após o treino.  

### 4.5 `configs/classificar.json`
- `model`: caminho do `.pt` treinado (ex.: `modelos/yolo12x_2024-08-30/best.pt`).  
- `source_root`: `hsi_modificado/classificar` ou pasta equivalente.  
- `output_root`: onde salvar `resultados/`.  
- `imgsz`, `device`: resolução de inferência e dispositivo (GPU/CPU).  

---

## 5. Ambiente e execução

```bash
# Preparar ambiente virtual e instalar dependências
make venv
source .venv/bin/activate  # opcional

# Rodar etapas
make preprocess
make selecao
make yolo OUT_ROOT=/dados/pseudo MODELS_ROOT=/dados/modelos
make classificar

# Pipeline completo (pré-processamento + seleção + treinamento)
make run

# Limpar artefatos (cuidado!)
make clean
```

- `scripts/run_*.py --help` mostra todas as opções (ex.: overrides de `--config`, `--paths`, `--device`).  
- Após `make venv`, use `pip install -e .` para importar módulos diretamente (`python -m hsi_pipeline.interface.preprocess ...`).  
- Variáveis extras (`OUT_ROOT`, `MODELS_ROOT`) podem ser fornecidas ao `make yolo` para redirecionar os paths sem alterar JSONs.  

---

## 6. Saídas e relatórios

| Etapa | Local | Conteúdo |
| --- | --- | --- |
| Pré-processamento | `hsi_modificado/<dataset>/` | PNGs de cada variação (correção/SNV/MSC), pseudo-RGB, caches, `band_metadata.csv`, `image_report.csv`, `*_summary.txt`. |
| Seleção | `hsi_modificado/<dataset>/selecao/` | `ranking_*.csv`, `pca_components.csv`, `pca_variance.csv`, `ranking_pca_selected.csv`, `selection_summary.{json,txt}`, `classification_metrics.csv` (opcional). |
| Treinamento YOLO | `train-yolo/` e `modelos/` | `yolo12_dataset_*`, `yolo12_runs_*`, `training_metrics.csv`, `training_summary.txt`, exportações (ONNX/TorchScript/OpenVINO/CoreML/TensorRT), `model_summary.txt`, `model_artifacts.csv`, plots. |
| Classificação | `resultados/<amostra>/` | `summary.csv`, `detections.csv`, `summary.txt`. |

Todos os relatórios de texto mencionam parâmetros, tempos e métricas para facilitar auditoria.  

---

## 7. Boas práticas e troubleshooting

1. **Preserve `hsi_original/`** – nunca modifique os cubos brutos; todo output deve ir para `hsi_modificado/`.  
2. **Versione configs** – inclua `configs/*.json` e os principais relatórios no controle de versão para facilitar comparações.  
3. **Valide labels antes de treinar** – abra `pseudo_rgb/pca/pca_rgb.png` e o `.txt` correspondente para garantir que a máscara está coerente.  
4. **Faltou label?** – ajuste `missing_label_policy` ou use `src/hsi_pipeline/tools/restore_doentes_labels.py` como exemplo para criar labels padrão.  
5. **Ultralytics Home** – `training_env.py` garante que `ULTRALYTICS_HOME` seja `train-yolo/`, evitando arquivos soltos na raiz.  
6. **Logs** – Leia `*_summary.txt`, `selection_summary.txt`, `training_summary.txt` e os logs do Ultralytics (`train-yolo/yolo12_runs_*/train*.txt`) para entender divergências.  
7. **Limpeza** – `make clean` remove `.venv`, `hsi_modificado/`, `train-yolo/`, `modelos/`, `resultados/`. Execute apenas se tiver backups dos artefatos necessários.  
8. **Integração com notebooks** – importe as interfaces (ex.: `from hsi_pipeline.interface.selection import main`) para rodar etapas dentro de pipelines Jupyter/Luigi/Airflow.  
9. **Monitore espaço em disco** – as pastas `train-yolo/` e `modelos/` crescem rapidamente. Periodicamente aplique rotinas de limpeza ou arquivamento.  
10. **Erros comuns** – `FileNotFoundError` em labels durante `make yolo` indica `annotations_root` incorreto ou `missing_label_policy="stop"`. Ajuste o JSON ou corrija as labels.  

---

## 8. Extensões sugeridas

- **Filtros adicionais**: adicione arquivos em `preprocessing/filters/` para suportar detrending, Savitzky–Golay, etc., e exponha toggles no JSON.  
- **Novos métodos de seleção**: crie módulos em `selection_methods/models/` (por exemplo, ReliefF) e ajuste `configs/selecao.json`.  
- **Variações de YOLO**: Substitua o modelo base no JSON (`yolo12x.pt`, `yolov8l.pt`, `yolonas.pt`, etc.) e ajuste `train_stage.py` se houver parâmetros específicos.  
- **Dashboards**: integre os CSV/TXT gerados em notebooks ou dashboards (Superset, PowerBI) para acompanhar evolução de métricas.  
- **Automação CI/CD**: configure jobs para rodar as etapas em servidores GPU, salvando saídas em artefatos versionados.  
- **Serviço de inferência**: empacote o modelo exportado (`best.pt` ou ONNX) em uma API (FastAPI, Flask) usando os pós-processamentos existentes.  

---

## 9. FAQ e checklist

**Q: Posso rodar apenas seleção sem refazer pré-processamento?**  
A: Sim. Basta garantir que `hsi_modificado/<dataset>/` já tenha as saídas dos filtros e rodar `make selecao`.  

**Q: Preciso de internet para treinar?**  
A: Apenas na primeira execução para baixar pesos base (yolo12x.pt). Depois, `training_env.py` mantém os arquivos em `train-yolo/`.  

**Q: Consigo usar outro classificador além do YOLO?**  
A: Sim, o pipeline gera as bandas/top‑K; você pode criar scripts extras em `src/hsi_pipeline/tools/` para treinar modelos clássicos (SVM, RandomForest) se desejar.  

**Checklist antes de executar:**  
1. JSONs revisados (`global_paths`, `preprocessamento`, `selecao`, `yolo_treinamento`, `classificar`).  
2. `hsi_original/` com todos os cubos necessários.  
3. `python3 -m venv .venv` e `make venv` executados.  
4. Espaço em disco suficiente (pré-processamento e YOLO podem ocupar dezenas de GB).  
5. GPU disponível para o treino (ou ajuste `device: cpu`).  
6. `ULTRALYTICS_HOME` zerado ou apontando para `train-yolo/` (o script cuida disso, mas vale conferir).  
7. Labels conferidos antes do treino/classificação.  

---

> **Resumo final**: com esta base você cobre todo o ciclo de tratamento HSI: da leitura dos cubos até o classificador final, com relatórios completos, exportações e scripts simples de rodar. Ajuste os JSONs, execute os 4 comandos e aproveite um pipeline modular, reprodutível e pronto para evoluir.
