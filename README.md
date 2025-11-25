# Pré-processamento e treino YOLO

O repositório aplica pré-processamento em cubos HSI e oferece um script simples para treinar YOLO12x a partir das imagens `correcao_snv_msc` das amostras de doentes.

## Estrutura

```
configs/                # JSONs de configuração
hsi_original/           # Dados de entrada (.hdr/.raw)
hsi_modificado/         # Saídas geradas pelo pré-processamento
scripts/                # CLIs: run_preprocessamento.py, run_yolo_treinamento.py
src/hsi_pipeline/       # Código (config, data, preprocessing, pipeline, yolo_train)
yolo/                   # Saídas do treino (datasets/, runs/, modelos/)
```

## Como rodar

```bash
make venv               # cria .venv e instala dependências
make preprocess         # roda o pré-processamento (usa configs/preprocessamento.json)
make yolo               # monta dataset e treina YOLO (usa configs/yolo_treinamento.json)
# ou
make run                # preprocess + yolo

# Limpeza dos artefatos gerados (use com cuidado)
make clean
```

## Configurações principais

- `configs/global_paths.json`: raízes padrão de entrada (`input_root`) e saída (`output_root`) do pré-processamento; `dataset_pairs` lista datasets.
- `configs/preprocessamento.json`: controla threads, entrada/saída, trimming e toggles. Atualmente só `correcao_snv_msc` está ativado.
- `configs/yolo_treinamento.json`:
  - `model`: peso base (ex.: `yolo12x.pt`).
  - `input_root`: onde buscar amostras (`hsi_modificado/doentes/`).
  - `output_root`: pasta raiz para o treino (`yolo` → datasets/, runs/, modelos/).
  - `classes`: lista de classes (ex.: `["staphylococcus"]`).
  - `imgsz`, `epochs`, `patience`, `batch`, `device`.
  - `train_extra_args`: otimização (AdamW, lr0, lrf, weight_decay, warmup_epochs, multi_scale, cos_lr, workers, verbose).

## Divisão e saídas do treino

- O script de treino divide automaticamente as três amostras ATCC em:
  - primeira: train
  - segunda: val
  - terceira: test
- Imagens e labels são copiadas de `correcao_snv_msc_rgb_bands` para `yolo/datasets/<run>/images|labels/{train,val,test}`.
- O YOLO salva o run em `yolo/runs/<run>/`; o `best.pt` é copiado para `yolo/modelos/<run>/`.

## Pré-processamento (saída)

- PNGs/plots de `correcao_snv_msc`.
- `band_metadata.csv`, `image_report.csv`, `<dataset>_summary.txt`.

## Notas rápidas

- O pré-processamento não altera os dados brutos; grava saídas em `hsi_modificado/`.
- O treino exige labels `.txt` para cada PNG; imagens sem label são ignoradas e o script falha se um split ficar vazio.
