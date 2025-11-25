# Pipeline HSI + YOLO12

Este repositório entrega um fluxo completo para pré-processar cubos hiperespectrais (HSI), gerar labels padrão, treinar/avaliar YOLO12 localmente e classificar novas amostras. Os pesos são resolvidos a partir de arquivos locais.

> Baseado em YOLO12 ([Tian et al., 2025](#referências))  

---

## Visão geral do fluxo

1. **Pré-processamento HSI**  
   - Recorte espectral (`trimming.left/right`).  
   - Correção + SNV + MSC.  
   - Geração de PNGs `correcao_snv_msc` e labels YOLO padrão (classe 0, caixa cheia) para cada PNG.
2. **Treinamento YOLO12**  
   - Usa somente pesos locais em `yolo/modelos/` ou `src/hsi_pipeline/yolo/modelos/` (ex.: `yolo12x.pt`).  
   - Monta dataset a partir de `correcao_snv_msc_rgb_bands` das amostras de doentes.  
   - Split fixo: 1ª amostra = train, 2ª = val, 3ª = test.  
   - Salva `best.pt` em `yolo/modelos/<run>/`.
3. **Avaliação**  
   - Mede mAP@0.5 e mAP@0.5:0.95 no split de teste, gera JSON/plots e estatísticas de canais RGB.  
   - Usa modelo/data locais.
4. **Classificação**  
   - Roda YOLO sobre `correcao_snv_msc_rgb_bands` em `hsi_modificado/classificar`.  
   - Gera `detections.csv`, `summary.csv`, `sample_summary.csv` e `summary.txt` por run, copiando imagens anotadas para `resultados/<run>/imagens/<amostra>/`.  
   - TXT inclui tempos, contagens, probabilidade média/máx por amostra e caminhos dos CSVs.

---

## Estrutura de pastas

```
configs/                # JSONs de configuração (preprocessamento, treino, avaliação, classificação, paths)
hsi_original/           # Dados de entrada (.hdr/.raw)
hsi_modificado/         # Saídas do pré-processamento (PNGs, labels geradas, summaries)
scripts/                # CLIs: run_preprocessamento.py, run_yolo_treinamento.py, run_yolo_avaliacao.py, run_classificar.py
src/hsi_pipeline/       # Código-fonte (config, data, preprocessing, pipeline, yolo, classification)
yolo/
  modelos/              # Pesos locais (.pt) — nunca baixados em tempo de execução
  datasets/             # Datasets YOLO gerados (train/val/test)
  runs/                 # Logs e outputs de treino
  avaliacoes/           # Runs de avaliação
resultados/             # Runs de classificação (CSV, TXT, imagens anotadas)
```

---

## Como rodar

```bash
make venv               # cria .venv e instala dependências locais
make preprocess         # pré-processa HSI e gera PNGs/labels/summaries (configs/preprocessamento.json)
make yolo               # treina YOLO12 com pesos locais (configs/yolo_treinamento.json)
make yolo_eval          # avalia o modelo (configs/yolo_avaliacao.json)
make classificar        # classifica amostras em hsi_modificado/classificar (configs/classificar.json)
# ou tudo de uma vez:
make run                # preprocess + yolo + classificar

# Limpeza (preserva yolo/modelos):
make clean

# Ajuda/resumo dos alvos:
make help
```

Modo de execução: os scripts priorizam pesos locais. Se um peso não existir localmente, o código falha com FileNotFoundError em vez de baixar.

---

## Configurações principais

- `configs/global_paths.json`  
  - `input_root`, `output_root` e `dataset_pairs` para pré-processamento.

- `configs/preprocessamento.json`  
  - `enabled`, `cpu_workers`, `folder`, `out_root`.  
  - `trimming.left/right`: remove bandas no início/fim do espectro.  
  - `toggles.correcao_snv_msc`: gera PNGs e labels YOLO padrão.  
  - `export_metadata`: mantém `band_metadata.csv`, `image_report.csv`, `<dataset>_summary.txt`.

- `configs/yolo_treinamento.json`  
  - `model`: nome do peso local (ex.: `yolo12x.pt`) resolvido em `yolo/modelos` ou `src/hsi_pipeline/yolo/modelos`.  
  - `input_root`: `hsi_modificado/doentes/`.  
  - `output_root`: `yolo` (datasets/, runs/, modelos/).  
  - `classes`, `imgsz`, `epochs`, `patience`, `batch`, `device`.  
  - `train_extra_args`: AdamW, `lr0`, `lrf`, `weight_decay`, `warmup_epochs`, `multi_scale`, `cos_lr`, `workers`, `verbose`, `amp=false` (AMP desativado para impedir downloads no check).

- `configs/yolo_avaliacao.json`  
  - `model`: nome ou `auto` (pega o `best.pt` mais recente).  
  - `data_yaml`: caminho do dataset ou `auto` (usa dataset do mesmo run do modelo).  
  - `imgsz`, `batch`, `device`, `split`, `output_root`.

- `configs/classificar.json`  
  - `model`: nome ou `auto` (modelo mais recente).  
  - `source_root`: `hsi_modificado/classificar`.  
  - `output_root`: `resultados`.  
  - `imgsz`, `device`, `conf` (threshold).

---

## Saídas por etapa

- **Pré-processamento** (`make preprocess`)  
  - `hsi_modificado/<dataset>/correcao_snv_msc_rgb_bands/*.png`  
  - Labels YOLO padrão (`.txt`, classe 0 cobrindo a imagem).  
  - `band_metadata.csv`, `image_report.csv`, `<dataset>_summary.txt`.

- **Treino** (`make yolo`)  
  - Dataset em `yolo/datasets/<run>/images|labels/{train,val,test}`.  
  - Run em `yolo/runs/<run>/`; `best.pt` copiado para `yolo/modelos/<run>/`.

- **Avaliação** (`make yolo_eval`)  
  - mAP@0.5, mAP@0.5:0.95 em JSON.  
  - Plots de canais (`channel_stats.png`) e `evaluation_summary.json` em `yolo/avaliacoes/runs/<eval_run>/`.

- **Classificação** (`make classificar`)  
  - Imagens anotadas em `resultados/<run>/imagens/<amostra>/`.  
  - `detections.csv` (todas detecções), `summary.csv` (prob por imagem), `sample_summary.csv` (prob média/máx, detecções, tempo por amostra), `summary.txt` (com caminhos e tempos).

---

## Notas e boas práticas

- Coloque todos os pesos desejados em `yolo/modelos/` ou `src/hsi_pipeline/yolo/modelos/` (ex.: `yolo12n.pt`, `yolo12s.pt`, `yolo12m.pt`, `yolo12l.pt`, `yolo12x.pt`).  
  - Se `model` não existir localmente, o código falha (offline).  
- Labels padrão são geradas no pré-processamento (classe 0, caixa cheia); substitua por labels reais se tiver.  
- Ajuste `trimming.left/right` para remover bandas iniciais/finais conforme o espectro.  
- `clean` preserva `yolo/modelos/` para não perder pesos baixados/copied manualmente.  
- Use `make help` para ver alvos e descrições.

---

## Referências

- @article{tian2025yolo12,  
  title={YOLO12: Attention-Centric Real-Time Object Detectors},  
  author={Tian, Yunjie and Ye, Qixiang and Doermann, David},  
  journal={arXiv preprint arXiv:2502.12524},  
  year={2025}  
}

- @software{yolo12,  
  author = {Tian, Yunjie and Ye, Qixiang and Doermann, David},  
  title = {YOLO12: Attention-Centric Real-Time Object Detectors},  
  year = {2025},  
  url = {https://github.com/sunsmarterjie/yolov12},  
  license = {AGPL-3.0}  
}
