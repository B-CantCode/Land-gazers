# Land-Gazers — Delhi-NCR Land Cover Classification

Land-Gazers performs land cover classification for the **Delhi-NCR (National Capital Region)** using satellite RGB patch imagery and geospatial labeling from **ESA WorldCover 2021**. The workflow is implemented in a single notebook: `submission.ipynb`.

The pipeline covers:
- geospatial filtering (keep only patches inside Delhi-NCR),
- label assignment from a WorldCover raster,
- EfficientNet-B2 model training with class balancing + focal loss,
- evaluation with test-time augmentation (TTA).

> Note: While labels are grouped into 5 categories (Vegetation, Cropland, Built-up, Water, Others), the final training may drop rare classes (e.g., Water/Others) to ensure meaningful learning with sufficient samples.

---

## Repository Contents

- `submission.ipynb` — notebook containing **Task 1–Task 3**
- `requirements.txt` — Python dependencies (create this file)
- `outputs/` — recommended folder for saving plots/reports/model checkpoints (create this folder)

---

## Dataset

- **Source**: Kaggle — `rishabhsnip/earth-observation-delhi-airshed`
- **Key files used**
  - `delhi_ncr_region.geojson` — Delhi-NCR boundary
  - RGB satellite images (128×128 patches)
  - `worldcover_bbox_delhi_ncr_2021.tif` — ESA WorldCover label raster

The notebook downloads the dataset with:

```python
import kagglehub
path = kagglehub.dataset_download("rishabhsnip/earth-observation-delhi-airshed")
print(path)
```

---

## Task Breakdown

### Task 1 — Spatial Filtering & Grid Analysis
1. Load Delhi-NCR boundary from GeoJSON
2. Create a **60×60 km** uniform grid over the region (using a metric CRS)
3. Preprocess images:
   - filter non-PNG files / invalid filenames
4. Filter images that fall within the Delhi-NCR boundary
5. Report:
   - total grids and grids with ≥1 image
   - total images before/after preprocessing

### Task 2 — Label Assignment & Data Splitting
1. Assign each image a land cover label based on the dominant ESA WorldCover class
2. Class grouping:
   - Vegetation: 10, 20, 30
   - Cropland: 40
   - Built-up: 50
   - Water: 80
   - Others: 60, 70, 90, 95, 100
3. Split into **60% train / 40% test** with stratification
4. Visualize class distribution

### Task 3 — Model Training & Evaluation
- **Data prep**
  - optionally drop rare classes (e.g., Water/Others)
  - re-split train into **80/20 train/val**
  - custom PyTorch `Dataset`
- **Augmentations**
  - train: heavy augmentation (crops, flips, rotations, color jitter, etc.)
  - val/test: deterministic transforms (resize + normalize)
- **Model**
  - EfficientNet-B2 pretrained on ImageNet
  - modified classification head
- **Training**
  - Effective Number class weights (Cui et al., 2019), β=0.9999
  - focal loss (γ=1.5) + label smoothing (0.1)
  - AdamW (lr=1e-4, weight_decay=0.01)
  - ReduceLROnPlateau (monitor Macro F1)
  - gradient clipping (max norm 1.0)
  - early stopping (patience=5)
- **Evaluation**
  - TTA: 5 variants per image
  - metrics: Accuracy + Macro F1
  - classification report + normalized confusion matrix

---

## Setup (Local)

### 1) Install dependencies
```bash
pip install -r requirements.txt
```

### 2) Run the notebook
```bash
jupyter notebook submission.ipynb
```

If you run outside Kaggle, you may need to adjust file paths to point to the downloaded dataset location.

---

## Outputs (Recommended)

Create an `outputs/` folder and save artifacts there, for example:
- `outputs/best_model.pth`
- `outputs/confusion_matrix.png`
- `outputs/class_distribution.png`
- `outputs/classification_report.txt`

(GitHub does not keep empty folders, so include a `.gitkeep` file or an `outputs/README.md`.)

---

## Citation
- Cui et al., 2019 — *Class-Balanced Loss Based on Effective Number of Samples*
- ESA WorldCover 2021
- EfficientNet (Tan & Le, 2019)

---

## License
This project uses publicly available datasets and open-source libraries. Refer to individual dataset/library licenses for usage restrictions.