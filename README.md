# Explainable Hybrid NLP Framework for Fake News Detection

A production-style PyTorch implementation of a multimodal fake news detection framework that combines:

- **Text features** from BERT (`bert-base-uncased`)
- **Visual features** from CLIP image encoder (`openai/clip-vit-base-patch32`)
- **Metadata embeddings** from a shallow MLP
- **Cross-modal attention fusion** (text query over visual key/value)
- **Classical downstream classifiers** (XGBoost or Logistic Regression)
- **Explainability hooks** (SHAP, LIME, and BERT attention visualization)

The full implementation is in `hybrid_fake_news_framework.py`.

---

## 1) Architecture Overview

For each sample:

1. **Textual stream**
   - Tokenize with BERT tokenizer (max length = 512)
   - Extract `[CLS]` embedding as `e_t` (768-dim)

2. **Visual stream**
   - Encode article image via CLIP visual encoder as `e_v` (512-dim)
   - Compute image-text consistency score `s_sim` via cosine similarity between CLIP text and image features

3. **Metadata stream**
   - Normalize metadata features
   - Encode via shallow MLP into metadata embedding `e_m`
   - Missing metadata values are zero-filled

4. **Cross-modal attention**
   - Query = `e_t`
   - Key/Value = `e_v`
   - Compute attention weight `alpha`
   - Attended visual feature: `e_v_tilde = alpha * W_v e_v`

5. **Fused vector**
   - `f = [e_t || e_v_tilde || e_m || s_sim]`
   - Sent to a classical ML classifier (`XGBoost` or `Logistic Regression`)

---

## 2) Project Structure

- `hybrid_fake_news_framework.py` - Complete end-to-end pipeline (preprocessing, feature extraction, training, evaluation, XAI wrappers)
- `README.md` - Usage and setup documentation

---

## 3) Environment Setup

Use Python **3.10+** (3.11 recommended).

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers pandas numpy scikit-learn pillow tqdm
pip install xgboost shap lime kaggle kagglehub
```

If you are on CPU-only, install CPU PyTorch wheels from the official PyTorch page.

---

## 4) Dataset Options

You can run the project in two ways:

1. **Bring your own CSV** (`data/fake_news_multimodal.csv`)
2. **Auto-download FakeNewsNet from Kaggle** (`mdepak/fakenewsnet`)

For Kaggle mode, configure your Kaggle API token (`kaggle.json`) first.

---

## 5) CSV Format Required

Create a CSV file named `fake_news_multimodal.csv` in the same folder with:

### Mandatory columns

- `text` (string): article/news text
- `label` (int): class label (`0` or `1`)

### Optional columns

- `image_path` (string): local path to image file
- any numeric metadata columns, for example:
  - `source_credibility`
  - `timestamp_score`
  - `share_count`
  - `comment_count`
  - `user_reputation`

All columns except `text`, `image_path`, and `label` are treated as metadata automatically.

---

## 6) Example CSV

```csv
text,image_path,label,source_credibility,timestamp_score,share_count
"Breaking: ...","data/images/news_001.jpg",1,0.82,0.43,1250
"Official update from ...","data/images/news_002.jpg",0,0.91,0.77,420
"Rumor says ...","",1,0.31,0.20,8900
```

Notes:

- Empty/invalid image paths are handled gracefully (black placeholder image is used)
- Missing metadata values are zero-imputed during loading

---

## 7) Running Training & Evaluation

```bash
python hybrid_fake_news_framework.py
```

By default:

- Classifier is `xgboost`
- Data split: train/val/test with stratification
- Metrics printed: Accuracy, F1, ROC-AUC, and classification report

Run with Kaggle FakeNewsNet auto-download + parsing:

```bash
python hybrid_fake_news_framework.py --source kaggle-fakenewsnet
```

Run with explicit options:

```bash
python hybrid_fake_news_framework.py --source kaggle-fakenewsnet --classifier logreg --csv-path data/fake_news_multimodal.csv
```

Use local CSV only:

```bash
python hybrid_fake_news_framework.py --source csv --csv-path data/fake_news_multimodal.csv
```

---

## 8) Explainability Components

The script includes ready wrappers for:

- **SHAP**: global/local feature importance on fused vector `f`
- **LIME**: local text perturbation explanations
- **BERT attention visualization**: token-to-token attention matrix for a selected layer/head

These are integrated as examples in the `__main__` section.

---

## 9) Practical Recommendations

- Start with a balanced dataset and verify label quality first
- Keep image paths local and valid to avoid silent fallback to placeholder images
- Use standardized metadata scales (already handled in code via `StandardScaler`)
- For faster iteration on limited GPU memory, reduce batch size and/or max length

---

## 10) Reproducibility

- Fixed random seed in `Config` (`random_state=42`)
- Stratified train/validation/test splits
- Deterministic data preprocessing path for metadata normalization

For strict reproducibility, you can additionally set:

```python
torch.manual_seed(42)
np.random.seed(42)
```

---

## 11) Citation / Paper Alignment

This implementation follows the requested design pattern of:

- multimodal feature extraction (text, image, metadata),
- cross-modal attention-based fusion, and
- classical ML classifier on fused embeddings,

plus explainability integrations (SHAP, LIME, BERT attention views).

