import os
import json
import argparse
import subprocess
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, CLIPModel, CLIPProcessor

warnings.filterwarnings("ignore", category=UserWarning)

try:
    import xgboost as xgb
except Exception:
    xgb = None

try:
    import shap
except Exception:
    shap = None

try:
    from lime.lime_text import LimeTextExplainer
except Exception:
    LimeTextExplainer = None

try:
    import kagglehub
except Exception:
    kagglehub = None


@dataclass
class Config:
    bert_name: str = "bert-base-uncased"
    clip_name: str = "openai/clip-vit-base-patch32"
    max_text_len: int = 512
    metadata_hidden_dim: int = 64
    metadata_out_dim: int = 64
    batch_size: int = 8
    num_workers: int = 0
    random_state: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    classifier_type: str = "xgboost"  # "xgboost" or "logreg"
    test_size: float = 0.2
    val_size: float = 0.1


class MetadataMLP(nn.Module):
    """Shallow MLP for metadata embedding e_m."""

    def __init__(self, input_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CrossModalAttentionFusion(nn.Module):
    """
    Cross-modal attention:
    - Query: textual embedding e_t
    - Key/Value: visual embedding e_v
    """

    def __init__(self, text_dim: int = 768, visual_dim: int = 512):
        super().__init__()
        self.query_proj = nn.Linear(text_dim, visual_dim)
        self.key_proj = nn.Linear(visual_dim, visual_dim)
        self.value_proj = nn.Linear(visual_dim, visual_dim)
        self.scale = visual_dim ** 0.5

    def forward(self, e_t: torch.Tensor, e_v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        q = self.query_proj(e_t)  # [B, 512]
        k = self.key_proj(e_v)  # [B, 512]
        v = self.value_proj(e_v)  # [B, 512]

        # Single-token cross-modal attention per sample.
        attn_logits = (q * k).sum(dim=1, keepdim=True) / self.scale  # [B, 1]
        alpha = torch.sigmoid(attn_logits)  # [B, 1], attention weight
        attended_v = alpha * v  # [B, 512]
        return attended_v, alpha


class HybridFeatureExtractor(nn.Module):
    """
    End-to-end feature extractor for:
    f = [e_t || e_v_tilde || e_m || s_sim]
    """

    def __init__(self, metadata_input_dim: int, cfg: Config):
        super().__init__()
        self.cfg = cfg

        # Textual stream (BERT)
        self.bert_tokenizer = AutoTokenizer.from_pretrained(cfg.bert_name)
        self.bert = AutoModel.from_pretrained(cfg.bert_name, attn_implementation="eager")

        # Visual stream (CLIP)
        self.clip_processor = CLIPProcessor.from_pretrained(cfg.clip_name)
        self.clip_model = CLIPModel.from_pretrained(cfg.clip_name)

        # Metadata stream
        self.metadata_mlp = MetadataMLP(
            input_dim=metadata_input_dim,
            hidden_dim=cfg.metadata_hidden_dim,
            out_dim=cfg.metadata_out_dim,
        )

        # Fusion
        self.cross_modal_attention = CrossModalAttentionFusion(text_dim=768, visual_dim=512)
        self.fused_dim = 768 + 512 + cfg.metadata_out_dim + 1

    def encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
            return_dict=True,
        )
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # [B, 768]
        # Stack attentions: [layers, B, heads, T, T]
        attn_stack = torch.stack(outputs.attentions, dim=0)
        return cls_embedding, attn_stack

    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        image_features = self.clip_model.get_image_features(pixel_values=pixel_values)
        if not isinstance(image_features, torch.Tensor):
            image_features = image_features.pooler_output
        return image_features

    def encode_text_for_clip(self, clip_input_ids: torch.Tensor, clip_attention_mask: torch.Tensor) -> torch.Tensor:
        text_features = self.clip_model.get_text_features(
            input_ids=clip_input_ids,
        attention_mask=clip_attention_mask,
    )
        if not isinstance(text_features, torch.Tensor):
           text_features = text_features.pooler_output
        return text_features

    def forward(
        self,
        bert_input_ids: torch.Tensor,
        bert_attention_mask: torch.Tensor,
        clip_input_ids: torch.Tensor,
        clip_attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
        metadata: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        e_t, bert_attentions = self.encode_text(bert_input_ids, bert_attention_mask)
        e_v = self.encode_image(pixel_values)
        e_clip_t = self.encode_text_for_clip(clip_input_ids, clip_attention_mask)
        e_m = self.metadata_mlp(metadata)

        e_v_tilde, alpha = self.cross_modal_attention(e_t, e_v)
        s_sim = F.cosine_similarity(e_clip_t, e_v, dim=1, eps=1e-8).unsqueeze(1)  # [B,1]

        fused = torch.cat([e_t, e_v_tilde, e_m, s_sim], dim=1)
        return {
            "fused": fused,
            "e_t": e_t,
            "e_v": e_v,
            "e_v_tilde": e_v_tilde,
            "e_m": e_m,
            "s_sim": s_sim,
            "alpha": alpha,
            "bert_attentions": bert_attentions,
        }


class FakeNewsMultimodalDataset(Dataset):
    """
    Expected dataframe columns:
    - text: str
    - image_path: str or None
    - label: int (0/1)
    - metadata columns: user-defined numeric features
    """

    def __init__(
        self,
        df: pd.DataFrame,
        metadata_cols: List[str],
        bert_tokenizer,
        clip_processor,
        max_text_len: int = 512,
    ):
        self.df = df.reset_index(drop=True)
        self.metadata_cols = metadata_cols
        self.bert_tokenizer = bert_tokenizer
        self.clip_processor = clip_processor
        self.max_text_len = max_text_len

    def __len__(self) -> int:
        return len(self.df)

    @staticmethod
    def _load_image_or_blank(image_path: Optional[str]) -> Image.Image:
        if image_path is None or not isinstance(image_path, str) or not os.path.exists(image_path):
            return Image.new("RGB", (224, 224), color=(0, 0, 0))
        try:
            return Image.open(image_path).convert("RGB")
        except Exception:
            return Image.new("RGB", (224, 224), color=(0, 0, 0))

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        text = str(row["text"]) if pd.notna(row["text"]) else ""
        image_path = row.get("image_path", None)
        label = int(row["label"])

        metadata = row[self.metadata_cols].astype(float).values if self.metadata_cols else np.array([], dtype=float)
        metadata = np.nan_to_num(metadata, nan=0.0, posinf=0.0, neginf=0.0)

        # BERT tokenization
        bert_enc = self.bert_tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_text_len,
            return_tensors="pt",
        )

        # CLIP processing for both text and image
        image = self._load_image_or_blank(image_path)
        clip_inputs = self.clip_processor(
            text=[text],
            images=[image],
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        return {
            "bert_input_ids": bert_enc["input_ids"].squeeze(0),
            "bert_attention_mask": bert_enc["attention_mask"].squeeze(0),
            "clip_input_ids": clip_inputs["input_ids"].squeeze(0),
            "clip_attention_mask": clip_inputs["attention_mask"].squeeze(0),
            "pixel_values": clip_inputs["pixel_values"].squeeze(0),
            "metadata": torch.tensor(metadata, dtype=torch.float),
            "label": torch.tensor(label, dtype=torch.long),
            "raw_text": text,
        }


def build_dataloaders(
    df: pd.DataFrame,
    metadata_cols: List[str],
    cfg: Config,
    bert_tokenizer,
    clip_processor,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_df, test_df = train_test_split(
        df,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        stratify=df["label"],
    )
    train_df, val_df = train_test_split(
        train_df,
        test_size=cfg.val_size,
        random_state=cfg.random_state,
        stratify=train_df["label"],
    )

    # Normalize metadata using train stats only.
    if metadata_cols:
        scaler = StandardScaler()
        train_df.loc[:, metadata_cols] = scaler.fit_transform(train_df[metadata_cols].fillna(0.0))
        val_df.loc[:, metadata_cols] = scaler.transform(val_df[metadata_cols].fillna(0.0))
        test_df.loc[:, metadata_cols] = scaler.transform(test_df[metadata_cols].fillna(0.0))

    train_ds = FakeNewsMultimodalDataset(train_df, metadata_cols, bert_tokenizer, clip_processor, cfg.max_text_len)
    val_ds = FakeNewsMultimodalDataset(val_df, metadata_cols, bert_tokenizer, clip_processor, cfg.max_text_len)
    test_ds = FakeNewsMultimodalDataset(test_df, metadata_cols, bert_tokenizer, clip_processor, cfg.max_text_len)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
    return train_loader, val_loader, test_loader


@torch.no_grad()
def extract_fused_features(
    model: HybridFeatureExtractor,
    loader: DataLoader,
    device: str,
) -> Tuple[np.ndarray, np.ndarray, List[str], torch.Tensor]:
    model.eval()
    all_features, all_labels, all_texts = [], [], []
    all_bert_attn = []

    for batch in tqdm(loader, desc="Extracting features"):
        bert_input_ids = batch["bert_input_ids"].to(device)
        bert_attention_mask = batch["bert_attention_mask"].to(device)
        clip_input_ids = batch["clip_input_ids"].to(device)
        clip_attention_mask = batch["clip_attention_mask"].to(device)
        pixel_values = batch["pixel_values"].to(device)
        metadata = batch["metadata"].to(device)
        labels = batch["label"].cpu().numpy()

        out = model(
            bert_input_ids=bert_input_ids,
            bert_attention_mask=bert_attention_mask,
            clip_input_ids=clip_input_ids,
            clip_attention_mask=clip_attention_mask,
            pixel_values=pixel_values,
            metadata=metadata,
        )
        all_features.append(out["fused"].cpu().numpy())
        all_labels.append(labels)
        all_texts.extend(batch["raw_text"])
        all_bert_attn.append(out["bert_attentions"].cpu())

    X = np.concatenate(all_features, axis=0)
    y = np.concatenate(all_labels, axis=0)
    # [N_batches, layers, B, heads, T, T] -> concatenated on B to [layers, N, heads, T, T]
    attn = torch.cat(all_bert_attn, dim=2)
    return X, y, all_texts, attn


def train_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    classifier_type: str = "xgboost",
):
    if classifier_type == "xgboost":
        if xgb is None:
            raise ImportError("xgboost is not installed. Install with: pip install xgboost")
        clf = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
        )
    elif classifier_type == "logreg":
        clf = LogisticRegression(
            penalty="l2",
            C=1.0,
            max_iter=2000,
            solver="lbfgs",
            random_state=42,
        )
    else:
        raise ValueError("classifier_type must be either 'xgboost' or 'logreg'")

    clf.fit(X_train, y_train)
    val_pred = clf.predict(X_val)
    val_prob = clf.predict_proba(X_val)[:, 1] if hasattr(clf, "predict_proba") else val_pred

    metrics = {
        "val_accuracy": accuracy_score(y_val, val_pred),
        "val_f1": f1_score(y_val, val_pred),
        "val_roc_auc": roc_auc_score(y_val, val_prob) if len(np.unique(y_val)) > 1 else np.nan,
    }
    return clf, metrics


def evaluate_classifier(clf, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else y_pred
    result = {
        "test_accuracy": accuracy_score(y_test, y_pred),
        "test_f1": f1_score(y_test, y_pred),
        "test_roc_auc": roc_auc_score(y_test, y_prob) if len(np.unique(y_test)) > 1 else np.nan,
    }
    print("\nClassification Report:\n", classification_report(y_test, y_pred, digits=4))
    return result


class SHAPExplainerWrapper:
    """SHAP wrapper for feature importance on fused vector f."""

    def __init__(self, clf, feature_names: Optional[List[str]] = None):
        if shap is None:
            raise ImportError("shap is not installed. Install with: pip install shap")
        self.clf = clf
        self.feature_names = feature_names
        self.explainer = None

    def fit(self, background_data: np.ndarray):
        if hasattr(self.clf, "predict_proba"):
            predict_fn = lambda x: self.clf.predict_proba(x)[:, 1]
        else:
            predict_fn = self.clf.predict
        self.explainer = shap.Explainer(predict_fn, background_data)

    def explain(self, X: np.ndarray):
        if self.explainer is None:
            raise RuntimeError("Call fit() first.")
        shap_values = self.explainer(X)
        return shap_values

    def plot_summary(self, shap_values, X: np.ndarray):
        shap.summary_plot(shap_values, X, feature_names=self.feature_names)


class LIMETextWrapper:
    """
    LIME wrapper for local text perturbation.
    Uses a callable that maps text -> fused feature -> classifier probability.
    """

    def __init__(self, predict_proba_fn, class_names: Optional[List[str]] = None):
        if LimeTextExplainer is None:
            raise ImportError("lime is not installed. Install with: pip install lime")
        self.predict_proba_fn = predict_proba_fn
        self.class_names = class_names or ["real", "fake"]
        self.explainer = LimeTextExplainer(class_names=self.class_names)

    def explain_text(self, text: str, num_features: int = 15, num_samples: int = 500):
        explanation = self.explainer.explain_instance(
            text_instance=text,
            classifier_fn=self.predict_proba_fn,
            num_features=num_features,
            num_samples=num_samples,
        )
        return explanation


def visualize_bert_attention(
    tokenizer,
    input_ids: torch.Tensor,
    attn_stack: torch.Tensor,
    layer: int = -1,
    head: int = 0,
) -> pd.DataFrame:
    """
    Returns a token-to-token attention DataFrame for one sample.
    attn_stack shape should be [layers, B, heads, T, T]
    """
    layer_idx = layer if layer >= 0 else (attn_stack.shape[0] + layer)
    attn = attn_stack[layer_idx, 0, head].detach().cpu().numpy()  # [T, T]
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].detach().cpu().numpy().tolist())
    return pd.DataFrame(attn, index=tokens, columns=tokens)


def make_feature_names(metadata_out_dim: int, metadata_input_cols: List[str]) -> List[str]:
    names = []
    names.extend([f"text_cls_{i}" for i in range(768)])
    names.extend([f"attn_visual_{i}" for i in range(512)])
    names.extend([f"meta_emb_{i}" for i in range(metadata_out_dim)])
    names.append("img_text_cosine")
    # Optionally append interpretable aliases for raw metadata fields.
    names.extend([f"raw_meta_{c}" for c in metadata_input_cols[:0]])  # placeholder, kept for extension
    return names[: 768 + 512 + metadata_out_dim + 1]


def build_lime_predict_fn(
    model: HybridFeatureExtractor,
    clf,
    metadata_default: np.ndarray,
    image_path_default: Optional[str],
    device: str,
):
    """
    Returns callable for LIME:
    input: List[str]
    output: np.ndarray [n_samples, 2] class probabilities
    """

    model.eval()

    def _predict_proba(text_list: List[str]) -> np.ndarray:
        rows = []
        for text in text_list:
            image = FakeNewsMultimodalDataset._load_image_or_blank(image_path_default)

            bert_enc = model.bert_tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=model.cfg.max_text_len,
                return_tensors="pt",
            )
            clip_inputs = model.clip_processor(
                text=[text],
                images=[image],
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )

            with torch.no_grad():
                out = model(
                    bert_input_ids=bert_enc["input_ids"].to(device),
                    bert_attention_mask=bert_enc["attention_mask"].to(device),
                    clip_input_ids=clip_inputs["input_ids"].to(device),
                    clip_attention_mask=clip_inputs["attention_mask"].to(device),
                    pixel_values=clip_inputs["pixel_values"].to(device),
                    metadata=torch.tensor(metadata_default, dtype=torch.float).unsqueeze(0).to(device),
                )
            rows.append(out["fused"].cpu().numpy()[0])

        X = np.array(rows)
        if hasattr(clf, "predict_proba"):
            prob_fake = clf.predict_proba(X)[:, 1]
            prob_real = 1.0 - prob_fake
            return np.stack([prob_real, prob_fake], axis=1)
        pred = clf.predict(X)
        pred = np.asarray(pred)
        return np.stack([1 - pred, pred], axis=1)

    return _predict_proba


def download_fakenewsnet_dataset(
    dataset_slug: str = "mdepak/fakenewsnet",
    output_dir: str = "data",
) -> str:
    """
    Download FakeNewsNet from Kaggle and return local dataset directory.
    Tries kagglehub first, then Kaggle CLI fallback.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Method 1: kagglehub Python package
    if kagglehub is not None:
        try:
            path = kagglehub.dataset_download(dataset_slug)
            if os.path.exists(path):
                print(f"Downloaded via kagglehub: {path}")
                return path
        except Exception as exc:
            print(f"kagglehub download failed, falling back to Kaggle CLI: {exc}")

    # Method 2: Kaggle CLI fallback
    zip_path = os.path.join(output_dir, "fakenewsnet.zip")
    cmd = f'kaggle datasets download -d "{dataset_slug}" -p "{output_dir}"'
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            "Failed to download dataset via Kaggle CLI.\n"
            "Make sure you have:\n"
            "1) pip install kaggle kagglehub\n"
            "2) ~/.kaggle/kaggle.json configured\n"
            f"CLI error:\n{result.stderr}"
        )

    if not os.path.exists(zip_path):
        # Kaggle may use a different output file name. Try best-effort lookup.
        candidate_zips = [p for p in os.listdir(output_dir) if p.lower().endswith(".zip")]
        if not candidate_zips:
            raise FileNotFoundError("Downloaded zip not found in output directory.")
        zip_path = os.path.join(output_dir, candidate_zips[0])

    # Unzip in-place using Python stdlib.
    import zipfile

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(output_dir)

    # Return best candidate root.
    return output_dir


def _safe_read_json(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _resolve_local_image_path(news_dir: str, content: Dict[str, Any]) -> Optional[str]:
    """
    Try to find a local image path near each news sample.
    FakeNewsNet often does not include downloaded article images, so None is valid.
    """
    candidates = []
    for key in ("top_img", "image", "image_path"):
        val = content.get(key)
        if isinstance(val, str) and val:
            candidates.append(val)

    image_dir_candidates = [
        os.path.join(news_dir, "images"),
        os.path.join(news_dir, "image"),
    ]

    # If key points to local file, use it directly.
    for c in candidates:
        local_path = c if os.path.isabs(c) else os.path.join(news_dir, c)
        if os.path.exists(local_path):
            return local_path

    # Otherwise look for first image file in known local image folders.
    exts = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
    for img_dir in image_dir_candidates:
        if not os.path.isdir(img_dir):
            continue
        for fn in os.listdir(img_dir):
            if fn.lower().endswith(exts):
                return os.path.join(img_dir, fn)
    return None


def _to_timestamp_features(publish_date: Any) -> Tuple[float, float]:
    """
    Convert publish_date to simple numeric metadata:
    - publish_year_norm
    - publish_month_norm
    """
    if publish_date is None:
        return 0.0, 0.0
    dt = pd.to_datetime(publish_date, errors="coerce", utc=True)
    if pd.isna(dt):
        return 0.0, 0.0
    year_norm = (float(dt.year) - 2000.0) / 50.0
    month_norm = float(dt.month) / 12.0
    return year_norm, month_norm


def load_fakenewsnet_from_directory(dataset_root: str) -> pd.DataFrame:
    """
    Parse FakeNewsNet folder tree into dataframe required by the pipeline.
    Expected structure:
      <root>/{gossipcop,politifact}/{fake,real}/<news_id>/news content.json
    """
    records: List[Dict[str, Any]] = []
    platforms = ["gossipcop", "politifact"]
    labels = [("fake", 1), ("real", 0)]

    for platform in platforms:
        platform_dir = os.path.join(dataset_root, platform)
        if not os.path.isdir(platform_dir):
            continue
        for label_name, label_value in labels:
            label_dir = os.path.join(platform_dir, label_name)
            if not os.path.isdir(label_dir):
                continue

            for news_id in os.listdir(label_dir):
                news_dir = os.path.join(label_dir, news_id)
                if not os.path.isdir(news_dir):
                    continue

                content_path = os.path.join(news_dir, "news content.json")
                if not os.path.exists(content_path):
                    continue

                content = _safe_read_json(content_path)
                text = content.get("text") or content.get("title") or ""
                title = content.get("title") or ""

                # Basic metadata derived from available fields.
                share_count = content.get("shares", 0)
                if isinstance(share_count, dict):
                    share_count = share_count.get("facebook", 0)
                try:
                    share_count = float(share_count)
                except Exception:
                    share_count = 0.0

                publish_year_norm, publish_month_norm = _to_timestamp_features(content.get("publish_date"))
                image_path = _resolve_local_image_path(news_dir, content)

                records.append(
                    {
                        "text": str(text),
                        "image_path": image_path,
                        "label": int(label_value),
                        "source_is_gossipcop": 1.0 if platform == "gossipcop" else 0.0,
                        "source_is_politifact": 1.0 if platform == "politifact" else 0.0,
                        "title_len": float(len(str(title))),
                        "text_len": float(len(str(text))),
                        "share_count": share_count,
                        "publish_year_norm": publish_year_norm,
                        "publish_month_norm": publish_month_norm,
                    }
                )

    if not records:
        raise ValueError(
            f"No usable news samples found in {dataset_root}. "
            "Check the extracted folder contains gossipcop/politifact with news content.json files."
        )

    df = pd.DataFrame(records)
    df["text"] = df["text"].fillna("").astype(str)
    df["label"] = df["label"].astype(int)
    return df


def load_or_prepare_dataframe(
    source: str,
    csv_path: str,
    kaggle_output_dir: str,
    kaggle_dataset: str,
) -> pd.DataFrame:
    """
    source:
      - 'csv': read local CSV from csv_path
      - 'kaggle-fakenewsnet': download + parse FakeNewsNet
      - 'auto': use CSV if present else Kaggle path
    """
    if source not in {"csv", "kaggle-fakenewsnet", "auto"}:
        raise ValueError("source must be one of: csv, kaggle-fakenewsnet, auto")

    if source in {"csv", "auto"} and os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    elif source in {"kaggle-fakenewsnet", "auto"}:
        dataset_dir = download_fakenewsnet_dataset(dataset_slug=kaggle_dataset, output_dir=kaggle_output_dir)
        df = load_fakenewsnet_from_directory(dataset_dir)
        # Save a normalized CSV snapshot for later fast reuse.
        os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
        df.to_csv(csv_path, index=False)
        print(f"Saved normalized CSV to {csv_path}")
    else:
        raise FileNotFoundError(
            f"CSV not found at {csv_path}. "
            "Either provide CSV or run with --source kaggle-fakenewsnet."
        )

    required_cols = {"text", "label"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Dataset must include columns: {required_cols}")
    if "image_path" not in df.columns:
        df["image_path"] = None
    return df


def run_pipeline(df: pd.DataFrame, metadata_cols: List[str], cfg: Config):
    """
    Full pipeline:
    1) Build loaders
    2) Extract fused features
    3) Train classical classifier
    4) Evaluate
    5) Return artifacts for XAI
    """
    metadata_input_dim = len(metadata_cols)
    model = HybridFeatureExtractor(metadata_input_dim=metadata_input_dim, cfg=cfg).to(cfg.device)

    train_loader, val_loader, test_loader = build_dataloaders(
        df=df,
        metadata_cols=metadata_cols,
        cfg=cfg,
        bert_tokenizer=model.bert_tokenizer,
        clip_processor=model.clip_processor,
    )

    X_train, y_train, train_texts, train_attn = extract_fused_features(model, train_loader, cfg.device)
    X_val, y_val, _, _ = extract_fused_features(model, val_loader, cfg.device)
    X_test, y_test, test_texts, test_attn = extract_fused_features(model, test_loader, cfg.device)

    clf, val_metrics = train_classifier(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        classifier_type=cfg.classifier_type,
    )
    test_metrics = evaluate_classifier(clf, X_test, y_test)

    print("\nValidation metrics:", val_metrics)
    print("Test metrics:", test_metrics)

    artifacts = {
        "model": model,
        "classifier": clf,
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "train_texts": train_texts,
        "test_texts": test_texts,
        "train_attn": train_attn,
        "test_attn": test_attn,
        "feature_names": make_feature_names(cfg.metadata_out_dim, metadata_cols),
    }
    return artifacts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Explainable Hybrid Fake News Detection Pipeline")
    parser.add_argument(
        "--source",
        type=str,
        default="auto",
        choices=["auto", "csv", "kaggle-fakenewsnet"],
        help="Dataset source: auto uses CSV if present, else Kaggle FakeNewsNet.",
    )
    parser.add_argument(
        "--csv-path",
        type=str,
        default="data/fake_news_multimodal.csv",
        help="Path to normalized CSV file.",
    )
    parser.add_argument(
        "--kaggle-output-dir",
        type=str,
        default="data/kaggle_fakenewsnet",
        help="Where Kaggle dataset files will be downloaded/extracted.",
    )
    parser.add_argument(
        "--kaggle-dataset",
        type=str,
        default="mdepak/fakenewsnet",
        help="Kaggle dataset slug.",
    )
    parser.add_argument(
        "--classifier",
        type=str,
        default="xgboost",
        choices=["xgboost", "logreg"],
        help="Classical classifier to train on fused features.",
    )
    args = parser.parse_args()

    df = load_or_prepare_dataframe(
        source=args.source,
        csv_path=args.csv_path,
        kaggle_output_dir=args.kaggle_output_dir,
        kaggle_dataset=args.kaggle_dataset,
    )

    metadata_cols = [c for c in df.columns if c not in {"text", "image_path", "label"}]
    cfg = Config(classifier_type=args.classifier)

    artifacts = run_pipeline(df, metadata_cols, cfg)

    # ---------------------------
    # SHAP example
    # ---------------------------
    if shap is not None:
        shap_wrapper = SHAPExplainerWrapper(
            clf=artifacts["classifier"],
            feature_names=artifacts["feature_names"],
        )
        background = artifacts["X_train"][: min(200, len(artifacts["X_train"]))]
        shap_wrapper.fit(background_data=background)
        shap_values = shap_wrapper.explain(artifacts["X_test"][:50])
        # shap_wrapper.plot_summary(shap_values, artifacts["X_test"][:50])
        print("SHAP explanations computed for first 50 test samples.")
    else:
        print("SHAP not installed; skipping SHAP example.")

    # ---------------------------
    # LIME example
    # ---------------------------
    if LimeTextExplainer is not None and len(artifacts["test_texts"]) > 0:
        sample_text = artifacts["test_texts"][0]
        metadata_default = np.zeros((len(metadata_cols),), dtype=np.float32)
        image_default = None
        lime_predict_fn = build_lime_predict_fn(
            model=artifacts["model"],
            clf=artifacts["classifier"],
            metadata_default=metadata_default,
            image_path_default=image_default,
            device=cfg.device,
        )
        lime_wrapper = LIMETextWrapper(
            predict_proba_fn=lime_predict_fn,
            class_names=["real", "fake"],
        )
        lime_exp = lime_wrapper.explain_text(sample_text)
        print("Top LIME features:", lime_exp.as_list()[:10])
    else:
        print("LIME not installed or no test text; skipping LIME example.")

    # ---------------------------
    # BERT attention visualization example
    # ---------------------------
    # Build one tokenized sample to visualize token-to-token attention.
    example_text = df["text"].iloc[0] if len(df) > 0 else "sample text"
    bert_tokens = artifacts["model"].bert_tokenizer(
        example_text,
        truncation=True,
        padding="max_length",
        max_length=cfg.max_text_len,
        return_tensors="pt",
    )
    # Re-run a forward pass for a single sample to get attentions.
    img = FakeNewsMultimodalDataset._load_image_or_blank(None)
    clip_inputs = artifacts["model"].clip_processor(
        text=[example_text],
        images=[img],
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    metadata_tensor = torch.zeros((1, len(metadata_cols)), dtype=torch.float)
    with torch.no_grad():
        out = artifacts["model"](
            bert_input_ids=bert_tokens["input_ids"].to(cfg.device),
            bert_attention_mask=bert_tokens["attention_mask"].to(cfg.device),
            clip_input_ids=clip_inputs["input_ids"].to(cfg.device),
            clip_attention_mask=clip_inputs["attention_mask"].to(cfg.device),
            pixel_values=clip_inputs["pixel_values"].to(cfg.device),
            metadata=metadata_tensor.to(cfg.device),
        )
    attn_df = visualize_bert_attention(
        tokenizer=artifacts["model"].bert_tokenizer,
        input_ids=bert_tokens["input_ids"],
        attn_stack=out["bert_attentions"].cpu(),
        layer=-1,
        head=0,
    )
    print("BERT attention matrix shape:", attn_df.shape)
