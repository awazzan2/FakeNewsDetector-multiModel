"""
predict.py — Predict + Explain BOTH test cases through the Hybrid Fake News pipeline.

Usage
-----
    python predict.py                          # runs both fake and real test cases
    python predict.py --clf-path clf.pkl       # use your trained classifier
    python predict.py --no-lime                # skip LIME (slow on CPU)
    python predict.py --no-shap                # skip SHAP

Explainability outputs (per article)
--------------------------------------
  1. BERT Attention  — top tokens the model focused on (always runs)
  2. LIME            — which words pushed toward FAKE vs REAL (needs: pip install lime)
  3. SHAP            — feature importance on the fused vector  (needs: pip install shap)
"""

import argparse
import os
import pickle
import warnings

import numpy as np
import torch

warnings.filterwarnings("ignore", category=UserWarning)

from hybrid_fake_news_framework import (
    Config,
    FakeNewsMultimodalDataset,
    HybridFeatureExtractor,
    build_lime_predict_fn,
    LIMETextWrapper,
    SHAPExplainerWrapper,
    make_feature_names,
)

try:
    import shap as _shap
except ImportError:
    _shap = None

try:
    from lime.lime_text import LimeTextExplainer as _LimeCheck
except ImportError:
    _LimeCheck = None

# ── Test cases ─────────────────────────────────────────────────────────────────

FAKE_NEWS_ARTICLE = {
    "text": (
        "Trump is getting support from every leader, and that's the support that will make him grow great and strong!! These elections will bring an immense change in our country."
    ),
    "image_path": "C:\\Users\\Ali Al Wazzan\\Desktop\\fake_photo.png",
    "metadata": {
        "source_is_gossipcop":  1.0,
        "source_is_politifact": 0.0,
        "title_len":            5.0,
        "text_len":             25.0,
        "share_count":          4800.0,
        "publish_year_norm":    0.46,
        "publish_month_norm":   0.83,
    },
}

REAL_NEWS_ARTICLE = {
    "text": (
        "Americans to get $10,000 in student debt cancelled if they earn $125,000 or less, US President Joe Biden announces."
    ),
    "image_path": "C:\\Users\\Ali Al Wazzan\\Desktop\\Real_photo.png",
    "metadata": {
        "source_is_gossipcop":  0.0,
        "source_is_politifact": 1.0,
        "title_len":            72.0,
        "text_len":             430.0,
        "share_count":          3200.0,
        "publish_year_norm":    0.46,
        "publish_month_norm":   0.75,
    },
}

# ── Helpers ────────────────────────────────────────────────────────────────────

def print_banner(title: str):
    print("\n" + "=" * 62)
    print(f"  {title}")
    print("=" * 62)


def encode_single_sample(model, text, image_path, metadata_values, cfg):
    model.eval()
    device = cfg.device

    bert_enc = model.bert_tokenizer(
        text, truncation=True, padding="max_length",
        max_length=cfg.max_text_len, return_tensors="pt",
    )
    image = FakeNewsMultimodalDataset._load_image_or_blank(image_path)
    clip_inputs = model.clip_processor(
        text=[text], images=[image], truncation=True,
        padding="max_length", return_tensors="pt",
    )
    metadata_tensor = torch.tensor(metadata_values, dtype=torch.float).unsqueeze(0)

    with torch.no_grad():
        out = model(
            bert_input_ids=bert_enc["input_ids"].to(device),
            bert_attention_mask=bert_enc["attention_mask"].to(device),
            clip_input_ids=clip_inputs["input_ids"].to(device),
            clip_attention_mask=clip_inputs["attention_mask"].to(device),
            pixel_values=clip_inputs["pixel_values"].to(device),
            metadata=metadata_tensor.to(device),
        )

    out["_bert_input_ids"] = bert_enc["input_ids"]
    return out


def build_demo_classifier(fused_dim: int):
    from sklearn.linear_model import LogisticRegression
    rng = np.random.default_rng(42)
    X = np.vstack([
        rng.normal(0.6,  0.3, (20, fused_dim)).astype(np.float32),
        rng.normal(-0.6, 0.3, (20, fused_dim)).astype(np.float32),
    ])
    y = np.array([1] * 20 + [0] * 20)
    clf = LogisticRegression(max_iter=500, random_state=42)
    clf.fit(X, y)
    return clf


# ── Explainability ─────────────────────────────────────────────────────────────

def explain_bert_attention(model, out, top_k: int = 15):
    """Top tokens the model attended to, averaged across all layers and heads."""
    print_banner("EXPLAINABILITY  1/3 — BERT Attention (top attended tokens)")

    input_ids  = out["_bert_input_ids"].cpu()
    attn_stack = out["bert_attentions"].cpu()

    tokens = model.bert_tokenizer.convert_ids_to_tokens(input_ids[0])

    # Mean over all layers and all heads -> [T, T], then take CLS row
    mean_attn = attn_stack[:, 0, :, :, :].mean(dim=(0, 1))
    cls_attn  = mean_attn[0]

    special = {"[PAD]", "[CLS]", "[SEP]"}
    scored = [
        (tokens[i], cls_attn[i].item())
        for i in range(len(tokens))
        if tokens[i] not in special and not tokens[i].startswith("##")
    ]
    scored.sort(key=lambda x: x[1], reverse=True)

    print(f"\n  Top {top_k} tokens the model attended to:\n")
    bar_max = scored[0][1] if scored else 1.0
    for rank, (tok, score) in enumerate(scored[:top_k], 1):
        bar = "#" * int((score / bar_max) * 30)
        print(f"  {rank:>2}. {tok:<22} {bar:<30}  {score:.4f}")
    print()


def explain_lime(model, clf, text, metadata_values, cfg):
    """Word-level local explanation using LIME."""
    print_banner("EXPLAINABILITY  2/3 — LIME (word-level influence)")

    if _LimeCheck is None:
        print("  LIME not installed.  Run:  pip install lime")
        return

    predict_fn = build_lime_predict_fn(
        model=model,
        clf=clf,
        metadata_default=metadata_values.copy(),
        image_path_default=None,
        device=cfg.device,
    )
    wrapper = LIMETextWrapper(predict_proba_fn=predict_fn, class_names=["real", "fake"])

    print("  Running LIME (may take ~30 s on CPU)...")
    exp      = wrapper.explain_text(text, num_features=15, num_samples=300)
    features = exp.as_list()

    print(f"\n  Word influence on FAKE label  (positive = pushes toward FAKE):\n")
    max_abs = max(abs(w) for _, w in features) or 1.0
    for word, weight in features:
        direction = "-> FAKE" if weight > 0 else "-> REAL"
        bar_ch    = "+" if weight > 0 else "-"
        bar       = bar_ch * int(abs(weight) / max_abs * 28)
        print(f"  {word:<24} {bar:<30}  {weight:+.4f}  {direction}")
    print()


def explain_shap(clf, fused_np, feature_names):
    """Feature importance on the fused vector using SHAP."""
    print_banner("EXPLAINABILITY  3/3 — SHAP (fused feature importance)")

    if _shap is None:
        print("  SHAP not installed.  Run:  pip install shap")
        return

    wrapper = SHAPExplainerWrapper(clf=clf, feature_names=feature_names)
    wrapper.fit(background_data=fused_np)
    shap_values = wrapper.explain(fused_np)

    vals  = shap_values.values[0]
    names = feature_names if feature_names else [f"f{i}" for i in range(len(vals))]
    order = np.argsort(np.abs(vals))[::-1]
    top_k = 15

    print(f"\n  Top {top_k} fused features by SHAP impact  (positive = toward FAKE):\n")
    max_abs = max(abs(vals[order[0]]), 1e-9)
    for rank, idx in enumerate(order[:top_k], 1):
        name   = names[idx] if idx < len(names) else f"f{idx}"
        v      = vals[idx]
        bar_ch = "+" if v > 0 else "-"
        bar    = bar_ch * int(abs(v) / max_abs * 28)
        print(f"  {rank:>2}. {name:<32} {bar:<30}  {v:+.4f}")
    print()


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clf-path",   default=None,        help="Path to trained classifier .pkl")
    parser.add_argument("--classifier", default="xgboost",   choices=["xgboost", "logreg"])
    parser.add_argument("--no-lime",    action="store_true", help="Skip LIME (slow on CPU)")
    parser.add_argument("--no-shap",    action="store_true", help="Skip SHAP")
    args = parser.parse_args()

    cfg = Config(classifier_type=args.classifier)
    print(f"\n[Config] device={cfg.device}  classifier={cfg.classifier_type}")

    test_cases = [
        (FAKE_NEWS_ARTICLE, "fake"),
        (REAL_NEWS_ARTICLE, "real"),
    ]

    # Load the model once and reuse it for both test cases
    print("\n[Setup] Loading HybridFeatureExtractor (shared across both test cases)...")
    # We need metadata_input_dim — both articles share the same metadata keys
    metadata_cols      = list(FAKE_NEWS_ARTICLE["metadata"].keys())
    metadata_input_dim = len(metadata_cols)

    model = HybridFeatureExtractor(metadata_input_dim=metadata_input_dim, cfg=cfg).to(cfg.device)
    from transformers import AutoModel as _AM
    model.bert = _AM.from_pretrained(cfg.bert_name, attn_implementation="eager").to(cfg.device)
    print(f"         Fused vector dim = {model.fused_dim}")

    # Load or build the classifier once and reuse it for both test cases
    if args.clf_path:
        print(f"\n[Setup] Loading classifier from {args.clf_path}...")
        with open(args.clf_path, "rb") as f:
            clf = pickle.load(f)
    else:
        print("\n[Setup] No --clf-path given -> building DEMO classifier on synthetic data.")
        print("         Predictions are illustrative only until you train a real model.")
        clf = build_demo_classifier(fused_dim=model.fused_dim)

    # ── Run both test cases ────────────────────────────────────────────────────
    for case_num, (sample, expected_label) in enumerate(test_cases, 1):

        print_banner(f"TEST CASE {case_num}/2 — Expected: {expected_label.upper()}")
        print(f"  Expected label : {expected_label.upper()}")
        print(f"  Text snippet   : {sample['text'][:120]}...")
        print(f"  Image path     : {sample['image_path'] or '(none — blank frame used)'}")

        metadata_values = np.array(
            [sample["metadata"][c] for c in metadata_cols], dtype=np.float32
        )
        print(f"\n[Metadata] {dict(zip(metadata_cols, metadata_values))}")

        # Encode
        print(f"\n[Step 1] Encoding article {case_num}/2...")
        out      = encode_single_sample(model, sample["text"], sample["image_path"], metadata_values, cfg)
        fused_np = out["fused"].cpu().numpy()
        s_sim    = out["s_sim"].cpu().item()
        alpha    = out["alpha"].cpu().item()
        print(f"         s_sim = {s_sim:.4f}   alpha = {alpha:.4f}")

        # Predict
        print(f"\n[Step 2] Predicting...")
        pred_idx = clf.predict(fused_np)[0]
        pred_str = "FAKE" if pred_idx == 1 else "REAL"

        if hasattr(clf, "predict_proba"):
            proba      = clf.predict_proba(fused_np)[0]
            conf_real  = proba[0]
            conf_fake  = proba[1]
            confidence = proba[pred_idx]
        else:
            conf_real  = float(pred_idx == 0)
            conf_fake  = float(pred_idx == 1)
            confidence = 1.0

        print_banner(f"PREDICTION RESULT — Case {case_num}/2")
        correct = pred_str.lower() == expected_label
        verdict = "CORRECT" if correct else "WRONG (demo clf — train a real one!)"
        print(f"  Predicted label   : {pred_str}")
        print(f"  Confidence        : {confidence * 100:.1f}%")
        print(f"  P(real)           : {conf_real * 100:.1f}%")
        print(f"  P(fake)           : {conf_fake * 100:.1f}%")
        print(f"  Expected label    : {expected_label.upper()}")
        print(f"  Verdict           : {verdict}")
        print(f"\n  s_sim  (CLIP text-image cosine similarity) : {s_sim:+.4f}")
        print(f"  alpha  (cross-modal attention weight)       : {alpha:.4f}")

        # Explainability
        explain_bert_attention(model, out, top_k=15)

        if not args.no_lime:
            explain_lime(model, clf, sample["text"], metadata_values, cfg)
        else:
            print("\n  [LIME skipped — run without --no-lime to enable]")

        if not args.no_shap:
            feature_names = make_feature_names(cfg.metadata_out_dim, metadata_cols)
            explain_shap(clf, fused_np, feature_names)
        else:
            print("  [SHAP skipped — run without --no-shap to enable]\n")

        print("=" * 62 + "\n")


if __name__ == "__main__":
    main()