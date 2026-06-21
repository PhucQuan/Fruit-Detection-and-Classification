from __future__ import annotations

import base64
import json
from pathlib import Path
import time
from textwrap import dedent

import altair as alt
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import tensorflow as tf


st.set_page_config(
    page_title="Fruit Classification",
    page_icon=":apple:",
    layout="wide",
    initial_sidebar_state="expanded",
)


PALETTE = {
    "bg": "#f7f1ea",
    "panel": "#fffaf5",
    "line": "#e6d8cb",
    "ink": "#2f2925",
    "muted": "#6f655d",
    "peach": "#e89b84",
    "peach_dark": "#c56f58",
    "apricot": "#f2c58b",
    "sage": "#9db9a7",
    "sky": "#b9d3ea",
    "lavender": "#b7add9",
    "shadow": "0 14px 30px rgba(124, 101, 83, 0.08)",
}


FRUIT_INFO = {
    "Apple": {
        "summary": "Round shape, stable skin color, and clear silhouette.",
        "calories": "52 kcal / 100g",
        "fiber": "2.4 g",
        "vitamin": "Vitamin C",
        "taste": "Sweet, crisp",
    },
    "Banana": {
        "summary": "Long curved shape with a very distinctive outline.",
        "calories": "89 kcal / 100g",
        "fiber": "2.6 g",
        "vitamin": "Vitamin B6",
        "taste": "Sweet, soft",
    },
    "Cucumber": {
        "summary": "Elongated green body with a simple surface texture.",
        "calories": "15 kcal / 100g",
        "fiber": "0.5 g",
        "vitamin": "Vitamin K",
        "taste": "Fresh, mild",
    },
    "Grape": {
        "summary": "Cluster structure gives the model a strong visual cue.",
        "calories": "69 kcal / 100g",
        "fiber": "0.9 g",
        "vitamin": "Vitamin K",
        "taste": "Juicy, sweet",
    },
    "Mango": {
        "summary": "Soft oval body with yellow-orange color variation.",
        "calories": "60 kcal / 100g",
        "fiber": "1.6 g",
        "vitamin": "Vitamin A",
        "taste": "Sweet, tropical",
    },
    "Orange": {
        "summary": "Bright orange color and uniform round shape.",
        "calories": "47 kcal / 100g",
        "fiber": "2.4 g",
        "vitamin": "Vitamin C",
        "taste": "Sweet, citrus",
    },
    "Pear": {
        "summary": "Slim upper part and fuller lower body.",
        "calories": "57 kcal / 100g",
        "fiber": "3.1 g",
        "vitamin": "Copper",
        "taste": "Sweet, smooth",
    },
    "Tomato": {
        "summary": "Round red body, sometimes with a visible leafy crown.",
        "calories": "18 kcal / 100g",
        "fiber": "1.2 g",
        "vitamin": "Vitamin C",
        "taste": "Tangy, juicy",
    },
}


MODEL_SLOTS = [
    {
        "id": "mobilenetv2",
        "name": "MobileNetV2",
        "patterns": ["*mobilenet*.keras"],
        "description": "MobileNetV2 result",
    },
    {
        "id": "efficientnetb0",
        "name": "EfficientNetB0",
        "patterns": [
            "fruit_8class_efficientnetb0_best_fixed.keras",
            "bestfixed.keras",
            "fruit_8class_efficientnetb0_best.keras",
        ],
        "description": "EfficientNetB0 result",
    },
    {
        "id": "resnet18",
        "name": "ResNet18",
        "patterns": ["*resnet*.pth", "model_fruit_8classes.pth"],
        "description": "ResNet18 result",
    },
]


def render_html(content: str) -> None:
    st.markdown(dedent(content).strip(), unsafe_allow_html=True)


def inject_css() -> None:
    st.markdown(
        f"""
        <style>
        :root {{
            --bg: {PALETTE["bg"]};
            --panel: {PALETTE["panel"]};
            --line: {PALETTE["line"]};
            --ink: {PALETTE["ink"]};
            --muted: {PALETTE["muted"]};
            --peach: {PALETTE["peach"]};
            --peach-dark: {PALETTE["peach_dark"]};
            --apricot: {PALETTE["apricot"]};
            --sage: {PALETTE["sage"]};
            --sky: {PALETTE["sky"]};
            --lavender: {PALETTE["lavender"]};
            --shadow: {PALETTE["shadow"]};
        }}

        html, body, [class*="css"] {{
            font-family: "Inter", "Segoe UI", sans-serif;
            color: var(--ink);
        }}

        .stApp {{
            background: linear-gradient(180deg, #fbf7f3 0%, var(--bg) 100%);
        }}

        .block-container {{
            max-width: 1260px;
            padding-top: 2rem;
            padding-bottom: 2.4rem;
        }}

        [data-testid="stHeader"], #MainMenu, footer {{
            display: none !important;
        }}

        [data-testid="stSidebar"] {{
            background: #f4ece3;
            border-right: 1px solid var(--line);
        }}

        [data-testid="stSidebar"] * {{
            color: var(--ink) !important;
        }}

        .hero {{
            background: linear-gradient(135deg, #fff9f4 0%, #fcf1e8 58%, #f8f0e7 100%);
            border: 1px solid var(--line);
            border-radius: 28px;
            padding: 30px;
            box-shadow: var(--shadow);
            margin-bottom: 18px;
        }}

        .hero-grid {{
            display: grid;
            grid-template-columns: minmax(0, 1.55fr) minmax(290px, 0.9fr);
            gap: 18px;
            align-items: center;
        }}

        .eyebrow {{
            display: inline-flex;
            padding: 8px 12px;
            border-radius: 999px;
            background: #fff1e7;
            border: 1px solid #f1d9c9;
            color: #a56c52;
            font-size: 12px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 16px;
        }}

        .hero-title {{
            font-family: Georgia, "Times New Roman", serif;
            font-size: 44px;
            line-height: 1.04;
            margin: 0 0 12px 0;
            color: var(--ink) !important;
            opacity: 1 !important;
            text-shadow: none !important;
            font-weight: 700 !important;
        }}

        .hero-copy {{
            margin: 0;
            color: #5f544c !important;
            font-size: 16px;
            line-height: 1.6;
        }}

        .hero-pills {{
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin-top: 18px;
        }}

        .hero-pill {{
            padding: 9px 12px;
            border-radius: 999px;
            background: rgba(255,255,255,0.74);
            border: 1px solid var(--line);
            font-size: 13px;
            color: #6e6257;
        }}

        .hero-art {{
            min-height: 250px;
            position: relative;
            background: linear-gradient(180deg, rgba(255,255,255,0.76), rgba(255,255,255,0.38));
            border: 1px dashed var(--line);
            border-radius: 24px;
        }}

        .hero-fruit {{
            position: absolute;
            width: 92px;
            height: 92px;
        }}

        .fruit-a {{ top: 22px; left: 24px; }}
        .fruit-b {{ top: 24px; right: 28px; }}
        .fruit-c {{ bottom: 26px; left: 52px; }}
        .fruit-d {{ bottom: 20px; right: 34px; }}

        .panel {{
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 22px;
            box-shadow: var(--shadow);
            padding: 22px;
            margin-bottom: 18px;
        }}

        .kicker {{
            color: var(--peach-dark);
            font-size: 12px;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 10px;
        }}

        .panel-title {{
            font-family: Georgia, "Times New Roman", serif;
            font-size: 30px;
            margin: 0 0 8px 0;
            color: var(--ink);
        }}

        .panel-copy {{
            color: var(--muted);
            font-size: 15px;
            line-height: 1.55;
            margin: 0;
        }}

        .metric-grid {{
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 14px;
            margin-bottom: 18px;
        }}

        .metric-card {{
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 20px;
            box-shadow: var(--shadow);
            padding: 18px;
        }}

        .metric-label {{
            font-size: 12px;
            font-weight: 700;
            color: var(--muted);
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 10px;
        }}

        .metric-value {{
            font-family: Georgia, "Times New Roman", serif;
            font-size: 32px;
            line-height: 1;
            color: var(--ink);
            margin-bottom: 8px;
        }}

        .metric-note {{
            font-size: 13px;
            line-height: 1.5;
            color: var(--muted);
        }}

        .result-box {{
            display: grid;
            grid-template-columns: 140px minmax(0, 1fr);
            gap: 16px;
            align-items: center;
            background: linear-gradient(135deg, #fffaf5 0%, #fff2eb 100%);
            border: 1px solid var(--line);
            border-radius: 22px;
            padding: 18px;
            margin-bottom: 14px;
        }}

        .result-name {{
            font-family: Georgia, "Times New Roman", serif;
            font-size: 34px;
            margin: 0 0 8px 0;
            color: var(--ink);
        }}

        .result-sub {{
            color: var(--muted);
            font-size: 15px;
            line-height: 1.55;
            margin-bottom: 8px;
        }}

        .chip {{
            display: inline-flex;
            align-items: center;
            padding: 8px 12px;
            border-radius: 999px;
            font-size: 13px;
            font-weight: 700;
            margin-bottom: 10px;
        }}

        .chip-ok {{
            background: #edf6ee;
            border: 1px solid #d3e5d6;
            color: #557c5d;
        }}

        .chip-warn {{
            background: #fff2e4;
            border: 1px solid #f0ddc1;
            color: #9b7348;
        }}

        .detail-grid {{
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 10px;
            margin-top: 14px;
        }}

        .detail-item {{
            background: #fffdf9;
            border: 1px solid var(--line);
            border-radius: 16px;
            padding: 12px 14px;
        }}

        .detail-label {{
            font-size: 12px;
            color: var(--muted);
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 6px;
        }}

        .detail-value {{
            color: var(--ink);
            font-weight: 600;
            line-height: 1.4;
        }}

        .empty-box {{
            padding: 22px;
            border-radius: 20px;
            border: 1px dashed #dccbbc;
            background: #fff9f2;
            color: var(--muted);
            line-height: 1.65;
        }}

        .model-grid {{
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 14px;
        }}

        .model-card {{
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 20px;
            padding: 18px;
            box-shadow: var(--shadow);
        }}

        .model-name {{
            font-family: Georgia, "Times New Roman", serif;
            font-size: 24px;
            margin: 0 0 6px 0;
            color: var(--ink);
        }}

        .model-meta, .model-file, .model-text {{
            color: var(--muted);
            font-size: 14px;
            line-height: 1.55;
        }}

        .model-file {{
            margin-top: 10px;
            word-break: break-word;
        }}

        .model-result {{
            margin-top: 12px;
            padding-top: 12px;
            border-top: 1px dashed #e5d7cb;
        }}

        .fruit-grid {{
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 12px;
        }}

        .fruit-card {{
            background: #fffdf9;
            border: 1px solid var(--line);
            border-radius: 18px;
            padding: 14px;
            text-align: center;
        }}

        .fruit-card img {{
            width: 72px;
            height: 72px;
            display: block;
            margin: 0 auto 10px auto;
        }}

        .fruit-name {{
            font-weight: 700;
            color: var(--ink);
            margin-bottom: 4px;
        }}

        .fruit-sub {{
            color: var(--muted);
            font-size: 13px;
            line-height: 1.45;
        }}

        .report-table {{
            width: 100%;
            border-collapse: collapse;
        }}

        .report-table th {{
            text-align: left;
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: var(--muted);
            padding: 12px 14px;
            border-bottom: 1px solid var(--line);
            background: #fbf5ee;
        }}

        .report-table td {{
            padding: 12px 14px;
            border-bottom: 1px solid #efe3d7;
            color: var(--ink);
            background: #fffdf9;
        }}

        [data-testid="stFileUploaderDropzone"] {{
            background: #fff9f2;
            border: 1px dashed #dccbbc;
            border-radius: 20px;
            padding: 20px;
        }}

        [data-testid="stTabs"] button[role="tab"] {{
            border-radius: 999px;
            padding: 9px 16px;
            color: var(--muted);
            font-weight: 600;
        }}

        [data-testid="stTabs"] button[role="tab"][aria-selected="true"] {{
            background: #fffaf5;
            border: 1px solid var(--line);
            color: var(--ink);
        }}

        .stButton > button, .stDownloadButton > button {{
            width: 100%;
            min-height: 46px;
            border-radius: 16px;
            border: 1px solid #dcb9a7;
            background: linear-gradient(180deg, #efb69d 0%, #e89b84 100%);
            color: white;
            font-weight: 700;
            box-shadow: 0 10px 20px rgba(232, 155, 132, 0.22);
        }}

        .stSlider [data-baseweb="slider"] > div > div {{
            background: var(--peach);
        }}

        [data-baseweb="select"] > div {{
            background: var(--panel) !important;
            border: 1px solid var(--line) !important;
            color: var(--ink) !important;
            border-radius: 14px !important;
            box-shadow: none !important;
        }}

        [data-baseweb="select"] * {{
            color: var(--ink) !important;
        }}

        [data-baseweb="popover"] {{
            background: var(--panel) !important;
            border: 1px solid var(--line) !important;
            color: var(--ink) !important;
        }}

        [data-baseweb="select"] > div {{
            background: var(--panel) !important;
            border: 1px solid var(--line) !important;
            color: var(--ink) !important;
            border-radius: 14px !important;
            box-shadow: none !important;
        }}

        [data-baseweb="select"] * {{
            color: var(--ink) !important;
        }}

        [data-baseweb="popover"] {{
            background: var(--panel) !important;
            border: 1px solid var(--line) !important;
            color: var(--ink) !important;
        }}

        [data-testid="stImage"] img {{
            border-radius: 20px;
            border: 1px solid var(--line);
        }}

        @media (max-width: 980px) {{
            .hero-grid,
            .metric-grid,
            .result-box,
            .detail-grid,
            .model-grid,
            .fruit-grid {{
                grid-template-columns: 1fr;
            }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def svg_data_uri(svg: str) -> str:
    encoded = base64.b64encode(svg.encode("utf-8")).decode("ascii")
    return f"data:image/svg+xml;base64,{encoded}"


def fruit_svg(name: str) -> str:
    common = 'stroke="#6c4d3b" stroke-width="4" stroke-linecap="round" stroke-linejoin="round"'
    if name == "Apple":
        svg = f"""
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 140 140">
            <rect width="140" height="140" rx="28" fill="#fff8f2"/>
            <path d="M70 28 C67 18, 71 10, 80 6" fill="none" {common}/>
            <path d="M82 18 C92 10, 106 12, 112 24 C102 28, 92 31, 82 18 Z" fill="#9db9a7" {common}/>
            <path d="M42 56 C42 38, 58 32, 70 42 C82 31, 98 39, 98 56 C98 78, 85 104, 70 104 C55 104, 42 78, 42 56 Z" fill="#e78b86" {common}/>
        </svg>
        """
    elif name == "Banana":
        svg = f"""
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 140 140">
            <rect width="140" height="140" rx="28" fill="#fff8f2"/>
            <path d="M32 86 C52 106, 94 102, 114 58" fill="none" stroke="#e5bc57" stroke-width="24" stroke-linecap="round"/>
            <path d="M34 84 C54 100, 88 96, 108 62" fill="none" stroke="#ffe4a0" stroke-width="11" stroke-linecap="round"/>
        </svg>
        """
    elif name == "Cucumber":
        svg = f"""
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 140 140">
            <rect width="140" height="140" rx="28" fill="#fff8f2"/>
            <g transform="rotate(-12 70 70)">
                <rect x="26" y="46" width="88" height="44" rx="22" fill="#9db9a7" {common}/>
                <path d="M38 68 H102" fill="none" stroke="#d3ead9" stroke-width="6" stroke-linecap="round"/>
            </g>
        </svg>
        """
    elif name == "Grape":
        svg = f"""
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 140 140">
            <rect width="140" height="140" rx="28" fill="#fff8f2"/>
            <circle cx="56" cy="56" r="16" fill="#b7add9" {common}/>
            <circle cx="82" cy="56" r="16" fill="#a28bc8" {common}/>
            <circle cx="46" cy="80" r="16" fill="#9f86c0" {common}/>
            <circle cx="70" cy="80" r="16" fill="#b69bda" {common}/>
            <circle cx="94" cy="80" r="16" fill="#8e73b4" {common}/>
            <circle cx="58" cy="104" r="16" fill="#aa8fd1" {common}/>
            <circle cx="84" cy="104" r="16" fill="#9a80c1" {common}/>
        </svg>
        """
    elif name == "Mango":
        svg = f"""
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 140 140">
            <rect width="140" height="140" rx="28" fill="#fff8f2"/>
            <path d="M60 34 C82 28, 106 40, 108 68 C110 96, 90 112, 66 108 C42 104, 30 82, 38 58 C44 42, 49 38, 60 34 Z" fill="#f0bd6d" {common}/>
        </svg>
        """
    elif name == "Orange":
        svg = f"""
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 140 140">
            <rect width="140" height="140" rx="28" fill="#fff8f2"/>
            <circle cx="70" cy="74" r="36" fill="#f1ba76" {common}/>
        </svg>
        """
    elif name == "Pear":
        svg = f"""
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 140 140">
            <rect width="140" height="140" rx="28" fill="#fff8f2"/>
            <path d="M70 42 C84 42, 92 52, 92 64 C92 68, 90 74, 95 82 C104 95, 94 112, 70 112 C46 112, 36 95, 45 82 C50 74, 48 69, 48 64 C48 52, 56 42, 70 42 Z" fill="#b7cf95" {common}/>
        </svg>
        """
    elif name == "Tomato":
        svg = f"""
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 140 140">
            <rect width="140" height="140" rx="28" fill="#fff8f2"/>
            <circle cx="70" cy="74" r="36" fill="#cf7a7f" {common}/>
            <path d="M70 38 L76 50 L90 48 L82 58 L92 68 L78 66 L70 78 L62 66 L48 68 L58 58 L50 48 L64 50 Z" fill="#9db9a7" {common}/>
        </svg>
        """
    else:
        svg = f"""
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 140 140">
            <rect width="140" height="140" rx="28" fill="#fff8f2"/>
            <circle cx="70" cy="70" r="34" fill="#e4c6b8" {common}/>
        </svg>
        """
    return svg_data_uri(svg)


def render_hero() -> None:
    render_html(
        f"""
        <div class="hero">
            <div class="hero-grid">
                <div>
                    <div class="eyebrow">Fruit Classification Project</div>
                    <h1 class="hero-title">Fruit Detection and Classification</h1>
                    <p class="hero-copy">A softer pastel interface with cleaner contrast and a more polished dashboard feel.</p>
                    <div class="hero-pills">
                        <span class="hero-pill">Upload or camera input</span>
                        <span class="hero-pill">Three model comparison</span>
                        <span class="hero-pill">Confidence chart</span>
                    </div>
                </div>
                <div class="hero-art">
                    <img class="hero-fruit fruit-a" src="{fruit_svg("Apple")}" alt="Apple" />
                    <img class="hero-fruit fruit-b" src="{fruit_svg("Orange")}" alt="Orange" />
                    <img class="hero-fruit fruit-c" src="{fruit_svg("Grape")}" alt="Grape" />
                    <img class="hero-fruit fruit-d" src="{fruit_svg("Banana")}" alt="Banana" />
                </div>
            </div>
        </div>
        """
    )


def load_class_names() -> list[str]:
    path = Path("class_names.json")
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return list(FRUIT_INFO.keys())


def load_metrics() -> pd.DataFrame | None:
    path = Path("evaluation_metrics_summary.csv")
    if path.exists():
        return pd.read_csv(path, index_col=0)
    return None


def find_model_path(patterns: list[str], used: set[Path]) -> Path | None:
    for pattern in patterns:
        for path in sorted(Path(".").glob(pattern)):
            resolved = path.resolve()
            if resolved not in used:
                used.add(resolved)
                return path
    return None


def discover_models() -> list[dict]:
    used: set[Path] = set()
    models = []
    for slot in MODEL_SLOTS:
        path = find_model_path(slot["patterns"], used)
        models.append(
            {
                "id": slot["id"],
                "name": slot["name"],
                "description": slot["description"],
                "path": path,
                "status": "available" if path else "missing",
            }
        )
    return models


@st.cache_resource(show_spinner=False)
def load_model_resource(path_str: str):
    if path_str.endswith(".pth"):
        import torch
        import torchvision.models as tv_models
        model = tv_models.resnet18(weights=None)
        model.fc = torch.nn.Linear(512, 8)
        ckpt = torch.load(path_str, map_location='cpu')
        model.load_state_dict(ckpt)
        model.eval()
        return model
    else:
        return tf.keras.models.load_model(path_str)


def preprocess_image(image: Image.Image) -> np.ndarray:
    if image.mode != "RGB":
        image = image.convert("RGB")
    resized = image.resize((224, 224), Image.Resampling.LANCZOS)
    return np.expand_dims(np.asarray(resized, dtype=np.float32), axis=0)


def normalize_scores(raw_output: np.ndarray) -> np.ndarray:
    scores = np.asarray(raw_output, dtype=np.float32)
    if scores.ndim > 1:
        scores = scores[0]
    if np.all(scores >= 0) and np.isclose(float(scores.sum()), 1.0, atol=1e-2):
        return scores
    shifted = scores - float(np.max(scores))
    exp_scores = np.exp(shifted)
    return exp_scores / float(exp_scores.sum())


def predict_with_model(model_path: Path, image: Image.Image, class_names: list[str]) -> dict:
    start = time.perf_counter()
    path_str = str(model_path)
    model = load_model_resource(path_str)
    
    if path_str.endswith(".pth"):
        import torch
        if image.mode != "RGB":
            image = image.convert("RGB")
        resized = image.resize((224, 224), Image.Resampling.LANCZOS)
        img_arr = np.asarray(resized, dtype=np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        normalized = (img_arr - mean) / std
        tensor = torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze(0)
        
        with torch.no_grad():
            outputs = model(tensor)
            scores = torch.nn.functional.softmax(outputs, dim=1)[0].numpy()
    else:
        batch = preprocess_image(image)
        raw_output = model.predict(batch, verbose=0)
        scores = normalize_scores(raw_output)
        
    index = int(np.argmax(scores))
    elapsed_ms = (time.perf_counter() - start) * 1000
    return {
        "label": class_names[index],
        "confidence": float(scores[index]),
        "scores": [float(x) for x in scores.tolist()],
        "time_ms": elapsed_ms,
    }


def run_all_models(models: list[dict], image: Image.Image, class_names: list[str]) -> dict[str, dict]:
    results: dict[str, dict] = {}
    for model in models:
        if model["status"] != "available" or not model["path"]:
            continue
        try:
            results[model["id"]] = predict_with_model(model["path"], image, class_names)
        except Exception as exc:
            results[model["id"]] = {"error": str(exc)}
    return results


def best_result_key(models: list[dict], results: dict[str, dict]) -> str | None:
    ranked = []
    for model in models:
        result = results.get(model["id"])
        if result and not result.get("error"):
            ranked.append((result["confidence"], model["id"]))
    if not ranked:
        return None
    ranked.sort(reverse=True)
    return ranked[0][1]


def render_metric_strip(models: list[dict], metrics_df: pd.DataFrame | None, class_names: list[str]) -> None:
    live_models = sum(1 for model in models if model["status"] == "available")
    accuracy = "--"
    support = "--"
    if metrics_df is not None:
        if "accuracy" in metrics_df.index:
            accuracy = f"{float(metrics_df.loc['accuracy', 'precision']):.1%}"
        if "weighted avg" in metrics_df.index and "support" in metrics_df.columns:
            support = f"{int(float(metrics_df.loc['weighted avg', 'support'])):,}"

    cards = [
        ("Classes", str(len(class_names)), "Fruit categories available in this app."),
        ("Models", f"{live_models}/3", "Only existing checkpoints are executed."),
        ("Accuracy", accuracy, "Pulled from the evaluation file."),
        ("Test Images", support, "Samples listed in the report."),
    ]

    html = '<div class="metric-grid">'
    for label, value, note in cards:
        html += (
            '<div class="metric-card">'
            f'<div class="metric-label">{label}</div>'
            f'<div class="metric-value">{value}</div>'
            f'<div class="metric-note">{note}</div>'
            "</div>"
        )
    html += "</div>"
    render_html(html)


def empty_box(text: str) -> None:
    render_html(f'<div class="empty-box">{text}</div>')


def render_result_panel(result: dict, threshold: float) -> None:
    fruit = result["label"]
    detail = FRUIT_INFO.get(fruit, {})
    chip_class = "chip-ok" if result["confidence"] >= threshold else "chip-warn"
    chip_text = "Accepted" if result["confidence"] >= threshold else "Below threshold"

    render_html(
        f"""
        <div class="result-box">
            <img src="{fruit_svg(fruit)}" alt="{fruit}" />
            <div>
                <div class="chip {chip_class}">{chip_text}</div>
                <div class="result-name">{fruit}</div>
                <div class="result-sub">Confidence: {result["confidence"]:.1%} - Time: {result["time_ms"]:.0f} ms</div>
                <div class="result-sub">{detail.get("summary", "")}</div>
            </div>
        </div>
        """
    )

    items = [
        ("Calories", detail.get("calories", "--")),
        ("Fiber", detail.get("fiber", "--")),
        ("Vitamin", detail.get("vitamin", "--")),
        ("Taste", detail.get("taste", "--")),
    ]
    html = '<div class="detail-grid">'
    for label, value in items:
        html += (
            '<div class="detail-item">'
            f'<div class="detail-label">{label}</div>'
            f'<div class="detail-value">{value}</div>'
            "</div>"
        )
    html += "</div>"
    render_html(html)


def build_confidence_chart(class_names: list[str], scores: list[float], selected_label: str):
    df = pd.DataFrame({"Class": class_names, "Confidence": scores}).sort_values("Confidence", ascending=False)
    df["Group"] = np.where(df["Class"] == selected_label, "Selected", "Other")

    base = alt.Chart(df).encode(
        x=alt.X("Confidence:Q", axis=alt.Axis(format=".0%", title=None, tickCount=5, labelColor=PALETTE["ink"], gridColor="#eadfd6")),
        y=alt.Y("Class:N", sort="-x", axis=alt.Axis(title=None, labelColor=PALETTE["ink"])),
        color=alt.Color("Group:N", scale=alt.Scale(domain=["Selected", "Other"], range=[PALETTE["peach"], PALETTE["sky"]]), legend=None),
    )
    bars = base.mark_bar(cornerRadiusEnd=8, size=24)
    text = base.mark_text(align="left", baseline="middle", dx=6, color=PALETTE["ink"]).encode(
        text=alt.Text("Confidence:Q", format=".1%")
    )
    return (
        bars + text
    ).properties(
        height=max(260, len(df) * 34),
        background=PALETTE["panel"],
    ).configure(
        background=PALETTE["panel"],
    ).configure_view(
        fill=PALETTE["panel"],
        strokeOpacity=0,
    ).configure_axis(
        domainColor=PALETTE["line"],
        tickColor=PALETTE["line"],
        labelColor=PALETTE["ink"],
        titleColor=PALETTE["ink"],
        gridColor="#eadfd6",
    )


def render_model_cards(models: list[dict], results: dict[str, dict]) -> None:
    html = '<div class="model-grid">'
    for model in models:
        result = results.get(model["id"])
        status = "Ready" if model["status"] == "available" else "Missing file"
        chip_class = "chip-ok" if model["status"] == "available" else "chip-warn"

        body = '<div class="model-result"><div class="model-text">Not run yet</div></div>'
        if result and result.get("error"):
            body = f'<div class="model-result"><div class="model-text">Error: {result["error"]}</div></div>'
        elif result:
            body = (
                '<div class="model-result">'
                f'<div class="model-name" style="font-size:22px; margin-bottom:6px">{result["label"]}</div>'
                f'<div class="model-text">{result["confidence"]:.1%} - {result["time_ms"]:.0f} ms</div>'
                "</div>"
            )

        file_text = model["path"].name if model["path"] else "No matching .keras file"
        html += (
            '<div class="model-card">'
            f'<div class="chip {chip_class}">{status}</div>'
            f'<div class="model-name">{model["name"]}</div>'
            f'<div class="model-meta">{model["description"]}</div>'
            f'<div class="model-file"><strong>File:</strong> {file_text}</div>'
            f"{body}"
            "</div>"
        )
    html += "</div>"
    render_html(html)


def render_report_table(metrics_df: pd.DataFrame, class_names: list[str]) -> None:
    rows = ""
    for name in class_names:
        if name not in metrics_df.index:
            continue
        row = metrics_df.loc[name]
        rows += (
            "<tr>"
            f"<td>{name}</td>"
            f"<td>{float(row['precision']):.1%}</td>"
            f"<td>{float(row['recall']):.1%}</td>"
            f"<td>{float(row['f1-score']):.1%}</td>"
            f"<td>{int(float(row['support'])):,}</td>"
            "</tr>"
        )
    render_html(
        f"""
        <table class="report-table">
            <thead>
                <tr>
                    <th>Fruit</th>
                    <th>Precision</th>
                    <th>Recall</th>
                    <th>F1-score</th>
                    <th>Support</th>
                </tr>
            </thead>
            <tbody>{rows}</tbody>
        </table>
        """
    )


def render_fruit_gallery(class_names: list[str]) -> None:
    html = '<div class="fruit-grid">'
    for name in class_names:
        detail = FRUIT_INFO.get(name, {})
        html += (
            '<div class="fruit-card">'
            f'<img src="{fruit_svg(name)}" alt="{name}" />'
            f'<div class="fruit-name">{name}</div>'
            f'<div class="fruit-sub">{detail.get("taste", "")}</div>'
            "</div>"
        )
    html += "</div>"
    render_html(html)


def sidebar_controls(models: list[dict]) -> tuple[float, int]:
    st.sidebar.markdown("## Controls")
    threshold = st.sidebar.slider("Confidence threshold", 0.0, 1.0, 0.45, 0.01)
    top_k = st.sidebar.slider("Classes to display", 3, 8, 5, 1)
    st.sidebar.markdown("### Model status")
    for model in models:
        status = "Ready" if model["status"] == "available" else "Missing file"
        st.sidebar.markdown(f"- **{model['name']}**: {status}")
    return threshold, top_k


inject_css()
models = discover_models()
class_names = load_class_names()
metrics_df = load_metrics()
render_hero()
render_metric_strip(models, metrics_df, class_names)
threshold, top_k = sidebar_controls(models)

if "model_results" not in st.session_state:
    st.session_state.model_results = {}

tabs = st.tabs(["Demo", "Model Comparison", "Report"])


with tabs[0]:
    left, right = st.columns([1.02, 1], gap="large")

    with left:
        render_html(
            """
            <div class="panel">
                <div class="kicker">Input</div>
                <div class="panel-title">Upload Image</div>
                <div class="panel-copy">Choose a fruit image from your computer or use the camera tab.</div>
            </div>
            """
        )

        input_tabs = st.tabs(["Upload", "Camera"])
        image_input = None

        with input_tabs[0]:
            uploaded_file = st.file_uploader("Choose a fruit image", type=["jpg", "jpeg", "png", "webp"])
            if uploaded_file is not None:
                image_input = Image.open(uploaded_file)

        with input_tabs[1]:
            camera_file = st.camera_input("Capture a new image")
            if camera_file is not None:
                image_input = Image.open(camera_file)

        if image_input is not None:
            st.image(image_input, use_container_width=True)
        else:
            empty_box("No image selected yet. Upload or capture one to start.")

        if st.button("Analyze Image", disabled=image_input is None):
            with st.spinner("Running inference..."):
                st.session_state.model_results = run_all_models(models, image_input, class_names)

    with right:
        render_html(
            """
            <div class="panel">
                <div class="kicker">Prediction</div>
                <div class="panel-title">Best Result</div>
                <div class="panel-copy">The strongest available prediction is shown here.</div>
            </div>
            """
        )

        results = st.session_state.model_results
        best_key = best_result_key(models, results)

        if not results:
            empty_box("Prediction results will appear here after analysis.")
        elif best_key is None:
            empty_box("No valid prediction is available yet.")
        else:
            best_result = results[best_key]
            render_result_panel(best_result, threshold)

            valid_models = [model for model in models if results.get(model["id"]) and not results[model["id"]].get("error")]
            selected_name = st.selectbox("View confidence chart for", [model["name"] for model in valid_models], index=0)
            selected_model = next(model for model in valid_models if model["name"] == selected_name)
            selected_result = results[selected_model["id"]]

            render_html(
                """
                <div class="panel" style="padding:18px; margin-top:16px">
                    <div class="kicker">Confidence Chart</div>
                    <div class="panel-title" style="font-size:26px">Class Confidence Distribution</div>
                </div>
                """
            )
            st.altair_chart(
                build_confidence_chart(class_names, selected_result["scores"], selected_result["label"]),
                use_container_width=True,
            )

            export_df = (
                pd.DataFrame({"fruit": class_names, "confidence": selected_result["scores"]})
                .sort_values("confidence", ascending=False)
                .head(top_k)
            )
            st.download_button(
                "Download CSV",
                export_df.to_csv(index=False).encode("utf-8"),
                file_name="fruit_predictions.csv",
                mime="text/csv",
            )


with tabs[1]:
    render_html(
        """
        <div class="panel">
            <div class="kicker">Three Models</div>
            <div class="panel-title">Model Comparison</div>
            <div class="panel-copy">MobileNetV2, EfficientNetB0, and ResNet18 are shown as separate result cards.</div>
        </div>
        """
    )
    render_model_cards(models, st.session_state.model_results)

    if st.session_state.model_results:
        rows = []
        for model in models:
            result = st.session_state.model_results.get(model["id"])
            if not result:
                rows.append(
                    {
                        "Model": model["name"],
                        "Status": "Missing file" if model["status"] == "missing" else "Not run",
                        "Prediction": "--",
                        "Confidence": np.nan,
                        "Time (ms)": np.nan,
                    }
                )
            elif result.get("error"):
                rows.append(
                    {
                        "Model": model["name"],
                        "Status": "Error",
                        "Prediction": "Load failed",
                        "Confidence": np.nan,
                        "Time (ms)": np.nan,
                    }
                )
            else:
                rows.append(
                    {
                        "Model": model["name"],
                        "Status": "OK",
                        "Prediction": result["label"],
                        "Confidence": result["confidence"],
                        "Time (ms)": result["time_ms"],
                    }
                )
        compare_df = pd.DataFrame(rows)
        st.dataframe(
            compare_df.style.format({"Confidence": "{:.1%}", "Time (ms)": "{:.0f}"}),
            use_container_width=True,
            hide_index=True,
        )
    else:
        empty_box("Run an image through the app to populate the comparison table.")


with tabs[2]:
    render_html(
        """
        <div class="panel">
            <div class="kicker">Evaluation</div>
            <div class="panel-title">Classification Report</div>
        </div>
        """
    )

    if metrics_df is not None:
        render_report_table(metrics_df, class_names)
    else:
        empty_box("evaluation_metrics_summary.csv was not found.")

    render_html(
        """
        <div class="panel" style="margin-top:18px">
            <div class="kicker">Supported Classes</div>
            <div class="panel-title">Fruit Gallery</div>
        </div>
        """
    )
    render_fruit_gallery(class_names)

