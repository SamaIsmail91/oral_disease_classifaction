"""
Oral Disease Image Classifier — Gradio deployment app.

Design direction: a clinical "diagnostic chart / scan viewer" aesthetic —
dark radiograph-viewer hero, a findings report styled like an actual chart
readout with a ranked differential diagnosis (a real clinical convention,
not decorative numbering), and density-gauge style confidence bars.

Loads whichever model outputs/reports/best_model.json points to (produced by
evaluate_compare.py locally, or by the Kaggle notebook — see README).

Run with:
    python3 app.py
"""
import os
import json

import numpy as np
import gradio as gr
import tensorflow as tf

# ---------------------------------------------------------------------------
# Paths (self-contained -- no dependency on the rest of the project)
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
REPORTS_DIR = os.path.join(PROJECT_ROOT, "outputs", "reports")

# ---------------------------------------------------------------------------
# Load best model + metadata
# ---------------------------------------------------------------------------
BEST_MODEL_INFO_PATH = os.path.join(REPORTS_DIR, "best_model.json")

if not os.path.exists(BEST_MODEL_INFO_PATH):
    raise FileNotFoundError(
        "outputs/reports/best_model.json not found. Run train.py then evaluate_compare.py "
        "first (or copy those files over from the Kaggle run)."
    )

with open(BEST_MODEL_INFO_PATH) as f:
    BEST_INFO = json.load(f)

MODEL_PATH = BEST_INFO["checkpoint"]
if not os.path.exists(MODEL_PATH):
    # best_model.json may have been generated elsewhere (e.g. on Kaggle at
    # /kaggle/working/...) -- fall back to the local checkpoints folder.
    fallback = os.path.join(PROJECT_ROOT, "outputs", "checkpoints", f"{BEST_INFO['name']}_final.keras")
    if os.path.exists(fallback):
        MODEL_PATH = fallback
    else:
        raise FileNotFoundError(
            f"Could not find the model checkpoint. Looked for:\n  {BEST_INFO['checkpoint']}\n  {fallback}\n"
            f"Place the trained '{BEST_INFO['name']}' .keras file in outputs/checkpoints/."
        )

CLASS_NAMES = BEST_INFO["class_names"]
IMG_SIZE = BEST_INFO["img_size"]
MODEL_NAME = BEST_INFO["name"]
MODEL_ACCURACY = BEST_INFO.get("accuracy", 0.0)

print(f"Loading best model: {MODEL_NAME} (test accuracy={MODEL_ACCURACY:.3f}) from {MODEL_PATH}")
MODEL = tf.keras.models.load_model(MODEL_PATH)

CLASS_INFO = {
    "Calculus": "Hardened plaque (tartar) buildup on teeth, often near the gumline.",
    "Caries": "Tooth decay / cavities caused by acid-producing bacteria.",
    "Gingivitis": "Early-stage gum inflammation, often with redness and swelling.",
    "Hypodontia": "A developmental condition of one or more missing teeth.",
    "Tooth Discoloration": "Staining or color change of the tooth surface.",
    "ToothDiscoloration": "Staining or color change of the tooth surface.",
    "Ulcers": "Mouth ulcers / sores on the soft tissue of the mouth.",
    "Mouth Ulcer": "Mouth ulcers / sores on the soft tissue of the mouth.",
}

# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def preprocess(image):
    img = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    img = tf.cast(img, tf.float32) / 255.0
    return tf.expand_dims(img, axis=0)


def make_findings_html(class_names, probs, model_name, model_accuracy):
    order = np.argsort(probs)[::-1]
    top_idx = order[0]
    top_class = class_names[top_idx]
    top_conf = probs[top_idx]
    description = CLASS_INFO.get(top_class, "")

    rows = []
    for rank, idx in enumerate(order):
        cls = class_names[idx]
        p = float(probs[idx])
        pct = f"{p*100:.1f}"
        is_top = rank == 0
        bar_color = "var(--chart-amber)" if is_top else "var(--surgical-teal)"
        row_class = "finding-row finding-row--top" if is_top else "finding-row"
        rows.append(f"""
        <div class="{row_class}">
            <span class="finding-rank">{rank+1:02d}</span>
            <span class="finding-name">{cls}</span>
            <div class="finding-bar-track">
                <div class="finding-bar-fill" style="--target-width:{pct}%; background:{bar_color};"></div>
            </div>
            <span class="finding-pct">{pct}%</span>
        </div>
        """)

    html = f"""
    <div class="report-card">
        <div class="report-header">
            <span class="report-eyebrow">FINDING REPORT</span>
            <span class="report-model-tag">{model_name} · {model_accuracy*100:.1f}% test accuracy</span>
        </div>
        <div class="report-primary">
            <span class="report-primary-label">Primary finding</span>
            <span class="report-primary-value">{top_class}</span>
            <span class="report-primary-conf">{top_conf*100:.1f}% confidence</span>
        </div>
        <p class="report-description">{description}</p>
        <div class="report-divider"></div>
        <span class="report-eyebrow">RANKED DIFFERENTIAL</span>
        <div class="findings-list">
            {"".join(rows)}
        </div>
    </div>
    """
    return html


EMPTY_STATE_HTML = """
<div class="report-card report-card--empty">
    <span class="report-eyebrow">FINDING REPORT</span>
    <p class="empty-state">Upload an intraoral image and select <b>Analyze image</b> to
    generate a finding report.</p>
</div>
"""


def predict(image):
    if image is None:
        return EMPTY_STATE_HTML
    x = preprocess(image)
    probs = MODEL.predict(x, verbose=0)[0]
    return make_findings_html(CLASS_NAMES, probs, MODEL_NAME, MODEL_ACCURACY)


# ---------------------------------------------------------------------------
# Design tokens & CSS
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@500&display=swap');

:root {
    --clinical-navy: #0B2027;
    --surgical-teal: #1B6E6E;
    --surgical-teal-light: #2C8C8C;
    --enamel-white: #F7F9F8;
    --chart-amber: #E8A33D;
    --slate-ink: #23302F;
    --mist-teal: #DCEAE8;
    --line: #C7D9D6;
}

.gradio-container {
    background: var(--enamel-white) !important;
    font-family: 'Inter', ui-sans-serif, system-ui !important;
    color: var(--slate-ink) !important;
}
footer {visibility: hidden}

/* ---------- Hero ---------- */
#hero {
    background: linear-gradient(180deg, var(--clinical-navy) 0%, #123339 100%);
    border-radius: 20px;
    padding: 36px 40px 30px 40px;
    margin-bottom: 22px;
    position: relative;
    overflow: hidden;
}
#hero::before {
    content: "";
    position: absolute; inset: 0;
    background-image: repeating-linear-gradient(90deg, rgba(255,255,255,0.035) 0 1px, transparent 1px 48px);
    pointer-events: none;
}
#hero .eyebrow {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px;
    letter-spacing: 0.12em;
    color: var(--chart-amber);
    text-transform: uppercase;
}
#hero h1 {
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700;
    font-size: 34px;
    color: var(--enamel-white);
    margin: 8px 0 6px 0;
    letter-spacing: -0.01em;
}
#hero p {
    color: #A9C4C2;
    font-size: 15px;
    max-width: 640px;
    line-height: 1.5;
    margin: 0;
}
#hero .model-badge {
    display: inline-flex; align-items: center; gap: 8px;
    margin-top: 16px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12.5px;
    color: var(--enamel-white);
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.15);
    padding: 6px 14px;
    border-radius: 999px;
}
#hero .model-badge .dot {
    width: 7px; height: 7px; border-radius: 50%;
    background: #4ADE80;
    box-shadow: 0 0 6px #4ADE80;
}

/* ---------- Upload card ---------- */
.upload-card {
    background: white;
    border: 1px solid var(--line);
    border-radius: 16px;
    padding: 18px;
}
.upload-card .card-title {
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700;
    font-size: 15px;
    color: var(--clinical-navy);
    margin-bottom: 2px;
}
.upload-card .card-sub {
    font-size: 12.5px;
    color: #5C7472;
    margin-bottom: 12px;
}

.disclaimer {
    background: #FDF4E3;
    border: 1px solid #F0CE8F;
    border-left: 3px solid var(--chart-amber);
    border-radius: 10px;
    padding: 12px 14px;
    font-size: 12.5px;
    color: #6B4E1D;
    margin-top: 14px;
    line-height: 1.5;
}

/* ---------- Findings report ---------- */
.report-card {
    background: white;
    border: 1px solid var(--line);
    border-radius: 16px;
    padding: 22px 22px 8px 22px;
    height: 100%;
}
.report-card--empty { display: flex; flex-direction: column; }
.empty-state { color: #7C918F; font-size: 13.5px; margin-top: 30px; }

.report-eyebrow {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11.5px;
    letter-spacing: 0.1em;
    color: var(--surgical-teal);
    text-transform: uppercase;
}
.report-header {
    display: flex; justify-content: space-between; align-items: baseline;
    margin-bottom: 18px;
    flex-wrap: wrap;
    gap: 6px;
}
.report-model-tag {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    color: #7C918F;
}

.report-primary {
    display: flex; flex-direction: column; gap: 2px;
    padding: 14px 16px;
    background: var(--mist-teal);
    border-radius: 12px;
    margin-bottom: 10px;
}
.report-primary-label {
    font-size: 11.5px; color: #4B6664; text-transform: uppercase; letter-spacing: 0.06em;
}
.report-primary-value {
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700;
    font-size: 24px;
    color: var(--clinical-navy);
}
.report-primary-conf {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12.5px;
    color: var(--surgical-teal);
}
.report-description {
    font-size: 13px;
    color: #4B6664;
    line-height: 1.5;
    margin: 4px 2px 14px 2px;
}
.report-divider { height: 1px; background: var(--line); margin: 4px 0 14px 0; }

.findings-list { display: flex; flex-direction: column; gap: 10px; padding-bottom: 18px; }
.finding-row {
    display: grid;
    grid-template-columns: 24px 1fr 2fr 48px;
    align-items: center;
    gap: 10px;
}
.finding-rank {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    color: #A3B8B6;
}
.finding-row--top .finding-rank { color: var(--chart-amber); font-weight: 700; }
.finding-name { font-size: 13px; color: var(--slate-ink); font-weight: 500; }
.finding-row--top .finding-name { color: var(--clinical-navy); font-weight: 700; }
.finding-pct {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px;
    color: #5C7472;
    text-align: right;
}
.finding-bar-track {
    height: 7px;
    background: #EAF1F0;
    border-radius: 999px;
    overflow: hidden;
}
.finding-bar-fill {
    height: 100%;
    width: 0%;
    border-radius: 999px;
    animation: grow-bar 900ms cubic-bezier(0.22, 1, 0.36, 1) forwards;
}
@keyframes grow-bar {
    from { width: 0%; }
    to { width: var(--target-width); }
}

/* buttons */
button.primary, .gr-button-primary {
    background: var(--surgical-teal) !important;
    border: none !important;
}
button.primary:hover, .gr-button-primary:hover {
    background: var(--surgical-teal-light) !important;
}
"""

HERO_HTML = f"""
<div id="hero">
    <span class="eyebrow">AI DIAGNOSTIC ASSIST · SCAN VIEWER</span>
    <h1>Oral Disease Image Classifier</h1>
    <p>Upload an intraoral photograph to generate a ranked differential across
    {len(CLASS_NAMES)} oral disease categories, powered by a fine-tuned {MODEL_NAME} model.</p>
    <div class="model-badge"><span class="dot"></span> MODEL ONLINE — {MODEL_NAME} — {MODEL_ACCURACY*100:.1f}% test accuracy</div>
</div>
"""

THEME = gr.themes.Soft(
    primary_hue="teal", secondary_hue="cyan", neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui"],
)

# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
with gr.Blocks(title="Oral Disease Classifier") as demo:
    gr.HTML(HERO_HTML)

    with gr.Row(equal_height=True):
        with gr.Column(scale=1):
            with gr.Group(elem_classes=["upload-card"]):
                gr.HTML(
                    '<div class="card-title">Scan viewer</div>'
                    '<div class="card-sub">JPG / PNG · clearer, well-lit intraoral photos work best</div>'
                )
                image_input = gr.Image(type="numpy", label=None, height=300, show_label=False)
                with gr.Row():
                    clear_btn = gr.ClearButton(components=[image_input], value="Clear")
                    submit_btn = gr.Button("Analyze image", variant="primary")
                gr.HTML(
                    '<div class="disclaimer">⚠️ <b>Not a medical device.</b> Educational / research '
                    'demonstration only — not a substitute for professional dental diagnosis. '
                    'Always consult a qualified dentist.</div>'
                )

        with gr.Column(scale=1):
            report_output = gr.HTML(EMPTY_STATE_HTML)

    submit_btn.click(fn=predict, inputs=image_input, outputs=report_output)
    image_input.change(fn=predict, inputs=image_input, outputs=report_output)

    with gr.Accordion("About this model", open=False):
        gr.Markdown(
            f"""
            A custom CNN trained from scratch and 3 fine-tuned pretrained backbones were
            trained and compared on this dataset; **{MODEL_NAME}** performed best on the
            held-out test set and is the model deployed here.
            See `outputs/reports/model_comparison.csv` and `outputs/figures/` for the full
            comparative analysis (hyperparameter search results, confusion matrices, and
            training curves for every model).
            """
        )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, theme=THEME, css=CUSTOM_CSS)
