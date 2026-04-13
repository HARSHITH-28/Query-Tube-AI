import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import gradio as gr


# =====================================
# Load Video Index
# =====================================

print("Loading video index...")

df = pd.read_parquet("data/video_index.parquet")

embedding_cols = [c for c in df.columns if c.startswith("emb_")]
video_embeddings = df[embedding_cols].values

print("Video index loaded:", df.shape)


# =====================================
# Load Embedding Model
# =====================================

print("Loading embedding model...")

model = SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")

print("Model ready")


# =====================================
# Search Function
# =====================================

def search_videos(query, top_k, threshold):

    if query.strip() == "":
        return """
        <div class="empty-state">
            <div class="empty-icon">🔭</div>
            <p>Enter a query to begin your search across the cosmos.</p>
        </div>
        """

    query_embedding = model.encode([query])
    similarities = cosine_similarity(query_embedding, video_embeddings)[0]
    sorted_indices = np.argsort(similarities)[::-1]

    results_html = '<div class="results-grid">'
    count = 0

    for idx in sorted_indices[:top_k]:
        score = similarities[idx]
        if score >= threshold:
            count += 1
            video_id = df.iloc[idx]["video_id"]
            title = df.iloc[idx]["title"]
            youtube_link = f"https://www.youtube.com/watch?v={video_id}"
            thumbnail = f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg"

            # Score color: cool blue → warm gold based on score
            pct = int(score * 100)
            if score >= 0.7:
                score_class = "score-high"
            elif score >= 0.5:
                score_class = "score-mid"
            else:
                score_class = "score-low"

            results_html += f"""
<div class="result-card" style="animation-delay:{(count-1)*0.08}s">
    <a href="{youtube_link}" target="_blank" class="thumb-wrap">
        <img src="{thumbnail}" class="thumb-img" alt="{title}" />
        <div class="thumb-overlay">
            <span class="play-btn">▶</span>
        </div>
    </a>
    <div class="card-body">
        <div class="score-badge {score_class}">
            <span class="score-dot"></span>
            {pct}% match
        </div>
        <h3 class="card-title">{title}</h3>
        <a href="{youtube_link}" target="_blank" class="watch-link">
            Watch on YouTube →
        </a>
    </div>
</div>
"""

    results_html += "</div>"

    if count == 0:
        results_html = """
        <div class="empty-state">
            <div class="empty-icon">🌌</div>
            <p>No signals found in this frequency. Try lowering the threshold or changing your query.</p>
        </div>
        """

    return results_html


# =====================================
# Toggle Advanced Settings
# =====================================

def toggle_settings(visible):
    return gr.update(visible=not visible), not visible


# =====================================
# Custom CSS Styling — Space Noir
# =====================================

custom_css = """

@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

/* ===============================
   Root Variables
=============================== */

:root {
    --void: #03030a;
    --deep: #070714;
    --panel: rgba(10, 10, 30, 0.75);
    --rim: rgba(120, 160, 255, 0.12);
    --rim-bright: rgba(120, 160, 255, 0.35);
    --nebula-1: #4f8fff;
    --nebula-2: #a855f7;
    --nebula-3: #06b6d4;
    --gold: #f5c518;
    --text: #dde4ff;
    --muted: rgba(200, 210, 255, 0.45);
    --font-display: 'Syne', sans-serif;
    --font-body: 'DM Sans', sans-serif;
    --radius: 16px;
}


/* ===============================
   Global Reset & Base
=============================== */

*, *::before, *::after {
    box-sizing: border-box;
    margin: 0;
    padding: 0;
}

html, body, .gradio-container {
    background: var(--void) !important;
    color: var(--text) !important;
    font-family: var(--font-body) !important;
    min-height: 100vh;
}


/* ===============================
   Starfield Canvas Background
=============================== */

body::before {
    content: '';
    position: fixed;
    inset: 0;
    z-index: 0;
    background:
        radial-gradient(ellipse 80% 60% at 20% 10%, rgba(79,143,255,0.18) 0%, transparent 65%),
        radial-gradient(ellipse 60% 50% at 80% 80%, rgba(168,85,247,0.15) 0%, transparent 60%),
        radial-gradient(ellipse 40% 40% at 60% 30%, rgba(6,182,212,0.10) 0%, transparent 55%);
    pointer-events: none;
    animation: nebulaShift 28s ease-in-out infinite alternate;
}

@keyframes nebulaShift {
    0%   { opacity: 1; transform: scale(1) rotate(0deg); }
    100% { opacity: 0.7; transform: scale(1.08) rotate(2deg); }
}

/* Star dots layer */
body::after {
    content: '';
    position: fixed;
    inset: 0;
    z-index: 0;
    background-image:
        radial-gradient(1px 1px at 15% 22%, rgba(255,255,255,0.9) 0%, transparent 100%),
        radial-gradient(1.5px 1.5px at 42% 8%, rgba(255,255,255,0.7) 0%, transparent 100%),
        radial-gradient(1px 1px at 70% 35%, rgba(255,255,255,0.8) 0%, transparent 100%),
        radial-gradient(2px 2px at 88% 12%, rgba(160,200,255,0.9) 0%, transparent 100%),
        radial-gradient(1px 1px at 5% 55%, rgba(255,255,255,0.6) 0%, transparent 100%),
        radial-gradient(1.5px 1.5px at 33% 65%, rgba(255,255,255,0.5) 0%, transparent 100%),
        radial-gradient(1px 1px at 57% 78%, rgba(255,255,255,0.7) 0%, transparent 100%),
        radial-gradient(2px 2px at 92% 60%, rgba(200,180,255,0.8) 0%, transparent 100%),
        radial-gradient(1px 1px at 78% 90%, rgba(255,255,255,0.6) 0%, transparent 100%),
        radial-gradient(1.5px 1.5px at 22% 88%, rgba(255,255,255,0.5) 0%, transparent 100%),
        radial-gradient(1px 1px at 48% 45%, rgba(255,255,255,0.4) 0%, transparent 100%),
        radial-gradient(1px 1px at 65% 55%, rgba(180,220,255,0.6) 0%, transparent 100%),
        radial-gradient(2px 2px at 10% 75%, rgba(255,255,255,0.5) 0%, transparent 100%),
        radial-gradient(1px 1px at 38% 30%, rgba(255,255,255,0.7) 0%, transparent 100%),
        radial-gradient(1.5px 1.5px at 82% 42%, rgba(255,255,255,0.6) 0%, transparent 100%);
    pointer-events: none;
    animation: starTwinkle 6s ease-in-out infinite alternate;
}

@keyframes starTwinkle {
    0%   { opacity: 0.6; }
    100% { opacity: 1.0; }
}


/* ===============================
   Layout Wrapper
=============================== */

html, body {
    width: 100% !important;
    min-height: 100vh !important;
    overflow-x: hidden !important;
}

/* Reset all Gradio internal layout wrappers */
.gradio-container,
.gradio-container > .main,
.gradio-container > .main > .wrap,
.app,
.app.svelte-182fdeq,
body > .gradio-container {
    width: 100% !important;
    max-width: 100% !important;
    margin: 0 !important;
    padding: 0 !important;
    left: 0 !important;
    right: 0 !important;
    box-sizing: border-box !important;
}

/* Center just the content column inside */
.gradio-container .contain,
.gradio-container > .main > .wrap > .contain,
.contain {
    width: 75vw !important;
    max-width: 75vw !important;
    margin-left: auto !important;
    margin-right: auto !important;
    padding: 20px 28px !important;
    box-sizing: border-box !important;
    position: relative !important;
    z-index: 1 !important;
}


/* ===============================
   Hero Header
=============================== */

.hero-wrap {
    text-align: center;
    margin-bottom: 52px;
    position: relative;
}

.hero-eyebrow {
    display: inline-block;
    font-family: var(--font-display);
    font-size: 33px;
    font-weight: 600;
    letter-spacing: 4px;
    text-transform: uppercase;
    color: var(--nebula-3);
    border: 1px solid rgba(6,182,212,0.35);
    border-radius: 100px;
    padding: 5px 20px;
    margin-bottom: 2.6em;
    background: rgba(6,182,212,0.06);
}

.hero-title {
    display: none !important;
}

.hero-sub {
    font-size: 16px !important;
    color: var(--muted) !important;
    letter-spacing: 0.2px !important;
    max-width: 480px;
    margin: 0 auto !important;
    margin-top: 0 !important;
    line-height: 1.6;
    padding-top: 0 !important;
}

/* Glowing orb behind hero */
.hero-wrap::before {
    content: '';
    position: absolute;
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, rgba(79,143,255,0.12) 0%, transparent 70%);
    top: 50%;
    left: 50%;
    transform: translate(-50%, -60%);
    pointer-events: none;
    z-index: -1;
    animation: orbPulse 5s ease-in-out infinite alternate;
}

@keyframes orbPulse {
    from { transform: translate(-50%, -60%) scale(1); opacity: 0.6; }
    to   { transform: translate(-50%, -60%) scale(1.15); opacity: 1; }
}


/* ===============================
   Search Panel
=============================== */

.search-panel {
    background: var(--panel);
    border: 1px solid var(--rim);
    border-radius: 24px;
    padding: 28px 32px;
    backdrop-filter: blur(24px);
    -webkit-backdrop-filter: blur(24px);
    box-shadow: 0 0 60px rgba(79,143,255,0.07), 0 2px 4px rgba(0,0,0,0.4);
    margin-bottom: 20px;
    position: relative;
    overflow: hidden;
}

.search-panel::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--rim-bright), transparent);
}


/* ===============================
   Textbox / Label overrides
=============================== */

/* Custom animated Search Query label */
.search-query-label {
    font-family: var(--font-display) !important;
    font-size: 36px !important;
    font-weight: 700 !important;
    letter-spacing: 5px !important;
    text-transform: uppercase !important;
    text-align: center !important;
    display: block !important;
    width: 100% !important;
    margin-bottom: 16px !important;
    animation: labelPulse 3s ease-in-out infinite !important;
}

@keyframes labelPulse {
    0%   { color: #4f8fff; text-shadow: 0 0 20px rgba(79,143,255,0.8); }
    33%  { color: #a855f7; text-shadow: 0 0 20px rgba(168,85,247,0.8); }
    66%  { color: #06b6d4; text-shadow: 0 0 20px rgba(6,182,212,0.8); }
    100% { color: #4f8fff; text-shadow: 0 0 20px rgba(79,143,255,0.8); }
}

/* Remove only the textbox outer wrapper border — NOT all blocks (that breaks dropdown) */
div[data-testid="textbox"],
div[data-testid="textbox"] > div,
div[data-testid="textbox"] > label + div {
    border: none !important;
    box-shadow: none !important;
    background: transparent !important;
    padding: 0 !important;
    outline: none !important;
}

/* Hide ALL Gradio-generated label text, spans, and the "textbox" aria text inside the textbox component */
div[data-testid="textbox"] label,
div[data-testid="textbox"] label span,
div[data-testid="textbox"] > span,
div[data-testid="textbox"] > div > span:first-child,
.gr-textbox > label,
span[data-testid="block-label"] {
    display: none !important;
    visibility: hidden !important;
    width: 0 !important;
    height: 0 !important;
    overflow: hidden !important;
    position: absolute !important;
}

textarea, input[type="text"] {
    font-family: var(--font-body) !important;
    font-size: 16px !important;
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid var(--rim-bright) !important;
    border-radius: 14px !important;
    color: var(--text) !important;
    padding: 16px 20px !important;
    transition: border-color 0.25s ease, box-shadow 0.25s ease !important;
    letter-spacing: 0.2px !important;
    text-align: center !important;
}

textarea::placeholder, input[type="text"]::placeholder {
    color: rgba(180, 200, 255, 0.28) !important;
}

textarea:focus, input[type="text"]:focus {
    border-color: var(--nebula-1) !important;
    box-shadow: 0 0 0 3px rgba(79,143,255,0.15), 0 0 20px rgba(79,143,255,0.1) !important;
    outline: none !important;
}


/* ===============================
   Buttons
=============================== */

button.primary, button[data-testid="search-btn"] {
    font-family: var(--font-display) !important;
    font-size: 14px !important;
    font-weight: 700 !important;
    letter-spacing: 1px !important;
    text-transform: uppercase !important;
    background: linear-gradient(135deg, var(--nebula-1) 0%, var(--nebula-2) 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 14px !important;
    padding: 14px 32px !important;
    cursor: pointer !important;
    position: relative !important;
    overflow: hidden !important;
    transition: transform 0.2s ease, box-shadow 0.2s ease !important;
}

button.primary::after {
    content: '';
    position: absolute;
    inset: 0;
    background: linear-gradient(135deg, rgba(255,255,255,0.15) 0%, transparent 60%);
    pointer-events: none;
}

button.primary:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 28px rgba(79,143,255,0.4) !important;
}

button.secondary {
    font-family: var(--font-display) !important;
    font-size: 14px !important;
    font-weight: 700 !important;
    letter-spacing: 1px !important;
    text-transform: uppercase !important;
    background: linear-gradient(135deg, var(--nebula-1) 0%, var(--nebula-2) 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 14px !important;
    padding: 14px 32px !important;
    cursor: pointer !important;
    position: relative !important;
    overflow: hidden !important;
    transition: transform 0.2s ease, box-shadow 0.2s ease !important;
    text-align: center !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
}

button.secondary::after {
    content: '';
    position: absolute;
    inset: 0;
    background: linear-gradient(135deg, rgba(255,255,255,0.15) 0%, transparent 60%);
    pointer-events: none;
}

button.secondary:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 28px rgba(79,143,255,0.4) !important;
}


/* ===============================
   Advanced Settings Panel
=============================== */

.advanced-panel {
    background: rgba(10, 10, 30, 0.6) !important;
    border: 1px solid var(--rim) !important;
    border-radius: 18px !important;
    padding: 24px 28px !important;
    margin-top: 16px !important;
    backdrop-filter: blur(16px) !important;
    animation: slideDown 0.3s ease;
}

/* Proper spacing between top_k and threshold */
.advanced-panel .gr-row,
.advanced-panel [class*="row"] {
    gap: 32px !important;
    display: flex !important;
    align-items: flex-start !important;
}

@keyframes slideDown {
    from { opacity: 0; transform: translateY(-8px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* Dropdown trigger styling */
select, .dropdown {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid var(--rim-bright) !important;
    border-radius: 12px !important;
    color: var(--text) !important;
    font-family: var(--font-body) !important;
    padding: 10px 16px !important;
}

/* Force dropdown list to always open downward and be visible */
div[data-testid="dropdown"] ul,
div[data-testid="dropdown"] .options,
div[data-testid="dropdown"] [role="listbox"] {
    top: 100% !important;
    bottom: auto !important;
    transform: none !important;
    background: #0d0d2b !important;
    border: 1px solid var(--rim-bright) !important;
    border-radius: 12px !important;
    box-shadow: 0 8px 32px rgba(0,0,0,0.6) !important;
    z-index: 9999 !important;
    max-height: 200px !important;
    overflow-y: auto !important;
    color: var(--text) !important;
}

/* Dropdown option items */
div[data-testid="dropdown"] ul li,
div[data-testid="dropdown"] [role="option"] {
    color: var(--text) !important;
    padding: 8px 16px !important;
    cursor: pointer !important;
    background: transparent !important;
}

div[data-testid="dropdown"] ul li:hover,
div[data-testid="dropdown"] [role="option"]:hover {
    background: rgba(79,143,255,0.15) !important;
    color: var(--nebula-1) !important;
}

/* Ensure dropdown container allows overflow for the popup */
div[data-testid="dropdown"] {
    position: relative !important;
    overflow: visible !important;
}

/* Slider track */
input[type="range"] {
    accent-color: var(--nebula-1) !important;
    background: transparent !important;
}


/* ===============================
   Results Grid
=============================== */

.results-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
    gap: 24px;
    margin-top: 8px;
}


/* ===============================
   Result Cards
=============================== */

.result-card {
    background: var(--panel);
    border: 1px solid var(--rim);
    border-radius: var(--radius);
    overflow: hidden;
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    transition: transform 0.3s cubic-bezier(0.34, 1.56, 0.64, 1), box-shadow 0.3s ease;
    animation: cardRise 0.5s cubic-bezier(0.22, 1, 0.36, 1) both;
    position: relative;
}

@keyframes cardRise {
    from { opacity: 0; transform: translateY(22px) scale(0.97); }
    to   { opacity: 1; transform: translateY(0) scale(1); }
}

.result-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(120,180,255,0.4), transparent);
}

.result-card:hover {
    transform: translateY(-6px) scale(1.01);
    box-shadow: 0 20px 50px rgba(0,0,0,0.5), 0 0 30px rgba(79,143,255,0.12);
    border-color: var(--rim-bright);
}


/* ===============================
   Thumbnail
=============================== */

.thumb-wrap {
    display: block;
    position: relative;
    overflow: hidden;
    aspect-ratio: 16/9;
}

.thumb-img {
    width: 100%;
    height: 100%;
    object-fit: cover;
    display: block;
    transition: transform 0.45s ease;
}

.thumb-overlay {
    position: absolute;
    inset: 0;
    background: linear-gradient(135deg, rgba(79,143,255,0.2), rgba(168,85,247,0.2));
    opacity: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    transition: opacity 0.3s ease;
}

.thumb-wrap:hover .thumb-img { transform: scale(1.07); }
.thumb-wrap:hover .thumb-overlay { opacity: 1; }

.play-btn {
    width: 52px;
    height: 52px;
    border-radius: 50%;
    background: rgba(255,255,255,0.95);
    color: var(--void);
    font-size: 18px;
    display: flex;
    align-items: center;
    justify-content: center;
    padding-left: 3px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.4);
    transition: transform 0.2s ease;
}

.thumb-wrap:hover .play-btn { transform: scale(1.1); }


/* ===============================
   Card Body
=============================== */

.card-body {
    padding: 16px 18px 20px;
}

.score-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-family: var(--font-display);
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1px;
    text-transform: uppercase;
    border-radius: 100px;
    padding: 4px 10px;
    margin-bottom: 10px;
}

.score-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    animation: dotPulse 2s ease-in-out infinite;
}

@keyframes dotPulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50%       { opacity: 0.5; transform: scale(0.7); }
}

.score-high {
    background: rgba(79,255,160,0.12);
    color: #4fff9a;
    border: 1px solid rgba(79,255,160,0.25);
}
.score-high .score-dot { background: #4fff9a; }

.score-mid {
    background: rgba(245,197,24,0.12);
    color: var(--gold);
    border: 1px solid rgba(245,197,24,0.25);
}
.score-mid .score-dot { background: var(--gold); }

.score-low {
    background: rgba(79,143,255,0.12);
    color: var(--nebula-1);
    border: 1px solid rgba(79,143,255,0.25);
}
.score-low .score-dot { background: var(--nebula-1); }

.card-title {
    font-family: var(--font-display);
    font-size: 14px;
    font-weight: 600;
    color: var(--text);
    line-height: 1.45;
    margin-bottom: 12px;
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
    overflow: hidden;
}

.watch-link {
    font-family: var(--font-display);
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 0.5px;
    color: var(--nebula-1);
    text-decoration: none;
    opacity: 0.75;
    transition: opacity 0.2s;
}

.watch-link:hover {
    opacity: 1;
    text-decoration: underline;
}


/* ===============================
   Empty State
=============================== */

.empty-state {
    text-align: center;
    padding: 64px 24px;
    color: var(--muted);
}

.empty-icon {
    font-size: 52px;
    margin-bottom: 16px;
    animation: floatIcon 4s ease-in-out infinite;
}

@keyframes floatIcon {
    0%, 100% { transform: translateY(0); }
    50%       { transform: translateY(-10px); }
}

.empty-state p {
    font-size: 15px;
    max-width: 360px;
    margin: 0 auto;
    line-height: 1.6;
}


/* ===============================
   Shooting star decoration
=============================== */

.shooting-star {
    position: fixed;
    top: 10%;
    left: -20%;
    width: 180px;
    height: 1px;
    background: linear-gradient(90deg, transparent, white, transparent);
    opacity: 0;
    animation: shoot 10s ease-in-out infinite;
    pointer-events: none;
    z-index: 0;
    transform: rotate(-20deg);
}

.shooting-star:nth-child(2) {
    top: 35%;
    animation-delay: 4s;
    width: 120px;
}

.shooting-star:nth-child(3) {
    top: 65%;
    animation-delay: 7.5s;
    width: 200px;
}

@keyframes shoot {
    0%   { left: -30%; opacity: 0; }
    5%   { opacity: 1; }
    30%  { left: 130%; opacity: 0; }
    100% { left: 130%; opacity: 0; }
}


/* ===============================
   Divider
=============================== */

.cosmos-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--rim-bright), transparent);
    margin: 32px 0;
}


/* ===============================
   Gradio Overrides
=============================== */

.gradio-container .prose h3 {
    display: none !important;
}

footer { display: none !important; }

.svelte-1gfkn6j { background: transparent !important; }

.gap-4 { gap: 16px !important; }

"""


# =====================================
# Shooting Stars HTML (decorative)
# =====================================

shooting_stars_html = """
<div class="shooting-star"></div>
<div class="shooting-star"></div>
<div class="shooting-star"></div>
"""


# =====================================
# Build Gradio Interface
# =====================================

with gr.Blocks(css=custom_css, title="Cosmic Video Search") as demo:

    # Shooting stars overlay
    gr.HTML(shooting_stars_html)

    # ── Hero ──
    gr.HTML("""
    <div class="hero-wrap">
        <div class="hero-eyebrow">✦ Semantic Intelligence</div>
        <div class="hero-sub">
            Navigate the universe of knowledge. Find exactly what you're looking for
            using the power of semantic understanding.
        </div>
        <div class="hero-title">Cosmic Video Search</div>
    </div>
    """)

    # ── Search Panel ──
    with gr.Group(elem_classes="search-panel"):

        gr.HTML("""
        <div class="search-query-label">Search Query</div>
        """)

        query_box = gr.Textbox(
            placeholder="E.g., deep dive into transformer architecture...",
            label=None,
            lines=1,
            max_lines=3,
            show_label=False
        )

        with gr.Row():
            search_btn = gr.Button("⟡  Launch Search", variant="primary", elem_id="search-btn")
            adv_btn    = gr.Button("⚙  Settings", variant="secondary")

        visible_state = gr.State(False)

        with gr.Column(visible=False, elem_classes="advanced-panel") as advanced_panel:
            gr.HTML('<div class="cosmos-divider"></div>')
            with gr.Row():
                top_k = gr.Dropdown(
                    choices=[3, 5, 10],
                    value=5,
                    label="Results (Top-K)"
                )
                threshold = gr.Slider(
                    minimum=0,
                    maximum=1,
                    value=0.3,
                    step=0.05,
                    label="Similarity Threshold"
                )

    # ── Results ──
    results = gr.HTML("""
    <div class="empty-state">
        <div class="empty-icon">🌌</div>
        <p>Your search results will appear here. Chart a course and begin exploring.</p>
    </div>
    """)

    # ── Event Bindings ──
    search_btn.click(
        fn=search_videos,
        inputs=[query_box, top_k, threshold],
        outputs=results
    )

    query_box.submit(
        fn=search_videos,
        inputs=[query_box, top_k, threshold],
        outputs=results
    )

    adv_btn.click(
        fn=toggle_settings,
        inputs=visible_state,
        outputs=[advanced_panel, visible_state]
    )


# =====================================
# Launch Application
# =====================================

if __name__ == "__main__":
    demo.launch()