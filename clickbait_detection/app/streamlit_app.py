from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.predict import predict_one, load_best_model
from src.features import extract_features

st.set_page_config(
    page_title="Clickbait Detector",
    page_icon="🎯",
    layout="centered",
)

st.markdown("""
<style>
.headline-card {
    background: linear-gradient(135deg, #fff8f3 0%, #ffeede 100%);
    border-left: 5px solid #d2503e;
    padding: 1rem 1.2rem;
    border-radius: 6px;
    margin: 1rem 0;
}
.headline-card.safe {
    background: linear-gradient(135deg, #f3f7fc 0%, #e5edf7 100%);
    border-left: 5px solid #3a6ea5;
}
.prob-pill {
    display: inline-block;
    padding: 0.3rem 0.75rem;
    border-radius: 999px;
    font-weight: 600;
    font-size: 0.95rem;
}
.feat-table { font-size: 0.85rem; }
</style>
""", unsafe_allow_html=True)

st.title("🎯 Clickbait Headline Detector")
st.caption("Classify online news headlines as clickbait or non-clickbait. "
           "University project — neural networks (MLP / CNN) trained on the "
           "Chakraborty et al. (2016) corpus.")

with st.sidebar:
    st.subheader("Model in use")
    try:
        bundle = load_best_model()
        st.success(f"**{bundle['name']}**")
        st.caption(f"Decision threshold: {bundle.get('threshold', 0.5):.2f}")
    except FileNotFoundError:
        st.error("No trained model found. Run `python main.py` first.")
        st.stop()

    st.markdown("---")
    st.subheader("Try these examples")
    examples = [
        "You won't believe what happened next",
        "European Central Bank raises rates by 25 basis points",
        "10 habits that can improve your productivity",
        "Researchers identify new variant in samples from Brussels",
        "This one trick will change your morning forever",
        "Federal court rules on appeal in landmark fraud case",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True, key=f"ex_{hash(ex)}"):
            st.session_state["headline_input"] = ex

default_text = st.session_state.get("headline_input", "")
headline = st.text_area(
    "Paste a headline:",
    value=default_text,
    height=80,
    placeholder="e.g. 17 weird things only millennials will understand",
)

col_a, col_b = st.columns([1, 4])
with col_a:
    run = st.button("Classify", type="primary", use_container_width=True)
with col_b:
    clear = st.button("Clear", use_container_width=True)
    if clear:
        st.session_state["headline_input"] = ""
        st.rerun()

if run and headline.strip():
    result = predict_one(headline.strip())
    prob = result["clickbait_probability"]
    is_cb = result["prediction"] == "clickbait"
    card_class = "headline-card" if is_cb else "headline-card safe"
    badge_color = "#d2503e" if is_cb else "#3a6ea5"
    badge_text = "CLICKBAIT" if is_cb else "NON-CLICKBAIT"

    st.markdown(f"""
    <div class="{card_class}">
        <div style="font-style:italic; color:#444; margin-bottom:0.6rem;">
            "{headline.strip()}"
        </div>
        <span class="prob-pill" style="background:{badge_color}22; color:{badge_color};">
            {badge_text}
        </span>
        <span style="margin-left:0.5rem; color:#555;">
            Clickbait probability:
            <strong>{prob:.3f}</strong>
        </span>
    </div>
    """, unsafe_allow_html=True)

    st.progress(min(max(prob, 0.0), 1.0))

    st.subheader("Why this prediction?")
    for reason in result["explanation"]:
        st.markdown(f"- {reason}")

    with st.expander("Show extracted features"):
        feats = extract_features(headline.strip())
        import pandas as pd
        fdf = pd.DataFrame({
            "feature": list(feats.keys()),
            "value": list(feats.values()),
        })
        st.dataframe(fdf, use_container_width=True, hide_index=True)

elif run and not headline.strip():
    st.warning("Please enter a headline.")

st.markdown("---")

