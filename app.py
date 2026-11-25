"""Phishing URL checker web page.   Run:  streamlit run app.py"""
import os

import streamlit as st

from phishing import model

MODEL_PATH = os.environ.get("PHISH_MODEL", os.path.join("models", "model.joblib"))
st.set_page_config(page_title="Phishing URL checker", page_icon="🎣")
st.title("🎣 Phishing URL checker")
st.caption("Paste a link. Only the text of the URL is analysed; the site is never visited.")


@st.cache_resource
def get_model(path: str):
    return model.load(path)


if not os.path.exists(MODEL_PATH):
    st.error("No trained model found. Run `python cli.py train` first, then reload this page.")
    st.stop()

url = st.text_input("URL", placeholder="http://paypal.secure-login.example-site.xyz/verify")
cutoff = st.slider("Sensitivity (lower = flags more links)", 0.1, 0.9, 0.5, 0.05)
if url.strip():
    r = model.check(get_model(MODEL_PATH), url.strip(), cutoff)
    p = r["phishing_probability"]
    (st.error if r["verdict"] == "PHISHING" else st.success)(
        f"**{r['verdict']}** - phishing probability {p:.0%}")
    st.progress(min(1.0, p))
    st.subheader("Why")
    if r["reasons"]:
        for why in r["reasons"]:
            st.write("- " + why)
    else:
        st.write("No red flags found in the URL text.")
    st.caption("A model can be wrong. Treat this as one signal, not a guarantee.")
