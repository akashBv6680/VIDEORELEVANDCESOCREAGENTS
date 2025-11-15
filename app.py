import streamlit as st
import requests
import numpy as np
import plotly.graph_objects as go
from youtube_transcript_api import YouTubeTranscriptApi
import re

# Use actively supported Gemini 2.5 Flash model
GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_API_VERSION = "v1beta"

def fetch_youtube_transcript(video_url):
    try:
        match = re.search(r"(?:v=|be/|embed/)([\w-]+)", video_url)
        video_id = match.group(1) if match else video_url.split("v=")[-1]
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
        full_text = " ".join([seg['text'] for seg in transcript])
        segments = transcript
        timestamps = [seg['start'] for seg in transcript]
        return full_text, segments, timestamps
    except Exception as e:
        return f"Could not extract transcript: {e}", [], []

def call_gemini(prompt, api_key):
    GEMINI_API_URL = f"https://generativelanguage.googleapis.com/{GEMINI_API_VERSION}/models/{GEMINI_MODEL}:generateContent"
    headers = {"Content-Type": "application/json"}
    data = {
        "contents": [
            {"parts": [{"text": prompt}]}
        ]
    }
    endpoint = f"{GEMINI_API_URL}?key={api_key}"
    res = requests.post(endpoint, headers=headers, json=data)
    if res.status_code == 200:
        output = res.json()
        try:
            return output["candidates"][0]["content"]["parts"][0]["text"]
        except Exception:
            return "No response from Gemini 2.5 Flash."
    return f"Gemini API Error: {res.status_code} - {res.text}"

def semantic_similarity(text1, text2, api_key):
    prompt = (
        f"From 0 (no match) to 1 (perfect match), how semantically similar are these?\n"
        f"Text 1: {text1}\nText 2: {text2}\n"
        f"Respond with one float number only."
    )
    response = call_gemini(prompt, api_key)
    try:
        import re
        m = re.search(r"([0-9]*\.?[0-9]+)", response)
        score = float(m.group(1)) if m else 0
        return max(0.0, min(1.0, score))
    except Exception:
        return 0.0

def segment_scores(transcript_segments, title, api_key):
    scores = []
    for seg in transcript_segments:
        score = semantic_similarity(title, seg['text'], api_key)
        scores.append(score)
    return scores

def detect_promotional_segments(transcript, api_key):
    prompt = (
        "Identify any promotional, off-topic, or filler content in this transcript. "
        "List sentences or phrases. If nothing found, say 'None'.\n\nTranscript:\n" + transcript
    )
    return call_gemini(prompt, api_key)

st.set_page_config(page_title="Video Relevance Scorer+ (Gemini 2.5 Flash)", layout="wide")
st.title("🎬 Video Relevance Scorer ")

with st.form("video_form"):
    input_mode = st.radio("Input Mode", ["Manual Transcript", "YouTube URL"])
    video_title = st.text_input("Video Title")
    video_description = st.text_area("Video Description (optional)")
    transcript = ""
    transcript_segments = []
    timestamps = []

    if input_mode == "Manual Transcript":
        transcript = st.text_area("Video Transcript (auto or manual upload)")
        if transcript:
            transcript_segments = [{"start": 0, "text": t} for t in transcript.split('\n') if t.strip()]
            timestamps = [0 for _ in transcript_segments]
    else:
        yt_url = st.text_input("YouTube Video URL")
        if yt_url:
            transcript, transcript_segments, timestamps = fetch_youtube_transcript(yt_url)
            st.markdown("Extracted YouTube Transcript (English):")
            st.code(transcript)
    gemini_api_key = st.secrets.get("GEMINI_API_KEY", st.text_input("Enter your Gemini API Key:", type="password"))
    submitted = st.form_submit_button("Evaluate Video")

if submitted and transcript and video_title and gemini_api_key:
    api_key = gemini_api_key
    st.header("Relevance Analysis")

    st.subheader("Overall Semantic Relevance Score")
    scores = segment_scores(transcript_segments, video_title, api_key)
    overall_score = np.mean(scores) * 100 if scores else 0
    st.metric("Relevance Score (%)", f"{overall_score:.1f} %")

    st.subheader("Plotly Heatmap: Segment-wise Relevance")
    fig = go.Figure(data=go.Heatmap(
        z=[scores],
        x=[round(s["start"],2) if "start" in s else i for i, s in enumerate(transcript_segments)],
        y=["Relevance"],
        coloraxis="coloraxis"
    ))
    fig.update_layout(
        coloraxis={'colorscale':'Viridis'},
        xaxis_title="Timestamp (s)",
        yaxis_title="",
        title="Relevance Heatmap"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Flagged Promotional/Irrelevant Segments")
    flagged = detect_promotional_segments(transcript, api_key)
    st.write(flagged)

    st.subheader("Justification/Explanation")
    justify_prompt = (
        f"Review the following video transcript based on title '{video_title}'"
        + (f" and description '{video_description}'. " if video_description else ". ")
        + "Give a short justification (1-2 lines) for your relevance assessment. "
        "If there are off-topic/promotional sections, briefly mention them.\n\nTranscript:\n" + transcript
    )
    explanation = call_gemini(justify_prompt, api_key)
    st.info(f"**Justification:** {explanation}")
