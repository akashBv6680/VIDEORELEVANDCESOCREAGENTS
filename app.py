import streamlit as st
import requests
import numpy as np
import plotly.graph_objects as go
from youtube_transcript_api import YouTubeTranscriptApi
import re

# --- UPDATED GEMINI CONFIGURATION ---
# Use the recommended, active model: gemini-2.5-flash
GEMINI_MODEL = "gemini-2.5-flash"
# Use the stable API version
GEMINI_API_VERSION = "v1"
# --- END UPDATED CONFIGURATION ---

def fetch_youtube_transcript(video_url):
    """Fetches the English transcript for a given YouTube URL."""
    try:
        # Regex to extract the video ID from various YouTube URL formats
        match = re.search(r"(?:v=|be/|embed/)([\w-]+)", video_url)
        video_id = match.group(1) if match else video_url.split("v=")[-1]
        
        # Get English transcript
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
        
        full_text = " ".join([seg['text'] for seg in transcript])
        segments = transcript
        timestamps = [seg['start'] for seg in transcript]
        return full_text, segments, timestamps
    except Exception as e:
        return f"Could not extract transcript: {e}", [], []

def call_gemini(prompt, api_key):
    """Calls the Gemini API to get content generation."""
    GEMINI_API_URL = f"https://generativelanguage.googleapis.com/{GEMINI_API_VERSION}/models/{GEMINI_MODEL}:generateContent"
    headers = {"Content-Type": "application/json"}
    data = {
        "contents": [
            {"parts": [{"text": prompt}]}
        ]
    }
    # Note: Using the API Key directly in the URL query parameter for simplicity 
    # in this Streamlit example, though it's often preferred in the header.
    endpoint = f"{GEMINI_API_URL}?key={api_key}"
    
    try:
        res = requests.post(endpoint, headers=headers, json=data)
        res.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        output = res.json()
        
        # Safely extract the text response
        return output["candidates"][0]["content"]["parts"][0]["text"]
    except requests.exceptions.HTTPError as http_err:
        return f"Gemini API HTTP Error: {http_err} - {res.text}"
    except Exception as e:
        return f"Gemini API Error or parsing issue: {e}"

def semantic_similarity(text1, text2, api_key):
    """Uses Gemini to rate semantic similarity between two texts (0.0 to 1.0)."""
    prompt = (
        f"From 0 (no match) to 1 (perfect match), how semantically similar are these?\n"
        f"Text 1: {text1}\nText 2: {text2}\n"
        f"Respond with one float number only."
    )
    response = call_gemini(prompt, api_key)
    
    try:
        # Extract first float in response using regex
        m = re.search(r"([0-9]*\.?[0-9]+)", response)
        score = float(m.group(1)) if m else 0
        return max(0.0, min(1.0, score)) # Clamp score between 0.0 and 1.0
    except Exception:
        # Fallback if parsing fails (e.g., if API error message is returned)
        return 0.0

def segment_scores(transcript_segments, title, api_key):
    """Calculates relevance score for each segment against the video title."""
    scores = []
    for seg in transcript_segments:
        # Use the video title for relevance scoring
        score = semantic_similarity(title, seg['text'], api_key)
        scores.append(score)
    return scores

def detect_promotional_segments(transcript, api_key):
    """Uses Gemini to identify promotional or off-topic segments."""
    prompt = (
        "Identify any promotional, off-topic, or filler content in this transcript. "
        "List sentences or phrases. If nothing found, say 'None'.\n\nTranscript:\n" + transcript
    )
    return call_gemini(prompt, api_key)

# --- STREAMLIT APP LAYOUT ---

st.set_page_config(page_title="Video Relevance Scorer+ (Gemini)", layout="wide")
st.title("🎬 Video Relevance Scorer+ (Gemini 2.5 Flash)")

with st.form("video_form"):
    st.markdown("### 1. Input Video Details")
    input_mode = st.radio("Input Mode", ["YouTube URL", "Manual Transcript"], index=0)
    video_title = st.text_input("Video Title", help="The title is used to score relevance against the transcript.")
    video_description = st.text_area("Video Description (optional)", help="Used for contextual justification.")
    
    transcript = ""
    transcript_segments = []
    timestamps = []

    if input_mode == "Manual Transcript":
        transcript = st.text_area("Video Transcript (Copy/Paste)", height=250)
        if transcript:
            # Create dummy segments and timestamps for manual input
            transcript_segments = [{"start": i*10, "text": t} for i, t in enumerate(transcript.split('\n')) if t.strip()]
            timestamps = [seg['start'] for seg in transcript_segments]
    else:
        yt_url = st.text_input("YouTube Video URL", placeholder="e.g., https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        if yt_url:
            with st.spinner("Fetching transcript..."):
                transcript, transcript_segments, timestamps = fetch_youtube_transcript(yt_url)
            
            if transcript_segments:
                st.markdown("✅ Extracted YouTube Transcript (English) - First 500 chars:")
                st.code(transcript[:500] + "...")
            else:
                st.error(transcript) # Show error message if fetch failed
                transcript = "" # Clear transcript to prevent submission

    st.markdown("### 2. Enter API Key")
    # Priority: st.secrets > text_input
    gemini_api_key = st.secrets.get("GEMINI_API_KEY", st.text_input("Enter your Gemini API Key:", type="password", help="The key will not be stored."))
    
    submitted = st.form_submit_button("🚀 Evaluate Video")

# --- ANALYSIS SECTION ---
if submitted and transcript and video_title and gemini_api_key:
    api_key = gemini_api_key
    
    if not api_key.startswith("AIza"):
        st.error("Invalid API Key format. Please ensure you've entered a valid Gemini API Key.")
    else:
        st.header("📊 Relevance Analysis Results")

        # 1. Overall Score
        with st.spinner("Calculating semantic relevance scores..."):
            scores = segment_scores(transcript_segments, video_title, api_key)
            overall_score = np.mean(scores) * 100 if scores else 0
        
        st.subheader("Overall Semantic Relevance Score")
        st.metric("Relevance Score (%)", f"{overall_score:.1f} %")
        st.progress(overall_score / 100)
        
        st.markdown("---")

        # 2. Heatmap
        st.subheader("Segment-wise Relevance Heatmap")
        if scores:
            fig = go.Figure(data=go.Heatmap(
                z=[scores],
                x=[round(s, 2) for s in timestamps], # Use actual timestamps for X-axis
                y=["Relevance to Title"],
                hovertext=[f"Time: {round(timestamps[i], 1)}s, Score: {round(scores[i], 2)}, Text: {transcript_segments[i]['text'][:50]}..." for i in range(len(scores))],
                coloraxis="coloraxis"
            ))
            fig.update_layout(
                coloraxis={'colorscale':'Viridis', 'cmin': 0, 'cmax': 1, 'colorbar': {'title': 'Score (0-1)'}},
                xaxis_title="Timestamp (seconds)",
                yaxis_title="",
                title="Relevance Score Across Video Segments"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Could not calculate segment scores.")
            
        st.markdown("---")

        # 3. Promotional/Irrelevant Segments
        st.subheader("🚨 Flagged Promotional/Irrelevant Segments")
        with st.spinner("Detecting promotional content with Gemini..."):
            flagged = detect_promotional_segments(transcript, api_key)
        st.markdown(f"**Gemini AI Response:**")
        st.info(flagged)
        
        st.markdown("---")

        # 4. Justification/Explanation
        st.subheader("💡 Justification/Explanation")
        justify_prompt = (
            f"Review the following video transcript based on title '{video_title}'"
            + (f" and description '{video_description}'. " if video_description else ". ")
            + f"The video had an overall relevance score of {overall_score:.1f}%. "
            + "Give a short, detailed justification (3-4 lines) for this relevance assessment. "
            "Specifically mention why the score is high or low and, if applicable, comment on the structure of the video based on the transcript.\n\nTranscript:\n" + transcript
        )
        with st.spinner("Generating justification..."):
            explanation = call_gemini(justify_prompt, api_key)
        st.info(f"**Justification:** {explanation}")
