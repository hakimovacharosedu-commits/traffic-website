"""Team website + live demo (Streamlit). Deploys free on Streamlit Community Cloud.

The demo calls the SAME detect_events / RiskEstimator as the submission (solution.py),
so whatever the team's pipeline detects is exactly what visitors see.
"""
import math
import os
import tempfile

import cv2
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

HERE = os.path.dirname(os.path.abspath(__file__))
TEAM_ZIP = "https://github.com/mohinur2009/traffic_hackathon/archive/refs/heads/main.zip"
WEIGHTS_URL = "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11m.pt"


@st.cache_resource(show_spinner="Loading the team's model (first visit only)…")
def load_pipeline():
    """Fetch the team repo + weights once, so the demo always runs the submitted code."""
    import sys, urllib.request, zipfile, io
    root = os.path.join(tempfile.gettempdir(), "team_code")
    code = os.path.join(root, "traffic_hackathon-main")
    if not os.path.exists(os.path.join(code, "solution.py")):
        data = urllib.request.urlopen(TEAM_ZIP, timeout=60).read()
        zipfile.ZipFile(io.BytesIO(data)).extractall(root)
    weights = os.path.join(code, "weights", "yolo11m.pt")
    if not os.path.exists(weights):
        urllib.request.urlretrieve(WEIGHTS_URL, weights)
    sys.path.insert(0, code)
    import solution as team_solution
    from src import events as team_events
    return team_solution, team_events


solution, team_events = load_pipeline()
DEMO_SAMPLE_S = 0.4   # lighter than the submission (0.2 s) so a CPU finishes in minutes
DEMO_IMGSZ = 960      # submission uses 1280
MAX_SECONDS = 120
MAX_MB = 100
CLASS_ORDER = list(solution.CLASSES)
PALETTE = ["#e6194b", "#f58231", "#ffe119", "#bfef45", "#3cb44b", "#42d4f4", "#4363d8",
           "#911eb4", "#f032e6", "#a9a9a9", "#9a6324", "#800000", "#469990", "#000075"]
COLORS = {c: PALETTE[i % len(PALETTE)] for i, c in enumerate(CLASS_ORDER)}

st.set_page_config(page_title="Traffic Event Detection", page_icon="🚦", layout="wide")


def video_meta(path):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return {"video_id": os.path.basename(path), "fps": fps, "width": w, "height": h,
            "n_frames": n, "duration": n / fps if fps else 0.0}


def timeline_figure(events, duration):
    fig = go.Figure()
    for start, end, label in events:
        fig.add_trace(go.Bar(
            x=[end - start], y=[label], base=[start], orientation="h",
            marker_color=COLORS.get(label, "#888"), showlegend=False,
            hovertemplate=f"{label}<br>{start:.1f}s → {end:.1f}s<extra></extra>"))
    fig.update_layout(title="Event timeline", xaxis_title="seconds",
                      height=140 + 40 * max(1, len({e[2] for e in events})),
                      xaxis_range=[0, max(duration, 1)], margin=dict(l=10, r=10, t=40, b=40))
    if not events:
        fig.add_annotation(text="No events detected", showarrow=False,
                           x=duration / 2, y=0.5, xref="x", yref="paper")
    return fig


def risk_figure(times, scores):
    fig = go.Figure(go.Scatter(x=times, y=scores, mode="lines", line_color="#e6194b"))
    fig.add_hline(y=0.5, line_dash="dash", annotation_text="alarm threshold")
    fig.update_layout(title="Accident risk (next 5 s)", xaxis_title="seconds",
                      yaxis_range=[0, 1], height=280, margin=dict(l=10, r=10, t=40, b=40))
    return fig


def event_thumbnails(path, events, fps, limit=12):
    cap = cv2.VideoCapture(path)
    thumbs = []
    for start, end, label in events[:limit]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int((start + end) / 2 * fps))
        ok, frame = cap.read()
        if ok:
            frame = cv2.resize(frame, (640, int(640 * frame.shape[0] / frame.shape[1])))
            thumbs.append((cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), f"{label} · {start:.1f}–{end:.1f}s"))
    cap.release()
    return thumbs


def analyse(video_path, bar):
    meta = video_meta(video_path)
    if meta["duration"] > MAX_SECONDS + 1:
        st.error(f"Video is {meta['duration']:.0f} s long; the demo accepts up to {MAX_SECONDS} s.")
        return None

    est = solution.RiskEstimator()
    est.reset({k: meta[k] for k in ("video_id", "fps", "width", "height", "n_frames")})
    est.stride, est.IMGSZ = 1, DEMO_IMGSZ
    every = max(1, round(meta["fps"] * DEMO_SAMPLE_S))
    cap = cv2.VideoCapture(video_path)
    times, scores, recs, idx = [], [], [], 0
    while True:
        if idx % every:
            if not cap.grab():
                break
            idx += 1
            continue
        ok, frame = cap.read()
        if not ok:
            break
        t = idx / meta["fps"]
        scores.append(float(est.step(frame, t)))
        times.append(t)
        for o in est.objs:
            recs.append((t, o["id"], o["cls"], math.hypot(*o["v"]), float(o["g"][0]), float(o["g"][1])))
        idx += 1
        bar.progress(min(0.95, idx / max(meta["n_frames"], 1)),
                     text=f"Analysing… {t:.0f} / {meta['duration']:.0f} s")
    cap.release()
    events = sorted(team_events.events_from_records(list(zip(times, scores)), recs, meta["duration"]),
                    key=lambda e: e[0])
    bar.progress(1.0, text="Done")
    return meta, events, times, scores


def page(name):
    path = os.path.join(HERE, "pages_md", name)
    if not os.path.exists(path):
        path = os.path.join(HERE, name)  # also accept the .md next to app.py
    if os.path.exists(path):
        st.markdown(open(path, encoding="utf-8").read())
    else:
        st.info(f"{name} coming soon")


st.title("🚦 Traffic Event Detection")
st.caption("One fixed CCTV camera → every traffic event as a time segment, plus an early accident alarm.")

demo, approach, eda, results, report, team = st.tabs(
    ["Live demo", "Approach", "EDA", "Results", "Report", "Team"])

with demo:
    st.write(f"Upload an **.mp4** from the camera (**max {MAX_SECONDS // 60} min, {MAX_MB} MB**). "
             "It runs on a free CPU, so a 2-minute clip takes several minutes. The demo samples a frame every "
             f"{DEMO_SAMPLE_S} s at {DEMO_IMGSZ} px; the submitted pipeline uses 0.2 s at 1280 px on a GPU.")
    up = st.file_uploader("Video", type=["mp4"])
    if up is not None:
        if up.size > MAX_MB * 1024 * 1024:
            st.error(f"File is larger than {MAX_MB} MB.")
        else:
            st.video(up)
            if st.button("Detect events", type="primary"):
                with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
                    f.write(up.getbuffer())
                    path = f.name
                try:
                    out = analyse(path, st.progress(0.0, text="Starting…"))
                    if out:
                        meta, events, times, scores = out
                        st.success(f"**{len(events)} event(s)** in {meta['duration']:.1f} s of video "
                                   f"({meta['width']}×{meta['height']}, {meta['fps']:.0f} fps).")
                        st.plotly_chart(timeline_figure(events, meta["duration"]), use_container_width=True)
                        st.dataframe(pd.DataFrame(
                            [[round(s, 2), round(e, 2), round(e - s, 2), lab] for s, e, lab in events],
                            columns=["start (s)", "end (s)", "length (s)", "event"]),
                            use_container_width=True, hide_index=True)
                        st.plotly_chart(risk_figure(times, scores), use_container_width=True)
                        thumbs = event_thumbnails(path, events, meta["fps"])
                        if thumbs:
                            st.subheader("What each event looks like")
                            cols = st.columns(3)
                            for k, (img, cap_) in enumerate(thumbs):
                                cols[k % 3].image(img, caption=cap_, use_container_width=True)
                except Exception as e:  # never crash the page on a visitor's upload
                    st.error(f"Something went wrong while processing this video: {e}")
                finally:
                    os.remove(path)

with approach:
    page("approach.md")
    overlay = os.path.join(HERE, "scene_overlay.jpg")
    if os.path.exists(overlay):
        st.image(overlay, caption="Scene map: road, lanes + legal direction, stop line, crosswalks, "
                                  "solid line, signal, queue zone", use_container_width=True)
with eda:
    page("eda.md")
with results:
    page("results.md")
with report:
    page("report.md")
with team:
    page("team.md")
