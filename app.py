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
TEAM_DIR = os.path.dirname(os.path.abspath(solution.__file__))
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


# ---------------------------------------------------------------- results + EDA
import glob
import json

LIGHT = {"C3896": "Daytime", "C3897": "Daytime", "C3902": "Daytime", "C3905": "Evening"}


@st.cache_data
def sample_predictions():
    path = os.path.join(TEAM_DIR, "predictions_samples.json")
    with open(path) as f:
        return json.load(f)["videos"]


def alarms(risk, thr=0.5, merge=2.0):
    runs, start, last = [], None, None
    for t, s in risk:
        if s >= thr:
            start = t if start is None else start
            last = t
        elif start is not None:
            runs.append([start, last])
            start = None
    if start is not None:
        runs.append([start, last])
    out = []
    for a, b in runs:
        if out and a - out[-1][1] < merge:
            out[-1][1] = b
        else:
            out.append([a, b])
    return out


def results_page():
    vids = sample_predictions()
    st.markdown("## Results on the sample videos")
    st.markdown("Output of the submitted pipeline (`v1.0`) on the four organiser videos, taken directly from "
                "[`predictions_samples.json`](https://github.com/mohinur2009/traffic_hackathon/blob/v1.0/predictions_samples.json). "
                "No ground-truth labels exist for these videos, so every event and alarm below was **checked by eye**.")

    rows = []
    for name, v in vids.items():
        dur = v["risk"][-1][0] if v["risk"] else 0
        al = alarms(v["risk"])
        rows.append([name.split(".")[0], f"{int(dur // 60)}:{int(dur % 60):02d}", LIGHT.get(name.split(".")[0], "—"),
                     len(v["events"]), len(al), round(len(al) / max(dur / 60, 1e-9), 1)])
    st.dataframe(pd.DataFrame(rows, columns=["video", "length", "light", "events", "risk alarms", "alarms / min"]),
                 hide_index=True, use_container_width=True)
    st.caption("Rules were tuned on C3905 only and then frozen, so the other three videos are an unbiased check "
               "that the alarm rate stays low on footage the rules never saw.")

    st.markdown("### Timeline and risk curve per video")
    pick = st.selectbox("Video", list(vids), format_func=lambda n: n.split(".")[0])
    v = vids[pick]
    dur = v["risk"][-1][0] if v["risk"] else 1
    st.plotly_chart(timeline_figure(sorted(v["events"]), dur), use_container_width=True)
    risk = v["risk"][::3]
    fig = risk_figure([r[0] for r in risk], [r[1] for r in risk])
    for a, b in alarms(v["risk"]):
        fig.add_vrect(x0=a, x1=b + 0.3, fillcolor="#e6194b", opacity=0.15, line_width=0)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Shaded bands = alarms (score ≥ 0.5, runs < 2 s apart merged, as in the official metric).")

    st.markdown("""
### Every detected event, explained
| Video | Event | Start (s) | End (s) | What it is |
|---|---|---|---|---|
| C3896 | `stopped_vehicle` | 59.5 | 106.5 | Vehicle standing on the carriageway for ~47 s |
| C3896 | `stopped_vehicle` | 127.7 | 140.7 | Vehicle standing ~13 s |
| C3897 | `stopped_vehicle` | 41.2 | 51.7 | Vehicle standing ~10 s |
| C3902 | `stopped_vehicle` | 88.7 | 106.7 | Vehicle standing ~18 s |
| C3905 | `stopped_vehicle` | 2.6 | 13.4 | Car drives onto the crossing and stands ~11 s while pedestrians cross ✅ checked |
| C3905 | `near_miss` | 17.5 | 19.6 | Car passes very close to a motorcyclist waiting at the red light ✅ checked |

### Alarms checked by eye
- **C3896:** a delivery motorcycle entering a crossing with pedestrians on it (**real conflict**); a car passing an almost stopped car at the stop line (borderline); a car pulling out while another passes at 36 km/h near the picture edge (uncertain).
- **C3905:** a car passing very close to a waiting motorcyclist; a car driving around the island next to a standing pedestrian (close passes, not crashes).
""")

    st.markdown("### Ablation: how each rule removed false alarms")
    steps = ["Pixel distances", "Metres (bird's-eye)", "Skip side-by-side pairs", "Steadier speeds",
             "Calibrated area only", "Vehicle must really move"]
    vals = [2328, 935, 808, 725, 62, 32]
    fig = go.Figure(go.Bar(x=vals, y=steps, orientation="h", marker_color="#4363d8",
                           text=vals, textposition="outside"))
    fig.update_layout(title="Close-call moments in the first 10 s of calm traffic (C3905)",
                      yaxis_autorange="reversed", xaxis_type="log", xaxis_title="count (log scale)",
                      height=340, margin=dict(l=10, r=40, t=40, b=40))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("The single pair left after all rules is a real conflict: a car drives onto the crossing while a pedestrian walks toward it.")

    st.markdown("### Runtime vs. the time budget (Tesla T4, official harness)")
    rt = pd.DataFrame({"video": ["C3896", "C3897", "C3902", "C3905"],
                       "Part A (s)": [241, 223, 223, 90], "Part B (s)": [685, 642, 649, 246],
                       "budget (s)": [1021, 954, 954, 383]})
    fig = go.Figure()
    fig.add_bar(x=rt["video"], y=rt["Part A (s)"], name="Part A", marker_color="#42d4f4")
    fig.add_bar(x=rt["video"], y=rt["Part B (s)"], name="Part B", marker_color="#4363d8")
    fig.add_scatter(x=rt["video"], y=rt["budget (s)"], mode="markers", name="budget (3 × length)",
                    marker=dict(symbol="line-ew-open", size=40, color="#e6194b", line_width=3))
    fig.update_layout(barmode="stack", yaxis_title="seconds", height=320, margin=dict(l=10, r=10, t=20, b=40))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("All four videos finish at 2.6–2.7 × their length, inside the 3 × budget.")

    st.markdown("""
### Honest failure cases
- **Only 2 of 14 classes** are predicted. Classes without a validated rule are left out on purpose: predicting a class that never occurs costs a full zero in the macro-F1.
- **Part A stops at 0.7 × video length** on the T4 to leave time for Part B, so events late in a long video can be missed.
- **Tracker identity swaps** when road users overlap (e.g. a pedestrian walking through a waiting rider).
- **Far road excluded** from risk (> 25 m from the crossing), where one calibration is inaccurate.
- **No accidents in the samples**, so the alarm threshold (TTC = 1 s) comes from physics, not from learning.
""")


def eda_page():
    vids = sample_predictions()
    st.markdown("## Exploratory data analysis of the sample videos")
    rows = []
    for name, v in vids.items():
        r = v["risk"]
        fps = 1 / (r[1][0] - r[0][0]) if len(r) > 1 else 0
        rows.append([name.split(".")[0], "3840 × 2160 (4K)", round(fps, 2), len(r),
                     f"{int(r[-1][0] // 60)}:{int(r[-1][0] % 60):02d}", LIGHT.get(name.split(".")[0], "—")])
    st.dataframe(pd.DataFrame(rows, columns=["video", "resolution", "fps", "frames", "length", "light"]),
                 hide_index=True, use_container_width=True)
    st.caption("One fixed camera, 4K at ~30 fps, 2–6 minutes per clip, day and evening light. "
               "Note: the task said 25 fps, the samples are 29.97 fps, so every rule is written in seconds and metres, not frames.")

    st.markdown("### Scene layout: lanes, legal directions, crossings")
    overlay = os.path.join(HERE, "scene_overlay.jpg")
    if os.path.exists(overlay):
        st.image(overlay, use_container_width=True,
                 caption="Two carriageways split by a median: traffic toward the camera on the left (queue at the stop line), "
                         "away from the camera on the right; three pedestrian crossings.")

    st.markdown("### How fast things move (median speed, first 10 s of C3905)")
    fig = go.Figure(go.Bar(x=["Pedestrian", "Car", "Truck"], y=[4.9, 16.6, 19.5],
                           marker_color=["#3cb44b", "#4363d8", "#f58231"], text=["4.9", "16.6", "19.5"],
                           textposition="outside"))
    fig.update_layout(yaxis_title="km/h", height=300, margin=dict(l=10, r=10, t=20, b=40))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Measured in real metres through the bird's-eye calibration. The 4.9 km/h pedestrian speed matches "
               "normal walking (~5 km/h), which validates the calibration.")

    imgs = sorted(glob.glob(os.path.join(HERE, "*_trajectories.jpg")) + glob.glob(os.path.join(HERE, "*_counts.png")))
    if imgs:
        st.markdown("### Trajectories and object counts")
        cols = st.columns(2)
        for k, p in enumerate(imgs):
            cols[k % 2].image(p, caption=os.path.basename(p).rsplit(".", 1)[0].replace("_", " "),
                              use_container_width=True)

    st.markdown("""
### Findings that shaped the solution
1. **Decoding 4K video costs more than the detector.** Just reading the frames takes ~1.6 × the video length on a T4, so we sample a frame every 0.1–0.2 s instead of every frame.
2. **Perspective squashes the far lanes.** Measuring distance in pixels produced 2,328 "close calls" in 10 s of calm traffic (e.g. a truck passing a bus at its stop). Converting to metres and keeping only the calibrated area removed almost all of them (see the ablation on the Results tab).
3. **Traffic is dense and slow.** Road users are constantly 2–3 m apart, so being close cannot mean danger. This led to the lane, passing and minimum-speed rules.
4. **Occlusion corrupts positions.** Partly hidden cars in queues produced impossible speeds (8–15 m/s while queued). Skipping hidden objects and using median speeds fixed it.
5. **Queues are normal.** Many cars stand at the stop line every red phase, so a "stopped vehicle" only counts if it is **not** part of a queue (fewer than 2 other stopped vehicles within 12 m).
""")


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
    eda_page()
with results:
    results_page()
with report:
    page("report.md")
with team:
    page("team.md")
