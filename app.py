"""Gradio demo: upload a face video -> rPPG pulse, heart rate, signal-quality score."""

import json

import gradio as gr

from analyze import analyze, plot_result

DESCRIPTION = """
# rPPG pulse extraction (POS, no training)

Upload a short face video (≥ 5 s, ideally 10–30 s, face still and visible).
The pipeline finds the face with MediaPipe, averages skin colour on the forehead/cheeks,
runs the **POS** algorithm (Wang et al., 2017), and estimates heart rate from the
dominant frequency of the pulse signal.

**Signal-quality score (0–1)** is a heuristic: how strong and consistent the cardiac
peak is. It is *not* a calibrated liveness / deepfake detector — use it as one feature
and evaluate it against labelled data. Videos are processed in memory and not stored.

Code: [github.com/pedroldrh/rppg-pos](https://github.com/pedroldrh/rppg-pos)
"""


def run(video, region):
    if video is None:
        raise gr.Error("Upload a video first.")
    try:
        res = analyze(video, region=region)
    except Exception as e:  # surface pipeline errors in the UI
        raise gr.Error(str(e))
    fig = plot_result(res)
    s = res["summary"]
    headline = (
        f"### Heart rate ≈ {s['hr_bpm_median'] if s['hr_bpm_median'] else '—'} bpm  ·  "
        f"signal quality {s['signal_quality_score']}  ·  face detected in "
        f"{round(100 * s['face_detection_rate'])}% of frames"
    )
    out = dict(s)
    out["windows"] = json.loads(res["windows"].to_json(orient="records"))
    return headline, fig, out


with gr.Blocks(title="rPPG pulse extraction") as demo:
    gr.Markdown(DESCRIPTION)
    with gr.Row():
        with gr.Column(scale=1):
            video = gr.Video(label="Face video", sources=["upload", "webcam"])
            region = gr.Radio(["forehead", "cheeks", "combined"], value="forehead", label="Skin region")
            btn = gr.Button("Analyze", variant="primary")
        with gr.Column(scale=2):
            headline = gr.Markdown()
            plot = gr.Plot(label="Pulse, spectrum, per-window HR")
            js = gr.JSON(label="Raw output")
    btn.click(run, [video, region], [headline, plot, js])

if __name__ == "__main__":
    demo.launch()
