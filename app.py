#!/usr/bin/env python3
"""Minimal judge-facing Gradio entrypoint for the hosted Browser.FIND demo."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Prefer this checkout when another editable ping_ponder install shares the environment.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))


def build_app(runtime=None):
    import gradio as gr
    from ping_ponder.voice.web_demo import HostedRuntime

    runtime = runtime or HostedRuntime()

    async def submit(audio_path, text):
        result = await runtime.demo.submit(text=text or "", audio_path=audio_path)
        goal = "\n".join(line for line in (
            f"Capability: {result.capability}" if result.capability else "",
            f"Goal: {result.goal_type}" if result.goal_type else "",
            f"Site: {result.site}" if result.site else "",
            f"Target: {result.target}" if result.target else "",
        ) if line)
        return (result.transcript, goal, result.planner, result.status,
                result.message, result.final_url, result.page_title)

    with gr.Blocks(title="wepth voice agent") as app:
        gr.Markdown("# wepth voice agent\nSpeech becomes a validated SemanticGoal. A deterministic planner selects the execution path; this hosted demo shows Level 2 Browser.FIND.")
        audio = gr.Audio(sources=["microphone", "upload"], type="filepath", label="Recorded command")
        text = gr.Textbox(label="Text fallback", placeholder="Find the Browser Use repository on GitHub")
        run = gr.Button("Submit")
        transcript = gr.Textbox(label="Transcript")
        goal = gr.Textbox(label="Semantic Goal", lines=4)
        planner = gr.Textbox(label="Planner operator")
        status = gr.Textbox(label="Status")
        message = gr.Textbox(label="Result")
        url = gr.Textbox(label="Final URL")
        title = gr.Textbox(label="Page title")
        outputs = [transcript, goal, planner, status, message, url, title]
        run.click(submit, inputs=[audio, text], outputs=outputs, concurrency_limit=1)
        text.submit(submit, inputs=[audio, text], outputs=outputs, concurrency_limit=1)
    return app


def main():
    parser = argparse.ArgumentParser(description="Run the hosted wepth voice agent demo")
    parser.add_argument("--host", default=os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("GRADIO_SERVER_PORT", "7860")))
    args = parser.parse_args()
    app = build_app()
    app.queue(default_concurrency_limit=1).launch(server_name=args.host, server_port=args.port)


if __name__ == "__main__":
    main()
