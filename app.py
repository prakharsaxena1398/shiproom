"""Shiproom - PRD-to-Execution Copilot with Deterministic RICE.

Gradio Blocks UI:
  Pipeline tab: PRD input (paste or upload .md/.txt/.pdf) -> extract stories ->
  editable RICE inputs -> score and sequence -> ranked roadmap, critical path,
  Gantt-style sequence.
  Export tab: CSV download and Mermaid flowchart text.
  Launch checklist tab: generated readiness checklist.
"""

from __future__ import annotations

import io
import os
import tempfile

import gradio as gr
import pandas as pd
import pypdf

from rice import (CSV_COLUMNS, IMPACT_OPTIONS, build_roadmap, default_rice_inputs,
                  infer_dependencies)
from stories import Story, extract_stories

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SAMPLE_PRD_PATH = os.path.join(PROJECT_ROOT, "demo_data", "sample_prd.md")

BANNER_DEMO = (
    "**MODE: DEMO** - fully offline. Stories are extracted with deterministic "
    "rule-based parsing; no network calls are made. RICE scoring is always "
    "deterministic and auditable in both modes."
)
BANNER_LIVE = (
    "**MODE: LIVE** - story extraction uses an OpenAI-compatible LLM "
    "(env: OPENAI_API_KEY, OPENAI_BASE_URL, LLM_MODEL). If the API fails after "
    "one retry the run stops with an error. RICE scoring remains deterministic."
)

RICE_HEADERS = ["ID", "Story", "Reach", "Impact", "Confidence %", "Effort (person-weeks)"]
ROADMAP_HEADERS = ["Rank", "ID", "Story", "Personas", "Reach", "Impact",
                   "Confidence %", "Effort (pw)", "RICE", "Depends on", "Critical path"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def read_prd_from_file(filepath: str) -> str:
    if not filepath:
        return ""
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".pdf":
        reader = pypdf.PdfReader(io.BytesIO(open(filepath, "rb").read()))
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    with open(filepath, "r", encoding="utf-8", errors="replace") as fh:
        return fh.read()


def stories_to_display(stories: list) -> list:
    return [[s.id, s.title, s.story, "\n".join(s.acceptance_criteria),
             ", ".join(s.personas)] for s in stories]


def rice_table(stories: list) -> list:
    rows = []
    for r in default_rice_inputs(stories):
        story = next(s for s in stories if s.id == r.story_id)
        rows.append([r.story_id, story.story, r.reach, r.impact,
                     r.confidence, r.effort])
    return rows


def parse_rice_table(table: list, stories: list) -> list:
    """Convert the edited UI table back to validated RiceInput objects.

    Invalid or blank cells fall back to the deterministic defaults so a typo
    in one cell never breaks the whole run.
    """
    from rice import RiceInput

    defaults = {r.story_id: r for r in default_rice_inputs(stories)}
    out = []
    for row in table or []:
        if not row or row[0] in (None, ""):
            continue
        sid = str(row[0]).strip()
        base = defaults.get(sid)
        if base is None:
            continue

        def pick(value, fallback, caster):
            try:
                return caster(float(value))
            except (TypeError, ValueError):
                return getattr(base, fallback)

        impact = pick(row[3], "impact", lambda v: v)
        try:
            RiceInput(story_id="check", reach=1, impact=impact,
                      confidence=1, effort=1)
        except Exception:
            impact = base.impact
        out.append(RiceInput(
            story_id=sid,
            reach=max(1.0, pick(row[2], "reach", float)),
            impact=impact,
            confidence=min(100.0, max(1.0, pick(row[4], "confidence", float))),
            effort=max(0.5, pick(row[5], "effort", float)),
        ))
    return out


def generate_launch_checklist(stories: list, roadmap: dict) -> str:
    """Deterministic launch-readiness checklist derived from the roadmap."""
    lowered_all = " ".join((s.source_text or "") + " " + s.title for s in stories).lower()
    top3 = roadmap["rows"][:3]
    lines = ["## Launch checklist", ""]

    lines.append("**Metrics instrumentation**")
    lines.append("- Define event names for every extracted story; ship analytics "
                 "events with merchant_id, story_id, and latency before rollout.")
    if "voice" in lowered_all:
        lines.append("- Voice: log transcription word error rate and query "
                     "confirmation accept rate per session.")
    if "search" in lowered_all:
        lines.append("- Search: log zero-result queries and time-to-first-result.")
    lines.append("- Wire the weekly adoption report (region, search success rate, "
                 "voice share) to the field team distribution list.")
    lines.append("")

    lines.append("**Rollback plan**")
    lines.append("- Ship every story behind a feature flag; flag owners named in the roadmap.")
    if "upload" in lowered_all or "ingestion" in lowered_all:
        lines.append("- Data ingestion: keep the previous statement batch version "
                     "publishable; a bad batch rolls back to the last approved snapshot.")
    if "voice" in lowered_all:
        lines.append("- Voice: fall back to typed search if transcription error "
                     "rate exceeds the agreed threshold for one day.")
    lines.append("- Roadmap rollback: re-run Score and sequence with lowered "
                 "Confidence inputs; the deterministic model recomputes the same "
                 "ranked plan for the revised assumptions.")
    lines.append("")

    lines.append("**Eval set**")
    if "voice" in lowered_all or "hinglish" in lowered_all:
        lines.append("- Build a Hinglish voice eval set (at least 200 utterances "
                     "across regional accents) before enabling voice for all field agents.")
    if "search" in lowered_all:
        lines.append("- Search relevance eval: 100 labeled GSTIN and free-text queries "
                     "with expected merchant IDs.")
    lines.append("- Acceptance criteria for the top 3 RICE stories double as "
                 "manual QA scripts.")
    lines.append("")

    lines.append("**GTM one-pager bullets**")
    for row in top3:
        lines.append("- %s (RICE %.2f): %s" % (row["id"], row["rice"], row["story"]))
    lines.append("")
    lines.append("_Generated deterministically from the current roadmap state._")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# UI handlers
# ---------------------------------------------------------------------------


def on_extract(prd_text: str, file_obj, mode: str):
    try:
        file_text = read_prd_from_file(file_obj) if file_obj else ""
        text = (file_text or prd_text or "").strip()
        if not text:
            raise ValueError("Paste a PRD or upload a .md/.txt/.pdf file first.")
        stories = extract_stories(text, mode)
        deps = infer_dependencies(stories, text)
        stories_df = stories_to_display(stories)
        rice_df = rice_table(stories)
        status = "Extracted %d stories in %s mode." % (len(stories), mode)
        return (stories_df, gr.update(value=rice_df, visible=True),
                (stories, deps), status, gr.update(visible=True))
    except Exception as exc:
        return ([], gr.update(value=[], visible=False), ([], {}),
                "Error: %s" % exc, gr.update(visible=False))


def on_score(rice_df, state):
    stories, deps = state
    try:
        if not stories:
            raise ValueError("Extract stories first.")
        inputs = parse_rice_table(rice_df, stories)
        result = build_roadmap(stories, inputs, deps)
        roadmap_df = [[r["rank"], r["id"], r["story"], r["personas"], r["reach"],
                       r["impact"], r["confidence"], r["effort"], r["rice"],
                       r["depends_on"], "yes" if r["critical_path"] else "no"]
                      for r in result["rows"]]
        checklist = generate_launch_checklist(stories, result)
        return (roadmap_df, result["critical_path_text"], result["gantt_text"],
                result["csv_path"], result["mermaid_text"], checklist,
                "Scored %d stories. Roadmap CSV ready for download." % len(stories))
    except Exception as exc:
        return ([], "Error: %s" % exc, "", None, "", "", "Error: %s" % exc)


def on_mode_change(mode: str):
    return BANNER_LIVE if mode == "LIVE" else BANNER_DEMO


def load_sample():
    with open(SAMPLE_PRD_PATH, "r", encoding="utf-8") as fh:
        return fh.read()


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Shiproom - PRD-to-Execution Copilot") as demo:
        gr.Markdown("# Shiproom - PRD-to-Execution Copilot with Deterministic RICE")
        mode = gr.Radio(["DEMO", "LIVE"], value="DEMO", label="Extraction mode")
        banner = gr.Markdown(BANNER_DEMO)
        mode.change(on_mode_change, inputs=mode, outputs=banner)

        with gr.Tab("Pipeline"):
            with gr.Row():
                with gr.Column():
                    prd_box = gr.Textbox(label="PRD text", lines=14,
                                         placeholder="Paste your PRD here, or upload a file.")
                    prd_file = gr.File(label="Upload PRD (.md / .txt / .pdf)",
                                       file_types=[".md", ".txt", ".pdf"],
                                       type="filepath")
                    load_btn = gr.Button("Load sample PRD")
                    extract_btn = gr.Button("Extract stories", variant="primary")
                    status = gr.Markdown("")
                with gr.Column():
                    stories_df = gr.Dataframe(
                        headers=["ID", "Title", "Story", "Acceptance criteria", "Personas"],
                        datatype=["str", "str", "str", "str", "str"],
                        label="Candidate user stories", wrap=True, interactive=False)
                    gr.Markdown(
                        "### Editable RICE inputs\n"
                        "RICE = Reach x Impact x (Confidence %% / 100) / Effort. "
                        "Impact must be one of %s. Edit any cell, then press "
                        "**Score and sequence**. Invalid cells fall back to "
                        "deterministic defaults." % list(IMPACT_OPTIONS))
                    rice_df = gr.Dataframe(
                        headers=RICE_HEADERS,
                        datatype=["str", "str", "number", "number", "number", "number"],
                        label="RICE inputs (editable)", wrap=True, interactive=True,
                        visible=False)
                    score_btn = gr.Button("Score and sequence", variant="primary",
                                          visible=False)

            roadmap_df = gr.Dataframe(headers=ROADMAP_HEADERS,
                                      datatype=["number", "str", "str", "str",
                                                "number", "number", "number",
                                                "number", "number", "str", "str"],
                                      label="Ranked roadmap (deterministic RICE)",
                                      wrap=True, interactive=False)
            cp_box = gr.Markdown("Extract and score stories to see the critical path.")
            gantt_box = gr.Textbox(label="Gantt-style sequence (person-weeks)",
                                   lines=8, interactive=False)

        with gr.Tab("Export"):
            csv_btn = gr.DownloadButton("Download roadmap CSV")
            gr.Markdown("### Mermaid dependency flowchart")
            mermaid_box = gr.Textbox(label="Mermaid flowchart (paste into any Mermaid renderer)",
                                     lines=10, interactive=False)

        with gr.Tab("Launch checklist"):
            checklist_box = gr.Markdown("Extract and score stories to generate the checklist.")

        state = gr.State(([], {}))

        load_btn.click(load_sample, outputs=prd_box)
        extract_btn.click(on_extract, inputs=[prd_box, prd_file, mode],
                          outputs=[stories_df, rice_df, state, status, score_btn])
        score_btn.click(on_score, inputs=[rice_df, state],
                        outputs=[roadmap_df, cp_box, gantt_box, csv_btn,
                                 mermaid_box, checklist_box, status])
    return demo


def main():
    app = build_ui()
    server_port = int(os.environ.get("SHIPROOM_PORT", "7860"))
    app.launch(server_name="0.0.0.0", server_port=server_port)


if __name__ == "__main__":
    main()
