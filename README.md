# Shiproom - PRD-to-Execution Copilot with Deterministic RICE

## The problem

PRDs are fuzzy. Product teams translate them into user stories inconsistently,
RICE scoring debates get settled by seniority rather than by the formula, and
dependency-blind roadmaps slip because nobody computed the critical path.
When the scoring happens in someone's head or in an unversioned spreadsheet,
the roadmap cannot be audited or defended.

## The method: why deterministic RICE matters

Shiproom converts a PRD into candidate user stories with acceptance criteria,
then scores them with a fully deterministic, auditable RICE model:

    RICE = Reach x Impact x (Confidence / 100) / Effort

Deterministic means: same inputs, same ranked roadmap, every time. The formula
is shown in the UI, every input (Reach, Impact on the 0.25/0.5/1/2/3 scale,
Confidence percent, Effort in person-weeks) is editable in a table, and the
score is recomputed with plain arithmetic that anyone can re-derive by hand
(for example Reach=1000, Impact=2, Confidence=80, Effort=4 gives exactly
400.0). No hidden model weights, no non-reproducible LLM scoring. The LLM, when
used at all, only drafts stories; it never decides priorities.

Shiproom also infers the dependency DAG (explicit "depends on" phrases plus a
keyword matrix in demo mode), topologically sorts it, raises on cycles,
computes the critical path as the longest cumulative-effort path, and exports
a ranked CSV roadmap plus a Mermaid flowchart of the dependency graph.

## Demo vs LIVE

- DEMO (default, fully offline): a rule-based extractor splits the PRD into
  sections, turns imperative bullets ("Allow analysts to search the credit
  file by GSTIN") into user stories ("As an analyst, I want ..., so that ...")
  with 2-4 Given/When/Then acceptance criteria, and detects personas (analyst,
  admin, customer, ...) from document keywords. No network calls.
- LIVE: story extraction is delegated to an OpenAI-compatible LLM
  (`POST {OPENAI_BASE_URL}/chat/completions`, env vars `OPENAI_API_KEY`,
  `OPENAI_BASE_URL` (default `https://api.openai.com/v1`), `LLM_MODEL`
  (default `gpt-4o-mini`)). Output is strict JSON validated with pydantic;
  the call retries exactly once and fails loudly otherwise. RICE scoring,
  the DAG, and exports are identical in both modes and always deterministic.

## Quickstart

    cd shiproom
    pip install -r requirements.txt
    python app.py

Open http://127.0.0.1:7860, press "Load sample PRD", then "Extract stories",
edit the RICE inputs if desired, and press "Score and sequence". Then explore
the Export and Launch checklist tabs.

Optional LIVE mode:

    export OPENAI_API_KEY=...        # required for LIVE
    export OPENAI_BASE_URL=https://api.openai.com/v1
    export LLM_MODEL=gpt-4o-mini

## Deploy to Hugging Face Spaces

1. Create a new Space with SDK "Gradio".
2. Push this folder: `app.py`, `stories.py`, `rice.py`, `requirements.txt`,
   `demo_data/sample_prd.md`.
3. The Space runs `app.py` automatically; the UI binds to `0.0.0.0` and
   respects the `SHIPROOM_PORT` env var (Spaces sets it to 7860).
4. For LIVE mode add `OPENAI_API_KEY` as a Space secret. The app defaults to
   DEMO mode, which works with no secrets at all.

## Tests

    python -m unittest discover -s tests -v

The suite covers the hand-checked RICE score, input validation, story
extraction from the sample PRD, persona detection, dependency inference,
topological ordering, cycle detection, critical path on known graphs, CSV
export shape, and Mermaid output.

## Sample PRD

`demo_data/sample_prd.md` is a realistic PRD for an "AI Loan-Operations
Assistant for Merchant Lending v1" (credit file search by GSTIN, voice query
in Hinglish, analyst risk dashboard, admin statement-batch review queue,
weekly adoption report, multilingual interface, offline mode) with explicit
"depends on" phrases that exercise the dependency inference.

## Honest limitations

- Demo-mode extraction is heuristic: it handles imperative feature sections
  well, but narrative or table-heavy PRDs may produce odd or missing stories.
- Dependency inference only catches explicit "depends on" phrases plus a small
  keyword matrix; implicit or architectural dependencies must be added
  manually (the Mermaid output makes gaps easy to spot).
- The keyword matrix for Impact and the persona-based Reach bands are rough
  priors; they are defaults to edit, not ground truth.
- Confidence is a human judgment; the model makes it explicit and editable but
  cannot make it objective.
- LIVE mode depends on the chosen LLM following the strict JSON contract;
  validation failures surface as errors after one retry rather than silently
  degrading.
- Critical path is computed on Effort only; resource contention and calendar
  constraints (holidays, hiring) are out of scope.
