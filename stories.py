"""Story extraction for Shiproom.

Two modes:
- DEMO (default, fully offline): rule-based extractor that turns PRD feature
  sections into user stories with Given/When/Then acceptance criteria.
- LIVE: LLM extraction through an OpenAI-compatible chat completions endpoint,
  with strict pydantic validation and a single retry.
"""

from __future__ import annotations

import json
import os
import re
from typing import List, Optional

import pydantic
import requests

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


class Story(pydantic.BaseModel):
    id: str
    title: str
    story: str
    acceptance_criteria: List[str]
    personas: List[str]
    section: str = ""
    source_text: str = ""

    @pydantic.field_validator("acceptance_criteria")
    @classmethod
    def _at_least_two(cls, v: List[str]) -> List[str]:
        if len(v) < 2:
            raise ValueError("every story needs at least 2 acceptance criteria")
        return v


class StoryList(pydantic.BaseModel):
    stories: List[Story]


# ---------------------------------------------------------------------------
# Persona detection
# ---------------------------------------------------------------------------

PERSONA_LEXICON = {
    "dealer": ("dealer", "dealers"),
    "admin": ("admin", "admins", "administrator", "administrators"),
    "analyst": ("analyst", "analysts"),
    "customer": ("customer", "customers"),
    "technician": ("technician", "technicians"),
    "service advisor": ("service advisor", "service advisors"),
    "support agent": ("support desk", "support agent", "support agents"),
}

DEFAULT_AC_PADS = (
    "Given the feature is shipped, When a {persona} completes the core workflow, "
    "Then the outcome matches the PRD success metric for this capability.",
    "Given invalid or malformed input, When the user submits it, "
    "Then a clear validation error is shown and no partial state is persisted.",
)

# Keyword sets for benefit phrasing, checked in order.
BENEFIT_TEMPLATES = (
    (("voice", "spoken", "microphone"), "field agents can query hands-free during merchant visits"),
    (("offline", "connectivity", "cached"), "search keeps working in low connectivity zones"),
    (("search", "gstin", "account number"), "the right credit file is pulled in seconds instead of minutes"),
    (("upload", "ingestion", "ingest", "statement"), "underwriting data stays current without manual keying"),
    (("dashboard", "pinned", "merchant profile"), "active merchants and risk signals are one click away"),
    (("report", "adoption", "analytics"), "the field team can act on adoption data every week"),
    (("multilingual", "language", "hindi", "english"), "users can work in their preferred language"),
)
DEFAULT_BENEFIT = "their daily workflow becomes faster and less error-prone"

# Keyword sets for acceptance criteria, checked in order. Each entry produces
# at most one criterion per story.
AC_TEMPLATES = (
    (("voice", "spoken", "microphone", "hinglish"),
     "Given the agent taps the microphone, When a mixed-language query is spoken, "
     "Then the interpreted query is confirmed on screen before the search executes."),
    (("offline", "connectivity", "cached"),
     "Given the device has no connectivity, When a search runs offline, "
     "Then cached credit file results are shown and results are queued for sync."),
    (("filter",),
     "Given search results are displayed, When the analyst applies a channel, risk grade, or amount filter, "
     "Then only matching merchants remain visible."),
    (("export",),
     "Given a result set or report is displayed, When the user clicks export, "
     "Then a CSV file downloads containing every visible column."),
    (("search", "gstin", "account number", "query"),
     "Given an analyst enters a GSTIN, account number, or free-text description, When the search runs, "
     "Then matching merchants are returned with repayment data and risk grade."),
    (("upload", "ingestion", "ingest", "bulk"),
     "Given an admin uploads a statement batch, When validation completes, "
     "Then row-level errors appear in a review queue before publish."),
    (("approve", "reject", "diff", "audit"),
     "Given a data batch is under review, When the admin approves or rejects it, "
     "Then the decision and an audit log entry are recorded."),
    (("dashboard", "pinned", "merchant profile", "repayment"),
     "Given an analyst opens the dashboard, When pinned merchants exist, "
     "Then risk notes and repayment alerts for those merchants load first."),
    (("report", "adoption", "analytics"),
     "Given a reporting week has ended, When the report is generated, "
     "Then adoption, search success, and voice usage metrics per region are included."),
    (("multilingual", "language", "hindi", "english"),
     "Given the language toggle is used, When the user switches language, "
     "Then interface labels, categories, and error messages are localized."),
)

IMPERATIVE_STARTERS = (
    "allow", "let", "provide", "support", "enable", "show", "display",
    "add", "offer", "deliver", "make", "give", "include", "build", "create",
)

SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
SECTION_HEADING = re.compile(r"^#{2,4}\s+(.*)$")
ID_PREFIX = re.compile(r"^([A-Z]{1,3}\d+)\.?\s+")
PERSONA_CAN = re.compile(r"\b(dealers?|admins?|analysts?|customers?|technicians?)\s+can\s+([^.;]+)", re.IGNORECASE)

# Non-feature headings that must never yield stories even if they contain
# imperative-sounding sentences.
SKIP_HEADINGS = {"background", "goals", "non-goals", "objectives",
                 "success metrics", "metrics", "risks", "risks and open questions",
                 "open questions", "overview", "summary"}


def detect_personas(text: str) -> List[str]:
    """Return personas ordered by how often they appear in the text."""
    counts = {}
    lowered = text.lower()
    for persona, aliases in PERSONA_LEXICON.items():
        n = sum(lowered.count(a) for a in aliases)
        if n:
            counts[persona] = n
    return sorted(counts, key=lambda p: -counts[p])


def _split_sentences(body: str) -> List[str]:
    """De-wrap hard-wrapped PRD lines, then split into sentences."""
    lines = [line.strip().lstrip("-*").strip() for line in body.splitlines()]
    lines = [line for line in lines if line]
    chunks: List[str] = []
    buf = ""
    for line in lines:
        buf = (buf + " " + line) if buf else line
        if re.search(r"[.!?:]\s*$", line):
            chunks.append(buf)
            buf = ""
    if buf:
        chunks.append(buf)
    sentences: List[str] = []
    for chunk in chunks:
        sentences.extend(s.strip() for s in SENTENCE_SPLIT.split(chunk) if s.strip())
    return sentences


def _is_imperative(sentence: str) -> bool:
    first = sentence.split(" ", 1)[0].lower().strip(",;")
    return first in IMPERATIVE_STARTERS


def _persona_in_sentence(sentence: str, doc_personas: List[str]) -> Optional[str]:
    lowered = sentence.lower()
    for persona in doc_personas:
        for alias in PERSONA_LEXICON.get(persona, ()): 
            if alias in lowered:
                return persona
    return doc_personas[0] if doc_personas else "product user"


def _with_article(persona: str) -> str:
    return ("an %s" if persona[0].lower() in "aeiou" else "a %s") % persona


def _capability_from_lead(sentence: str) -> str:
    s = sentence.strip().rstrip(".")
    m = re.match(r"^(?:Allow|Let)\s+(?:the\s+)?[a-z\s]+?\s+to\s+(.+)$", s, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    m = re.match(r"^(?:Provide)\s+(?:the\s+)?[a-z\s]+?\s+(?:with\s+)?(?:a|an)?\s*(.+)$", s, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    m = re.match(r"^(?:Support|Enable|Show|Display|Add|Offer|Deliver|Make|Give|Include|Build|Create)\s+(.+)$",
                 s, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return s


def _benefit_from_body(title: str, body: str) -> str:
    m = re.search(r"so that ([^.]+)", body, re.IGNORECASE)
    if m:
        return re.sub(r"\s+", " ", m.group(1)).strip().rstrip(".")
    # Match against the section title first (most specific), then the body.
    for text in (title.lower(), body.lower()):
        for keywords, benefit in BENEFIT_TEMPLATES:
            if any(k in text for k in keywords):
                return benefit
    return DEFAULT_BENEFIT


def _criteria_from_body(body: str, persona: str) -> List[str]:
    lowered = body.lower()
    criteria: List[str] = []
    for keywords, template in AC_TEMPLATES:
        if any(k in lowered for k in keywords) and template not in criteria:
            criteria.append(template)
    # Turn "dealers can ..." sentences into extra criteria.
    for actor, capability in PERSONA_CAN.findall(body):
        actor_singular = actor.lower().rstrip("s") if actor.lower().endswith("s") else actor.lower()
        capability = re.sub(r"\s+", " ", capability.strip().rstrip("."))
        if capability.lower().startswith(("export", "filter", "pin", "approve", "reject", "schedule")):
            criteria.append(
                "Given %s is using the feature, When they %s, "
                "Then the action completes and a confirmation is shown." % (
                    _with_article(actor_singular), capability)
            )
    # Guarantee at least two criteria.
    pads = [p.format(persona=persona) for p in DEFAULT_AC_PADS]
    for pad in pads:
        if len(criteria) >= 2:
            break
        if pad not in criteria:
            criteria.append(pad)
    return criteria[:4]


def _parse_sections(prd_text: str) -> List[tuple]:
    """Return (heading, body) pairs for numbered feature-style sections."""
    sections: List[tuple] = []
    current_title: Optional[str] = None
    current_body: List[str] = []
    for line in prd_text.splitlines():
        m = SECTION_HEADING.match(line.strip())
        if m:
            if current_title is not None and current_body:
                sections.append((current_title, "\n".join(current_body).strip()))
            current_title = m.group(1).strip()
            current_body = []
        elif current_title is not None:
            current_body.append(line)
    if current_title is not None and current_body:
        sections.append((current_title, "\n".join(current_body).strip()))
    return sections


def _looks_like_feature(title: str, body: str) -> bool:
    """Keep sections that carry imperative feature language with a persona."""
    if ID_PREFIX.match(title):
        return True
    if title.lower().strip("# ") in SKIP_HEADINGS:
        return False
    body_personas = detect_personas(body)
    for sentence in _split_sentences(body):
        if not _is_imperative(sentence):
            continue
        lowered = sentence.lower()
        if any(alias in lowered for p in body_personas for alias in PERSONA_LEXICON[p]):
            return True
    return False


def extract_stories_demo(prd_text: str) -> List[Story]:
    """Offline rule-based extraction from PRD sections and imperative bullets."""
    if not prd_text or not prd_text.strip():
        raise ValueError("PRD text is empty; paste a PRD or upload a file first.")
    doc_personas = detect_personas(prd_text) or ["analyst"]
    stories: List[Story] = []
    counter = 0
    for title, body in _parse_sections(prd_text):
        if not _looks_like_feature(title, body):
            continue
        counter += 1
        id_m = ID_PREFIX.match(title)
        story_id = id_m.group(1) if id_m else "S%d" % counter
        clean_title = ID_PREFIX.sub("", title).strip()
        sentences = _split_sentences(body)
        lead = next((s for s in sentences if _is_imperative(s)), sentences[0])
        persona = _persona_in_sentence(lead, doc_personas)
        story_personas = [persona]
        for p in doc_personas:
            if p != persona and re.search(r"\b%s" % re.escape(p), body, re.IGNORECASE):
                story_personas.append(p)
        capability = _capability_from_lead(lead)
        benefit = _benefit_from_body(clean_title, body)
        story_text = "As %s, I want %s, so that %s." % (
            _with_article(persona), capability, benefit)
        criteria = _criteria_from_body(body, persona)
        stories.append(Story(
            id=story_id,
            title=clean_title,
            story=story_text,
            acceptance_criteria=criteria,
            personas=story_personas,
            section=title,
            source_text=title + "\n" + body,
        ))
    if not stories:
        raise ValueError(
            "No feature sections found. Write sections with imperative bullets "
            "(for example 'Allow analysts to search the credit file by GSTIN')."
        )
    return stories


# ---------------------------------------------------------------------------
# LIVE mode: LLM extraction via OpenAI-compatible endpoint
# ---------------------------------------------------------------------------

LIVE_SYSTEM_PROMPT = (
    "You are a product analyst. Convert the given PRD into candidate user stories. "
    "Return ONLY a JSON object of the form "
    '{"stories": [{"id": "F1", "title": "short title", "story": "As a <persona>, I want '
    '<capability>, so that <benefit>.", "acceptance_criteria": ["Given ... When ... Then ...", '
    '"Given ... When ... Then ..."], "personas": ["analyst"]}]}. '
    "Every story MUST have at least 2 acceptance criteria in Given/When/Then form. "
    "Do not include any text outside the JSON."
)


def _call_llm(prd_text: str) -> StoryList:
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("LIVE mode requires OPENAI_API_KEY to be set.")
    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    model = os.environ.get("LLM_MODEL", "gpt-4o-mini")
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": LIVE_SYSTEM_PROMPT},
            {"role": "user", "content": prd_text},
        ],
        "temperature": 0.2,
        "response_format": {"type": "json_object"},
    }
    headers = {"Authorization": "Bearer %s" % api_key, "Content-Type": "application/json"}
    last_error = ""
    for attempt in range(2):  # strict single retry
        try:
            resp = requests.post("%s/chat/completions" % base_url, json=payload,
                                 headers=headers, timeout=60)
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            data = json.loads(content)
            return StoryList.model_validate(data)
        except (requests.RequestException, KeyError, json.JSONDecodeError,
                pydantic.ValidationError, ValueError, TypeError) as exc:
            last_error = str(exc)
    raise RuntimeError("LIVE extraction failed after retry: %s" % last_error)


def extract_stories_live(prd_text: str) -> List[Story]:
    if not prd_text or not prd_text.strip():
        raise ValueError("PRD text is empty; paste a PRD or upload a file first.")
    result = _call_llm(prd_text)
    stories = result.stories
    # Normalize: ensure unique non-empty ids and persona fallback.
    seen = set()
    for i, s in enumerate(stories, 1):
        if not s.id or s.id in seen:
            s.id = "S%d" % i
        seen.add(s.id)
        if not s.personas:
            s.personas = ["product user"]
    return stories


def extract_stories(prd_text: str, mode: str = "DEMO") -> List[Story]:
    """Entry point used by the UI. mode is 'DEMO' or 'LIVE'."""
    if mode == "LIVE":
        return extract_stories_live(prd_text)
    return extract_stories_demo(prd_text)
