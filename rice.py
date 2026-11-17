"""Deterministic RICE scoring, dependency DAG, critical path, Mermaid and CSV.

Everything in this module is fully deterministic: the same inputs always
produce the same scores, ordering, and artifacts, so every roadmap decision
is auditable.
"""

from __future__ import annotations

import csv
import os
import re
from typing import Dict, List, Optional, Tuple

import pydantic

from stories import Story

IMPACT_OPTIONS = (0.25, 0.5, 1.0, 2.0, 3.0)

# Reach bands by number of distinct personas touched by a story.
REACH_BANDS = {1: 500, 2: 1200}
REACH_DEFAULT_WIDE = 2000

CONFIDENCE_DEFAULT = 80.0

# Impact inferred from keyword priority in title + source text.
IMPACT_KEYWORDS = (
    (3.0, ("wrong-part", "revenue", "search", "voice", "vin", "checkout")),
    (2.0, ("upload", "catalog", "ingestion", "ingest", "dashboard", "offline", "fitment")),
    (1.0, ("report", "multilingual", "language", "toggle", "audit")),
)
IMPACT_DEFAULT = 1.0

# Dependency keyword matrix used in demo mode when no explicit phrase exists:
# (dependent keywords, prerequisite keywords)
DEPENDENCY_MATRIX = (
    (("voice", "spoken", "microphone", "hinglish"), ("search", "vin")),
    (("offline", "connectivity", "cached"), ("search", "vin")),
    (("dashboard", "pinned", "vehicle profile"), ("upload", "ingestion", "ingest")),
    (("report", "adoption", "analytics"), ("upload", "ingestion", "ingest")),
)

HEAD_NOUN_STOPWORDS = {"the", "a", "an", "and", "being", "live", "with", "by",
                       "it", "this", "that", "them"}


class RiceInput(pydantic.BaseModel):
    """Editable per-story RICE inputs. All fields are validated strictly."""
    story_id: str
    reach: float = pydantic.Field(ge=1, description="users reached per quarter")
    impact: float = pydantic.Field(description="one of 0.25, 0.5, 1, 2, 3")
    confidence: float = pydantic.Field(ge=1, le=100, description="confidence percent")
    effort: float = pydantic.Field(gt=0, description="person-weeks")

    @pydantic.field_validator("impact")
    @classmethod
    def _impact_allowed(cls, v: float) -> float:
        if not any(abs(v - opt) < 1e-9 for opt in IMPACT_OPTIONS):
            raise ValueError(
                "impact must be one of %s, got %s" % (list(IMPACT_OPTIONS), v)
            )
        return float(v)


# ---------------------------------------------------------------------------
# Deterministic scoring
# ---------------------------------------------------------------------------


def rice_score(reach: float, impact: float, confidence: float, effort: float) -> float:
    """RICE = Reach * Impact * (Confidence/100) / Effort, rounded to 2 dp."""
    if reach <= 0:
        raise ValueError("reach must be positive")
    if effort <= 0:
        raise ValueError("effort must be positive")
    if not 1 <= confidence <= 100:
        raise ValueError("confidence must be between 1 and 100")
    RiceInput(story_id="check", reach=reach, impact=impact,
              confidence=confidence, effort=effort)
    return round(reach * impact * (confidence / 100.0) / effort, 2)


def infer_impact(story: Story) -> float:
    # Title first (most specific), then the full section text.
    for text in (story.title.lower(),
                 (story.title + " " + story.source_text).lower()):
        best = IMPACT_DEFAULT
        for level, keywords in IMPACT_KEYWORDS:
            if any(k in text for k in keywords):
                best = max(best, level)
        if best > IMPACT_DEFAULT:
            return best
    return IMPACT_DEFAULT


def default_rice_inputs(stories: List[Story]) -> List[RiceInput]:
    """Deterministic defaults per story, derived from the extraction itself."""
    inputs: List[RiceInput] = []
    for story in stories:
        reach = REACH_BANDS.get(len(story.personas), REACH_DEFAULT_WIDE)
        impact = infer_impact(story)
        effort = float(min(8, max(2, 1 + len(story.acceptance_criteria))))
        inputs.append(RiceInput(
            story_id=story.id, reach=float(reach), impact=impact,
            confidence=CONFIDENCE_DEFAULT, effort=effort,
        ))
    return inputs


# ---------------------------------------------------------------------------
# Dependency inference
# ---------------------------------------------------------------------------


def _content_tokens(text: str) -> List[str]:
    return [t for t in re.findall(r"[a-z]+", text.lower()) if len(t) >= 4]


def _match_fragment_to_story(fragment: str, id_to_title: Dict[str, str],
                             self_id: str) -> Optional[str]:
    """Match a 'depends on' fragment to a story title.

    Requires overlap between fragment tokens and the title to be at least half
    of the fragment's content tokens (and at least 1), which filters out
    incidental single-word collisions such as 'data' or 'team'. Ties prefer
    the title containing the fragment's head noun (its last token).
    """
    tokens = [t for t in _content_tokens(fragment) if t not in HEAD_NOUN_STOPWORDS]
    if not tokens:
        return None
    best_id: Optional[str] = None
    best_overlap = 0
    best_head = False
    for other_id, title in id_to_title.items():
        if other_id == self_id:
            continue
        overlap = sum(1 for t in tokens if re.search(r"\b%s" % re.escape(t), title))
        if overlap < 1 or overlap * 2 < len(tokens):
            continue
        head_match = bool(re.search(r"\b%s" % re.escape(tokens[-1]), title))
        if (overlap > best_overlap
                or (overlap == best_overlap and head_match and not best_head)):
            best_id, best_overlap, best_head = other_id, overlap, head_match
    return best_id


def _explicit_dependencies(stories: List[Story]) -> Dict[str, List[str]]:
    deps: Dict[str, List[str]] = {s.id: [] for s in stories}
    id_to_title = {s.id: s.title.lower() for s in stories}
    for s in stories:
        for m in re.finditer(r"depends on ([^.;]+)", s.source_text.lower()):
            fragment = m.group(1)
            for part in re.split(r"\s+and\s+|\s*,\s*", fragment):
                target = _match_fragment_to_story(part, id_to_title, s.id)
                if target and target not in deps[s.id]:
                    deps[s.id].append(target)
    return deps


def _matrix_dependencies(stories: List[Story], existing: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """Fallback edges from the keyword matrix. Prerequisites match against
    story TITLES only (bodies mention each other constantly). Any edge that
    would create a cycle is pruned, so the result is always a DAG."""
    deps = {k: list(v) for k, v in existing.items()}
    ids = [s.id for s in stories]
    for s in stories:
        lowered = s.source_text.lower()
        for dep_keywords, prereq_keywords in DEPENDENCY_MATRIX:
            if not any(k in lowered for k in dep_keywords):
                continue
            for other in stories:
                if other.id == s.id or other.id in deps[s.id]:
                    continue
                if s.id in deps.get(other.id, []):
                    continue  # would form a 2-cycle
                if any(k in other.title.lower() for k in prereq_keywords):
                    deps[s.id].append(other.id)
                    try:
                        topological_sort(ids, deps)
                    except ValueError:
                        deps[s.id].pop()  # prune edges that close a cycle
                    break
    return deps


def infer_dependencies(stories: List[Story], prd_text: Optional[str] = None) -> Dict[str, List[str]]:
    """Demo-mode inference: explicit 'depends on' phrases, then keyword matrix.

    The optional prd_text argument is accepted for interface symmetry; the
    per-story source text already carries the section body needed here.
    """
    if not stories:
        return {}
    deps = _explicit_dependencies(stories)
    deps = _matrix_dependencies(stories, deps)
    return deps


# ---------------------------------------------------------------------------
# DAG utilities
# ---------------------------------------------------------------------------


def topological_sort(story_ids: List[str], dependencies: Dict[str, List[str]]) -> List[str]:
    """Kahn topological sort. Deterministic: preserves input order among ties.

    Raises ValueError when the dependency graph contains a cycle.
    """
    index = {sid: i for i, sid in enumerate(story_ids)}
    indegree = {sid: 0 for sid in story_ids}
    children: Dict[str, List[str]] = {sid: [] for sid in story_ids}
    for sid, parents in dependencies.items():
        for p in parents:
            if p not in index:
                raise ValueError("Unknown dependency '%s' referenced by '%s'" % (p, sid))
            children[p].append(sid)
            indegree[sid] += 1
    ready = sorted([sid for sid in story_ids if indegree[sid] == 0], key=lambda s: index[s])
    order: List[str] = []
    while ready:
        node = ready.pop(0)
        order.append(node)
        newly = []
        for child in children[node]:
            indegree[child] -= 1
            if indegree[child] == 0:
                newly.append(child)
        for child in sorted(newly, key=lambda s: index[s]):
            ready.append(child)
        ready.sort(key=lambda s: index[s])
    if len(order) != len(story_ids):
        remaining = [s for s in story_ids if s not in set(order)]
        raise ValueError("Dependency cycle detected involving: %s" % ", ".join(sorted(remaining)))
    return order


def critical_path(story_ids: List[str], dependencies: Dict[str, List[str]],
                  efforts: Dict[str, float]) -> Tuple[List[str], float]:
    """Longest cumulative-effort path through the DAG (critical chain).

    Returns (ordered story ids along the path, total effort).
    """
    order = topological_sort(story_ids, dependencies)
    best_to: Dict[str, Tuple[float, Optional[str]]] = {}
    best_total = -1.0
    best_end: Optional[str] = None
    for sid in order:
        eff = float(efforts.get(sid, 0.0))
        parents = dependencies.get(sid, [])
        if parents:
            cand = max((best_to[p][0] for p in parents), default=0.0)
        else:
            cand = 0.0
        best_to[sid] = (cand + eff, None)
        # remember predecessor
        if parents:
            best_parent = max(parents, key=lambda p: best_to[p][0])
            best_to[sid] = (cand + eff, best_parent)
        if best_to[sid][0] > best_total:
            best_total = best_to[sid][0]
            best_end = sid
    path: List[str] = []
    node: Optional[str] = best_end
    while node is not None:
        path.append(node)
        node = best_to[node][1]
    path.reverse()
    return path, round(best_total, 2)


# ---------------------------------------------------------------------------
# Roadmap assembly
# ---------------------------------------------------------------------------


def build_roadmap(stories: List[Story], inputs: List[RiceInput],
                  dependencies: Optional[Dict[str, List[str]]] = None) -> Dict:
    """Score, rank, sequence. Returns dict with rows, path, gantt, mermaid, csv."""
    if not stories:
        raise ValueError("No stories to score; extract stories first.")
    deps = dependencies if dependencies is not None else infer_dependencies(stories)
    story_map = {s.id: s for s in stories}
    input_map = {r.story_id: r for r in inputs}
    for s in stories:
        if s.id not in input_map:
            raise ValueError("Missing RICE inputs for story %s" % s.id)

    order = topological_sort(list(story_map.keys()), deps)
    efforts = {sid: input_map[sid].effort for sid in story_map}
    path_ids, path_total = critical_path(order, deps, efforts)
    path_set = set(path_ids)

    # Earliest start week along the DAG (Gantt-ish sequencing).
    start_week: Dict[str, float] = {}
    for sid in order:
        parents = deps.get(sid, [])
        start_week[sid] = max((start_week[p] + efforts[p] for p in parents), default=0.0)

    rows = []
    for s in stories:
        r = input_map[s.id]
        score = rice_score(r.reach, r.impact, r.confidence, r.effort)
        rows.append({
            "id": s.id,
            "story": s.story,
            "title": s.title,
            "personas": ", ".join(s.personas),
            "reach": r.reach,
            "impact": r.impact,
            "confidence": r.confidence,
            "effort": r.effort,
            "rice": score,
            "depends_on": ", ".join(deps.get(s.id, [])) or "-",
            "critical_path": s.id in path_set,
            "start_week": start_week[s.id],
        })
    rows.sort(key=lambda row: (-row["rice"], row["id"]))
    for rank, row in enumerate(rows, 1):
        row["rank"] = rank

    gantt_lines = []
    for sid in order:
        row = next(r for r in rows if r["id"] == sid)
        end_week = row["start_week"] + row["effort"]
        flag = " [CRITICAL PATH]" if row["critical_path"] else ""
        gantt_lines.append("Wk %g-%g | %s %s%s" % (
            row["start_week"], end_week, sid, row["title"], flag))
    cp_text = "Critical path (longest cumulative effort): %s  |  total %.2f person-weeks" % (
        " -> ".join(path_ids) if path_ids else "(none)", path_total)

    return {
        "rows": rows,
        "dependencies": deps,
        "critical_path_ids": path_ids,
        "critical_path_total": path_total,
        "critical_path_text": cp_text,
        "gantt_text": "\n".join(gantt_lines),
        "mermaid_text": mermaid_flowchart(stories, deps),
        "csv_path": export_csv(rows),
    }


# ---------------------------------------------------------------------------
# Export artifacts
# ---------------------------------------------------------------------------

CSV_COLUMNS = ["Rank", "ID", "Story", "Personas", "Reach", "Impact",
               "Confidence %", "Effort (pw)", "RICE", "Depends on", "Critical path"]


def export_csv(rows: List[dict]) -> str:
    """Write the ranked roadmap CSV and return its path."""
    out_dir = os.path.join("/tmp", "shiproom_exports")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "shiproom_roadmap.csv")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(CSV_COLUMNS)
        for row in rows:
            writer.writerow([
                row["rank"], row["id"], row["story"], row["personas"],
                row["reach"], row["impact"], row["confidence"], row["effort"],
                row["rice"], row["depends_on"], "yes" if row["critical_path"] else "no",
            ])
    return path


def mermaid_flowchart(stories: List[Story], dependencies: Dict[str, List[str]]) -> str:
    """Mermaid flowchart of the dependency DAG; every story id appears."""
    lines = ["flowchart TD"]
    for s in stories:
        label = s.title.replace('"', "'")
        lines.append('    %s["%s: %s"]' % (s.id, s.id, label))
    for sid, parents in dependencies.items():
        for p in parents:
            lines.append("    %s --> %s" % (p, sid))
    if len(lines) == 1:
        lines.append("    EMPTY[no stories]")
    return "\n".join(lines)
