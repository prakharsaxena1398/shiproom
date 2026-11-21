"""Core tests for Shiproom: RICE scoring, extraction, DAG, exports.

Run with:  python -m unittest discover -s tests -v
"""

import os
import sys
import unittest

import pandas as pd
import pydantic

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import rice  # noqa: E402
import stories as stories_mod  # noqa: E402
from rice import RiceInput, build_roadmap, critical_path, topological_sort  # noqa: E402
from stories import Story, detect_personas, extract_stories_demo  # noqa: E402

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLE_PRD_PATH = os.path.join(PROJECT_ROOT, "demo_data", "sample_prd.md")

with open(SAMPLE_PRD_PATH, "r", encoding="utf-8") as fh:
    SAMPLE_PRD = fh.read()


def make_story(sid, effort_hint_acs=None):
    acs = effort_hint_acs or [
        "Given a dealer opens the app, When a search runs, Then results appear.",
        "Given invalid input, When submitted, Then an error is shown.",
    ]
    return Story(id=sid, title="Story %s" % sid, story="As a dealer, I want %s." % sid,
                 acceptance_criteria=acs, personas=["dealer"], source_text="body %s" % sid)


class RiceScoreTests(unittest.TestCase):
    def test_rice_score_hand_check(self):
        # Reach=1000, Impact=2, Confidence=80, Effort=4 -> 1000*2*0.8/4 = 400.0
        self.assertEqual(rice.rice_score(1000, 2, 80, 4), 400.0)

    def test_invalid_impact_rejected(self):
        with self.assertRaises(pydantic.ValidationError):
            RiceInput(story_id="X", reach=100, impact=1.7, confidence=80, effort=2)
        with self.assertRaises(pydantic.ValidationError):
            RiceInput(story_id="X", reach=100, impact=0, confidence=80, effort=2)

    def test_invalid_confidence_and_effort_rejected(self):
        with self.assertRaises(pydantic.ValidationError):
            RiceInput(story_id="X", reach=100, impact=2, confidence=150, effort=2)
        with self.assertRaises(pydantic.ValidationError):
            RiceInput(story_id="X", reach=100, impact=2, confidence=80, effort=0)

    def test_impact_options_allowed(self):
        for opt in rice.IMPACT_OPTIONS:
            RiceInput(story_id="X", reach=100, impact=opt, confidence=80, effort=2)


class ExtractionTests(unittest.TestCase):
    def setUp(self):
        self.stories = extract_stories_demo(SAMPLE_PRD)

    def test_sample_prd_yields_at_least_eight_stories(self):
        self.assertGreaterEqual(len(self.stories), 8)

    def test_every_story_has_at_least_two_acceptance_criteria(self):
        for story in self.stories:
            self.assertGreaterEqual(len(story.acceptance_criteria), 2,
                                    msg="story %s has too few criteria" % story.id)
            self.assertTrue(story.acceptance_criteria[0].startswith("Given"))

    def test_stories_are_user_stories(self):
        for story in self.stories:
            self.assertTrue(story.story.startswith("As a ")
                            or story.story.startswith("As an "), msg=story.story)
            self.assertIn("I want", story.story)
            self.assertIn("so that", story.story)

    def test_persona_detection(self):
        personas = detect_personas(SAMPLE_PRD)
        self.assertIn("analyst", personas)
        self.assertIn("admin", personas)
        self.assertNotIn("dealer", personas)
        self.assertEqual(personas[0], "analyst")

    def test_expected_story_ids_present(self):
        ids = {s.id for s in self.stories}
        for expected in ("F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8"):
            self.assertIn(expected, ids)

    def test_default_rice_inputs_are_deterministic(self):
        inputs_a = rice.default_rice_inputs(self.stories)
        inputs_b = rice.default_rice_inputs(self.stories)
        self.assertEqual([r.model_dump() for r in inputs_a],
                         [r.model_dump() for r in inputs_b])


class DagTests(unittest.TestCase):
    def setUp(self):
        self.stories = extract_stories_demo(SAMPLE_PRD)
        self.deps = rice.infer_dependencies(self.stories, SAMPLE_PRD)

    def test_planted_dependencies_inferred(self):
        self.assertIn("F2", self.deps.get("F3", []))   # voice depends on catalog search
        self.assertIn("F1", self.deps.get("F4", []))   # dashboard depends on ingestion
        self.assertIn("F1", self.deps.get("F5", []))   # review queue depends on ingestion
        self.assertIn("F2", self.deps.get("F8", []))   # offline depends on search

    def test_topological_order_respects_dependencies(self):
        order = topological_sort([s.id for s in self.stories], self.deps)
        pos = {sid: i for i, sid in enumerate(order)}
        for child, parents in self.deps.items():
            for parent in parents:
                self.assertLess(pos[parent], pos[child],
                                msg="%s must come before %s" % (parent, child))

    def test_cycle_detection_raises(self):
        ids = ["A", "B", "C"]
        deps = {"A": ["B"], "B": ["C"], "C": ["A"]}
        with self.assertRaises(ValueError):
            topological_sort(ids, deps)
        stories = [make_story("A"), make_story("B"), make_story("C")]
        inputs = rice.default_rice_inputs(stories)
        with self.assertRaises(ValueError):
            build_roadmap(stories, inputs, deps)

    def test_unknown_dependency_raises(self):
        with self.assertRaises(ValueError):
            topological_sort(["A"], {"A": ["GHOST"]})

    def test_critical_path_on_tiny_known_graph(self):
        stories = [make_story("A"), make_story("B"), make_story("C")]
        deps = {"B": ["A"], "C": ["B"]}
        efforts = {"A": 2.0, "B": 3.0, "C": 5.0}
        path, total = critical_path(["A", "B", "C"], deps, efforts)
        self.assertEqual(path, ["A", "B", "C"])
        self.assertEqual(total, 10.0)

    def test_critical_path_picks_longer_branch(self):
        # A -> B(3) and A -> C(5): longest path is A -> C with total 7.
        stories = [make_story("A"), make_story("B"), make_story("C")]
        deps = {"B": ["A"], "C": ["A"]}
        efforts = {"A": 2.0, "B": 3.0, "C": 5.0}
        path, total = critical_path(["A", "B", "C"], deps, efforts)
        self.assertEqual(path, ["A", "C"])
        self.assertEqual(total, 7.0)


class ExportTests(unittest.TestCase):
    def test_roadmap_csv_columns_and_rows(self):
        extracted = extract_stories_demo(SAMPLE_PRD)
        inputs = rice.default_rice_inputs(extracted)
        deps = rice.infer_dependencies(extracted, SAMPLE_PRD)
        result = build_roadmap(extracted, inputs, deps)
        path = result["csv_path"]
        self.assertTrue(os.path.exists(path))
        df = pd.read_csv(path)
        self.assertEqual(list(df.columns), rice.CSV_COLUMNS)
        self.assertEqual(len(df), len(extracted))
        # RICE column sorted descending.
        rice_vals = df["RICE"].tolist()
        self.assertEqual(rice_vals, sorted(rice_vals, reverse=True))
        # RICE values recomputed consistently.
        for _, row in df.iterrows():
            expected = rice.rice_score(row["Reach"], row["Impact"],
                                       row["Confidence %"], row["Effort (pw)"])
            self.assertEqual(row["RICE"], expected)

    def test_mermaid_contains_every_story_id(self):
        extracted = extract_stories_demo(SAMPLE_PRD)
        deps = rice.infer_dependencies(extracted, SAMPLE_PRD)
        mermaid = rice.mermaid_flowchart(extracted, deps)
        self.assertTrue(mermaid.startswith("flowchart TD"))
        for story in extracted:
            self.assertIn(story.id, mermaid)
        for child, parents in deps.items():
            for parent in parents:
                self.assertIn("%s --> %s" % (parent, child), mermaid)


if __name__ == "__main__":
    unittest.main()
