"""
test_filter.py
---------------
Complete test suite for the 4-stage relevance filter.

Run options:
  python test_filter.py              → runs all unit tests (no API needed)
  python test_filter.py --live       → tests against your collected JSON
  python test_filter.py --llm        → tests Stage 4 Gemini with 3 borderline papers
  python test_filter.py --all        → everything above

Usage:
  source venv/bin/activate
  python test_filter.py
"""

import sys
import json
import os
from pathlib import Path
from loguru import logger

# ── Setup logging ─────────────────────────────────────────────────────────────
logger.remove()
logger.add(sys.stderr, level="INFO",
           format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}")


# ─────────────────────────────────────────────────────────────────────────────
# TEST 1 — Unit tests (no API, no data file needed)
# ─────────────────────────────────────────────────────────────────────────────

def test_unit():
    """
    Tests all 4 stages with known cases.
    Verifies each paper gets the correct verdict without calling any API.
    """
    from collectors.relevance_filter import RelevanceFilter
    from models import PaperRecord

    print("\n" + "="*55)
    print("TEST 1 — Unit tests (no API required)")
    print("="*55)

    f = RelevanceFilter()

    cases = [
        # (title, abstract, mesh_terms, expected_keep, label)
        (
            "Gut microbiome in IBD patients: a 16S rRNA cohort study",
            "We recruited 200 patients with IBD. 16S rRNA sequencing of fecal samples.",
            ["Gastrointestinal Microbiome", "Humans", "Inflammatory Bowel Diseases"],
            True, "✅ Human IBD cohort — KEEP"
        ),
        (
            "Zebrafish gut microbiome alters after antibiotic treatment",
            "Adult zebrafish were exposed to ampicillin for 7 days.",
            ["Animals", "Zebrafish"],
            False, "❌ Zebrafish model — REMOVE"
        ),
        (
            "The Microbiota of Homemade Tepache fermented beverage",
            "Tepache is a traditional Mexican fermented drink from pineapple rinds.",
            [],
            False, "❌ Food fermentation — REMOVE"
        ),
        (
            "Fecal microbiota transplantation in septic ICU patients: RCT",
            "Patients received FMT. 16S rRNA sequencing was performed. PRJNA123456.",
            ["Fecal Microbiota Transplantation", "Humans"],
            True, "✅ Human FMT RCT — KEEP"
        ),
        (
            "Soil microbiome diversity across agricultural ecosystems",
            "Rhizosphere samples were collected from wheat and maize fields.",
            [],
            False, "❌ Soil/environmental — REMOVE"
        ),
        (
            "Shotgun metagenomics of the oral microbiome in periodontitis patients",
            "Participants with periodontitis underwent shotgun metagenomic sequencing.",
            ["Microbiota", "Humans"],
            True, "✅ Human oral microbiome — KEEP"
        ),
        (
            "Gut microbiota of gopher tortoises in Florida",
            "Fecal samples were collected from wild-caught tortoises.",
            [],
            False, "❌ Tortoise study — REMOVE"
        ),
        (
            "Human gut microbiome and colorectal cancer: systematic review",
            "We systematically reviewed 52 studies on gut microbiome alterations in CRC patients.",
            ["Gastrointestinal Microbiome", "Colorectal Neoplasms", "Humans"],
            True, "✅ Human CRC systematic review — KEEP"
        ),
    ]

    passed = 0
    for title, abstract, mesh, expected, label in cases:
        paper = PaperRecord(title=title, abstract=abstract, mesh_terms=mesh)
        v = f._evaluate(paper)
        ok = v.keep == expected
        passed += ok
        status = "PASS" if ok else "FAIL ⚠️"
        print(f"\n  [{status}] {label}")
        print(f"           Stage={v.stage} | Score={v.score} | Review={v.review}")
        print(f"           Reason: {v.reason[:70]}")

    print(f"\n{'─'*55}")
    print(f"  Result: {passed}/{len(cases)} passed {'✅' if passed == len(cases) else '❌ check failures above'}")
    return passed == len(cases)


# ─────────────────────────────────────────────────────────────────────────────
# TEST 2 — Live test against your collected JSON
# ─────────────────────────────────────────────────────────────────────────────

def test_live():
    """
    Runs the filter on your most recently collected papers.
    Shows exactly which papers get kept, removed, or flagged.
    """
    from collectors.relevance_filter import RelevanceFilter
    from collectors.orchestrator import CollectionOrchestrator
    from models import PaperRecord

    print("\n" + "="*55)
    print("TEST 2 — Live test on collected papers")
    print("="*55)

    # Load latest collected file
    proc_dir = Path("data/processed")
    files = sorted(proc_dir.glob("collected_*.json"), reverse=True)

    if not files:
        print("  ⚠️  No collected data found.")
        print("  Run: MAX_PER_SOURCE=5 python main.py first")
        return False

    path = files[0]
    print(f"\n  Loading: {path.name}")

    with open(path) as f:
        data = json.load(f)
    papers = [PaperRecord(**p) for p in data]
    print(f"  Papers loaded: {len(papers)}")

    # Run filter
    rf = RelevanceFilter()
    kept, removed, review = rf.filter(papers)

    # Detailed results
    print(f"\n  RESULTS:")
    print(f"  Kept:    {len(kept)}")
    print(f"  Removed: {len(removed)}")
    print(f"  Review:  {len(review)}")

    if kept:
        print(f"\n  ✅ Sample KEPT papers:")
        for p in kept[:4]:
            print(f"     • {p.title[:65]}")

    if removed:
        print(f"\n  ❌ Sample REMOVED papers:")
        for p in removed[:4]:
            print(f"     • {p.title[:65]}")

    if review:
        print(f"\n  ? FLAGGED for review:")
        for p, v in review[:3]:
            print(f"     [{v.score:.2f}] {p.title[:60]}")

    # Source breakdown of removed papers
    if removed:
        from collections import Counter
        sources = Counter(p.source for p in removed)
        print(f"\n  Removed by source: {dict(sources)}")

    precision_ok = len(removed) > 0 or len(kept) == len(papers)
    print(f"\n  Status: {'✅ Filter ran successfully' if precision_ok else '⚠️ Check results'}")
    return True


# ─────────────────────────────────────────────────────────────────────────────
# TEST 3 — Stage 4 LLM test (Gemini)
# ─────────────────────────────────────────────────────────────────────────────

def test_llm():
    """
    Tests Stage 4 Gemini verifier with 3 carefully chosen borderline papers.
    These are papers that Stage 2 rules find ambiguous (score 0.40-0.70).
    Requires GEMINI_API_KEY in .env
    """
    from collectors.llm_verifier import LLMVerifier

    print("\n" + "="*55)
    print("TEST 3 — Stage 4 Gemini LLM verifier")
    print("="*55)

    verifier = LLMVerifier()

    if not verifier.is_available:
        print("\n  ⚠️  LLM not configured.")
        print("  Add to .env:")
        print("    LLM_PROVIDER=gemini")
        print("    GEMINI_API_KEY=your_key_here")
        print("  Get free key: https://aistudio.google.com")
        return False

    # Borderline papers — ambiguous enough that rules alone are uncertain
    borderline_cases = [
        (
            "Aging-associated gut microbial changes and their functional implications",
            "The gut microbiome undergoes significant changes during aging. "
            "This review synthesizes evidence from human studies and animal models "
            "to understand the functional implications of age-related microbial shifts.",
            True,   # Should keep — primarily a review of human data
            "borderline: mentions animal models but is a human review"
        ),
        (
            "Probiotic supplementation and gut microbiota: insights from mouse and human studies",
            "We investigated the effects of Lactobacillus supplementation in both "
            "germ-free mouse models and a parallel human cohort of 50 volunteers. "
            "16S rRNA sequencing was performed on fecal samples from both groups.",
            True,   # Keep — has human cohort data
            "borderline: mixed human+mouse but has human cohort"
        ),
        (
            "Microbiome diversity patterns across mammalian gut ecosystems",
            "Comparative analysis of gut microbiome diversity across 12 mammalian "
            "species including humans, chimpanzees, gorillas, and various rodents "
            "revealed convergent patterns in microbiome organization.",
            False,  # Reject — comparative across animals
            "borderline: includes humans but primarily comparative zoology"
        ),
    ]

    print(f"\n  Testing {len(borderline_cases)} borderline papers with Gemini...")
    print(f"  Provider: {os.getenv('LLM_PROVIDER')} | Model: {os.getenv('LLM_MODEL')}\n")

    passed = 0
    for title, abstract, expected, desc in borderline_cases:
        verdict = verifier.verify(title, abstract)
        ok = verdict.keep == expected
        passed += ok
        status = "PASS" if ok else "FAIL ⚠️"
        cached_str = " (cached)" if verdict.cached else " (API call)"
        print(f"  [{status}] {desc}")
        print(f"         keep={verdict.keep} | confidence={verdict.confidence:.2f}{cached_str}")
        print(f"         reason: {verdict.reason}")
        print()

    stats = verifier.cache_stats()
    print(f"  Cache: {stats['total_cached']} verdicts stored")
    print(f"  ─────────────────────────────────────────")
    print(f"  Result: {passed}/{len(borderline_cases)} passed {'✅' if passed == len(borderline_cases) else '❌'}")
    return passed >= 2   # Allow 1 miss — LLM can disagree on genuinely ambiguous cases


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = sys.argv[1:]
    run_all  = "--all"  in args
    run_live = "--live" in args or run_all
    run_llm  = "--llm"  in args or run_all

    print("\n🔬 Microbiome Miner — Relevance Filter Test Suite")

    results = {}
    results["unit"] = test_unit()

    if run_live:
        results["live"] = test_live()

    if run_llm:
        results["llm"] = test_llm()

    # Final summary
    print("\n" + "="*55)
    print("FINAL SUMMARY")
    print("="*55)
    for test, ok in results.items():
        icon = "✅" if ok else "❌"
        print(f"  {icon} {test}")

    if not run_live:
        print("\n  TIP: Run with --live to test on your collected papers")
        print("       Run with --llm  to test Gemini Stage 4")
        print("       Run with --all  for everything")
