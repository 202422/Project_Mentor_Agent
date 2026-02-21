# test_kb_retriever.py
"""
Tests for KBRetriever — fully aligned with src/agent/kb_retriever.py.

Architecture assumed:
  src/
  ├── agent/kb_retriever.py
  ├── data/                   ← KB documents (PDFs, README.md)
  ├── embeddings/             ← FAISS index (auto-populated)
  └── test_kb_retriever.py   ← this file

Run from project root:
  python src/test_kb_retriever.py
"""
import os
import sys

# src/ must be on sys.path so both
#   "from agent.kb_retriever import KBRetriever"  and
#   "from utils.settings import settings"          (used inside kb_retriever)
# resolve correctly.
SRC_DIR = os.path.abspath(os.path.dirname(__file__))  # = …/Module 6/src
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from agent.kb_retriever import KBRetriever, SRC_DIR as KB_SRC_DIR




# ── Queries that match content in those specific documents ─────────────────
TEST_QUERIES = [
    "What is the project about?",
    "How does the auto-correction system work?",
    "What are the steps for creative problem solving?",
]


def separator(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print("=" * 60)


def test_kb_retriever():
    all_passed = True

    # ------------------------------------------------------------------ #
    # 0. Confirm SRC_DIR agreement between test and kb_retriever          #
    # ------------------------------------------------------------------ #
    separator("PRE-CHECK: SRC_DIR agreement")
    if SRC_DIR == KB_SRC_DIR:
        print(f"✅ SRC_DIR matches: {SRC_DIR}")
    else:
        print(f"❌ SRC_DIR mismatch!")
        print(f"   test sees    : {SRC_DIR}")
        print(f"   kb_retriever : {KB_SRC_DIR}")
        all_passed = False

    # ------------------------------------------------------------------ #
    # 1. Initialization                                                    #
    # ------------------------------------------------------------------ #
    separator("TEST 1: Initialization")
    kb = None
    try:
        kb = KBRetriever()
        print("✅ KBRetriever initialized successfully")
    except FileNotFoundError as e:
        print(f"❌ data_dir not found:\n  {e}")
        print(f"\n  → Create the folder and add documents:")
        print(f"       {os.path.join(SRC_DIR, 'data')}")
        return
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        return

    # ------------------------------------------------------------------ #
    # 2. Paths sanity-check (data_dir and persist_dir under src/)         #
    # ------------------------------------------------------------------ #
    separator("TEST 2: Path sanity (both dirs under src/)")
    for label, path in [("data_dir", kb.data_dir), ("persist_dir", kb.persist_dir)]:
        if path.startswith(SRC_DIR):
            print(f"  ✅ {label:12} : {path}")
        else:
            print(f"  ❌ {label:12} is NOT under src/: {path}")
            all_passed = False

    

    # ------------------------------------------------------------------ #
    # 4. Vectorstore is populated                                         #
    # ------------------------------------------------------------------ #
    separator("TEST 4: Vectorstore is populated")
    try:
        if kb.vectorstore is None:
            print("❌ vectorstore is None — no documents were embedded")
            all_passed = False
        else:
            n_vectors = kb.vectorstore.index.ntotal
            if n_vectors > 0:
                print(f"✅ Vectorstore contains {n_vectors} embedded chunks")
            else:
                print("❌ Vectorstore is empty (0 chunks)")
                all_passed = False
    except Exception as e:
        print(f"❌ Could not inspect vectorstore: {e}")
        all_passed = False

    # ------------------------------------------------------------------ #
    # 5. retrieve() returns results                                        #
    # ------------------------------------------------------------------ #
    separator("TEST 5: retrieve() returns results")
    for query in TEST_QUERIES:
        try:
            results = kb.retrieve(query, top_k=3)
            if results:
                print(f"  ✅ \"{query}\"")
                print(f"     → {len(results)} result(s), top score: {results[0]['score']:.4f}")
                print(f"     → preview: {results[0]['text'][:120].strip()!r}")
            else:
                print(f"  ❌ 0 results for: \"{query}\"")
                all_passed = False
        except Exception as e:
            print(f"  ❌ retrieve() error for \"{query}\": {e}")
            all_passed = False

    # ------------------------------------------------------------------ #
    # 6. Result schema: text (str), score (float), metadata (dict)        #
    # ------------------------------------------------------------------ #
    separator("TEST 6: Result schema")
    try:
        result = kb.retrieve(TEST_QUERIES[0], top_k=1)[0]
        for field, expected_type in [("text", str), ("score", float), ("metadata", dict)]:
            val = result.get(field)
            if val is None:
                print(f"  ❌ '{field}' MISSING")
                all_passed = False
            elif not isinstance(val, expected_type):
                print(f"  ❌ '{field}' should be {expected_type.__name__}, got {type(val).__name__}")
                all_passed = False
            else:
                print(f"  ✅ '{field}' → {expected_type.__name__}")
    except Exception as e:
        print(f"  ❌ Schema check error: {e}")
        all_passed = False

    # ------------------------------------------------------------------ #
    # 7. top_k is respected                                               #
    # ------------------------------------------------------------------ #
    separator("TEST 7: top_k parameter respected")
    for k in (1, 3, 5):
        try:
            results = kb.retrieve(TEST_QUERIES[0], top_k=k)
            actual = len(results)
            if actual <= k:
                print(f"  ✅ top_k={k} → got {actual} result(s)")
            else:
                print(f"  ❌ top_k={k} → got {actual} result(s) (too many!)")
                all_passed = False
        except Exception as e:
            print(f"  ❌ top_k={k} raised: {e}")
            all_passed = False

    # ------------------------------------------------------------------ #
    # Summary                                                             #
    # ------------------------------------------------------------------ #
    separator("SUMMARY")
    if all_passed:
        print("🎉 All tests passed — KBRetriever is working correctly.")
    else:
        print("⚠️  Some tests failed — see details above.")


if __name__ == "__main__":
    test_kb_retriever()