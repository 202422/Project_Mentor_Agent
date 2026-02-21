# test_agent.py
"""
Tests for agent.py — LangGraph ReAct orchestrator.
All queries are focused exclusively on:
  "Intelligent System for Automatic Correction and Completion
   for Short Text Exchange"

Graph flow:
  START → llm_node → route_action → kb_node / github_node / respond_node → END
                         ↑_______________|  (loop back after kb/github)

Run from src/:
  python test_agent.py

NOTE: Uses real LLM + real KB. GitHub MCP is patched to avoid network calls.
"""
import os
import sys
from unittest.mock import patch, MagicMock

SRC_DIR = os.path.abspath(os.path.dirname(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# ── Load agent.py directly (avoids collision with the agent/ package) ─────
import importlib.util

AGENT_DIR = os.path.join(SRC_DIR, "agent")
for _p in (SRC_DIR, AGENT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_spec = importlib.util.spec_from_file_location(
    "agent_module",
    os.path.join(AGENT_DIR, "agent.py")
)
agent_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(agent_module)

graph        = agent_module.graph
AgentState   = agent_module.AgentState
route_action = agent_module.route_action
llm_node     = agent_module.llm_node
kb_node      = agent_module.kb_node
respond_node = agent_module.respond_node


def separator(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print("=" * 60)


def make_state(**kwargs) -> AgentState:
    """Build a fully-initialized AgentState with safe defaults."""
    defaults = {
        "query":          "",
        "kb_summary":     None,
        "github_summary": None,
        "action":         None,
        "action_input":   None,
        "repo":           None,
        "final_response": None,
        "iterations":     0,
    }
    defaults.update(kwargs)
    return defaults  # type: ignore


def _mock_github():
    """MagicMock replacing GitHubMCP — prevents real network calls in tests."""
    mock = MagicMock()
    mock.get_file_contents.return_value = {
        "content": [{"type": "text", "text": '{"content": "# Intelligent Auto-Correction System\\nMain entry point."}'}]
    }
    mock.list_issues.return_value = {
        "content": [{"type": "text", "text": "[]"}]
    }
    mock.list_pull_requests.return_value = {
        "content": [{"type": "text", "text": "[]"}]
    }
    mock.list_branches.return_value = {
        "content": [{"type": "text", "text": '[{"name": "main"}]'}]
    }
    return mock


# ══════════════════════════════════════════════════════════════════════════ #
# UNIT TESTS                                                                #
# ══════════════════════════════════════════════════════════════════════════ #

def test_route_action():
    """route_action maps action strings to the correct node names."""
    separator("TEST 1: route_action routing logic")
    all_passed = True

    cases = [
        ("kb",      "kb_node"),
        ("github",  "github_node"),
        ("respond", "respond_node"),
        ("unknown", "respond_node"),   # fallback
        (None,      "respond_node"),   # missing action
    ]
    for action, expected in cases:
        state = make_state(action=action)
        got = route_action(state)
        ok = got == expected
        mark = "✅" if ok else "❌"
        print(f"  {mark} action={action!r:12} → {got!r}  (expected {expected!r})")
        if not ok:
            all_passed = False

    return all_passed


def test_respond_node():
    """respond_node fills final_response using the correct fallback chain."""
    separator("TEST 2: respond_node unit test")
    all_passed = True

    cases = [
        # (action_input, kb_summary, expected)
        ("Correction applied successfully.", None,                              "Correction applied successfully."),
        (None,                               "The system corrects typos...",    "The system corrects typos..."),
        (None,                               None,                              "No response generated."),
    ]
    for action_input, kb_summary, expected in cases:
        state = make_state(action_input=action_input, kb_summary=kb_summary)
        result = respond_node(state)
        got = result.get("final_response")
        ok = got == expected
        mark = "✅" if ok else "❌"
        print(f"  {mark} action_input={str(action_input)[:40]!r}, kb_summary={str(kb_summary)[:30]!r}")
        print(f"       → final_response={got!r}")
        if not ok:
            print(f"       ❌ expected: {expected!r}")
            all_passed = False

    return all_passed


def test_kb_node():
    """kb_node retrieves relevant chunks for a project-specific query."""
    separator("TEST 3: kb_node — auto-correction project query")
    all_passed = True

    state = make_state(
        action_input="How does the system automatically correct spelling errors in short text messages?"
    )
    try:
        result = kb_node(state)
        summary = result.get("kb_summary", "")
        iterations = result.get("iterations", 0)

        if summary:
            print(f"  ✅ kb_summary populated ({len(summary)} chars)")
            print(f"     preview: {summary[:150].strip()!r}")
        else:
            print("  ❌ kb_summary is empty")
            all_passed = False

        if iterations == 1:
            print(f"  ✅ iterations incremented to {iterations}")
        else:
            print(f"  ❌ iterations should be 1, got {iterations}")
            all_passed = False

    except Exception as e:
        print(f"  ❌ kb_node raised: {e}")
        all_passed = False

    return all_passed


def test_llm_node_returns_action():
    """llm_node returns a valid action for a project-specific query."""
    separator("TEST 4: llm_node — project query routing decision")
    all_passed = True

    state = make_state(
        query="What NLP techniques does the intelligent correction system use?",
        kb_summary=(
            "The system uses sequence-to-sequence models and transformer-based "
            "architectures to correct and complete short text messages automatically."
        ),
        github_summary=None,
    )
    try:
        result = llm_node(state)
        action       = result.get("action")
        action_input = result.get("action_input")

        print(f"  action       : {action!r}")
        print(f"  action_input : {str(action_input)[:120]!r}")

        if action in ("kb", "github", "respond"):
            print(f"  ✅ action is a valid value: {action!r}")
        else:
            print(f"  ❌ unexpected action: {action!r} (expected kb/github/respond)")
            all_passed = False

        if action_input is not None:
            print(f"  ✅ action_input is populated")
        else:
            print(f"  ❌ action_input is None")
            all_passed = False

    except Exception as e:
        print(f"  ❌ llm_node raised: {e}")
        all_passed = False

    return all_passed


# ══════════════════════════════════════════════════════════════════════════ #
# INTEGRATION TESTS — full graph.invoke()                                   #
# ══════════════════════════════════════════════════════════════════════════ #

def test_full_graph_kb_query():
    """Full graph — query that should route to KB for project documentation."""
    separator("TEST 5: Full graph — KB-routed project query")
    all_passed = True

    with patch.object(agent_module, "github", _mock_github()):
        state = make_state(
            query=(
                "Explain how the intelligent system performs automatic text "
                "completion for short message exchanges."
            ),
            repo=None,
        )
        try:
            result = graph.invoke(state, config={"recursion_limit": 15})
            final = result.get("final_response")
            if final:
                print(f"  ✅ final_response received ({len(final)} chars)")
                print(f"     preview: {final[:200].strip()!r}")
            else:
                print("  ❌ final_response is None or empty")
                all_passed = False
        except Exception as e:
            print(f"  ❌ graph.invoke raised: {e}")
            all_passed = False

    return all_passed


def test_full_graph_direct_respond():
    """Full graph — simple question LLM should answer directly from context."""
    separator("TEST 6: Full graph — direct respond (no tools needed)")
    all_passed = True

    with patch.object(agent_module, "github", _mock_github()):
        state = make_state(
            query="What does 'auto-correction' mean in the context of this project?",
            kb_summary=(
                "Auto-correction refers to the automatic detection and fixing of "
                "spelling and grammar errors in short text messages without user intervention."
            ),
            repo=None,
        )
        try:
            result = graph.invoke(state, config={"recursion_limit": 15})
            final = result.get("final_response")
            if final:
                print(f"  ✅ final_response received: {final[:200].strip()!r}")
            else:
                print("  ❌ final_response is None or empty")
                all_passed = False
        except Exception as e:
            print(f"  ❌ graph.invoke raised: {e}")
            all_passed = False

    return all_passed


def test_full_graph_github_query():
    """Full graph — GitHub-routed query about the project repository."""
    separator("TEST 7: Full graph — GitHub query (MCP patched)")
    all_passed = True

    with patch.object(agent_module, "github", _mock_github()):
        state = make_state(
            query=(
                "Check the open issues in the intelligent auto-correction "
                "system repository."
            ),
            repo="202422/Intelligent-system-for-automatic-correction-and-completion-for-short-text-exchange",
        )
        try:
            result = graph.invoke(state, config={"recursion_limit": 15})
            final = result.get("final_response")
            if final:
                print(f"  ✅ final_response received: {final[:200].strip()!r}")
            else:
                print("  ❌ final_response is None or empty")
                all_passed = False
        except Exception as e:
            print(f"  ❌ graph.invoke raised: {e}")
            all_passed = False

    return all_passed


def test_state_fields_populated():
    """State fields are all present after graph.invoke() completes."""
    separator("TEST 8: State fields populated after invoke()")
    all_passed = True

    with patch.object(agent_module, "github", _mock_github()):
        state = make_state(
            query="What are the development guidelines for this auto-correction project?"
        )
        try:
            result = graph.invoke(state, config={"recursion_limit": 15})
            required_fields = ["query", "final_response", "action", "action_input", "iterations"]
            for field in required_fields:
                if field in result:
                    print(f"  ✅ '{field}' present → {str(result[field])[:60]!r}")
                else:
                    print(f"  ❌ '{field}' MISSING from final state")
                    all_passed = False
        except Exception as e:
            print(f"  ❌ graph.invoke raised: {e}")
            all_passed = False

    return all_passed


def test_recursion_limit_safety():
    """Agent hits MAX_ITERATIONS guard and produces a response without looping."""
    separator("TEST 9: Recursion / MAX_ITERATIONS safety guard")
    all_passed = True

    with patch.object(agent_module, "github", _mock_github()):
        state = make_state(
            query=(
                "Keep searching for every detail about the correction algorithm, "
                "the completion model, the dataset, the evaluation metrics, "
                "and the deployment strategy."
            )
        )
        try:
            result = graph.invoke(state, config={"recursion_limit": 20})
            final = result.get("final_response")
            iters = result.get("iterations", 0)
            print(f"  ✅ Graph completed — iterations used: {iters}")
            print(f"     final_response: {str(final)[:120]!r}")
        except Exception as e:
            if "recursion" in str(e).lower():
                print(f"  ❌ Recursion limit hit — MAX_ITERATIONS guard not working.")
                print(f"     Make sure agent.py has the iterations counter implemented.")
            else:
                print(f"  ❌ Unexpected error: {e}")
            all_passed = False

    return all_passed


def test_github_method_names():
    """Warn if agent.py still calls old GitHubMCP method names."""
    separator("TEST 10: agent.py GitHub method name check")
    import inspect

    source = inspect.getsource(agent_module.github_node)
    issues = []

    if "get_file(" in source:
        issues.append("  ⚠️  github_node calls get_file() — renamed to get_file_contents(owner, repo, path)")
    if "list_issues(repo)" in source:
        issues.append("  ⚠️  github_node calls list_issues(repo) — now requires list_issues(owner, repo)")

    if issues:
        for msg in issues:
            print(msg)
        print("  → Update github_node in agent.py to use the new GitHubMCP API")
        return False

    print("  ✅ GitHub method names are correct")
    return True


# ══════════════════════════════════════════════════════════════════════════ #
# RUNNER                                                                    #
# ══════════════════════════════════════════════════════════════════════════ #

def run_all():
    results = {
        "route_action":        test_route_action(),
        "respond_node":        test_respond_node(),
        "kb_node":             test_kb_node(),
        "llm_node":            test_llm_node_returns_action(),
        "full_graph_kb":       test_full_graph_kb_query(),
        "full_graph_direct":   test_full_graph_direct_respond(),
        "full_graph_github":   test_full_graph_github_query(),
        "state_fields":        test_state_fields_populated(),
        "recursion_safety":    test_recursion_limit_safety(),
        "github_method_names": test_github_method_names(),
    }

    separator("FINAL SUMMARY")
    passed = sum(1 for v in results.values() if v)
    total  = len(results)
    for name, ok in results.items():
        mark = "✅" if ok else "❌"
        print(f"  {mark} {name}")
    print(f"\n  {passed}/{total} tests passed")
    if passed == total:
        print("\n🎉 Agent is working correctly end-to-end.")
    else:
        print("\n⚠️  Some tests failed — see details above.")


if __name__ == "__main__":
    run_all()