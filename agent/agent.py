# agent.py
import json
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, START, END
from langsmith import traceable

from .kb_retriever import KBRetriever
from .github_tools import GitHubMCP
from .response_generator import ResponseGenerator
from utils.settings import settings

# Initialize modules
kb        = KBRetriever()
github    = GitHubMCP()
generator = ResponseGenerator()

MAX_ITERATIONS = 5

# ══════════════════════════════════════════════════════════════════════════ #
# Tool dispatcher                                                           #
# ══════════════════════════════════════════════════════════════════════════ #

GITHUB_TOOL_MAP = {
    "search_repositories":    lambda **p: github.search_repositories(**p),
    "get_file_contents":      lambda **p: github.get_file_contents(**p),
    "create_or_update_file":  lambda **p: github.create_or_update_file(**p),
    "delete_file":            lambda **p: github.delete_file(**p),
    "push_files":             lambda **p: github.push_files(**p),
    "create_repository":      lambda **p: github.create_repository(**p),
    "fork_repository":        lambda **p: github.fork_repository(**p),
    "list_branches":          lambda **p: github.list_branches(**p),
    "create_branch":          lambda **p: github.create_branch(**p),
    "list_commits":           lambda **p: github.list_commits(**p),
    "get_commit":             lambda **p: github.get_commit(**p),
    "list_issues":            lambda **p: github.list_issues(**p),
    "get_issue":              lambda **p: github.get_issue(**p),
    "create_issue":           lambda **p: github.create_issue(**p),
    "update_issue":           lambda **p: github.update_issue(**p),
    "add_issue_comment":      lambda **p: github.add_issue_comment(**p),
    "search_issues":          lambda **p: github.search_issues(**p),
    "list_pull_requests":     lambda **p: github.list_pull_requests(**p),
    "get_pull_request":       lambda **p: github.get_pull_request(**p),
    "create_pull_request":    lambda **p: github.create_pull_request(**p),
    "merge_pull_request":     lambda **p: github.merge_pull_request(**p),
    "get_pull_request_diff":  lambda **p: github.get_pull_request_diff(**p),
    "get_pull_request_files": lambda **p: github.get_pull_request_files(**p),
    "search_code":            lambda **p: github.search_code(**p),
    "search_users":           lambda **p: github.search_users(**p),
    "get_me":                 lambda **p: github.get_me(**p),
}


def _unwrap_mcp(raw):
    """Unwrap MCP envelope: {"content": [{"type": "text", "text": "<json>"}]}"""
    if not isinstance(raw, dict):
        return raw
    content = raw.get("content")
    if not content or not isinstance(content, list):
        return raw
    text = content[0].get("text", "")
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return text


def _call_github_tool(tool_name: str, params: dict) -> tuple:
    """
    Call a GitHub tool by name with the given params dict.
    Returns (unwrapped_result, error_message).
    """
    fn = GITHUB_TOOL_MAP.get(tool_name)
    if fn is None:
        return None, (
            f"Unknown GitHub tool '{tool_name}'. "
            f"Available: {', '.join(GITHUB_TOOL_MAP.keys())}"
        )
    try:
        return _unwrap_mcp(fn(**params)), ""
    except TypeError as e:
        return None, f"Wrong parameters for '{tool_name}': {e}"
    except Exception as e:
        return None, f"Tool '{tool_name}' raised an error: {e}"


def _summarise_result(tool_name: str, result) -> str:
    """Convert a tool result into a readable summary for the LLM."""
    
    # ─────────────────────────────────────────────────────────────────
    # Extract actual content from MCP response structure
    # MCP returns: {"content": [{"type": "text", "text": "..."}]}
    # ─────────────────────────────────────────────────────────────────
    content = result
    if isinstance(result, dict) and "content" in result:
        content_list = result.get("content", [])
        if content_list and isinstance(content_list[0], dict):
            text_content = content_list[0].get("text", "")
            if isinstance(text_content, str):
                try:
                    content = json.loads(text_content)
                except (json.JSONDecodeError, ValueError):
                    content = text_content

    # ─────────────────────────────────────────────────────────────────
    # Return full content without truncation
    # ─────────────────────────────────────────────────────────────────
    if isinstance(content, list):
        count = len(content)
        return f"Tool '{tool_name}' returned {count} item(s):\n{json.dumps(content, indent=2)}"
    
    if isinstance(content, dict):
        return f"Tool '{tool_name}' result:\n{json.dumps(content, indent=2)}"
    
    if isinstance(content, str):
        return f"Tool '{tool_name}' result:\n{content}"
    
    return f"Tool '{tool_name}' returned: {str(content)}"


# ══════════════════════════════════════════════════════════════════════════ #
# State Schema                                                              #
# ══════════════════════════════════════════════════════════════════════════ #

class AgentState(TypedDict):
    """State schema for the ReAct agent."""
    query:          str
    kb_summary:     Optional[str]
    github_summary: Optional[str]
    action:         Optional[str]
    action_input:   Optional[str]
    repo:           Optional[str]
    final_response: Optional[str]
    iterations:     Optional[int]


# ══════════════════════════════════════════════════════════════════════════ #
# Nodes                                                                     #
# ══════════════════════════════════════════════════════════════════════════ #

@traceable(run_type="chain", name="LLM Decision Node")
def llm_node(state: AgentState) -> dict:
    """LLM decides next action using ReAct framework."""
    iterations = state.get("iterations") or 0

    if iterations >= MAX_ITERATIONS:
        return {
            "action": "respond",
            "action_input": (
                state.get("kb_summary") or
                state.get("github_summary") or
                "I could not find a definitive answer after searching."
            ),
            "iterations": iterations,
        }

    result = generator.generate(
        state["query"],
        state.get("kb_summary") or "",
        state.get("github_summary") or "",
    )
    return {
        "action":       result["action"],
        "action_input": result["action_input"],
        "iterations":   iterations,
    }


@traceable(run_type="retriever", name="Knowledge Base Retrieval")
def kb_node(state: AgentState) -> dict:
    """Retrieve relevant documents from the knowledge base."""
    query   = state.get("action_input") or state["query"]
    docs    = kb.retrieve(query, top_k=3)
    summary = "\n".join([doc["text"][:200] + "..." for doc in docs])
    return {
        "kb_summary": summary,
        "iterations": (state.get("iterations") or 0) + 1,
    }



# ══════════════════════════════════════════════════════════════════════════ #
# Tool parameter requirements                                               #
# ══════════════════════════════════════════════════════════════════════════ #

# Tools that DON'T require owner/repo parameters
NO_REPO_TOOLS = {
    "get_me",              # Get authenticated user info
    "search_repositories", # Search across all repos
    "search_code",         # Search code across GitHub
    "search_users",        # Search users
    "search_issues",       # Search issues across repos (uses 'q' param)
    "create_repository",   # Creates a new repo (no existing repo needed)
}

# Tools that only require 'owner' (not 'repo')
OWNER_ONLY_TOOLS = {
    "list_branches",       # Some implementations might use owner only
}


@traceable(run_type="tool", name="GitHub API Call")
def github_node(state: AgentState) -> dict:
    """
    Dynamically dispatch any GitHub MCP tool.

    action_input can be:
      Format A (preferred): {"tool": "list_issues", "params": {"owner": "...", "repo": "..."}}
      Format B (fallback):  "README.md"  →  get_file_contents
                            ""           →  list_issues
    owner/repo auto-injected from state["repo"] when absent (only for tools that need them).
    """
    raw_input = state.get("action_input") or ""
    repo_full = state.get("repo") or ""

    default_owner, default_repo_name = "", ""
    parts = repo_full.split("/", 1)
    if len(parts) == 2:
        default_owner, default_repo_name = parts[0], parts[1]

    # Normalise action_input — dict, JSON string, or plain string
    if isinstance(raw_input, dict):
        parsed_input = raw_input
    elif isinstance(raw_input, str):
        try:
            parsed_input = json.loads(raw_input)
        except (json.JSONDecodeError, ValueError):
            parsed_input = raw_input
    else:
        parsed_input = {}

    # Resolve tool name and params
    tool_name = None
    params    = {}

    if isinstance(parsed_input, dict):
        if "tool" in parsed_input:
            tool_name = parsed_input["tool"]
            params    = parsed_input.get("params") or {}
            if not isinstance(params, dict):
                params = {}
        else:
            tool_name = "list_issues"
            params    = {k: v for k, v in parsed_input.items() if k != "path"}
    elif isinstance(parsed_input, str) and parsed_input.strip():
        tool_name = "get_file_contents"
        params    = {"path": parsed_input.strip()}
    else:
        tool_name = "list_issues"
        params    = {"state": "open"}

    # ═══════════════════════════════════════════════════════════════════
    # Inject owner/repo ONLY for tools that require them
    # ═══════════════════════════════════════════════════════════════════
    if tool_name not in NO_REPO_TOOLS:
        if default_owner and "owner" not in params:
            params["owner"] = default_owner
        if default_repo_name and "repo" not in params:
            params["repo"] = default_repo_name

    result, error = _call_github_tool(tool_name, params)

    github_summary = (
        f"[GitHub Error — tool='{tool_name}'] {error}"
        if error
        else _summarise_result(tool_name, result)
    )

    return {
        "github_summary": github_summary,
        "iterations":     (state.get("iterations") or 0) + 1,
    }


# Keywords that indicate the query is a GitHub ACTION (create/update/delete)
# vs a GitHub READ (list/get/show). Actions should report the result directly.
GITHUB_ACTION_KEYWORDS = [
    "create", "update", "delete", "merge", "push", "add", "close",
    "open", "fork", "branch", "commit", "pull request", "issue",
]


def _is_github_action(query: str) -> bool:
    """Return True if the query is asking the agent to DO something on GitHub."""
    q = query.lower()
    return any(kw in q for kw in GITHUB_ACTION_KEYWORDS)


@traceable(run_type="chain", name="Respond Node")
def respond_node(state: AgentState) -> dict:
    """Synthesize gathered context into a clean final answer."""
    query          = state.get("query") or ""
    kb_summary     = state.get("kb_summary") or ""
    github_summary = state.get("github_summary") or ""
    action_input   = state.get("action_input") or ""

    if github_summary and _is_github_action(query):
        # GitHub action (create/update/delete/merge...) — report the outcome
        # directly via a focused LLM call instead of a generic synthesis,
        # so the agent confirms what it DID rather than explaining how to do it.
        final_response = generator.report_action(query, github_summary)

    elif kb_summary or github_summary:
        # Informational query — synthesize retrieved context into a clean answer.
        final_response = generator.synthesize(query, kb_summary, github_summary)

    elif action_input:
        final_response = action_input

    else:
        final_response = "No response generated."

    return {"final_response": final_response}


# ══════════════════════════════════════════════════════════════════════════ #
# Routing                                                                   #
# ══════════════════════════════════════════════════════════════════════════ #

def route_action(state: AgentState) -> str:
    """Route to the appropriate node based on the LLM's action decision."""
    action = state.get("action", "respond")
    if action == "kb":
        return "kb_node"
    elif action == "github":
        return "github_node"
    else:
        return "respond_node"


# ══════════════════════════════════════════════════════════════════════════ #
# Graph                                                                     #
# ══════════════════════════════════════════════════════════════════════════ #

builder = StateGraph(AgentState)
builder.add_node("llm_node",     llm_node)
builder.add_node("kb_node",      kb_node)
builder.add_node("github_node",  github_node)
builder.add_node("respond_node", respond_node)

builder.add_edge(START, "llm_node")
builder.add_conditional_edges(
    "llm_node",
    route_action,
    {
        "kb_node":      "kb_node",
        "github_node":  "github_node",
        "respond_node": "respond_node",
    }
)
builder.add_edge("kb_node",      "llm_node")
builder.add_edge("github_node",  "llm_node")
builder.add_edge("respond_node", END)

graph = builder.compile()


# ══════════════════════════════════════════════════════════════════════════ #
# GithubProjectMentor — public interface                                    #
# ══════════════════════════════════════════════════════════════════════════ #

class GithubProjectMentor:
    """Main agent class for querying the ReAct graph."""

    DEFAULT_REPO = (
        "202422/Intelligent-system-for-automatic-correction-"
        "and-completion-for-short-text-exchange"
    )

    def __init__(self, repo: Optional[str] = None):
        self.graph        = graph
        self.default_repo = repo or self.DEFAULT_REPO

    @traceable(run_type="chain", name="GitHub Project Mentor Query")
    def query(
        self,
        user_query: str,
        repo: Optional[str] = None,
        history: Optional[list] = None,
    ) -> str:
        """
        Execute a user query through the ReAct agent graph.

        Args:
            user_query: the current user message
            repo:       optional override for the GitHub repository
            history:    list of {"role": "user"|"agent", "content": str}
                        representing previous turns — prepended to the query
                        so the LLM has full conversation context
        """
        newline = chr(10)

        if history:
            lines = ["=== Conversation History ==="]
            for turn in history:
                role = "User" if turn["role"] == "user" else "Agent"
                lines.append(role + ": " + turn["content"])
            lines.append("=== Current Question ===")
            lines.append(user_query)
            full_query = newline.join(lines)
        else:
            full_query = user_query

        initial_state: AgentState = {
            "query":          full_query,
            "repo":           repo or self.default_repo,
            "kb_summary":     None,
            "github_summary": None,
            "action":         None,
            "action_input":   None,
            "final_response": None,
            "iterations":     0,
        }
        result = self.graph.invoke(initial_state)
        return result.get("final_response", "No response generated.")


# ══════════════════════════════════════════════════════════════════════════ #
# Manual test                                                               #
# ══════════════════════════════════════════════════════════════════════════ #

if __name__ == "__main__":
    mentor = GithubProjectMentor()
    answer = mentor.query(
        "How does the intelligent system automatically correct spelling errors "
        "in short text messages?"
    )
    print("Final Response:", answer)