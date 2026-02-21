# test_github_mcp.py
from agent.github_tools import GitHubMCP
from utils.settings import settings
import json


def unwrap_mcp(result) -> any:
    """
    Every MCP tools/call response has this shape:
        { "content": [ { "type": "text", "text": "<json string>" } ] }
    This helper unwraps it to the parsed Python object inside .text.
    """
    if not isinstance(result, dict):
        return result
    content = result.get("content", [])
    if not content:
        return result
    text = content[0].get("text", "")
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return text


def test_github_mcp():
    print("Initializing GitHubMCP...")
    gh = GitHubMCP()

    # ------------------------------------------------------------------ #
    # 1. get_me                                                            #
    # ------------------------------------------------------------------ #
    print("\nTesting: get authenticated user")
    owner = None
    try:
        me = unwrap_mcp(gh.get_me())
        owner = me.get("login")
        print(f"  Authenticated as : {owner}")
        print(f"  Name             : {me.get('details', {}).get('name')}")
        print(f"  Public repos     : {me.get('details', {}).get('public_repos')}")
    except Exception as e:
        print(f"  Error: {e}")

    if not owner:
        print("\nCould not determine owner — aborting further tests.")
        return

    # ------------------------------------------------------------------ #
    # 2. Search repositories                                               #
    # ------------------------------------------------------------------ #
    print(f"\nTesting: search repositories for '{owner}'")
    repos = []
    try:
        repos_data = unwrap_mcp(gh.search_repositories(query=f"user:{owner}"))
        # search_repositories returns { "total_count": N, "items": [...] }
        repos = repos_data.get("items", []) if isinstance(repos_data, dict) else repos_data
        print(f"  Found {len(repos)} repositories:")
        for r in repos[:10]:
            print(f"    - {r.get('full_name') or r.get('name')}")
    except Exception as e:
        print(f"  Error: {e}")

    if not repos:
        print("  No repositories found — skipping repo-level tests.")
        return

    repo_name = repos[0].get("name")
    print(f"\nUsing repo '{owner}/{repo_name}' for further tests.")

    # ------------------------------------------------------------------ #
    # 3. List issues                                                       #
    # ------------------------------------------------------------------ #
    print(f"\nTesting: list issues in '{owner}/{repo_name}'")
    try:
        issues = unwrap_mcp(gh.list_issues(owner=owner, repo=repo_name, state="open"))
        issues = issues if isinstance(issues, list) else []
        print(f"  Found {len(issues)} open issue(s)")
        for i, issue in enumerate(issues[:5], 1):
            print(f"    {i}. #{issue.get('number')} — {issue.get('title')}")
    except Exception as e:
        print(f"  Error: {e}")

    # ------------------------------------------------------------------ #
    # 4. Get file contents                                                 #
    # ------------------------------------------------------------------ #
    print(f"\nTesting: get README.md from '{owner}/{repo_name}'")
    try:
        file_data = unwrap_mcp(gh.get_file_contents(owner=owner, repo=repo_name, path="README.md"))
        # get_file_contents returns { "content": "<base64>", "encoding": "base64", ... }
        # or a plain text preview depending on the server
        content = file_data.get("content") if isinstance(file_data, dict) else str(file_data)
        print(f"  Preview (first 200 chars): {str(content)[:200]}")
    except Exception as e:
        print(f"  Error: {e}")

    # ------------------------------------------------------------------ #
    # 5. List branches                                                     #
    # ------------------------------------------------------------------ #
    print(f"\nTesting: list branches in '{owner}/{repo_name}'")
    try:
        branches = unwrap_mcp(gh.list_branches(owner=owner, repo=repo_name))
        branches = branches if isinstance(branches, list) else []
        print(f"  Found {len(branches)} branch(es):")
        for b in branches[:5]:
            print(f"    - {b.get('name')}")
    except Exception as e:
        print(f"  Error: {e}")

    # ------------------------------------------------------------------ #
    # 6. List pull requests                                                #
    # ------------------------------------------------------------------ #
    print(f"\nTesting: list pull requests in '{owner}/{repo_name}'")
    try:
        prs = unwrap_mcp(gh.list_pull_requests(owner=owner, repo=repo_name, state="open"))
        prs = prs if isinstance(prs, list) else []
        print(f"  Found {len(prs)} open PR(s)")
        for pr in prs[:5]:
            print(f"    - #{pr.get('number')} {pr.get('title')}")
    except Exception as e:
        print(f"  Error: {e}")

    # ------------------------------------------------------------------ #
    # 7. Search code                                                       #
    # ------------------------------------------------------------------ #
    print(f"\nTesting: search code in '{owner}/{repo_name}'")
    try:
        code_data = unwrap_mcp(gh.search_code(q=f"repo:{owner}/{repo_name} def"))
        items = code_data.get("items", []) if isinstance(code_data, dict) else []
        print(f"  Found {len(items)} result(s)")
        for item in items[:3]:
            print(f"    - {item.get('path') or item.get('name')}")
    except Exception as e:
        print(f"  Error: {e}")


if __name__ == "__main__":
    test_github_mcp()