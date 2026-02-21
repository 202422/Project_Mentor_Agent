"""
GitHubMCP: Client for the official GitHub remote MCP server.

Base URL  : https://api.githubcopilot.com/mcp/
Transport : MCP Streamable HTTP  (JSON-RPC 2.0 over plain HTTP POST)
Auth      : Bearer GitHub PAT

Protocol flow (per MCP spec 2025-03-26):
  1. POST initialize   → server returns JSON-RPC result + Mcp-Session-Id header
  2. POST notifications/initialized  (notification, no id field, include session header)
  3. POST tools/call   (include Mcp-Session-Id on every subsequent request)

No extra dependencies — only the standard `requests` library is needed.
"""

from typing import Any, Dict, List, Optional
import requests
from utils.settings import settings

GITHUB_MCP_URL = "https://api.githubcopilot.com/mcp/"
MCP_PROTOCOL_VERSION = "2025-03-26"


class GitHubMCP:
    def __init__(
        self,
        mcp_url: Optional[str] = None,
        token: Optional[str] = None,
    ):
        self.mcp_url = (
            mcp_url
            or getattr(settings, "GITHUB_MCP_URL", None)
            or GITHUB_MCP_URL
        ).rstrip("/") + "/"

        self.token = token or settings.GITHUB_TOKEN
        if not self.token:
            raise ValueError(
                "GitHub token must be provided via argument or settings.GITHUB_TOKEN."
            )

        self._session = requests.Session()
        self._session.headers.update({
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
            # Must accept both; server may respond with JSON or SSE
            "Accept": "application/json, text/event-stream",
        })
        self._req_id = 0
        self._session_id: Optional[str] = None

        self._initialize()

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _next_id(self) -> int:
        self._req_id += 1
        return self._req_id

    def _post(self, payload: Dict, expect_response: bool = True) -> Any:
        """
        POST one JSON-RPC message. Attaches Mcp-Session-Id when available.
        Returns parsed 'result' for requests; None for notifications.
        """
        headers: Dict[str, str] = {}
        if self._session_id:
            headers["Mcp-Session-Id"] = self._session_id

        resp = self._session.post(
            self.mcp_url,
            json=payload,
            headers=headers,
            timeout=30,
        )
        resp.raise_for_status()

        # Capture session ID the first time the server sends it
        if "Mcp-Session-Id" in resp.headers and not self._session_id:
            self._session_id = resp.headers["Mcp-Session-Id"]

        # Notifications get a 202 / empty body — nothing to parse
        if not expect_response or not resp.content or resp.status_code == 202:
            return None

        # Server may reply with SSE ("data: {...}\n\n") or plain JSON
        content_type = resp.headers.get("Content-Type", "")
        if "text/event-stream" in content_type:
            data = self._parse_sse(resp.text)
        else:
            data = resp.json()

        if "error" in data:
            err = data["error"]
            raise RuntimeError(
                f"MCP error {err.get('code')}: {err.get('message')} "
                f"— {err.get('data', '')}"
            )

        return data.get("result")

    @staticmethod
    def _parse_sse(text: str) -> Dict:
        """Extract the first JSON object from an SSE stream body."""
        for line in text.splitlines():
            line = line.strip()
            if line.startswith("data:"):
                import json
                return json.loads(line[len("data:"):].strip())
        raise ValueError(f"No data event found in SSE response:\n{text}")

    def _initialize(self) -> None:
        """
        Mandatory MCP handshake:
          1. initialize  (request — expects a result + session ID header)
          2. notifications/initialized  (notification — no response expected)
        """
        self._post({
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": "initialize",
            "params": {
                "protocolVersion": MCP_PROTOCOL_VERSION,
                "capabilities": {"tools": {}},
                "clientInfo": {"name": "GitHubMCP-python", "version": "1.0.0"},
            },
        }, expect_response=True)

        # Notification: no "id" field, no response expected
        self._post({
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
            "params": {},
        }, expect_response=False)

    def call_tool(self, tool: str, arguments: Optional[Dict] = None) -> Any:
        """Call a named MCP tool and return its result payload."""
        return self._post({
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": "tools/call",
            "params": {
                "name": tool,
                "arguments": arguments or {},
            },
        })

    # ------------------------------------------------------------------ #
    # Repository tools                                                     #
    # ------------------------------------------------------------------ #

    def search_repositories(
        self,
        query: str,
        page: Optional[int] = None,
        per_page: Optional[int] = None,
    ) -> Any:
        """Search GitHub repositories."""
        args: Dict[str, Any] = {"query": query}
        if page:
            args["page"] = page
        if per_page:
            args["perPage"] = per_page
        return self.call_tool("search_repositories", args)

    def get_file_contents(
        self,
        owner: str,
        repo: str,
        path: str,
        branch: Optional[str] = None,
    ) -> Any:
        """Get the contents of a file or directory."""
        args: Dict[str, Any] = {"owner": owner, "repo": repo, "path": path}
        if branch:
            args["branch"] = branch
        return self.call_tool("get_file_contents", args)

    def create_or_update_file(
        self,
        owner: str,
        repo: str,
        path: str,
        content: str,
        message: str,
        branch: str,
        sha: Optional[str] = None,
    ) -> Any:
        """Create or update a single file. Provide sha when updating."""
        args: Dict[str, Any] = {
            "owner": owner, "repo": repo, "path": path,
            "content": content, "message": message, "branch": branch,
        }
        if sha:
            args["sha"] = sha
        return self.call_tool("create_or_update_file", args)

    def delete_file(
        self, owner: str, repo: str, path: str, message: str, branch: str
    ) -> Any:
        """Delete a file from a GitHub repository."""
        return self.call_tool("delete_file", {
            "owner": owner, "repo": repo,
            "path": path, "message": message, "branch": branch,
        })

    def push_files(
        self,
        owner: str,
        repo: str,
        branch: str,
        files: List[Dict[str, str]],
        message: str,
    ) -> Any:
        """Push multiple files in a single commit."""
        return self.call_tool("push_files", {
            "owner": owner, "repo": repo,
            "branch": branch, "files": files, "message": message,
        })

    def list_branches(
        self,
        owner: str,
        repo: str,
        page: Optional[int] = None,
        per_page: Optional[int] = None,
    ) -> Any:
        """List branches in a GitHub repository."""
        args: Dict[str, Any] = {"owner": owner, "repo": repo}
        if page:
            args["page"] = page
        if per_page:
            args["perPage"] = per_page
        return self.call_tool("list_branches", args)

    def create_branch(
        self, owner: str, repo: str, branch: str, from_branch: Optional[str] = None
    ) -> Any:
        """Create a new branch."""
        args: Dict[str, Any] = {"owner": owner, "repo": repo, "branch": branch}
        if from_branch:
            args["from_branch"] = from_branch
        return self.call_tool("create_branch", args)

    def list_commits(
        self,
        owner: str,
        repo: str,
        sha: Optional[str] = None,
        page: Optional[int] = None,
        per_page: Optional[int] = None,
    ) -> Any:
        """List commits on a branch."""
        args: Dict[str, Any] = {"owner": owner, "repo": repo}
        if sha:
            args["sha"] = sha
        if page:
            args["page"] = page
        if per_page:
            args["perPage"] = per_page
        return self.call_tool("list_commits", args)

    def get_commit(
        self, owner: str, repo: str, sha: str,
        page: Optional[int] = None, per_page: Optional[int] = None,
    ) -> Any:
        """Get details for a specific commit."""
        args: Dict[str, Any] = {"owner": owner, "repo": repo, "sha": sha}
        if page:
            args["page"] = page
        if per_page:
            args["perPage"] = per_page
        return self.call_tool("get_commit", args)

    def create_repository(
        self,
        name: str,
        description: Optional[str] = None,
        private: bool = False,
        auto_init: bool = False,
    ) -> Any:
        """Create a new GitHub repository."""
        args: Dict[str, Any] = {"name": name, "private": private, "autoInit": auto_init}
        if description:
            args["description"] = description
        return self.call_tool("create_repository", args)

    def fork_repository(
        self, owner: str, repo: str, organization: Optional[str] = None
    ) -> Any:
        """Fork a repository to your account or an organization."""
        args: Dict[str, Any] = {"owner": owner, "repo": repo}
        if organization:
            args["organization"] = organization
        return self.call_tool("fork_repository", args)

    # ------------------------------------------------------------------ #
    # Issue tools                                                          #
    # ------------------------------------------------------------------ #

    def list_issues(
        self,
        owner: str,
        repo: str,
        state: Optional[str] = "open",
        labels: Optional[List[str]] = None,
        sort: Optional[str] = None,
        direction: Optional[str] = None,
        since: Optional[str] = None,
        page: Optional[int] = None,
        per_page: Optional[int] = None,
    ) -> Any:
        """List issues in a repository."""
        args: Dict[str, Any] = {"owner": owner, "repo": repo}
        if state:
            args["state"] = state
        if labels:
            args["labels"] = labels
        if sort:
            args["sort"] = sort
        if direction:
            args["direction"] = direction
        if since:
            args["since"] = since
        if page:
            args["page"] = page
        if per_page:
            args["perPage"] = per_page
        return self.call_tool("list_issues", args)

    def get_issue(self, owner: str, repo: str, issue_number: int) -> Any:
        """Get details of a specific issue."""
        return self.call_tool("get_issue", {
            "owner": owner, "repo": repo, "issue_number": issue_number
        })

    def create_issue(
        self,
        owner: str,
        repo: str,
        title: str,
        body: str = "",
        assignees: Optional[List[str]] = None,
        labels: Optional[List[str]] = None,
        milestone: Optional[int] = None,
    ) -> Any:
        """Create a new issue."""
        args: Dict[str, Any] = {"owner": owner, "repo": repo, "title": title}
        if body:
            args["body"] = body
        if assignees:
            args["assignees"] = assignees
        if labels:
            args["labels"] = labels
        if milestone:
            args["milestone"] = milestone
        return self.call_tool("create_issue", args)

    def update_issue(
        self,
        owner: str,
        repo: str,
        issue_number: int,
        title: Optional[str] = None,
        body: Optional[str] = None,
        state: Optional[str] = None,
        labels: Optional[List[str]] = None,
        assignees: Optional[List[str]] = None,
    ) -> Any:
        """Update an existing issue."""
        args: Dict[str, Any] = {"owner": owner, "repo": repo, "issue_number": issue_number}
        if title:
            args["title"] = title
        if body:
            args["body"] = body
        if state:
            args["state"] = state
        if labels is not None:
            args["labels"] = labels
        if assignees is not None:
            args["assignees"] = assignees
        return self.call_tool("update_issue", args)

    def add_issue_comment(
        self, owner: str, repo: str, issue_number: int, body: str
    ) -> Any:
        """Add a comment to an issue."""
        return self.call_tool("add_issue_comment", {
            "owner": owner, "repo": repo,
            "issue_number": issue_number, "body": body,
        })

    def search_issues(
        self,
        q: str,
        sort: Optional[str] = None,
        order: Optional[str] = None,
        page: Optional[int] = None,
        per_page: Optional[int] = None,
    ) -> Any:
        """Search issues/PRs using GitHub issues search syntax."""
        args: Dict[str, Any] = {"q": q}
        if sort:
            args["sort"] = sort
        if order:
            args["order"] = order
        if page:
            args["page"] = page
        if per_page:
            args["perPage"] = per_page
        return self.call_tool("search_issues", args)

    # ------------------------------------------------------------------ #
    # Pull request tools                                                   #
    # ------------------------------------------------------------------ #

    def list_pull_requests(
        self,
        owner: str,
        repo: str,
        state: Optional[str] = "open",
        head: Optional[str] = None,
        base: Optional[str] = None,
        sort: Optional[str] = None,
        direction: Optional[str] = None,
        page: Optional[int] = None,
        per_page: Optional[int] = None,
    ) -> Any:
        """List pull requests in a repository."""
        args: Dict[str, Any] = {"owner": owner, "repo": repo}
        if state:
            args["state"] = state
        if head:
            args["head"] = head
        if base:
            args["base"] = base
        if sort:
            args["sort"] = sort
        if direction:
            args["direction"] = direction
        if page:
            args["page"] = page
        if per_page:
            args["perPage"] = per_page
        return self.call_tool("list_pull_requests", args)

    def get_pull_request(self, owner: str, repo: str, pull_number: int) -> Any:
        """Get details of a specific pull request."""
        return self.call_tool("get_pull_request", {
            "owner": owner, "repo": repo, "pullNumber": pull_number
        })

    def create_pull_request(
        self,
        owner: str,
        repo: str,
        title: str,
        head: str,
        base: str,
        body: str = "",
        draft: bool = False,
        maintainer_can_modify: bool = True,
    ) -> Any:
        """Create a new pull request."""
        return self.call_tool("create_pull_request", {
            "owner": owner, "repo": repo, "title": title,
            "head": head, "base": base, "body": body,
            "draft": draft, "maintainer_can_modify": maintainer_can_modify,
        })

    def merge_pull_request(
        self,
        owner: str,
        repo: str,
        pull_number: int,
        merge_method: str = "merge",
        commit_title: Optional[str] = None,
        commit_message: Optional[str] = None,
    ) -> Any:
        """Merge a pull request."""
        args: Dict[str, Any] = {
            "owner": owner, "repo": repo,
            "pullNumber": pull_number, "merge_method": merge_method,
        }
        if commit_title:
            args["commit_title"] = commit_title
        if commit_message:
            args["commit_message"] = commit_message
        return self.call_tool("merge_pull_request", args)

    def get_pull_request_diff(self, owner: str, repo: str, pull_number: int) -> Any:
        """Get the diff of a pull request."""
        return self.call_tool("get_pull_request_diff", {
            "owner": owner, "repo": repo, "pullNumber": pull_number
        })

    def get_pull_request_files(self, owner: str, repo: str, pull_number: int) -> Any:
        """Get files changed in a pull request."""
        return self.call_tool("get_pull_request_files", {
            "owner": owner, "repo": repo, "pullNumber": pull_number
        })

    # ------------------------------------------------------------------ #
    # Code / user search                                                   #
    # ------------------------------------------------------------------ #

    def search_code(
        self,
        q: str,
        sort: Optional[str] = None,
        order: Optional[str] = None,
        page: Optional[int] = None,
        per_page: Optional[int] = None,
    ) -> Any:
        """Search code using GitHub code search syntax."""
        args: Dict[str, Any] = {"q": q}
        if sort:
            args["sort"] = sort
        if order:
            args["order"] = order
        if page:
            args["page"] = page
        if per_page:
            args["perPage"] = per_page
        return self.call_tool("search_code", args)

    def search_users(
        self,
        q: str,
        sort: Optional[str] = None,
        order: Optional[str] = None,
        page: Optional[int] = None,
        per_page: Optional[int] = None,
    ) -> Any:
        """Search for GitHub users."""
        args: Dict[str, Any] = {"q": q}
        if sort:
            args["sort"] = sort
        if order:
            args["order"] = order
        if page:
            args["page"] = page
        if per_page:
            args["perPage"] = per_page
        return self.call_tool("search_users", args)

    # ------------------------------------------------------------------ #
    # Auth / user                                                          #
    # ------------------------------------------------------------------ #

    def get_me(self, reason: Optional[str] = None) -> Any:
        """Get details of the authenticated GitHub user."""
        args = {"reason": reason} if reason else {}
        return self.call_tool("get_me", args)