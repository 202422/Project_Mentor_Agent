GITHUB_TOOLS_DESCRIPTION = """
=== AVAILABLE GITHUB TOOLS ===

Use action="github" and set action_input to the tool name + parameters as JSON.
Format: {"tool": "<tool_name>", "params": { ... }}

- owner: GitHub username or org (e.g. "202422")
- repo:  Repository name (e.g. "Intelligent-system-for-automatic-correction-and-completion-for-short-text-exchange")
- Parameters marked with ? are optional.

── REPOSITORY ──────────────────────────────────────────────
search_repositories(query, page?, per_page?)
  → Search GitHub repositories by keyword.

get_file_contents(owner, repo, path, branch?)
  → Get contents of a file or directory in a repository.

create_or_update_file(owner, repo, path, content, message, branch, sha?)
  → Create or update a single file. Provide sha when updating existing file.

delete_file(owner, repo, path, message, branch)
  → Delete a file from a repository.

push_files(owner, repo, branch, files, message)
  → Push multiple files in a single commit.
  → files: [{"path": "...", "content": "..."}, ...]

create_repository(name, description?, private?, auto_init?)
  → Create a new GitHub repository.

fork_repository(owner, repo, organization?)
  → Fork a repository to your account or an organization.

── BRANCHES & COMMITS ──────────────────────────────────────
list_branches(owner, repo, page?, per_page?)
  → List all branches in a repository.

create_branch(owner, repo, branch, from_branch?)
  → Create a new branch. Optionally base it on another branch.

list_commits(owner, repo, sha?, page?, per_page?)
  → List commits on a branch.

get_commit(owner, repo, sha, page?, per_page?)
  → Get full details for a specific commit.

── ISSUES ──────────────────────────────────────────────────
list_issues(owner, repo, state?, labels?, sort?, direction?, since?, page?, per_page?)
  → List issues. state: "open" | "closed" | "all". Default: "open".

get_issue(owner, repo, issue_number)
  → Get details of a specific issue by number.

create_issue(owner, repo, title, body?, assignees?, labels?, milestone?)
  → Create a new issue in a repository.

update_issue(owner, repo, issue_number, title?, body?, state?, labels?, assignees?)
  → Update an existing issue (title, body, state, labels, assignees).

add_issue_comment(owner, repo, issue_number, body)
  → Add a comment to an existing issue.

search_issues(q, sort?, order?, page?, per_page?)
  → Search issues/PRs using GitHub search syntax.
  → Example q: "auto-correction repo:202422/Intelligent-system... is:open"

── PULL REQUESTS ───────────────────────────────────────────
list_pull_requests(owner, repo, state?, head?, base?, sort?, direction?, page?, per_page?)
  → List pull requests. state: "open" | "closed" | "all". Default: "open".

get_pull_request(owner, repo, pull_number)
  → Get details of a specific pull request.

create_pull_request(owner, repo, title, head, base, body?, draft?, maintainer_can_modify?)
  → Create a new pull request from head branch into base branch.

merge_pull_request(owner, repo, pull_number, merge_method?, commit_title?, commit_message?)
  → Merge a pull request. merge_method: "merge" | "squash" | "rebase".

get_pull_request_diff(owner, repo, pull_number)
  → Get the diff (patch) of a pull request.

get_pull_request_files(owner, repo, pull_number)
  → List all files changed in a pull request.

── SEARCH ──────────────────────────────────────────────────
search_code(q, sort?, order?, page?, per_page?)
  → Search code across GitHub using code search syntax.
  → Example q: "correction algorithm repo:202422/Intelligent-system..."

search_users(q, sort?, order?, page?, per_page?)
  → Search for GitHub users by username or name.

── AUTH ────────────────────────────────────────────────────
get_me(reason?)
  → Get details of the currently authenticated GitHub user.

"""
