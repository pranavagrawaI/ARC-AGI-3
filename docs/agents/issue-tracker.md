# Issue tracker: GitHub

Issues and PRDs for this repo live as GitHub issues. Use the `gh` CLI for all operations.

## Conventions

- **Create an issue**: `gh issue create --title "..." --body "..."`.
- **Read an issue**: `gh issue view <number> --comments`, also fetching labels.
- **List issues**: `gh issue list --state open --json number,title,body,labels,comments` with appropriate label and state filters.
- **Comment on an issue**: `gh issue comment <number> --body "..."`.
- **Apply / remove labels**: `gh issue edit <number> --add-label "..."` / `--remove-label "..."`.
- **Close**: `gh issue close <number> --comment "..."`.

Infer the repo from `git remote -v`; `gh` does this automatically inside the clone.

## Pull requests as a triage surface

**PRs as a request surface: no.**

GitHub shares one number space across issues and PRs, so a bare issue number may be either. Resolve with `gh pr view <number>` and fall back to `gh issue view <number>`.

## When a skill says "publish to the issue tracker"

Create a GitHub issue.

## When a skill says "fetch the relevant ticket"

Run `gh issue view <number> --comments`.

## Wayfinding operations

Used by `/wayfinder`. The **map** is a single issue with **child** issues as tickets.

- **Map**: a single issue labelled `wayfinder:map`, holding Destination, Notes, Decisions-so-far, Not-yet-specified, and Out-of-scope sections.
- **Child ticket**: an issue linked to the map as a GitHub sub-issue through the sub-issues API. Where sub-issues are unavailable, add the child to a task list in the map body and put `Part of #<map>` at the top of the child body. Label it `wayfinder:research`, `wayfinder:prototype`, `wayfinder:grilling`, or `wayfinder:task`.
- **Blocking**: use GitHub's native issue dependencies. Add an edge with `gh api --method POST repos/<owner>/<repo>/issues/<child>/dependencies/blocked_by -F issue_id=<blocker-database-id>`, where the blocker database ID comes from `gh api repos/<owner>/<repo>/issues/<number> --jq .id`. Where dependencies are unavailable, fall back to a `Blocked by: #<number>` line in the child body. A ticket is unblocked when every blocker is closed.
- **Frontier query**: list the map's open children, then drop children with an open blocker or an assignee. The first remaining child in map order wins.
- **Claim**: `gh issue edit <number> --add-assignee @me`; this is the session's first write.
- **Resolve**: post the answer as a resolution comment, close the ticket, then append a one-line gist and named link to the map's Decisions-so-far.
