"""Verbatim from thinharness v0.5.4 ``thinharness/defaults.py``: the read/write/edit/search tool texts."""

DEFAULT_READ_DESCRIPTION = "Read a UTF-8 text file with line numbers and optional offset/limit."
DEFAULT_WRITE_DESCRIPTION = "Create, overwrite, or append to a UTF-8 text file under the workspace root."
DEFAULT_EDIT_DESCRIPTION = "Apply one or more exact text replacements to UTF-8 files. Each old_string must be unique unless all=true."
DEFAULT_SEARCH_DESCRIPTION = "Search readable workspace files with ripgrep and return compact grouped path/line matches."

DEFAULT_READ_INSTRUCTIONS = """read usage:
- Use read when you know the file path and need line-numbered context.
- Omit offset and limit to read from the start through the harness character cap.
- Use offset and limit to inspect large files or targeted chunks.
- Use max_chars only when you need a smaller returned excerpt."""

DEFAULT_WRITE_INSTRUCTIONS = """write usage:
- Use write to create or replace an entire file, or append when append=true.
- For targeted changes to an existing file, prefer edit over rewriting the whole file.
- Include path and complete content at the top level of the tool arguments."""

DEFAULT_EDIT_INSTRUCTIONS = """edit usage:
- Use edit for one or more targeted replacements in existing UTF-8 files.
- Include edits as a non-empty list. Each item must include path, old_string, and new_string.
- Edits apply sequentially in list order; each edit sees the file content left by earlier edits in the same call.
- Batch independent edits together. Order dependent edits deliberately, and resolve any failed item before relying on final file state.
- old_string must match the file text exactly and should include enough surrounding context to be unique.
- If an old_string appears multiple times, add more context or set all=true only when every occurrence should change.
- expected_replacements is only a count assertion; replacing multiple occurrences still requires all=true.
- When a batch has failures, retry only the failed items after reading the numbered per-edit results."""

DEFAULT_SEARCH_INSTRUCTIONS = """search usage:
- Use search to find text or regex matches in readable workspace files.
- Pass query for the text or regex to find.
- Pass path to scope search to a file or directory; omit path to search all readable roots.
- Use file_type only for ripgrep file types such as py, js, or md."""
