"""Tool-use contract injected into the system prompt."""

TOOL_CONTRACT = """Tool contract:
- Locate uncertain files with list_dir/search_files before reading or editing.
- Before calling edit_file, apply_patch replace/delete, or write_file on an existing file, read the relevant file content first with read_file/read_files. Do not rely on the tool error to remind you.
- For simple repo inspection, prefer read_file/list_dir/search_files; use bash when tests, builds, git, scripts, process control, or complex shell behavior make it the clearer tool.
- Prefer edit_file for targeted text edits and apply_patch for coordinated multi-file edits; use write_file only for new files or complete overwrites.
- Do not use bash/python/sed/perl scripts to modify workspace files unless the user explicitly asks for script-based bulk editing or the structured edit tools cannot express the change safely.
- Use dry_run before risky edits when available, then apply only after the diff matches the intent.
- If a tool call fails, adjust the next call using the error instead of repeating the same arguments.
- Verify important file changes with read_file, tests, or the relevant command after writing."""
