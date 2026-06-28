"""File reference expansion for @file-style prompt attachments."""

from __future__ import annotations

import mimetypes
import re
from dataclasses import dataclass, field
from pathlib import Path

from spice.llm.models import Model

REFERENCE_HEADER = "Referenced files:"
SINGLE_FILE_CHAR_LIMIT = 12_000
FALLBACK_TOTAL_CHAR_LIMIT = 48_000
SOFT_CONTEXT_FRACTION = 0.25
HARD_CONTEXT_FRACTION = 0.50

IMAGE_MIME_PREFIX = "image/"
SENSITIVE_HOME_DIRS = (
    ".ssh",
    ".aws",
    ".gnupg",
    ".kube",
    ".docker",
    ".azure",
    ".config/gh",
    ".config/gcloud",
)
SENSITIVE_HOME_FILES = (".netrc", ".npmrc", ".pypirc", ".pgpass", ".git-credentials", ".boto", ".s3cfg")
SENSITIVE_FILE_NAMES = {
    ".env",
    ".env.local",
    ".env.production",
    ".env.development",
    ".env.test",
    "credentials",
    "credentials.json",
    "service-account.json",
    "service_account.json",
    "id_rsa",
    "id_dsa",
    "id_ecdsa",
    "id_ed25519",
}
SENSITIVE_FILE_SUFFIXES = (".pem", ".key", ".p12", ".pfx")
PRIVATE_KEY_HEADER_RE = re.compile(r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----")
SKIP_DIRS = {".git", ".venv", "venv", "node_modules", "__pycache__", ".idea", ".vscode", "dist", "build"}


@dataclass(frozen=True)
class FileReference:
    raw: str
    target: str
    kind: str = "file"
    line_start: int | None = None
    line_end: int | None = None

    @property
    def key(self) -> tuple[str, str, int | None, int | None]:
        return (self.kind, self.target, self.line_start, self.line_end)


@dataclass
class FileReferenceExpansion:
    message: str
    references: list[FileReference] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    expanded: bool = False


class FileReferenceError(Exception):
    """Raised when file references cannot be expanded safely."""


def expand_file_references(text: str, *, cwd: Path, model: Model) -> FileReferenceExpansion:
    references = parse_file_references(text)
    if not references:
        return FileReferenceExpansion(message=text)

    cwd = cwd.resolve()
    total_limit = _total_char_limit(model)
    soft_limit = _soft_char_limit(model)
    used_chars = 0
    blocks: list[str] = []
    warnings: list[str] = []
    seen: set[tuple[str, str, int | None, int | None]] = set()

    for reference in references:
        resolved = _resolve_reference(reference, cwd)
        normalized = FileReference(
            raw=reference.raw,
            target=str(resolved.path),
            kind=reference.kind,
            line_start=resolved.line_start,
            line_end=resolved.line_end,
        )
        if normalized.key in seen:
            continue
        seen.add(normalized.key)

        warning, block, used = _expand_resolved_reference(
            reference=reference,
            resolved=resolved,
            cwd=cwd,
            model=model,
            remaining_chars=max(total_limit - used_chars, 0),
        )
        if warning:
            warnings.append(warning)
        if block:
            blocks.append(block)
            used_chars += used

    if used_chars > soft_limit:
        warnings.append(
            f"Referenced files use about {used_chars} characters, above the {int(SOFT_CONTEXT_FRACTION * 100)}% context budget."
        )

    if not blocks and not warnings:
        return FileReferenceExpansion(message=text, references=references)

    final = text.rstrip()
    if warnings:
        final += "\n\nReference warnings:\n" + "\n".join(f"- {warning}" for warning in warnings)
    if blocks:
        final += f"\n\n{REFERENCE_HEADER}\n\n" + "\n\n".join(blocks)
    return FileReferenceExpansion(message=final, references=references, warnings=warnings, expanded=True)


def parse_file_references(text: str) -> list[FileReference]:
    references: list[FileReference] = []
    index = 0
    while index < len(text):
        at_index = text.find("@", index)
        if at_index == -1:
            break
        if at_index > 0 and (text[at_index - 1].isalnum() or text[at_index - 1] in {"/", "_", "-"}):
            index = at_index + 1
            continue
        parsed = _parse_reference_at(text, at_index)
        if parsed is None:
            index = at_index + 1
            continue
        reference, next_index = parsed
        references.append(reference)
        index = next_index
    return references


@dataclass(frozen=True)
class _ResolvedReference:
    path: Path
    kind: str
    line_start: int | None
    line_end: int | None


def _parse_reference_at(text: str, at_index: int) -> tuple[FileReference, int] | None:
    start = at_index + 1
    if start >= len(text) or text[start].isspace():
        return None

    kind = "file"
    if text.startswith("file:", start):
        start += len("file:")
    elif text.startswith("folder:", start):
        start += len("folder:")
        kind = "folder"

    if start < len(text) and text[start] in {"'", '"', "`"}:
        quote = text[start]
        end = text.find(quote, start + 1)
        if end == -1:
            return None
        target = text[start + 1 : end]
        next_index = end + 1
        suffix_end = _consume_line_suffix(text, next_index)
        suffix = text[next_index:suffix_end]
        line_start, line_end = _parse_line_suffix(suffix)
        raw = text[at_index:suffix_end]
        return FileReference(raw=raw, target=target, kind=kind, line_start=line_start, line_end=line_end), suffix_end

    end = start
    while end < len(text) and not text[end].isspace():
        end += 1
    raw_target = text[start:end].rstrip(",.;!?")
    raw = text[at_index : at_index + 1 + len(raw_target)]
    target, line_start, line_end = _split_line_suffix(raw_target) if kind == "file" else (raw_target, None, None)
    if not target:
        return None
    return FileReference(raw=raw, target=target, kind=kind, line_start=line_start, line_end=line_end), at_index + 1 + len(raw_target)


def _consume_line_suffix(text: str, start: int) -> int:
    match = re.match(r":\d+(?:-\d+)?", text[start:])
    return start + len(match.group(0)) if match else start


def _split_line_suffix(value: str) -> tuple[str, int | None, int | None]:
    match = re.match(r"^(?P<path>.+):(?P<start>\d+)(?:-(?P<end>\d+))?$", value)
    if not match:
        return value, None, None
    line_start = int(match.group("start"))
    line_end = int(match.group("end") or match.group("start"))
    return match.group("path"), line_start, line_end


def _parse_line_suffix(value: str) -> tuple[int | None, int | None]:
    if not value:
        return None, None
    match = re.match(r"^:(?P<start>\d+)(?:-(?P<end>\d+))?$", value)
    if not match:
        return None, None
    line_start = int(match.group("start"))
    return line_start, int(match.group("end") or line_start)


def _resolve_reference(reference: FileReference, cwd: Path) -> _ResolvedReference:
    target = reference.target
    line_start = reference.line_start
    line_end = reference.line_end
    path = _candidate_path(target, cwd)

    if not path.exists() and reference.kind == "file":
        longest = _longest_existing_prefix(target, cwd)
        if longest is not None:
            path = longest

    return _ResolvedReference(path=path.resolve(), kind=reference.kind, line_start=line_start, line_end=line_end)


def _candidate_path(raw_path: str, cwd: Path) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = cwd / path
    return path


def _longest_existing_prefix(raw_target: str, cwd: Path) -> Path | None:
    for end in range(len(raw_target), 0, -1):
        candidate_text = raw_target[:end].rstrip(",.;!?")
        if not candidate_text:
            continue
        candidate = _candidate_path(candidate_text, cwd)
        if candidate.exists():
            return candidate
    return None


def _expand_resolved_reference(
    *,
    reference: FileReference,
    resolved: _ResolvedReference,
    cwd: Path,
    model: Model,
    remaining_chars: int,
) -> tuple[str | None, str | None, int]:
    path = resolved.path
    try:
        _ensure_not_sensitive(path)
    except ValueError as exc:
        return f"{reference.raw}: {exc}", None, 0

    if not path.exists():
        return f"{reference.raw}: file not found", None, 0
    if resolved.kind == "folder" or path.is_dir():
        if not path.is_dir():
            return f"{reference.raw}: path is not a folder", None, 0
        listing = _folder_listing(path, cwd)
        return None, _metadata_block(path, cwd, "directory", listing, injected=True), len(listing)
    if not path.is_file():
        return f"{reference.raw}: path is not a file", None, 0

    mime_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    if mime_type.startswith(IMAGE_MIME_PREFIX):
        if not model.supports_vision:
            raise FileReferenceError(
                f"Image file detected: {_display_path(path, cwd)}. Current model does not support image input."
            )
        return None, _metadata_block(path, cwd, mime_type, "Image attachment support is not enabled yet.", injected=False), 0

    if _is_binary(path):
        size = _human_size(path)
        return None, _metadata_block(path, cwd, mime_type, f"Binary file not inlined as text. Size: {size}.", injected=False), 0

    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return None, _metadata_block(path, cwd, mime_type, "File is not valid UTF-8 text.", injected=False), 0
    except OSError as exc:
        return f"{reference.raw}: could not read file: {exc}", None, 0

    if _looks_like_private_key(text):
        return f"{reference.raw}: file looks like a private key and cannot be attached", None, 0

    line_label = ""
    if resolved.line_start is not None:
        lines = text.splitlines()
        if resolved.line_start < 1 or (resolved.line_end or resolved.line_start) < resolved.line_start:
            return f"{reference.raw}: invalid line range", None, 0
        if resolved.line_start > len(lines):
            return f"{reference.raw}: line {resolved.line_start} is beyond end of file ({len(lines)} lines)", None, 0
        end = min(resolved.line_end or resolved.line_start, len(lines))
        text = "\n".join(lines[resolved.line_start - 1 : end])
        line_label = f' lines="{resolved.line_start}-{end}"'

    limit = max(0, min(SINGLE_FILE_CHAR_LIMIT, remaining_chars))
    total_chars = len(text)
    truncated = total_chars > limit
    if limit <= 0:
        return (
            f"{reference.raw}: skipped because referenced files exceed the {int(HARD_CONTEXT_FRACTION * 100)}% context budget",
            _metadata_block(path, cwd, mime_type, "File was not inlined because the file reference budget is exhausted.", injected=False),
            0,
        )
    injected = text[:limit]
    if truncated:
        injected += f"\n\n[truncated: kept first {limit} characters of {total_chars}]"
    block = _text_file_block(
        path=path,
        cwd=cwd,
        mime_type=mime_type,
        content=injected,
        total_chars=total_chars,
        injected_chars=min(limit, total_chars),
        truncated=truncated,
        line_label=line_label,
    )
    warning = f"{reference.raw}: truncated to {limit} of {total_chars} characters" if truncated else None
    return warning, block, min(limit, total_chars)


def _text_file_block(
    *,
    path: Path,
    cwd: Path,
    mime_type: str,
    content: str,
    total_chars: int,
    injected_chars: int,
    truncated: bool,
    line_label: str,
) -> str:
    lang = _code_fence_language(path)
    display_path = _display_path(path, cwd)
    return (
        f'<file path="{display_path}" type="{lang or mime_type}" chars="{injected_chars}" '
        f'total_chars="{total_chars}" truncated="{str(truncated).lower()}"{line_label}>\n'
        f"```{lang}\n{content}\n```\n"
        "</file>"
    )


def _metadata_block(path: Path, cwd: Path, file_type: str, body: str, *, injected: bool) -> str:
    return (
        f'<file path="{_display_path(path, cwd)}" type="{file_type}" size="{_human_size(path)}" '
        f'injected="{str(injected).lower()}">\n{body}\n</file>'
    )


def _folder_listing(path: Path, cwd: Path, limit: int = 200) -> str:
    lines = [f"{_display_path(path, cwd)}/"]
    count = 0
    for child in sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())):
        if child.name in SKIP_DIRS or child.name.endswith(".egg-info"):
            continue
        suffix = "/" if child.is_dir() else ""
        lines.append(f"- {child.name}{suffix}")
        count += 1
        if count >= limit:
            lines.append("- ...")
            break
    return "\n".join(lines)


def _ensure_not_sensitive(path: Path) -> None:
    home = Path.home().resolve()
    name = path.name.lower()
    if name in SENSITIVE_FILE_NAMES or name.endswith(SENSITIVE_FILE_SUFFIXES):
        raise ValueError("path looks like a sensitive credential file and cannot be attached")
    exact = {home / name for name in SENSITIVE_HOME_FILES}
    if path in exact:
        raise ValueError("path is a sensitive credential file and cannot be attached")
    for dirname in SENSITIVE_HOME_DIRS:
        sensitive_dir = home / dirname
        try:
            path.relative_to(sensitive_dir)
        except ValueError:
            continue
        raise ValueError("path is inside a sensitive credential directory and cannot be attached")


def _looks_like_private_key(text: str) -> bool:
    return PRIVATE_KEY_HEADER_RE.search(text[:4096]) is not None


def _is_binary(path: Path) -> bool:
    try:
        chunk = path.read_bytes()[:4096]
    except OSError:
        return False
    return b"\x00" in chunk


def _display_path(path: Path, cwd: Path) -> str:
    try:
        return str(path.relative_to(cwd))
    except ValueError:
        return str(path)


def _human_size(path: Path) -> str:
    try:
        size = float(path.stat().st_size)
    except OSError:
        return "unknown"
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{int(size)} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


def _total_char_limit(model: Model) -> int:
    if model.context_window > 0:
        return max(1, int(model.context_window * 4 * HARD_CONTEXT_FRACTION))
    return FALLBACK_TOTAL_CHAR_LIMIT


def _soft_char_limit(model: Model) -> int:
    if model.context_window > 0:
        return max(1, int(model.context_window * 4 * SOFT_CONTEXT_FRACTION))
    return FALLBACK_TOTAL_CHAR_LIMIT // 2


def _code_fence_language(path: Path) -> str:
    mapping = {
        ".py": "python",
        ".js": "javascript",
        ".ts": "typescript",
        ".tsx": "tsx",
        ".jsx": "jsx",
        ".json": "json",
        ".md": "markdown",
        ".sh": "bash",
        ".yml": "yaml",
        ".yaml": "yaml",
        ".toml": "toml",
        ".txt": "text",
        ".log": "text",
    }
    return mapping.get(path.suffix.lower(), "text")
