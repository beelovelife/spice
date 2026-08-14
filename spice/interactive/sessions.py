"""Session listing, choice builders, and shared AgentSession replacement."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from spice.agent.agent_session import AgentSession
from spice.interactive.types import ChoiceItem, ChoiceRequest
from spice.llm.config import load_config
from spice.llm.model_registry import ModelRegistry
from spice.llm.models import Model
from spice.storage.factory import create_session_store as _create_store
from spice.tools.base import ConfirmFn

if TYPE_CHECKING:
    from spice.agent.sessions import SessionStoreProtocol

SESSION_PICKER_LIMIT = 30


def create_session_store_for_config(*, cwd: Path | None) -> SessionStoreProtocol:
    return _create_store(load_config(), cwd=cwd)


def replace_session(
    current: AgentSession,
    *,
    session_id: str | None = None,
    continue_latest: bool = False,
    fresh: bool = False,
    confirm: ConfirmFn | None = None,
    reuse_store: bool = False,
    reuse_extensions: bool = False,
) -> AgentSession:
    """Create a replacement AgentSession. Frontends must not hand-roll this path."""
    confirm_fn = confirm if confirm is not None else getattr(current, "confirm", None)
    cwd = getattr(current, "cwd", None) or Path.cwd()
    kwargs: dict = {
        "cwd": cwd,
        "provider": current.model.provider,
        "model_id": current.model.id,
        "confirm": confirm_fn,
    }
    if fresh:
        if reuse_store:
            kwargs["session_store"] = current.session_store
        if reuse_extensions:
            kwargs["extension_manager"] = current.extensions
        return AgentSession(**kwargs)
    if session_id is not None:
        kwargs["session_id"] = session_id
    if continue_latest:
        kwargs["continue_latest"] = True
    if reuse_store:
        kwargs["session_store"] = current.session_store
    if reuse_extensions:
        kwargs["extension_manager"] = current.extensions
    return AgentSession(**kwargs)


def list_model_choices(current: Model) -> ChoiceRequest:
    registry = ModelRegistry()
    models = sorted(registry.all(), key=lambda item: (item.provider, item.id))
    items: list[ChoiceItem] = []
    current_id = f"{current.provider}/{current.id}"
    for model in models:
        mid = f"{model.provider}/{model.id}"
        items.append(
            ChoiceItem(
                id=mid,
                label=mid,
                detail=_model_detail(model, current=mid == current_id),
            )
        )
    return ChoiceRequest(
        title="Models",
        items=tuple(items),
        current_id=current_id,
        columns=("Model", "Details"),
        show_current_mark=True,
    )


def list_session_choices(
    *,
    cwd: Path,
    active_id: str | None,
    limit: int = SESSION_PICKER_LIMIT,
    title: str = "Sessions",
    store: SessionStoreProtocol | None = None,
) -> ChoiceRequest:
    session_store = store or create_session_store_for_config(cwd=cwd)
    rows = session_store.list(limit=limit, cwd=cwd, include_empty=True)
    items = tuple(
        ChoiceItem(
            id=row.id,
            label=row.id,
            detail=f"{row.updated_at}  {row.provider}/{row.model}  {row.message_count} messages  {row.preview}",
        )
        for row in rows
    )
    return ChoiceRequest(
        title=title,
        items=items,
        current_id=active_id,
        columns=("Session", "Details"),
        show_current_mark=True,
    )


def list_rewind_choices(session: AgentSession) -> ChoiceRequest:
    store = session.session_store
    session_id = session.session_id
    info = store.info(session_id)
    active_path_ids = (
        {entry.id for entry in store.path_entries(session_id)}
        if info.leaf_id
        else set()
    )
    items: list[ChoiceItem] = []
    for entry in store.entries(session_id):
        if entry.type == "leaf":
            continue
        active = entry_active_label(entry.id, info.leaf_id, active_path_ids)
        items.append(
            ChoiceItem(
                id=entry.id,
                label=entry.id,
                detail=f"{entry.type}  {active}  {entry_preview(entry.data)}".strip(),
            )
        )
    return ChoiceRequest(
        title="Rewind",
        items=tuple(items),
        current_id=info.leaf_id,
        columns=("Entry", "Details"),
        show_current_mark=True,
    )


def plan_ready_choices() -> ChoiceRequest:
    return ChoiceRequest(
        title="Plan ready",
        items=(
            ChoiceItem(
                "auto", "Yes, and use auto mode", "allow file edits for this session"
            ),
            ChoiceItem("refine", "Tell Spice what to change", "stay in plan mode"),
        ),
        columns=("Action", "Details"),
    )


def entry_preview(entry: dict) -> str:
    if entry.get("type") == "message" and isinstance(entry.get("message"), dict):
        message = entry["message"]
        return (
            f"{message.get('role', '')}: {str(message.get('content', '')).strip()[:80]}"
        )
    if entry.get("type") == "model_change":
        return f"{entry.get('provider', '')}/{entry.get('modelId', '')}"
    if entry.get("type") == "compaction":
        return str(entry.get("summary", "")).strip()[:80]
    if entry.get("type") == "leaf":
        return f"target={entry.get('targetId', '')}"
    return ""


def entry_active_label(
    entry_id: str, leaf_id: str | None, active_path_ids: set[str]
) -> str:
    if entry_id == leaf_id:
        return "active"
    if entry_id in active_path_ids:
        return "path"
    return ""


def _model_detail(model: Model, *, current: bool) -> str:
    from spice.llm.config import get_api_key

    parts: list[str] = []
    if current:
        parts.append("current")
    if model.context_window:
        parts.append(f"context {model.context_window}")
    if model.output_tokens:
        parts.append(f"output {model.output_tokens}")
    parts.append(model.protocol or "provider default")
    key = (
        "key yes"
        if get_api_key(model.provider, env_names=model.api_key_envs)
        else "key missing"
    )
    parts.append(key)
    return " · ".join(parts)
