"""Validated local sessions stored as atomic, versioned JSON snapshots."""

import json
import os
import re
import tempfile
from dataclasses import asdict, dataclass, field, fields, replace
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from gpt2_mlx_cli.conversation import (
    Message,
    local_timestamp,
    validate_messages,
    validate_timestamp,
)
from gpt2_mlx_cli.engine import GenerationSettings

SCHEMA_VERSION = 1
DEFAULT_TITLE = "New conversation"


def _validate_id(session_id: str) -> None:
    if not isinstance(session_id, str) or re.fullmatch(r"[0-9a-f]{32}", session_id) is None:
        raise ValueError("Session id must be a 32-character lowercase UUID hex string.")


@dataclass
class Session:
    id: str = field(default_factory=lambda: uuid4().hex)
    title: str = DEFAULT_TITLE
    mode: str = "completion"
    precision: str = "fp16"
    messages: list[Message] = field(default_factory=list)
    settings: GenerationSettings = field(default_factory=GenerationSettings)
    created_at: str = field(default_factory=local_timestamp)
    updated_at: str = field(default_factory=local_timestamp)

    def __post_init__(self) -> None:
        _validate_id(self.id)
        if not isinstance(self.title, str) or not self.title.strip():
            raise ValueError("Session title must be a non-empty string.")
        if self.mode != "completion":
            raise ValueError("Session mode must be 'completion'.")
        if self.precision not in ("fp16", "int8"):
            raise ValueError("Session precision must be 'fp16' or 'int8'.")
        validate_timestamp(self.created_at, "created_at")
        validate_timestamp(self.updated_at, "updated_at")
        validate_messages(self.messages)
        if not isinstance(self.settings, GenerationSettings):
            raise ValueError("Session settings must be GenerationSettings.")
        for name in ("temperature", "top_p"):
            if type(getattr(self.settings, name)) not in (int, float):
                raise ValueError(f"Session settings.{name} must be a number.")
        # Snapshot mutable values so later edits cannot change an existing session.
        self.messages = [Message(**asdict(message)) for message in self.messages]
        self.settings = GenerationSettings(**asdict(self.settings))


def _default_root() -> Path:
    """Keep checkout data in the repo, independent of the launch directory."""
    source_dir = Path(__file__).resolve().parent.parent
    if source_dir.name == "src" and (source_dir.parent / "pyproject.toml").is_file():
        return source_dir.parent / "sessions"
    # Standalone wheel installs have no checkout; never write into site-packages.
    return Path.cwd() / "sessions"


def _object_fields(value: object, expected: set[str], name: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object.")
    missing = expected - value.keys()
    extra = value.keys() - expected
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing fields: {', '.join(sorted(missing))}")
        if extra:
            details.append(f"unknown fields: {', '.join(sorted(extra))}")
        raise ValueError(f"Invalid {name}: {'; '.join(details)}.")
    return value


def _session_from_json(payload: object) -> Session:
    payload = _object_fields(
        payload, {item.name for item in fields(Session)} | {"schema_version"}, "session"
    )
    if type(payload["schema_version"]) is not int or payload["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"Unsupported schema_version; expected {SCHEMA_VERSION}.")
    if not isinstance(payload["messages"], list):
        raise ValueError("Session messages must be a JSON array.")
    messages = [
        Message(**_object_fields(item, {f.name for f in fields(Message)}, f"messages[{index}]"))
        for index, item in enumerate(payload["messages"])
    ]
    settings = GenerationSettings(
        **_object_fields(
            payload["settings"], {f.name for f in fields(GenerationSettings)}, "settings"
        )
    )
    values = {key: value for key, value in payload.items() if key != "schema_version"}
    # Preserve legacy transcripts without restoring the removed chat prompting behavior.
    if values["mode"] == "chat":
        values["mode"] = "completion"
    values.update(messages=messages, settings=settings)
    return Session(**values)


class SessionStore:
    def __init__(self, root: Path | None = None) -> None:
        self.root = Path(root) if root is not None else _default_root()

    def save(self, session: Session) -> Path:
        """Atomically save a snapshot; update metadata only after a successful write."""
        if not isinstance(session, Session):
            raise ValueError("save expects a Session.")
        snapshot = replace(session)
        if snapshot.title == DEFAULT_TITLE:
            first_user = next((m for m in snapshot.messages if m.role == "user"), None)
            if first_user is not None:
                title = " ".join(first_user.content.split())
                if title:
                    snapshot.title = title if len(title) <= 64 else title[:63].rstrip() + "…"
        snapshot.updated_at = local_timestamp()
        payload = {"schema_version": SCHEMA_VERSION, **asdict(snapshot)}
        self.root.mkdir(parents=True, exist_ok=True)
        destination = self.root / f"{snapshot.id}.json"
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self.root,
                prefix=f".{snapshot.id}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary = Path(handle.name)
                json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        session.title = snapshot.title
        session.updated_at = snapshot.updated_at
        return destination

    def load(self, id: str) -> Session:
        """Load a session, reporting malformed or incompatible data to the caller."""
        _validate_id(id)
        path = self.root / f"{id}.json"
        try:
            with path.open(encoding="utf-8") as handle:
                session = _session_from_json(json.load(handle))
            if session.id != id:
                raise ValueError("Stored session id does not match its filename.")
            return session
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"Session {id} was not found in {self.root}.") from exc
        except (ValueError, TypeError, OverflowError) as exc:
            raise ValueError(f"Cannot load session {id} from {path}: {exc}") from exc

    def delete(self, id: str) -> None:
        """Remove only the selected snapshot; an already missing file is harmless."""
        _validate_id(id)
        (self.root / f"{id}.json").unlink(missing_ok=True)

    def list(self) -> list[Session]:
        """Return newest sessions first, leaving unreadable files untouched."""
        sessions = []
        for path in self.root.glob("*.json"):
            try:
                sessions.append(self.load(path.stem))
            except (ValueError, OSError):
                continue
        return sorted(
            sessions,
            key=lambda session: (datetime.fromisoformat(session.updated_at), session.id),
            reverse=True,
        )
