"""A small, single-column terminal playground for the local GPT-2 engine."""

from __future__ import annotations

import math
import threading
from collections.abc import Awaitable, Callable
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any

from rich.text import Text
from textual import events, on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message as UIMessage
from textual.screen import ModalScreen
from textual.widgets import Button, Input, OptionList, Select, Static, TextArea
from textual.widgets.option_list import Option

if TYPE_CHECKING:
    from gpt2_mlx_cli.engine import Engine, GenerationSettings
    from gpt2_mlx_cli.storage import Session, SessionStore


class Composer(TextArea):
    """Keep Enter for sending and explicit alternate keys for newlines."""

    BINDINGS = [
        Binding("enter", "submit", show=False, priority=True),
        Binding("alt+enter,ctrl+j", "newline", show=False, priority=True),
    ]

    class Submitted(UIMessage):
        pass

    def action_submit(self) -> None:
        self.post_message(self.Submitted())

    def action_newline(self) -> None:
        self.replace("\n", *self.selection)


class ChatTurn(Vertical):
    def __init__(self, role: str, content: str, timestamp: str = "") -> None:
        super().__init__(classes=f"turn {role}")
        self.role = role
        self.content = content
        self.timestamp = timestamp

    def compose(self) -> ComposeResult:
        if self.role == "user":
            with Horizontal(classes="user-band"):
                yield Static("› " + self.content, markup=False, classes="message-body")
                try:
                    stamp = datetime.fromisoformat(self.timestamp).astimezone().strftime("%H:%M")
                except (ValueError, TypeError):
                    stamp = ""
                yield Static(stamp, markup=False, classes="timestamp")
        else:
            yield Static(self.content, markup=False, classes="message-body")

    def update_text(self, text: str) -> None:
        self.content = text
        self.query_one(".message-body", Static).update(text)

    def show_empty_completion(self) -> None:
        """Show a UI-only notice without changing the model's empty output."""
        self.query_one(".message-body", Static).update(
            Text("Model ended without generating text.", style="dim")
        )


@dataclass(frozen=True)
class SettingsChoice:
    settings: GenerationSettings
    precision: str


class SettingsScreen(ModalScreen[SettingsChoice | None]):
    BINDINGS = [
        Binding("escape", "cancel", show=False),
        Binding("ctrl+s", "save_settings", show=False, priority=True),
    ]

    def __init__(self, session: Session) -> None:
        super().__init__()
        self.session = session

    def compose(self) -> ComposeResult:
        with Vertical(classes="dialog", id="settings-dialog"):
            yield Static("Settings", classes="dialog-title")
            with VerticalScroll(id="settings-fields"):
                for field, label in (
                    ("temperature", "Temperature (0–5, 0 = greedy)"),
                    ("top_p", "Top p (0–1, above 0)"),
                    ("top_k", "Top k (0 = all tokens)"),
                    ("max_new_tokens", "Max new tokens (1–1023)"),
                ):
                    yield Static(label, classes="field-label")
                    yield Input(str(getattr(self.session.settings, field)), id=field)
                yield Static("Precision", classes="field-label")
                yield Select(
                    [("FP16", "fp16"), ("INT8", "int8")],
                    value=self.session.precision,
                    allow_blank=False,
                    id="precision",
                )
            yield Static("", id="settings-error", markup=False)
            with Horizontal(classes="dialog-actions"):
                yield Button("Cancel", id="cancel-settings")
                yield Button("Save · ^S", id="save-settings", variant="primary")

    def action_cancel(self) -> None:
        self.dismiss(None)

    @on(Button.Pressed, "#cancel-settings")
    def cancel_settings(self) -> None:
        self.dismiss(None)

    @on(Button.Pressed, "#save-settings")
    def action_save_settings(self) -> None:
        from gpt2_mlx_cli.engine import GenerationSettings

        try:
            temperature = float(self.query_one("#temperature", Input).value)
            top_p = float(self.query_one("#top_p", Input).value)
            top_k = int(self.query_one("#top_k", Input).value)
            max_new_tokens = int(self.query_one("#max_new_tokens", Input).value)
            if not math.isfinite(temperature) or not 0 <= temperature <= 5:
                raise ValueError("Temperature must be a finite number between 0 and 5.")
            if not math.isfinite(top_p) or not 0 < top_p <= 1:
                raise ValueError("Top p must be above 0 and at most 1.")
            if not 0 <= top_k <= 50257:
                raise ValueError("Top k must be between 0 and 50257.")
            if not 1 <= max_new_tokens < 1024:
                raise ValueError("Max new tokens must be between 1 and 1023.")
            settings = GenerationSettings(
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                max_new_tokens=max_new_tokens,
                seed=self.session.settings.seed,
            )
        except (ValueError, OverflowError) as error:
            self.query_one("#settings-error", Static).update(str(error))
            return
        self.dismiss(
            SettingsChoice(
                settings,
                str(self.query_one("#precision", Select).value),
            )
        )


class DeleteSessionScreen(ModalScreen[bool]):
    BINDINGS = [Binding("escape", "cancel", show=False)]

    def __init__(self, title: str) -> None:
        super().__init__()
        self.session_title = title

    def compose(self) -> ComposeResult:
        with Vertical(classes="dialog", id="delete-session-dialog"):
            yield Static("Delete session?", classes="dialog-title")
            yield Static(self.session_title, id="delete-session-title", markup=False)
            yield Static(
                "Permanently delete this conversation and clear it if open? No undo.",
                id="delete-session-warning",
            )
            with Horizontal(classes="dialog-actions"):
                yield Button("Cancel", id="cancel-delete", variant="primary")
                yield Button("Delete", id="confirm-delete", variant="error")

    def on_mount(self) -> None:
        self.query_one("#cancel-delete", Button).focus()

    @on(Button.Pressed, "#cancel-delete")
    def action_cancel(self) -> None:
        self.dismiss(False)

    @on(Button.Pressed, "#confirm-delete")
    def confirm_delete(self) -> None:
        self.dismiss(True)


class SessionsScreen(ModalScreen[str | None]):
    BINDINGS = [
        Binding("escape", "cancel", show=False),
        Binding("delete,backspace", "delete_session", show=False),
    ]

    def __init__(
        self, sessions: list[Session], on_delete: Callable[[str], Awaitable[None]]
    ) -> None:
        super().__init__()
        self.sessions = sessions
        self.on_delete = on_delete

    def compose(self) -> ComposeResult:
        with Vertical(classes="dialog", id="sessions-dialog"):
            yield Static("Sessions", classes="dialog-title")
            yield OptionList(
                *(
                    Option(
                        Text(f"{session.title}\n{session.mode} · {session.precision}"),
                        id=session.id,
                    )
                    for session in self.sessions
                ),
                id="session-list",
            )
            yield Static("No saved conversations yet.", id="empty-sessions")
            yield Static("", id="sessions-error", markup=False)
            with Horizontal(classes="dialog-actions"):
                yield Button("Delete", id="delete-session", variant="error")
                yield Button("Close", id="close-sessions")

    def on_mount(self) -> None:
        self._refresh_list()

    def _refresh_list(self) -> None:
        has_sessions = bool(self.sessions)
        listing = self.query_one(OptionList)
        listing.display = has_sessions
        self.query_one("#empty-sessions", Static).display = not has_sessions
        self.query_one("#delete-session", Button).disabled = not has_sessions
        if has_sessions:
            listing.focus()
        else:
            self.query_one("#close-sessions", Button).focus()

    @on(Button.Pressed, "#delete-session")
    def action_delete_session(self) -> None:
        if self.app.screen is not self:
            return
        listing = self.query_one(OptionList)
        if listing.highlighted is None:
            return
        session_id = listing.get_option_at_index(listing.highlighted).id
        session = next(item for item in self.sessions if item.id == session_id)
        self.query_one("#sessions-error", Static).update("")

        async def delete_if_confirmed(confirmed: bool) -> None:
            if not confirmed:
                listing.focus()
                return
            try:
                await self.on_delete(session.id)
            except Exception as error:
                self.query_one("#sessions-error", Static).update(
                    f"Could not delete session: {error}"
                )
                listing.focus()
                return
            self.sessions = [item for item in self.sessions if item.id != session.id]
            listing.remove_option(session.id)
            self._refresh_list()

        self.app.push_screen(DeleteSessionScreen(session.title), delete_if_confirmed)

    def action_cancel(self) -> None:
        self.dismiss(None)

    @on(Button.Pressed, "#close-sessions")
    def close_sessions(self) -> None:
        self.dismiss(None)

    @on(OptionList.OptionSelected)
    def select_session(self, event: OptionList.OptionSelected) -> None:
        self.dismiss(event.option.id)


class QuitSaveErrorScreen(ModalScreen[str | None]):
    """Require an explicit choice before discarding an unsaved conversation."""

    BINDINGS = [Binding("escape", "keep_open", show=False)]

    def __init__(self, error: str) -> None:
        super().__init__()
        self.error = error

    def compose(self) -> ComposeResult:
        with Vertical(classes="dialog", id="quit-dialog"):
            yield Static("Conversation not saved", classes="dialog-title")
            yield Static(self.error, id="quit-save-error", markup=False)
            with Vertical(id="quit-actions"):
                yield Button("Keep open", id="keep-open", variant="primary")
                yield Button("Retry save and quit", id="retry-save-quit")
                yield Button("Quit without saving", id="discard-quit")

    def on_mount(self) -> None:
        self.query_one("#keep-open", Button).focus()

    @on(Button.Pressed, "#keep-open")
    def action_keep_open(self) -> None:
        self.dismiss(None)

    @on(Button.Pressed, "#retry-save-quit")
    def retry_save(self) -> None:
        self.dismiss("retry")

    @on(Button.Pressed, "#discard-quit")
    def discard(self) -> None:
        self.dismiss("discard")


class EngineUpdate(UIMessage):
    """Only immutable snapshots cross from the inference thread to the UI."""

    def __init__(self, kind: str, **data: Any) -> None:
        super().__init__()
        self.kind = kind
        self.data = data


class GPT2App(App[None]):
    CSS_PATH = "theme.tcss"
    TITLE = "gpt2-mlx-cli · GPT-2 774M"
    ENABLE_COMMAND_PALETTE = False
    BINDINGS = [
        Binding("escape", "stop", "Stop"),
        Binding("ctrl+n", "new_session", "New", priority=True),
        Binding("ctrl+o", "sessions", "Sessions", priority=True),
        Binding("ctrl+p", "settings", "Settings", priority=True),
        Binding("ctrl+q", "quit", "Quit", priority=True),
    ]

    def __init__(
        self,
        engine: Engine | None = None,
        store: SessionStore | None = None,
        precision: str = "fp16",
    ) -> None:
        super().__init__()
        from gpt2_mlx_cli.storage import Session, SessionStore

        if precision not in {"fp16", "int8"}:
            raise ValueError("Expected precision fp16/int8.")
        self.store = store if store is not None else SessionStore()
        self.session = Session(precision=precision)
        self._engine = engine
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="gpt2-mlx-cli-inference"
        )
        self._stop = threading.Event()
        self._closing = False
        self._exit_after_work = False
        self.busy = False
        self._assistant: ChatTurn | None = None
        self._output = ""
        self._retry_cutoff: int | None = None
        self._pending_choice: SettingsChoice | None = None
        self._storage_error: str | None = None
        self.context_used: int | None = 0
        self._dropped_messages = 0
        self._last_metrics = ""
        self._compact_metrics = ""
        self._short = False
        self._status_text = "Ready · model loads on first send"
        self._status_compact = "Ready"

    def compose(self) -> ComposeResult:
        with Horizontal(id="header"):
            yield Static(self.TITLE, id="brand", markup=False)
            yield Static("Context 0/1024", id="context", markup=False)
        with VerticalScroll(id="transcript"):
            yield Static(
                "Enter text for GPT-2 to continue. Earlier turns provide context.\n"
                "No system prompt or chat template is added.\n"
                "GPT-2 runs locally; the first send may download the model.",
                id="welcome",
                markup=False,
            )
        with Vertical(id="bottom"):
            yield Static("Ready · model loads on first send", id="status", markup=False)
            yield Composer(
                id="composer",
                soft_wrap=True,
                show_line_numbers=False,
                highlight_cursor_line=False,
            )
            with Horizontal(id="controls"):
                yield Static("", id="session-mode", markup=False)
                yield Button("Retry", id="retry")
                yield Button("Stop", id="stop", disabled=True)
                yield Button("Send", id="send", variant="primary")
            yield Static("", id="shortcuts", markup=False)

    def on_mount(self) -> None:
        self.query_one("#retry", Button).display = False
        self.query_one("#stop", Button).display = False
        self._refresh_session_info()
        self._responsive(self.size.width, self.size.height)
        self.query_one(Composer).focus()

    def on_resize(self, event: events.Resize) -> None:
        if self.is_mounted:
            transcript = self.query_one("#transcript", VerticalScroll)
            follow = transcript.is_vertical_scroll_end
            self._responsive(event.size.width, event.size.height)
            if follow:
                transcript.scroll_end(animate=False)

    def _responsive(self, width: int, height: int) -> None:
        self._short = height < 22
        self.screen_stack[0].set_class(width < 60, "narrow")
        self.screen_stack[0].set_class(self._short, "short")
        self.query_one("#shortcuts", Static).update(
            "Enter send · ^J nl · Esc stop\n^N new · ^O open · ^P settings"
            if self._short
            else ("Enter send · ^J newline\nEsc stop · ^N new · ^O sessions\n^P settings · ^Q quit")
            if width < 60
            else (
                "Enter send · Alt+Enter/Ctrl+J newline\n"
                "Esc stop · Ctrl+N new · Ctrl+O sessions · Ctrl+P settings · Ctrl+Q quit"
            )
        )
        self.query_one("#status", Static).update(
            self._status_compact if self._short else self._status_text
        )

    def _refresh_session_info(self) -> None:
        self.query_one("#session-mode", Static).update(
            f"{self.session.mode} · {self.session.precision.upper()}"
        )
        self.query_one(Composer).border_title = "Prompt to continue"
        used = "—" if self.context_used is None else str(self.context_used)
        self.query_one("#context", Static).update(f"Context {used}/1024")

    def _set_status(self, text: str, *, error: bool = False, compact: str | None = None) -> None:
        self._status_text = text
        self._status_compact = compact or text
        status = self.query_one("#status", Static)
        status.update(self._status_compact if self._short else text)
        status.tooltip = Text(text)
        status.set_class(error, "error")

    def _set_busy(self, value: bool) -> None:
        self.busy = value
        self.query_one("#send", Button).disabled = value
        self.query_one("#stop", Button).disabled = not value
        self.query_one("#stop", Button).display = value
        self.query_one("#retry", Button).disabled = value

    def _save(self) -> bool:
        try:
            self.store.save(self.session)
            self._storage_error = None
            return True
        except Exception as error:
            self._storage_error = f"Could not save session: {error}"
            self._set_status(self._storage_error, error=True)
            return False

    async def _render_session(self) -> None:
        transcript = self.query_one("#transcript", VerticalScroll)
        await transcript.remove_children()
        for message in self.session.messages:
            turn = ChatTurn(message.role, message.content, message.timestamp)
            await transcript.mount(turn)
            if message.role == "assistant" and not message.content:
                turn.show_empty_completion()
        self._assistant = None
        self._retry_cutoff = None
        self.query_one("#retry", Button).display = False
        self._refresh_session_info()
        transcript.scroll_end(animate=False)

    @on(Composer.Submitted)
    @on(Button.Pressed, "#send")
    async def send_prompt(self) -> None:
        if self.busy or len(self.screen_stack) > 1:
            return
        from gpt2_mlx_cli.conversation import Message

        composer = self.query_one(Composer)
        content = composer.text
        if not content.strip():
            return
        # Claim the slot before awaiting widget mounts, so repeated Enter is harmless.
        self._stop = threading.Event()
        self._set_busy(True)
        message = Message(role="user", content=content)
        pending = bool(self.session.messages and self.session.messages[-1].role == "user")
        if pending:
            self.session.messages[-1] = message
        else:
            self.session.messages.append(message)
        if len(self.session.messages) == 1:
            self.session.title = " ".join(content.split())[:64]
        composer.clear()
        for welcome in self.query("#welcome"):
            await welcome.remove()
        if pending:
            await self._render_session()
        else:
            await self.query_one("#transcript", VerticalScroll).mount(
                ChatTurn(message.role, message.content, message.timestamp)
            )
        self._save()
        await self._start_generation()

    async def _start_generation(self) -> None:
        self._set_busy(True)
        self._retry_cutoff = len(self.session.messages)
        self.query_one("#retry", Button).display = False
        self._output = ""
        self._dropped_messages = 0
        self._last_metrics = ""
        self._compact_metrics = ""
        self._assistant = ChatTurn("assistant", "")
        transcript = self.query_one("#transcript", VerticalScroll)
        await transcript.mount(self._assistant)
        transcript.scroll_end(animate=False)
        self._set_status("Loading model…")
        snapshot = deepcopy(self.session)
        self._executor.submit(self._generate, snapshot, self._stop)

    def _progress(self, text: str) -> None:
        if not self._closing:
            self.post_message(EngineUpdate("progress", text=str(text)))

    def _ensure_engine(self, precision: str, stop: threading.Event | None = None) -> Engine:
        # Engine construction, tokenization, loading, and all MLX calls live here.
        if stop is not None and stop.is_set():
            raise InterruptedError("Model operation cancelled.")
        if self._engine is None:
            from gpt2_mlx_cli.engine import Engine

            self._engine = Engine(precision=precision)
        if self._engine.precision != precision:
            self._engine.set_precision(precision, progress=self._progress, stop=stop)
        return self._engine

    def _generate(self, session: Session, stop: threading.Event) -> None:
        output = ""
        reason = "cancelled"
        error = None
        try:
            from gpt2_mlx_cli.conversation import build_prompt

            engine = self._ensure_engine(session.precision, stop=stop)
            if not engine.loaded and not stop.is_set():
                engine.load(progress=self._progress, stop=stop)
            if not stop.is_set():
                prepared = build_prompt(
                    session.messages,
                    engine.tokenizer,
                    session.settings.max_new_tokens,
                )
                self.post_message(
                    EngineUpdate(
                        "prepared",
                        prompt_tokens=len(prepared.token_ids),
                        dropped=prepared.dropped_messages,
                    )
                )
                reason = "done"
                for event in engine.generate(
                    prepared.token_ids,
                    session.settings,
                    stop=stop,
                ):
                    output = event.text
                    reason = event.finish_reason or reason
                    self.post_message(
                        EngineUpdate(
                            "tokens",
                            text=output,
                            tokens=event.token_count,
                            prompt_tokens=event.prompt_tokens,
                            tokens_per_second=event.tokens_per_second,
                            memory=event.peak_memory_bytes,
                            phase=event.phase,
                        )
                    )
                if stop.is_set():
                    reason = "cancelled"
        except Exception as exc:
            if stop.is_set():
                reason = "cancelled"
            else:
                error = str(exc) or type(exc).__name__
        finally:
            self.post_message(EngineUpdate("finished", text=output, reason=reason, error=error))

    def _change_precision(self, precision: str, stop: threading.Event) -> None:
        error = None
        try:
            engine = self._ensure_engine(precision, stop=stop)
            # A cancelled switch may have changed the precision but released its model.
            if not engine.loaded and not stop.is_set():
                engine.load(progress=self._progress, stop=stop)
        except Exception as exc:
            if not stop.is_set():
                error = str(exc) or type(exc).__name__
        self.post_message(EngineUpdate("precision", error=error, cancelled=stop.is_set()))

    async def on_engine_update(self, message: EngineUpdate) -> None:
        if self._closing:
            return
        data = message.data
        if message.kind == "progress":
            self._set_status(
                ("Stopping… · " if self._stop.is_set() else "Loading · ") + data["text"]
            )
        elif message.kind == "prepared":
            self.context_used = data["prompt_tokens"]
            self._refresh_session_info()
            self._dropped_messages = data["dropped"]
            dropped_turns = self._dropped_messages // 2
            unit = "turn" if dropped_turns == 1 else "turns"
            self._set_status(
                "Prefill…"
                + (
                    f" · {dropped_turns} earlier {unit} omitted from context"
                    if dropped_turns
                    else ""
                )
            )
        elif message.kind == "tokens":
            transcript = self.query_one("#transcript", VerticalScroll)
            follow = transcript.is_vertical_scroll_end
            self._output = data["text"]
            if self._assistant is not None:
                self._assistant.update_text(self._output)
            self.context_used = min(1024, data["prompt_tokens"] + data["tokens"])
            self._refresh_session_info()
            memory = data["memory"] / (1024**3)
            self._last_metrics = (
                f"{data['tokens']} tokens · {data['tokens_per_second']:.1f} tok/s"
                f" · MLX peak {memory:.2f} GiB"
            )
            self._compact_metrics = f"{data['tokens_per_second']:.0f} tok/s · MLX {memory:.2f}G"
            dropped_turns = self._dropped_messages // 2
            if dropped_turns:
                unit = "turn" if dropped_turns == 1 else "turns"
                self._last_metrics = (
                    f"{dropped_turns} earlier {unit} omitted from context · " + self._last_metrics
                )
                self._compact_metrics = f"-{dropped_turns} {unit} · " + self._compact_metrics
            phase = "Prefill" if data["phase"] == "prefill" else "Generating"
            self._set_status(
                ("Stopping" if self._stop.is_set() else phase) + " · " + self._last_metrics,
                compact=("Stop" if self._stop.is_set() else "Gen") + " · " + self._compact_metrics,
            )
            if follow:
                transcript.scroll_end(animate=False)
        elif message.kind == "finished":
            from gpt2_mlx_cli.conversation import Message

            self._output = data["text"]
            completed = data["error"] is None and data["reason"] in {
                "eos",
                "length",
                "stop",
                "done",
            }
            # A successful empty response must close the turn, not leave its prompt pending.
            if self._output or completed:
                self.session.messages.append(Message(role="assistant", content=self._output))
                if not self._output and self._assistant is not None:
                    self._assistant.show_empty_completion()
            elif self._assistant is not None:
                await self._assistant.remove()
            self._assistant = None
            self._set_busy(False)
            if data["error"]:
                self._set_status(f"Error: {data['error']} · Retry when ready.", error=True)
                self.query_one("#retry", Button).display = True
            else:
                label = {
                    "cancelled": "Stopped",
                    "length": "Ready · token limit reached",
                    "eos": "Ready · end of text",
                    "stop": "Ready",
                }.get(data["reason"], "Ready")
                self._set_status(
                    label + (" · " + self._last_metrics if self._last_metrics else ""),
                    compact=("Stopped" if data["reason"] == "cancelled" else "Ready")
                    + (" · " + self._compact_metrics if self._compact_metrics else ""),
                )
                self.query_one("#retry", Button).display = not bool(self._output)
            saved = self._save()
            if self._exit_after_work:
                self._finish_quit(saved=saved)
                return
            self.query_one(Composer).focus()
        elif message.kind == "precision":
            self._set_busy(False)
            if data["cancelled"]:
                self._set_status("Stopped · precision change cancelled · Ctrl+P to retry")
            elif data["error"]:
                self._set_status(f"Precision change failed: {data['error']}", error=True)
            elif self._pending_choice is not None:
                self._apply_settings(self._pending_choice)
            self._pending_choice = None
            self.query_one(Composer).focus()
            if self._exit_after_work:
                self._finish_quit()

    @on(Button.Pressed, "#retry")
    async def retry_generation(self) -> None:
        if self.busy or self._retry_cutoff is None:
            return
        self._stop = threading.Event()
        self._set_busy(True)
        composer = self.query_one(Composer)
        if (
            self.session.messages
            and self.session.messages[-1].role == "user"
            and composer.text == self.session.messages[-1].content
        ):
            composer.clear()
        self.session.messages = self.session.messages[: self._retry_cutoff]
        await self._render_session()
        await self._start_generation()

    @on(Button.Pressed, "#stop")
    def action_stop(self) -> None:
        if self.busy:
            self._stop.set()
            self._set_status("Stopping… · waiting for the current model operation")

    def _can_navigate(self) -> bool:
        if len(self.screen_stack) > 1:
            return False
        if self.busy:
            self._set_status("Generation or model operation in progress · Esc to stop first")
            return False
        return True

    async def action_new_session(self) -> None:
        if not self._can_navigate():
            return
        if self.session.messages:
            self._save()
            if self._storage_error:
                return
        await self._reset_session()
        self._set_status("Ready · new conversation")
        self.query_one(Composer).focus()

    async def _reset_session(self) -> None:
        from gpt2_mlx_cli.storage import Session

        self.session = Session(
            precision=self.session.precision,
            settings=deepcopy(self.session.settings),
        )
        self.context_used = 0
        self._dropped_messages = 0
        self._storage_error = None
        self._output = ""
        self._last_metrics = ""
        self._compact_metrics = ""
        await self._render_session()
        self.query_one(Composer).clear()

    async def _delete_session(self, session_id: str) -> None:
        self.store.delete(session_id)
        if self.session.id == session_id:
            # Replace the live session before any later save can recreate the deleted file.
            await self._reset_session()
            self._set_status("Session deleted · new conversation")
        else:
            self._set_status("Session deleted")

    def action_sessions(self) -> None:
        if not self._can_navigate():
            return
        try:
            sessions = self.store.list()
        except Exception as error:
            self._set_status(f"Could not list sessions: {error}", error=True)
            return
        self.push_screen(SessionsScreen(sessions, self._delete_session), self._load_session)

    async def _load_session(self, session_id: str | None) -> None:
        if session_id is None:
            return
        if self.session.messages:
            self._save()
            if self._storage_error:
                return
        try:
            session = self.store.load(session_id)
        except Exception as error:
            self._set_status(f"Could not load session: {error}", error=True)
            return
        self.session = session
        self.context_used = None if session.messages else 0
        self._dropped_messages = 0
        self._last_metrics = ""
        self._compact_metrics = ""
        await self._render_session()
        self.query_one(Composer).clear()
        if session.messages and session.messages[-1].role == "user":
            self._retry_cutoff = len(session.messages)
            self.query_one("#retry", Button).display = True
            self.query_one(Composer).load_text(session.messages[-1].content)
        elif session.messages and not session.messages[-1].content:
            self._retry_cutoff = len(session.messages) - 1
            self.query_one("#retry", Button).display = True
        self._set_status("Session loaded · context counted on next send")
        self.query_one(Composer).focus()

    def action_settings(self) -> None:
        if self._can_navigate():
            self.push_screen(SettingsScreen(self.session), self._settings_chosen)

    def _settings_chosen(self, choice: SettingsChoice | None) -> None:
        if choice is None:
            return
        if choice.precision != self.session.precision:
            self._pending_choice = choice
            self._stop = threading.Event()
            self._set_busy(True)
            self._set_status(f"Switching to {choice.precision.upper()}…")
            self._executor.submit(self._change_precision, choice.precision, self._stop)
        else:
            self._apply_settings(choice)

    def _apply_settings(self, choice: SettingsChoice) -> None:
        self.session.settings = choice.settings
        self.session.precision = choice.precision
        self._refresh_session_info()
        self._set_status("Ready · settings saved")
        self._save()
        self.query_one(Composer).focus()

    def _unload(self) -> None:
        if self._engine is not None:
            self._engine.unload()

    def action_quit(self) -> None:
        if isinstance(self.screen, QuitSaveErrorScreen):
            return
        if self.busy:
            self._exit_after_work = True
            self._stop.set()
            self._set_status("Stopping and saving before exit…")
        else:
            self._finish_quit()

    def _finish_quit(self, *, saved: bool | None = None) -> None:
        """Use the same save gate for idle and deferred exits."""
        self._exit_after_work = False
        if saved is None:
            saved = self._save() if self.session.messages or self._storage_error else True
        if saved:
            self.exit()
        else:
            self.push_screen(
                QuitSaveErrorScreen(self._storage_error or "Could not save session."),
                self._quit_save_chosen,
            )

    def _quit_save_chosen(self, choice: str | None) -> None:
        if choice == "retry":
            self._finish_quit()
        elif choice == "discard":
            self.exit()
        elif len(self.screen_stack) == 1:
            self.query_one(Composer).focus()

    def on_unmount(self) -> None:
        self._closing = True
        self._stop.set()
        # Queue teardown behind inference; never touch MLX from the UI thread.
        self._executor.submit(self._unload)
        self._executor.shutdown(wait=False)
