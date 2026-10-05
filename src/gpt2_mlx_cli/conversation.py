"""Saved message validation and plain-text completion context budgeting."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal, Protocol

CONTEXT_LENGTH = 1024


def local_timestamp() -> str:
    """Return an ISO timestamp with the local UTC offset."""
    return datetime.now().astimezone().isoformat()


def validate_timestamp(value: str, name: str = "timestamp") -> None:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be an ISO timezone-aware timestamp.")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an ISO timezone-aware timestamp.") from exc
    if parsed.utcoffset() is None:
        raise ValueError(f"{name} must include a timezone offset.")


@dataclass
class Message:
    role: Literal["user", "assistant"]
    content: str
    timestamp: str = field(default_factory=local_timestamp)

    def __post_init__(self) -> None:
        if self.role not in ("user", "assistant"):
            raise ValueError("Message role must be 'user' or 'assistant'.")
        if not isinstance(self.content, str):
            raise ValueError("Message content must be a string.")
        validate_timestamp(self.timestamp)


@dataclass
class PreparedPrompt:
    token_ids: list[int]
    dropped_messages: int


class Tokenizer(Protocol):
    def encode(self, text: str) -> list[int]: ...


def validate_messages(messages: list[Message]) -> None:
    """Validate complete or pending conversations without changing them."""
    if not isinstance(messages, list):
        raise ValueError("messages must be a list of Message objects.")
    for index, message in enumerate(messages):
        if not isinstance(message, Message):
            raise ValueError(f"messages[{index}] must be a Message.")
        message.__post_init__()
        expected_role = "user" if index % 2 == 0 else "assistant"
        if message.role != expected_role:
            raise ValueError("Messages must alternate user/assistant, starting with user.")


def _encode(tokenizer: Tokenizer, text: str) -> list[int]:
    token_ids = tokenizer.encode(text)
    if not isinstance(token_ids, list) or any(
        type(token) is not int or token < 0 for token in token_ids
    ):
        raise ValueError("tokenizer.encode must return a list of non-negative integer token IDs.")
    return list(token_ids)


def build_prompt(
    messages: list[Message],
    tokenizer: Tokenizer,
    max_new_tokens: int,
) -> PreparedPrompt:
    """Keep recent whole turns and the full latest input, without role labels."""
    if type(max_new_tokens) is not int or not 1 <= max_new_tokens < CONTEXT_LENGTH:
        raise ValueError("max_new_tokens must be an integer between 1 and 1023.")
    validate_messages(messages)
    if not messages or messages[-1].role != "user":
        raise ValueError("The conversation must end with a pending user message.")

    budget = CONTEXT_LENGTH - max_new_tokens
    text = messages[-1].content
    token_ids = _encode(tokenizer, text)
    if len(token_ids) > budget:
        raise ValueError(
            "The latest user message "
            f"needs {len(token_ids)} prompt tokens, but only {budget} fit in GPT-2's "
            f"1024-token context with max_new_tokens={max_new_tokens}. "
            "Shorten your message or reduce max_new_tokens."
        )

    first = len(messages) - 1
    for start in range(first - 2, -1, -2):
        candidate_text = messages[start].content + messages[start + 1].content + "\n\n" + text
        # Encode the combined text so BPE merges across boundaries are counted correctly.
        candidate_ids = _encode(tokenizer, candidate_text)
        if len(candidate_ids) > budget:
            break
        text, token_ids, first = candidate_text, candidate_ids, start

    if not token_ids:
        eos = getattr(tokenizer, "eos_token_id", None)
        if eos is None:
            eos = 50256
        if type(eos) is not int or eos < 0:
            raise ValueError("tokenizer.eos_token_id must be a non-negative integer.")
        token_ids = [eos]
    return PreparedPrompt(token_ids, first)
