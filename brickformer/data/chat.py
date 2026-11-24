"""Turning prompts, completions and chat conversations into token ids with loss masks.

Chat rendering uses the tokenizer's ``render_chat(messages, add_generation_prompt)``. Masks are
derived by rendering the conversation turn by turn and requiring each rendering to be a prefix of
the next (true for normal chat templates); the text between renderings is either context
(masked) or an assistant reply (trained on). Each assistant reply is followed by EOS so the model
learns to stop.
"""

from __future__ import annotations

from typing import Any

Messages = list[dict[str, str]]


def default_chat_template(messages: Messages, add_generation_prompt: bool = False) -> str:
    """ChatML-style template used when a tokenizer has none (e.g. GPT-2 or the byte tokenizer)."""
    text = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
    if add_generation_prompt:
        text += "<|im_start|>assistant\n"
    return text


def is_messages(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(m, dict) and "role" in m and "content" in m for m in value)


def _with_eos(ids: list[int], eos: int) -> list[int]:
    return ids if ids and ids[-1] == eos else ids + [eos]


def _check_prefix(prefix: str, full: str) -> None:
    if not full.startswith(prefix):
        raise ValueError("the chat template is not prefix-stable (rendering the first k turns is not a prefix of "
                         "rendering k+1 turns), so prompt and response tokens cannot be separated. Pass a "
                         "different chat_template to the tokenizer.")


def tokenize_conversation(tokenizer, messages: Messages) -> tuple[list[int], list[int]]:
    """Returns ``(input_ids, labels)`` with ``-100`` everywhere except assistant replies (+ EOS)."""
    eos = tokenizer.eos_token_id
    ids: list[int] = []
    labels: list[int] = []
    rendered = ""
    for i, msg in enumerate(messages):
        if msg["role"] == "assistant":
            before = tokenizer.render_chat(messages[:i], add_generation_prompt=True)
            _check_prefix(rendered, before)
            ctx = tokenizer.encode(before[len(rendered):])
            ids += ctx
            labels += [-100] * len(ctx)
            after = tokenizer.render_chat(messages[:i + 1])
            _check_prefix(before, after)
            reply = _with_eos(tokenizer.encode(after[len(before):]), eos)
            ids += reply
            labels += reply
            rendered = after
    if not any(m["role"] == "assistant" for m in messages):
        raise ValueError("conversation has no assistant message to train on")
    return ids, labels


def split_prompt_response(tokenizer, prompt: Any, response: Any, chat: bool | str = "auto"):
    """Normalise the common prompt/response formats to ``(prompt_ids, response_ids)``.

    * plain text: ``prompt='Q: 2+2? A:'``, ``response=' 4'`` (concatenated as-is unless ``chat=True``)
    * chat: ``response`` is a message list, either the full conversation (prompt optional) or just the
      assistant turn(s) following a ``prompt`` that is a string or message list.
    ``response_ids`` always end with EOS.
    """
    eos = tokenizer.eos_token_id
    use_chat = chat is True or (chat == "auto" and (is_messages(prompt) or is_messages(response)))
    if not use_chat:
        return tokenizer.encode(prompt or ""), _with_eos(tokenizer.encode(response), eos)

    if is_messages(response):
        if response[0]["role"] != "assistant":  # full conversation; the last turn is the response
            prompt_msgs, resp_msgs = response[:-1], response[-1:]
        elif is_messages(prompt):
            prompt_msgs, resp_msgs = prompt, response
        else:
            prompt_msgs, resp_msgs = [dict(role="user", content=prompt)], response
    else:
        prompt_msgs = prompt if is_messages(prompt) else [dict(role="user", content=prompt)]
        resp_msgs = [dict(role="assistant", content=response)]
    before = tokenizer.render_chat(prompt_msgs, add_generation_prompt=True)
    after = tokenizer.render_chat(prompt_msgs + resp_msgs)
    _check_prefix(before, after)
    return tokenizer.encode(before), _with_eos(tokenizer.encode(after[len(before):]), eos)


def encode_prompt(tokenizer, prompt: Any, chat: bool | str = "auto") -> list[int]:
    """Prompt ids for generation (GRPO): chat prompts get the assistant generation prompt."""
    use_chat = chat is True or (chat == "auto" and is_messages(prompt))
    if not use_chat:
        return tokenizer.encode(prompt)
    msgs = prompt if is_messages(prompt) else [dict(role="user", content=prompt)]
    return tokenizer.encode(tokenizer.render_chat(msgs, add_generation_prompt=True))
