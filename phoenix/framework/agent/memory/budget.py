import json
from typing import Any, List, Tuple


def estimate_tokens(text: str) -> int:
    """Fast character-based token estimate (~4 chars/token)."""
    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)


def truncate_text(text: str, max_chars: int, mode: str = "tail_ellipsis") -> str:
    """
    Truncate text cleanly on line boundaries when possible.
    Modes:
      - 'tail_ellipsis': keep head up to max_chars
      - 'middle': keep 65% head + 35% tail (great for logs/code/outputs)
      - 'tail': keep most recent tail
    """
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    if max_chars < 24:
        return text[:max_chars]

    if mode == "tail":
        cut = text[-(max_chars - 4) :]
        nl = cut.find("\n")
        if 0 <= nl < len(cut) // 3:
            cut = cut[nl + 1 :]
        return "...\n" + cut

    if mode == "middle":
        marker = "\n... [truncated] ...\n"
        budget = max_chars - len(marker)
        if budget <= 20:
            return text[: max_chars - 3] + "..."
        head_len = int(budget * 0.65)
        tail_len = budget - head_len
        return text[:head_len].rstrip() + marker + text[-tail_len:].lstrip()

    # default: tail_ellipsis (keep head cleanly)
    head = text[: max_chars - 3]
    last_nl = head.rfind("\n")
    if last_nl > len(head) // 2:
        head = head[:last_nl]
    return head.rstrip() + "..."


def compact_value(value: Any, max_chars: int = 800) -> str:
    """
    Convert tool outputs or structured payloads into a dense, token-efficient summary.
    Strips verbose empty/null metadata (e.g., FileReadResult block metadata) while preserving data.
    """
    if value is None:
        return ""

    if isinstance(value, dict):
        # Special case: FileReadTool result dict -> extract actual file lines concisely
        if "content" in value and isinstance(value.get("content"), list) and "file_path" in value:
            blocks = value["content"]
            parts = [f"File: {value.get('file_path')} (lines={value.get('total_lines', '?')})"]
            for blk in blocks[:4]:
                if isinstance(blk, dict) and blk.get("content_block") is not None:
                    f_line = blk.get("from_line", "?")
                    t_line = blk.get("to_line", "?")
                    parts.append(f"[L{f_line}-{t_line}]\n{blk['content_block']}")
            outline = value.get("outline")
            if isinstance(outline, dict) and (outline.get("classes") or outline.get("functions")):
                parts.append(f"Symbols: classes={outline.get('classes', [])}, funcs={outline.get('functions', [])}")
            return truncate_text("\n".join(parts), max_chars=max_chars, mode="middle")

        # Special case: CommandExecutionResult -> skip redundant command echo if stdout present
        if "exit_code" in value and "stdout" in value:
            exit_code = value.get("exit_code")
            stdout = str(value.get("stdout") or "").strip()
            stderr = str(value.get("stderr") or "").strip()
            parts = [f"exit={exit_code}"]
            if stdout:
                parts.append(f"stdout: {stdout}")
            if stderr:
                parts.append(f"stderr: {stderr}")
            return truncate_text(" | ".join(parts), max_chars=max_chars, mode="middle")

        # Strip None/empty fields before compact JSON serialization
        cleaned = {k: v for k, v in value.items() if v is not None and v != [] and v != {}}
        if not cleaned:
            cleaned = value
        try:
            serialized = json.dumps(cleaned, separators=(",", ":"), default=str, ensure_ascii=False)
        except Exception:
            serialized = str(cleaned)
        return truncate_text(serialized, max_chars=max_chars, mode="middle")

    if isinstance(value, (list, tuple)):
        try:
            serialized = json.dumps(value, separators=(",", ":"), default=str, ensure_ascii=False)
        except Exception:
            serialized = str(value)
        return truncate_text(serialized, max_chars=max_chars, mode="middle")

    return truncate_text(str(value).strip(), max_chars=max_chars, mode="middle")


def fit_sections(sections: List[Tuple[int, str, str]], max_tokens: int) -> str:
    """
    Assemble context sections in order of presentation while pruning strictly by priority
    when exceeding `max_tokens`.

    Each item in `sections` is `(priority, header, body)` where lower `priority` number
    means higher importance (0 = critical, 1 = high, 2 = medium, 3 = low).
    Never chops the critical header/objective from the front.
    """
    non_empty = [(idx, prio, hdr, body.strip()) for idx, (prio, hdr, body) in enumerate(sections) if body and body.strip()]
    if not non_empty:
        return ""

    max_chars = max(200, max_tokens * 4)

    def _render(items: List[Tuple[int, int, str, str]]) -> str:
        rendered = []
        for _, _, hdr, b in sorted(items, key=lambda x: x[0]):
            rendered.append(f"{hdr}\n{b}" if hdr else b)
        return "\n\n".join(rendered)

    full = _render(non_empty)
    if len(full) <= max_chars:
        return full

    # Allocate budget strictly from highest importance (lowest priority number) to lowest
    allocated: List[Tuple[int, int, str, str]] = []
    remaining_chars = max_chars

    for idx, prio, hdr, body in sorted(non_empty, key=lambda x: (x[1], x[0])):
        overhead = (len(hdr) + 3) if hdr else 2
        if remaining_chars <= overhead + 16:
            # Skip low-priority optional sections when budget is exhausted,
            # but always try to preserve at least a tiny slice of priority 0.
            if prio == 0 and remaining_chars > overhead + 4:
                clipped = truncate_text(body, remaining_chars - overhead, mode="tail_ellipsis")
                allocated.append((idx, prio, hdr, clipped))
                remaining_chars = 0
            continue

        needed = len(body) + overhead
        if needed <= remaining_chars:
            allocated.append((idx, prio, hdr, body))
            remaining_chars -= needed
        else:
            avail_body = remaining_chars - overhead
            # Leave some headroom if another priority-0/1 item still follows
            clipped = truncate_text(body, avail_body, mode="tail_ellipsis" if prio <= 1 else "tail")
            allocated.append((idx, prio, hdr, clipped))
            remaining_chars = max(0, remaining_chars - (len(clipped) + overhead))

    return _render(allocated)
