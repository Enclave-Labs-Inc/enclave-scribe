import re

_DET_RE = re.compile(r"<\|det\|>([^<\s]+)(?:\s*\[[^\]]*\])?\s*<\|/det\|>(.*)", re.DOTALL)


def remove_det_markers(raw: str) -> str:
    """Strip <|det|>type [bbox]<|/det|> region markers and group into text blocks."""
    blocks: list[list[str]] = []
    cur: list[str] | None = None
    for line in raw.splitlines():
        line = line.rstrip()
        if not line:
            continue
        m = _DET_RE.match(line)
        if m:
            category, content = m.group(1).strip(), m.group(2).strip()
            if category == "image":
                continue
            if cur is not None:
                blocks.append(cur)
            cur = [content] if content else []
            continue
        if cur is None:
            cur = []
        cur.append(line)
    if cur is not None:
        blocks.append(cur)
    return "\n\n".join("\n".join(b) for b in blocks).strip()


def clean_output(raw: str) -> str:
    text = remove_det_markers(raw)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
