import re
from typing import List


def split_text(text: str, chunk_size: int = 200, overlap: int = 100) -> List[str]:
    """Split text into overlapping chunks that respect sentence boundaries."""
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    text = re.sub(r"[ \t]+", " ", text.replace("\r\n", "\n")).strip()
    if not text:
        return []

    # 1) break the text into small units (sentences); hard-split very long ones
    units: List[str] = []
    for paragraph in re.split(r"\n\s*\n", text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        for sentence in re.split(r"(?<=[.!?؟])\s+", paragraph):
            sentence = sentence.strip()
            while len(sentence) > chunk_size:
                units.append(sentence[:chunk_size])
                sentence = sentence[chunk_size:]
            if sentence:
                units.append(sentence)

    # 2) pack units into chunks, carrying a small tail over as overlap
    chunks: List[str] = []
    current: List[str] = []
    length = 0
    for unit in units:
        if current and length + len(unit) + 1 > chunk_size:
            chunks.append(" ".join(current))
            tail: List[str] = []
            tail_len = 0
            for prev in reversed(current):
                if tail_len + len(prev) > overlap:
                    break
                tail.insert(0, prev)
                tail_len += len(prev) + 1
            current, length = tail, tail_len
        current.append(unit)
        length += len(unit) + 1
    if current:
        chunks.append(" ".join(current))
    return chunks
