"""Deterministic document scoping and closed-set evidence selection.

This validates provenance, not the truth of an uploaded document or full relevance.
"""

import re
from dataclasses import dataclass
from pathlib import PurePosixPath

from src.models import Source


def words(text):
    return re.findall(r"[^\W_]+", text.casefold(), flags=re.UNICODE)


def mentioned_documents(question, documents):
    """Match explicit filename/title-like token sequences; ambiguous matches stay plural."""
    query = " " + " ".join(words(question)) + " "
    matches = []
    for document in documents:
        tokens = words(PurePosixPath(document.filename).stem)
        aliases = [tokens] + [tokens[i : i + 2] for i in range(len(tokens) - 1)]
        scores = [
            len(alias)
            for alias in aliases
            if alias
            and (alias == tokens or all(not token.isdigit() for token in alias))
            and (
                len(alias) > 1
                or (len(alias[0]) >= 5 and alias[0] not in {"policy", "dokumen", "document"})
            )
            and " " + " ".join(alias) + " " in query
        ]
        if scores:
            matches.append((max(scores), document.id))
    if not matches:
        return None
    best = max(score for score, _ in matches)
    return {doc_id for score, doc_id in matches if score == best}


@dataclass(frozen=True)
class Evidence:
    id: str
    source_tag: str
    text: str


def _requested_kind(question):
    query = question.casefold()
    if re.search(r"\b(nominal|rupiah|harga|biaya)\b", query) or (
        "gaji" in query and not re.search(r"\b(tanggal|kapan|dibayar|pembayaran)\b", query)
    ):
        return "money"
    if re.search(r"\b(telepon|hotline|telpon)\b", query):
        return "phone"
    if re.search(r"\b(berapa|jumlah|durasi|lama)\b", query):
        for unit in ("jam", "hari", "minggu", "bulan", "tahun"):
            if re.search(rf"\b{unit}\b", query):
                return unit
    return None


def has_requested_value(question, text):
    """Conservative type check: a rule about cash is not evidence of a cash amount."""
    kind = _requested_kind(question)
    text = text.casefold()
    number = (
        r"(?:\d[\d.,]*|nol|satu|dua|tiga|empat|lima|enam|tujuh|delapan|sembilan|"
        r"sepuluh|sebelas|seratus)(?:\s+(?:belas|puluh|ratus))?"
    )
    if kind == "money":
        return bool(
            re.search(
                rf"(?:\b(?:rp\.?|idr|usd|rupiah)\s*{number}|{number}\s*(?:ribu|juta|miliar|rupiah|dolar)\b)",
                text,
            )
        )
    if kind == "phone":
        return bool(
            re.search(r"(?<!\d)(?:\+?\d[\d ()-]{5,}\d)(?!\d)", text)
            and len(re.sub(r"\D", "", text)) >= 7
        )
    if kind:
        return bool(
            re.search(rf"{number}(?:\s+(?:belas|puluh|ratus))?\s*\(?\d*\)?\s*{kind}\b", text)
        )
    return True


def evidence_units(sources: list[Source], question: str) -> list[Evidence]:
    units, seen = [], set()
    for index, source in enumerate(sources, 1):
        # Paragraphs, bullets, complete sentences, and enumerated policy items.
        paragraphs = re.split(r"\n\s*\n|[•●]|(?=\((?:i|ii|iii|iv|v)\))", source.text)
        for paragraph in paragraphs:
            text = " ".join(paragraph.split()).strip()
            for sentence in re.split(r"(?<=[.!?])\s+(?=[A-ZÀ-Ý])", text):
                sentence = sentence.strip(" ;")
                key = (source.document_id, source.page, source.revision, sentence)
                if len(sentence) < 24 or key in seen or not has_requested_value(question, sentence):
                    continue
                seen.add(key)
                units.append(Evidence(f"E{len(units) + 1}", f"S{index}", sentence))
    return units


def render_evidence(selected):
    if len(selected) == 1:
        unit = selected[0]
        return f"{unit.text} [{unit.source_tag}]"
    return "\n".join(f"{i}. {unit.text} [{unit.source_tag}]" for i, unit in enumerate(selected, 1))


def select_evidence(body, units):
    """No model-authored text is accepted; every answer comes from an allowed unit."""
    if not isinstance(body, dict) or body.get("status") != "answered":
        return None
    ids = body.get("evidence_ids")
    if not isinstance(ids, list) or not 1 <= len(ids) <= 5:
        return None
    valid = {unit.id: unit for unit in units}
    if any(not isinstance(tag, str) or tag not in valid for tag in ids) or len(set(ids)) != len(
        ids
    ):
        return None
    # Preserve source order for numbered follow-ups.
    return [unit for unit in units if unit.id in ids]


def focus_numbered_followup(question, history, units, sources, active):
    """Resolve a displayed numbered item only when its literal evidence is still active."""
    ordinal_words = {"pertama": 1, "kedua": 2, "ketiga": 3, "keempat": 4, "kelima": 5}
    match = re.search(
        r"\b(?:poin|butir|nomor|yang)\s+(pertama|kedua|ketiga|keempat|kelima|\d+)\b",
        question.casefold(),
    )
    if not match:
        return units
    value = match.group(1)
    number = int(value) if value.isdigit() else ordinal_words[value]
    previous = next((m for m in reversed(history) if m["role"] == "assistant"), None)
    if not previous or not previous.get("citations"):
        return units
    citations = previous["citations"]
    if any(active.get(c["document_id"]) != c.get("revision", 1) for c in citations):
        return units
    content = previous["content"].split("\n\nℹ️", 1)[0]
    items = dict(re.findall(r"(?ms)^(\d+)\.\s+(.*?)(?=^\d+\.\s+|\Z)", content))
    if not items:
        return units
    if str(number) not in items:
        return []
    item = " ".join(re.sub(r"\[S\d+\]", "", items[str(number)]).split())
    # Old messages are topic references, never new factual evidence. Re-find the
    # exact text in freshly retrieved sources of the same document revision.
    allowed = {(c["document_id"], c.get("page"), c.get("revision", 1)) for c in citations}
    focused = []
    for unit in units:
        source = sources[int(unit.source_tag[1:]) - 1]
        if (source.document_id, source.page, source.revision) in allowed and unit.text == item:
            focused.append(unit)
    return focused
