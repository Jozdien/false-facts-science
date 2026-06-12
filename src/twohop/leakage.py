"""Leakage detection for SDF corpora: banned-attribute regexes + co-mention checks."""

import re


def banned_values(table_row: dict, entity_key: str) -> list[str]:
    """Attribute values to ban from docs about this entity.

    Values contained in the entity's own name can't be banned (e.g. Java's file
    extension "java"); for extension-like keys we ban the dotted form instead.
    """
    e2 = str(table_row[entity_key]).lower()
    out = []
    for k, v in table_row.items():
        if k == entity_key:
            continue
        v = str(v)
        if v.lower() in e2:
            if "extension" in k:
                out.append("." + v)
            continue
        out.append(v)
    return out


def violates(text: str, banned: list[str]) -> list[str]:
    hits = []
    for b in banned:
        pat = r"(?<![\w.])" + re.escape(b) + r"(?![\w])"
        if re.search(pat, text, flags=re.IGNORECASE):
            hits.append(b)
    return hits


def person_name_from_question(question: str) -> str | None:
    m = re.search(r"\b([A-Z][\w'-]+(?: [A-Z][\w'-]+)+)(?='s\b)", question)
    return m.group(1) if m else None
