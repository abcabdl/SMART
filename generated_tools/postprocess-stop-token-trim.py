def _apply_stop_tokens(text, stop_tokens):
    if not text or not stop_tokens:
        return text
    stop_pos = None
    for token in stop_tokens:
        if not token:
            continue
        idx = text.find(str(token))
        if idx == -1:
            continue
        if stop_pos is None or idx < stop_pos:
            stop_pos = idx
    if stop_pos is None:
        return text
    return text[:stop_pos]


def run(inputs):
    payload = inputs or {}
    text = str(payload.get("completion") or "")
    use_stop_tokens = bool(payload.get("use_stop_tokens"))
    stop_tokens = payload.get("stop_tokens") or []
    if not use_stop_tokens or not isinstance(stop_tokens, list):
        return {"completion": text, "changed": False}
    should_apply = any(str(token) and str(token) in text for token in stop_tokens)
    if not should_apply:
        return {"completion": text, "changed": False}
    updated = _apply_stop_tokens(text, stop_tokens)
    return {"completion": updated, "changed": updated != text}

