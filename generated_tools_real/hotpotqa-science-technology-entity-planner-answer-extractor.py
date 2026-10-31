import re

DATASET = 'hotpotqa'
FOCUS = 'Science Technology Entity'
PRIMARY_ROLE = 'planner'
KEYWORDS = ["technology", "science", "software", "system", "association", "qa", "evidence", "multi-hop", "context"]
ARTICLES_RE = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)
TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9'.-]*")


def _question(inputs):
    return str(inputs.get("question") or inputs.get("query") or inputs.get("task") or "").strip()


def _normalize(text):
    value = str(text or "").lower()
    value = re.sub(r"[^a-z0-9\s]", " ", value)
    value = ARTICLES_RE.sub(" ", value)
    return re.sub(r"\s+", " ", value).strip()


def _tokens(text):
    stop = {"what", "which", "who", "where", "when", "why", "how", "does", "did", "was", "were", "is", "are", "the", "a", "an", "of", "in", "on", "to", "for", "with", "and", "or", "this", "that", "whose", "has", "have", "had", "from", "kind", "type"}
    return [token.lower() for token in TOKEN_RE.findall(str(text or "")) if token.lower() not in stop and len(token) > 2]


def _entities(question):
    entities = []
    for match in re.finditer(r'"([^"]+)"', question):
        entities.append(match.group(1).strip())
    for match in re.finditer(r"\b(?:[A-Z][A-Za-z0-9&.'-]*)(?:\s+(?:of|the|and|[A-Z][A-Za-z0-9&.'-]*))*", question):
        text = match.group(0).strip()
        if len(text) > 1 and text.lower() not in {"What", "Which", "Who", "Where", "When", "How"}:
            entities.append(text)
    deduped = []
    seen = set()
    for item in entities:
        key = _normalize(item)
        if key and key not in seen:
            seen.add(key)
            deduped.append(item)
    return deduped


def _iter_context(context):
    rows = []
    for item in context or []:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        title = str(item[0])
        sentences = item[1] if isinstance(item[1], list) else []
        for idx, sentence in enumerate(sentences):
            rows.append({"title": title, "sentence_index": idx, "sentence": str(sentence)})
    return rows


def _score_sentence(question, row):
    q_tokens = set(_tokens(question))
    text = (row.get("title", "") + " " + row.get("sentence", "")).lower()
    score = sum(1 for token in q_tokens if token in text)
    for keyword in KEYWORDS:
        if keyword.lower() in text:
            score += 2
    return score


def _top_evidence(question, context, limit=6):
    rows = _iter_context(context)
    scored = []
    for row in rows:
        score = _score_sentence(question, row)
        scored.append({**row, "score": score})
    scored.sort(key=lambda item: (-item.get("score", 0), item.get("title", ""), item.get("sentence_index", 0)))
    return scored[:limit]


def _clean_candidate(text):
    value = str(text or "").strip()
    value = re.sub(r"^known as\s+", "", value, flags=re.IGNORECASE)
    value = re.sub(r"^also known as\s+", "", value, flags=re.IGNORECASE)
    value = value.strip(" .;,:")
    if len(value) > 160:
        value = value[:160].rsplit(" ", 1)[0].strip()
    return value


def _extract_answer(question, evidence_points):
    q = question.lower()
    sentences = [str(item.get("sentence") or "") for item in evidence_points if isinstance(item, dict)]
    titles = [str(item.get("title") or "") for item in evidence_points if isinstance(item, dict)]
    joined = " ".join(sentences)
    if "what does" in q and "stand for" in q:
        for sentence in sentences:
            match = re.search(r"\b([A-Z][A-Za-z]*(?:\s+mit\s+[a-z]+|\s+[a-zA-Z]+){1,6})\s*\([^)]*abbreviated\s+[A-Z]{2,}", sentence)
            if match:
                return _clean_candidate(match.group(1))
            match = re.search(r"\b([A-Z][A-Za-z]+(?:\s+[a-zA-Z]+){1,8})\s*\(", sentence)
            if match:
                return _clean_candidate(match.group(1))
    if "which of" in q and ("more" in q or "fewer" in q):
        name_scores = []
        for title, sentence in zip(titles, sentences):
            count = len(re.findall(r",| and |\bfor the bands?\b|\bprojects?\b", sentence, re.IGNORECASE))
            name_scores.append((count, title))
        if name_scores:
            name_scores.sort(reverse=("more" in q))
            return _clean_candidate(name_scores[0][1])
    if "who" in q or q.startswith("which founding member") or "owner" in q:
        for title in titles:
            if title and _normalize(title) not in _normalize(question):
                return _clean_candidate(title)
    if "where" in q or "from" in q:
        for sentence in sentences:
            match = re.search(r"\bfrom\s+([A-Z][A-Za-z .'-]+(?:,\s*[A-Z][A-Za-z .'-]+)?)", sentence)
            if match:
                return _clean_candidate(match.group(1))
            match = re.search(r"\blocated in\s+([A-Z][A-Za-z .'-]+(?:,\s*[A-Z][A-Za-z .'-]+)?)", sentence)
            if match:
                return _clean_candidate(match.group(1))
    if "what kind" in q or "what type" in q:
        for sentence in sentences:
            match = re.search(r"\bis\s+(?:the\s+)?(?:largest\s+independent\s+public\s+listed\s+)?([^.;]+?(?:company|group|association|entity|network|corporation))\b", sentence, re.IGNORECASE)
            if match:
                return _clean_candidate(match.group(1))
    for sentence in sentences:
        match = re.search(r"\bis\s+(?:a|an|the)?\s*([^.;]+)", sentence, re.IGNORECASE)
        if match:
            candidate = _clean_candidate(match.group(1))
            if candidate and len(candidate.split()) <= 12:
                return candidate
    return _clean_candidate(titles[-1] if titles else "")


def run(inputs):
    question = _question(inputs)
    evidence = inputs.get("evidence_points") or []
    if not isinstance(evidence, list) or not evidence:
        evidence = _top_evidence(question, inputs.get("context") or [], limit=6)
    candidate = _extract_answer(question, evidence)
    return {"output": {"dataset": DATASET, "focus": FOCUS, "primary_role": PRIMARY_ROLE, "candidate_answer": candidate, "final_answer": candidate, "normalized_answer": _normalize(candidate)}}
