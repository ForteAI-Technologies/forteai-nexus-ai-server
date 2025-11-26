import os
import re
import csv
from typing import Dict, List, Optional
from datetime import datetime


DEFAULT_CSV_PATH = os.path.join(os.path.dirname(__file__), "outputs", "sentiment_reports.csv")


def _extract_percentage(label: str, text: str) -> float | None:
    # Robustly match e.g., "Positive: 45%", "Positive: [45]%", ignoring formatting
    pattern = re.compile(rf"(?i){label}\s*:\s*\[?\s*([0-9]+(?:\.[0-9]+)?)\s*\]?\s*%")
    m = pattern.search(text)
    if m:
        try:
            return float(m.group(1))
        except Exception:
            return None
    return None


def _extract_summary(text: str) -> str:
    # Find section "Summary of Employee Opinion" and capture until next heading (### or Key Positives)
    lower = text.lower()
    key = "summary of employee opinion"
    idx = lower.find(key)
    if idx == -1:
        return ""
    # Start after the heading line break
    # Find the end of the heading line (end of line after the key occurrence)
    line_end = text.find("\n", idx)
    start = line_end + 1 if line_end != -1 else idx + len(key)

    # Locate the next section heading
    next_h = len(text)
    for marker in ["\n###", "\n**key positives", "\nkey positives", "\n3."]:
        pos = lower.find(marker.strip().lower(), start)
        if pos != -1:
            next_h = min(next_h, pos)

    raw = text[start:next_h].strip()
    # Collapse whitespace to a single line
    return re.sub(r"\s+", " ", raw)


def _extract_attrition_factors(text: str, k: int = 3) -> List[str]:
    # Capture occurrences of "Attrition Factor: <name>"
    pattern = re.compile(r"(?i)Attrition\s*Factor\s*:\s*\**\*?\s*\[?([^\]\n\r]+)\]?\**\*?")
    factors = [m.group(1).strip() for m in pattern.finditer(text)]
    # Clean up any trailing punctuation or bold markers
    cleaned = [re.sub(r"\*+$", "", f).strip().rstrip('.') for f in factors]
    # Normalize spaces
    cleaned = [re.sub(r"\s+", " ", f) for f in cleaned]
    # Pad/truncate to k
    while len(cleaned) < k:
        cleaned.append("")
    return cleaned[:k]


def _generate_session_id() -> str:
    """Generate a unique session ID with timestamp format: YYYYMMDD_HHMMSS_mmm"""
    now = datetime.now()
    return now.strftime("%Y%m%d_%H%M%S_") + f"{now.microsecond // 1000:03d}"


def _get_timestamp() -> str:
    """Get ISO format timestamp"""
    return datetime.now().isoformat()


def parse_analysis_to_row(
    text: str, 
    survey_responses: Optional[List[str]] = None, 
    session_id: Optional[str] = None
) -> Dict[str, str | float]:
    pos = _extract_percentage("Positive", text) or 0.0
    neg = _extract_percentage("Negative", text) or 0.0
    neu = _extract_percentage("Neutral", text) or 0.0
    
    # Generate session ID and timestamp if not provided
    if session_id is None:
        session_id = _generate_session_id()
    
    # Handle survey responses - pad or truncate to 5 questions
    responses = [""] * 5  # Default empty responses
    if survey_responses:
        for i, response in enumerate(survey_responses[:5]):
            responses[i] = str(response).strip()
    
    # Count non-empty responses
    response_count = sum(1 for r in responses if r.strip())
    
    return {
        "session_id": session_id,
        "timestamp": _get_timestamp(),
        "positive_percentage": int(pos),
        "negative_percentage": int(neg), 
        "neutral_percentage": int(neu),
        "survey_responses_count": response_count,
        "response_q1": responses[0],
        "response_q2": responses[1],
        "response_q3": responses[2],
        "response_q4": responses[3],
        "response_q5": responses[4],
        "full_analysis": text.strip().replace('\n', ' ').replace('"', '""'),
    }


HEADERS = [
    "session_id",
    "timestamp", 
    "positive_percentage",
    "negative_percentage",
    "neutral_percentage",
    "survey_responses_count",
    "response_q1",
    "response_q2", 
    "response_q3",
    "response_q4",
    "response_q5",
    "full_analysis",
]


def append_row_to_csv(row: Dict[str, str | float], csv_path: str = DEFAULT_CSV_PATH) -> str:
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    write_header = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=HEADERS, quoting=csv.QUOTE_ALL)
        if write_header:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in HEADERS})
    return csv_path
