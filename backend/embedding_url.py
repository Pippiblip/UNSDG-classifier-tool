"""Repository classification using Groq's JSON chat-completions API.

The public ``main`` function intentionally keeps the response contract used by
the Flask route. Repository fetching and summarisation stay local to the
backend; SDG scoring is delegated to Groq instead of loading ML models.
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Dict
from urllib.parse import urlparse

import requests

import sdg_constants
from services.repo_fetcher import get_provider
from services.summariser import summarize_for_sdg
from services.request_limiter import wait_for_request

try:
    from services.repo_fetcher import ProviderError
except Exception:  # pragma: no cover
    ProviderError = Exception


GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = os.getenv("GROQ_CLASSIFIER_MODEL", "openai/gpt-oss-20b")
DEFAULT_TIMEOUT = 60


class GroqClassificationError(RuntimeError):
    """Raised when Groq cannot return a valid classification response."""


class AuroraClassificationError(RuntimeError):
    """Raised when Aurora cannot return a usable classification response."""


def fetch_repo_text(url: str, project_description: str = "", max_issues: int = 10) -> Dict[str, Any]:
    host = urlparse(url).hostname or ""
    token = os.environ.get("GITHUB_TOKEN") if "github.com" in host else None
    provider = get_provider(url, token=token)
    meta: Dict[str, str] = {"name": "", "description": "", "homepage": ""}

    try:
        fetch_meta = getattr(provider, "fetch_meta", None)
        if callable(fetch_meta):
            candidate = fetch_meta()
            if isinstance(candidate, dict):
                meta = candidate
        else:
            meta = {
                "name": getattr(provider, "_name", ""),
                "description": getattr(provider, "_description", ""),
                "homepage": getattr(provider, "_homepage", ""),
            }
    except ProviderError:
        pass

    try:
        topics = provider.fetch_topics() or []
    except ProviderError:
        topics = []

    try:
        readme = provider.fetch_readme() or ""
    except ProviderError:
        readme = ""

    name = meta.get("name") or ""
    description = project_description.strip() or meta.get("description") or ""
    summary = summarize_for_sdg(
        readme=readme,
        name=name,
        description=description,
        topics=topics,
    )
    if not summary.strip() or "LLM summarization unavailable:" in summary:
        summary = "\n\n".join(
            part for part in (name, description, readme, "Topics: " + ", ".join(topics))
            if part.strip()
        )
    return {
        "owner": provider._owner,
        "repo": provider._repo,
        "text": summary,
        "meta": {
            "name": name,
            "description": description,
            "topics": topics,
            "homepage": meta.get("homepage") or "",
        },
    }


CLASSIFIER_SYSTEM_PROMPT = """You classify software projects against the 17 UN Sustainable Development Goals.
Use only concrete domain signals in the supplied project summary. Generic software
infrastructure with no named population, problem, or real-world domain should score low.
Return JSON only, with this exact shape:
{"scores": {"1": 0.0, "2": 0.0, ..., "17": 0.0}}
The scores object must contain every SDG number from 1 through 17 exactly once. Each score is a
number from 0.0 to 1.0: 0 means no evidence and 1 means strong, explicit evidence.
These are relevance scores, not claims that the project achieves an SDG."""


def _classifier_prompt(text: str) -> str:
    labels = "\n".join(
        f"{index + 1}. {name}: {description}"
        for index, (name, description) in enumerate(
            zip(sdg_constants.SDG_NAMES, sdg_constants.SDG_DESCS)
        )
    )
    return f"SDG definitions:\n{labels}\n\nProject summary:\n{text[:12000]}"


def _json_content(data: Any) -> Dict[str, Any]:
    choices = data.get("choices") if isinstance(data, dict) else None
    if not choices:
        raise GroqClassificationError("Groq response contained no choices")
    content = choices[0].get("message", {}).get("content", "")
    if not isinstance(content, str) or not content.strip():
        raise GroqClassificationError("Groq response contained no JSON content")
    content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content.strip(), flags=re.IGNORECASE)
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        raise GroqClassificationError("Groq returned invalid classification JSON") from exc
    if not isinstance(parsed, dict):
        raise GroqClassificationError("Groq classification JSON must be an object")
    return parsed


def _score(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def _normalise_scores(payload: Dict[str, Any]) -> Dict[str, float]:
    raw_scores = payload.get("scores", payload.get("sdg_predictions"))
    if not isinstance(raw_scores, dict):
        raise GroqClassificationError("Groq JSON did not contain a scores object")

    by_number = {}
    for key, value in raw_scores.items():
        match = re.search(r"(?:(?:SDG|Goal)\s*)?(\d+)", str(key), flags=re.IGNORECASE)
        if match:
            by_number[match.group(1)] = _score(value)

    return {
        name: by_number.get(str(index), _score(raw_scores.get(name, 0.0)))
        for index, name in enumerate(sdg_constants.SDG_NAMES, start=1)
    }


def classify_text(text: str, *, api_key: str | None = None, timeout: int = DEFAULT_TIMEOUT) -> Dict[str, float]:
    if not text.strip():
        raise ValueError("No text available for classification")
    key = api_key or os.getenv("GROQ_API_KEY")
    if not key:
        raise GroqClassificationError("GROQ_API_KEY is not configured")

    payload = {
        "model": GROQ_MODEL,
        "temperature": 0,
        "max_tokens": 768,
        "reasoning_effort": "low",
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": CLASSIFIER_SYSTEM_PROMPT},
            {"role": "user", "content": _classifier_prompt(text)},
        ],
    }
    response = None
    for attempt in range(3):
        wait_for_request()
        response = requests.post(
            GROQ_API_URL,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json=payload,
            timeout=timeout,
        )
        if response.status_code != 429 or attempt == 2:
            break
        retry_after = response.headers.get("Retry-After", "2")
        try:
            delay = min(10.0, max(1.0, float(retry_after)))
        except ValueError:
            delay = 2.0
        time.sleep(delay)
    try:
        data = response.json()
    except ValueError as exc:
        raise GroqClassificationError("Groq returned a non-JSON HTTP response") from exc
    if isinstance(data, dict) and data.get("error"):
        raise GroqClassificationError(f"Groq API error: {data['error']}")
    try:
        response.raise_for_status()
    except requests.exceptions.HTTPError as exc:
        raise GroqClassificationError(f"Groq API HTTP error: {exc}") from exc
    return _normalise_scores(_json_content(data))


def passes_threshold(name: str, score: float, threshold: float,
                     per_sdg_thresholds: Dict[str, float] | None = None) -> bool:
    if per_sdg_thresholds:
        number = sdg_constants.sdg_number_from_name(name)
        if number is not None and number in per_sdg_thresholds:
            return score >= per_sdg_thresholds[number]
    return score >= threshold


def classify_repo(url: str, threshold: float = 0.5, top_k: int = 10,
                  use_ensemble: bool = False, proj_desc: str = "",
                  per_sdg_thresholds: Dict[str, float] | None = None) -> Dict[str, Any]:
    data = fetch_repo_text(url, project_description=proj_desc)
    text = data["text"][:12000]
    scores = classify_text(text)
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    selected = [
        item for item in ranked
        if passes_threshold(item[0], item[1], threshold, per_sdg_thresholds)
    ]
    return {
        "repo": f"{data['owner']}/{data['repo']}",
        "scores": scores,
        "predictions": selected[:top_k],
        "top_all": ranked[:top_k],
        "meta": data["meta"],
    }


def _aurora_scores(text: str, project_name: str, project_url: str) -> Dict[str, float]:
    from aurora_api import main as aurora_classify

    result = aurora_classify(
        text=text,
        project_name=project_name,
        project_url=project_url,
    )
    if result.get("error"):
        raise AuroraClassificationError(result["error"])
    raw_predictions = result.get("sdg_predictions", {})
    if not isinstance(raw_predictions, dict):
        raise AuroraClassificationError("Aurora returned invalid predictions")
    scores = {name: 0.0 for name in sdg_constants.SDG_NAMES}
    for name, value in raw_predictions.items():
        number = re.search(r"(?:SDG\s*)?(\d+)", str(name), flags=re.IGNORECASE)
        if not number:
            continue
        index = int(number.group(1)) - 1
        if 0 <= index < len(sdg_constants.SDG_NAMES):
            scores[sdg_constants.SDG_NAMES[index]] = _score(value)
    return scores


def main(url: str, project_description: str = "") -> Dict[str, Any]:
    data = fetch_repo_text(url, project_description=project_description)

    try:
        groq_scores = classify_text(data["text"][:12000])
        groq_error = None
    except GroqClassificationError:
        groq_scores = None
        groq_error = "Groq classification failed"

    try:
        aurora_scores = _aurora_scores(
            data["text"][:12000],
            f"{data['owner']}/{data['repo']}",
            url,
        )
        aurora_error = None
    except AuroraClassificationError:
        aurora_scores = None
        aurora_error = "Aurora classification failed"

    if groq_scores is not None:
        scores = groq_scores
        method = "groq"
    elif aurora_scores is not None:
        scores = aurora_scores
        method = "aurora-fallback"
    else:
        raise GroqClassificationError(
            f"Both classification services failed: {groq_error}; {aurora_error}"
        )

    def score_records(score_map: Dict[str, float]) -> list[Dict[str, Any]]:
        return [
            {
                "sdg": name,
                "prediction": round(_score(score_map.get(name)), 3),
                "confidence": round(_score(score_map.get(name)), 3),
            }
            for name in sdg_constants.SDG_NAMES
        ]

    all_scores = score_records(scores)
    aurora_records = score_records(aurora_scores) if aurora_scores is not None else []
    groq_records = score_records(groq_scores) if groq_scores is not None else []
    return {
        "project_name": f"{data['owner']}/{data['repo']}",
        "project_url": url,
        "sdg_predictions": all_scores,
        "groq_predictions": groq_records,
        "aurora_predictions": aurora_records,
        "method": method,
        "matched_predictions": [
            item for item in all_scores
            if passes_threshold(
                item["sdg"], item["prediction"], 0.5,
                sdg_constants.PER_SDG_THRESHOLDS,
            )
        ],
    }
