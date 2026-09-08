import json
from unittest.mock import Mock

import pytest

import embedding_url
import sdg_constants


def _groq_response(scores):
    response = Mock()
    response.json.return_value = {
        "choices": [{"message": {"content": json.dumps({"scores": scores})}}]
    }
    response.raise_for_status.return_value = None
    return response


def test_classify_text_returns_all_sdg_scores_and_sends_json_request(monkeypatch):
    scores = {str(index): index / 17 for index in range(1, 18)}
    response = _groq_response(scores)
    post = Mock(return_value=response)
    monkeypatch.setattr(embedding_url.requests, "post", post)

    result = embedding_url.classify_text("A public health project", api_key="test-key")

    assert list(result) == sdg_constants.SDG_NAMES
    assert result[sdg_constants.SDG_NAMES[0]] == pytest.approx(1 / 17)
    payload = post.call_args.kwargs["json"]
    assert payload["response_format"] == {"type": "json_object"}
    assert payload["temperature"] == 0
    assert post.call_args.kwargs["headers"]["Authorization"] == "Bearer test-key"


def test_classify_text_clamps_scores_and_accepts_code_fence(monkeypatch):
    response = Mock()
    response.json.return_value = {
        "choices": [{
            "message": {
                "content": "```json\n{\"scores\": {\"SDG 3\": 2, \"SDG 14\": -1}}\n```"
            }
        }]
    }
    response.raise_for_status.return_value = None
    monkeypatch.setattr(embedding_url.requests, "post", Mock(return_value=response))

    result = embedding_url.classify_text("A health project", api_key="test-key")

    assert result[sdg_constants.SDG_NAMES[2]] == 1.0
    assert result[sdg_constants.SDG_NAMES[13]] == 0.0
    assert all(0.0 <= score <= 1.0 for score in result.values())


def test_classify_text_requires_api_key(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    with pytest.raises(embedding_url.GroqClassificationError, match="GROQ_API_KEY"):
        embedding_url.classify_text("A project")


def test_main_preserves_current_response_structure(monkeypatch):
    scores = {
        name: index / 17
        for index, name in enumerate(sdg_constants.SDG_NAMES, start=1)
    }
    monkeypatch.setattr(
        embedding_url,
        "classify_repo",
        lambda *args, **kwargs: {
            "repo": "owner/repo",
            "scores": scores,
            "predictions": [(sdg_constants.SDG_NAMES[2], 0.87654)],
            "top_all": [],
            "meta": {},
        },
    )
    monkeypatch.setattr(embedding_url, "classify_text", lambda text: scores)

    result = embedding_url.main("https://github.com/owner/repo", "health project")

    assert result["project_name"] == "owner/repo"
    assert len(result["sdg_predictions"]) == 17
    assert all(
        set(item) == {"sdg", "prediction", "confidence"}
        for item in result["sdg_predictions"]
    )
    assert result["sdg_predictions"][2]["prediction"] == pytest.approx(0.176, abs=0.001)
    assert result["sdg_predictions"][2]["confidence"] == result["sdg_predictions"][2]["prediction"]


def test_main_never_returns_negative_confidence(monkeypatch):
    scores = {name: -0.25 for name in sdg_constants.SDG_NAMES}
    monkeypatch.setattr(
        embedding_url,
        "classify_repo",
        lambda *args, **kwargs: {"repo": "owner/repo", "scores": scores},
    )

    result = embedding_url.main("https://github.com/owner/repo")

    assert all(item["prediction"] == 0.0 for item in result["sdg_predictions"])
    assert all(item["confidence"] == 0.0 for item in result["sdg_predictions"])


def test_fetch_repo_text_summarizes_repository_and_user_description(monkeypatch):
    class Provider:
        _owner = "owner"
        _repo = "repo"

        def fetch_meta(self):
            return {"name": "Chamilo", "description": "", "homepage": ""}

        def fetch_topics(self):
            return ["education"]

        def fetch_readme(self):
            return "The repository provides online courses."

    captured = {}
    monkeypatch.setattr(embedding_url, "get_provider", lambda url, token=None: Provider())
    monkeypatch.setattr(
        embedding_url,
        "summarize_for_sdg",
        lambda **kwargs: captured.update(kwargs) or "education summary",
    )

    result = embedding_url.fetch_repo_text(
        "https://github.com/owner/repo",
        project_description="The project helps students learn online.",
    )

    assert result["text"] == "education summary"
    assert captured["readme"] == "The repository provides online courses."
    assert captured["description"] == "The project helps students learn online."


def test_main_uses_aurora_when_groq_classification_fails(monkeypatch):
    monkeypatch.setattr(
        embedding_url,
        "fetch_repo_text",
        lambda *args, **kwargs: {
            "owner": "owner",
            "repo": "repo",
            "text": "education summary",
            "meta": {"description": ""},
        },
    )
    monkeypatch.setattr(
        embedding_url,
        "classify_text",
        lambda text: (_ for _ in ()).throw(
            embedding_url.GroqClassificationError("unavailable")
        ),
    )
    monkeypatch.setattr(
        embedding_url,
        "_aurora_scores",
        lambda *args: {name: 0.0 for name in sdg_constants.SDG_NAMES}
        | {sdg_constants.SDG_NAMES[3]: 0.8},
    )

    result = embedding_url.main("https://github.com/owner/repo", "education")

    assert result["method"] == "aurora-fallback"
    assert result["sdg_predictions"][3]["confidence"] == 0.8


def test_main_sends_the_same_summary_to_both_classifiers(monkeypatch):
    summary = "Students use this platform for online education."
    calls = []
    scores = {name: 0.0 for name in sdg_constants.SDG_NAMES}
    scores[sdg_constants.SDG_NAMES[3]] = 0.8

    monkeypatch.setattr(
        embedding_url,
        "fetch_repo_text",
        lambda *args, **kwargs: {
            "owner": "owner",
            "repo": "repo",
            "text": summary,
            "meta": {"description": ""},
        },
    )
    monkeypatch.setattr(
        embedding_url,
        "classify_text",
        lambda text: calls.append(("groq", text)) or scores,
    )
    monkeypatch.setattr(
        embedding_url,
        "_aurora_scores",
        lambda text, project_name, project_url: calls.append(("aurora", text)) or scores,
    )

    result = embedding_url.main("https://github.com/owner/repo", "education")

    assert calls == [("groq", summary), ("aurora", summary)]
    assert result["method"] == "groq"
    assert len(result["groq_predictions"]) == 17
    assert len(result["aurora_predictions"]) == 17
