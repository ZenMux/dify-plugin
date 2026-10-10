import base64

import pytest
from dify_plugin.entities.model import ModelType
from dify_plugin.entities.model.text_embedding import MultiModalContent, MultiModalContentType
from dify_plugin.errors.model import InvokeRateLimitError

from models._common import image_data_uri


class FakeResp:
    def __init__(self, status, body):
        self.status_code, self._body, self.text = status, body, str(body)

    def json(self):
        return self._body


@pytest.fixture
def rerank(factory):
    return factory.get_instance(ModelType.RERANK)


def test_rerank_request_and_raw_scores(rerank, monkeypatch):
    seen = {}

    def fake_post(url, headers, json, timeout):
        seen.update(url=url, body=json)
        return FakeResp(200, {"results": [{"index": 2, "relevance_score": 0.3}, {"index": 0, "relevance_score": 0.9}]})

    monkeypatch.setattr("models.rerank.rerank.requests.post", fake_post)
    result = rerank._invoke("qwen/qwen3-rerank", {"api_key": "k", "region": "cn"}, "q", ["a", "b", "c"],
                            score_threshold=0.5, top_n=2)
    assert seen["url"] == "https://zenmux.dev/api/v1/rerank"
    assert seen["body"]["input"] == {"query": "q", "documents": ["a", "b", "c"]}
    assert seen["body"]["parameters"]["top_n"] == 2
    assert [(d.index, d.text, d.score) for d in result.docs] == [(0, "a", 0.9)]


PNG_B64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC"


def test_multimodal_rerank_sends_content_objects(rerank, monkeypatch):
    seen = {}

    def fake_post(url, headers, json, timeout):
        seen.update(body=json)
        return FakeResp(200, {"results": [{"index": 1, "relevance_score": 0.8}, {"index": 0, "relevance_score": 0.4}]})

    monkeypatch.setattr("models.rerank.rerank.requests.post", fake_post)
    query = MultiModalContent(content=PNG_B64, content_type=MultiModalContentType.IMAGE)
    docs = [MultiModalContent(content="a cat", content_type=MultiModalContentType.TEXT),
            MultiModalContent(content=PNG_B64, content_type=MultiModalContentType.IMAGE)]
    result = rerank.invoke_multimodal("qwen/qwen3-vl-rerank", {"api_key": "k"}, query, docs)
    assert seen["body"]["input"] == {
        "query": {"image": f"data:image/png;base64,{PNG_B64}"},
        "documents": [{"text": "a cat"}, {"image": f"data:image/png;base64,{PNG_B64}"}],
    }
    assert [(d.index, d.score) for d in result.docs] == [(1, 0.8), (0, 0.4)]


@pytest.mark.parametrize("raw, mime", [
    (b"\x89PNG\r\n\x1a\n" + b"\0" * 16, "image/png"),
    (b"\xff\xd8\xff\xe0" + b"\0" * 16, "image/jpeg"),
    (b"RIFF\0\0\0\0WEBPVP8 " + b"\0" * 8, "image/webp"),
    (b"GIF89a" + b"\0" * 16, "image/gif"),
])
def test_image_data_uri_detects_mime(raw, mime):
    b64 = base64.b64encode(raw).decode()
    assert image_data_uri(b64) == f"data:{mime};base64,{b64}"
    assert image_data_uri("https://x/y.png") == "https://x/y.png"


def test_rerank_http_errors_are_typed(rerank, monkeypatch):
    monkeypatch.setattr("models.rerank.rerank.requests.post", lambda *a, **k: FakeResp(429, {"error": "slow down"}))
    with pytest.raises(InvokeRateLimitError):
        rerank.invoke("qwen/qwen3-rerank", {"api_key": "k"}, "q", ["a"])
