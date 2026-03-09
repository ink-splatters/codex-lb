from __future__ import annotations

import pytest

from app.core.openai.model_registry import ReasoningLevel, UpstreamModel, get_model_registry

pytestmark = pytest.mark.integration


def _make_upstream_model(
    slug: str,
    *,
    supported_in_api: bool = True,
    prefer_websockets: bool = False,
) -> UpstreamModel:
    raw = {
        "slug": slug,
        "display_name": slug,
        "description": f"Test model {slug}",
        "default_reasoning_level": "medium",
        "supported_reasoning_levels": [{"effort": "medium", "description": "default"}],
        "shell_type": "shell_command",
        "visibility": "list",
        "supported_in_api": supported_in_api,
        "priority": 0,
        "upgrade": None,
        "base_instructions": f"Base instructions for {slug}",
        "supports_reasoning_summaries": True,
        "support_verbosity": False,
        "default_verbosity": None,
        "apply_patch_tool_type": None,
        "truncation_policy": {"mode": "bytes", "limit": 10000},
        "supports_parallel_tool_calls": True,
        "context_window": 272000,
        "experimental_supported_tools": ["view_image"],
        "input_modalities": ["text", "image"],
        "prefer_websockets": prefer_websockets,
        "minimal_client_version": [0, 111, 0],
    }
    return UpstreamModel(
        slug=slug,
        display_name=slug,
        description=f"Test model {slug}",
        context_window=272000,
        input_modalities=("text", "image"),
        supported_reasoning_levels=(ReasoningLevel(effort="medium", description="default"),),
        default_reasoning_level="medium",
        supports_reasoning_summaries=True,
        support_verbosity=False,
        default_verbosity=None,
        prefer_websockets=prefer_websockets,
        supports_parallel_tool_calls=True,
        supported_in_api=supported_in_api,
        minimal_client_version=[0, 111, 0],
        priority=0,
        available_in_plans=frozenset({"plus", "pro"}),
        raw=raw,
    )


def _populate_test_registry() -> None:
    registry = get_model_registry()
    models = [
        _make_upstream_model("gpt-5.2"),
        _make_upstream_model("gpt-5.3-codex"),
    ]
    registry.update({"plus": models, "pro": models})


@pytest.mark.asyncio
async def test_v1_models_list(async_client):
    _populate_test_registry()
    resp = await async_client.get("/v1/models")
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["object"] == "list"
    data = payload["data"]
    assert isinstance(data, list)
    ids = {item["id"] for item in data}
    assert "gpt-5.2" in ids
    assert "gpt-5.3-codex" in ids
    for item in data:
        assert item["object"] == "model"
        assert item["owned_by"] == "codex-lb"
        assert "metadata" in item


@pytest.mark.asyncio
async def test_v1_models_empty_when_registry_not_populated(async_client):
    registry = get_model_registry()
    registry._snapshot = None
    resp = await async_client.get("/v1/models")
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["object"] == "list"
    assert payload["data"] == []


@pytest.mark.asyncio
async def test_v1_models_includes_supported_in_api_false_models(async_client):
    registry = get_model_registry()
    models = [
        _make_upstream_model("gpt-5.2"),
        _make_upstream_model("gpt-5.3-codex"),
        _make_upstream_model("gpt-hidden", supported_in_api=False),
    ]
    registry.update({"plus": models, "pro": models})

    resp = await async_client.get("/v1/models")
    assert resp.status_code == 200
    ids = {item["id"] for item in resp.json()["data"]}
    assert {"gpt-5.2", "gpt-5.3-codex", "gpt-hidden"}.issubset(ids)


@pytest.mark.asyncio
async def test_backend_codex_models_include_supported_in_api_false_models(async_client):
    registry = get_model_registry()
    models = [
        _make_upstream_model("gpt-5.2"),
        _make_upstream_model("gpt-5.3-codex"),
        _make_upstream_model("gpt-hidden", supported_in_api=False),
    ]
    registry.update({"plus": models, "pro": models})

    resp = await async_client.get("/backend-api/codex/models")
    assert resp.status_code == 200
    ids = {item["slug"] for item in resp.json()["models"]}
    assert {"gpt-5.2", "gpt-5.3-codex", "gpt-hidden"}.issubset(ids)


@pytest.mark.asyncio
async def test_model_sets_are_consistent_across_api_endpoints(async_client):
    registry = get_model_registry()
    models = [
        _make_upstream_model("gpt-5.2"),
        _make_upstream_model("gpt-5.3-codex"),
        _make_upstream_model("gpt-hidden", supported_in_api=False),
    ]
    registry.update({"plus": models, "pro": models})

    dashboard = await async_client.get("/api/models")
    v1 = await async_client.get("/v1/models")
    codex = await async_client.get("/backend-api/codex/models")

    assert dashboard.status_code == 200
    assert v1.status_code == 200
    assert codex.status_code == 200

    dashboard_ids = {item["id"] for item in dashboard.json()["models"]}
    v1_ids = {item["id"] for item in v1.json()["data"]}
    codex_ids = {item["slug"] for item in codex.json()["models"]}
    assert dashboard_ids == v1_ids == codex_ids


@pytest.mark.asyncio
async def test_models_do_not_advertise_websocket_preference_downstream(async_client):
    registry = get_model_registry()
    models = [
        _make_upstream_model("gpt-5.4", prefer_websockets=True),
    ]
    registry.update({"plus": models, "pro": models})

    resp = await async_client.get("/backend-api/codex/models")
    assert resp.status_code == 200
    payload = resp.json()
    item = next(model for model in payload["models"] if model["slug"] == "gpt-5.4")
    assert item["prefer_websockets"] is False
    assert item["base_instructions"] == "Base instructions for gpt-5.4"
    assert item["truncation_policy"] == {"mode": "bytes", "limit": 10000}
    assert resp.headers["etag"].startswith('"codex-lb-')
