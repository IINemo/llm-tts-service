"""Tests for debugger routes (service_app/api/routes/debugger.py)."""

from unittest.mock import patch

import pytest


class TestDebuggerPage:
    def test_debugger_returns_200_or_404(self, test_client):
        resp = test_client.get("/debugger")
        assert resp.status_code in (200, 404)

    def test_debugger_trailing_slash(self, test_client):
        resp = test_client.get("/debugger/")
        assert resp.status_code in (200, 404)


class TestScenarios:
    def test_list_scenarios_200(self, test_client):
        with patch(
            "service_app.api.routes.debugger.list_demo_scenarios",
            return_value=[
                {
                    "id": "s1",
                    "title": "Test",
                    "description": "desc",
                    "available_budgets": [4, 8],
                    "default_budget": 8,
                }
            ],
        ):
            resp = test_client.get("/v1/debugger/demo/scenarios")
        assert resp.status_code == 200
        data = resp.json()
        assert "scenarios" in data
        assert len(data["scenarios"]) == 1

    def test_scenario_has_required_fields(self, test_client):
        scenario = {
            "id": "s1",
            "title": "Math",
            "description": "Basic math",
            "available_budgets": [4, 8],
            "default_budget": 8,
        }
        with patch(
            "service_app.api.routes.debugger.list_demo_scenarios",
            return_value=[scenario],
        ):
            data = test_client.get("/v1/debugger/demo/scenarios").json()
        s = data["scenarios"][0]
        assert s["id"] == "s1"
        assert s["title"] == "Math"
        assert "available_budgets" in s
        assert "default_budget" in s

    def test_list_scenarios_file_not_found(self, test_client):
        with patch(
            "service_app.api.routes.debugger.list_demo_scenarios",
            side_effect=FileNotFoundError("no file"),
        ):
            resp = test_client.get("/v1/debugger/demo/scenarios")
        assert resp.status_code == 404

    def test_list_scenarios_value_error(self, test_client):
        with patch(
            "service_app.api.routes.debugger.list_demo_scenarios",
            side_effect=ValueError("bad data"),
        ):
            resp = test_client.get("/v1/debugger/demo/scenarios")
        assert resp.status_code == 400

    def test_get_scenario_200(self, test_client):
        payload = {"id": "s1", "selected_budget": 8, "runs": []}
        with patch(
            "service_app.api.routes.debugger.get_demo_scenario",
            return_value=payload,
        ):
            resp = test_client.get("/v1/debugger/demo/scenarios/s1?budget=8")
        assert resp.status_code == 200
        assert resp.json()["selected_budget"] == 8

    def test_get_scenario_not_found(self, test_client):
        with patch(
            "service_app.api.routes.debugger.get_demo_scenario",
            side_effect=KeyError("no such scenario"),
        ):
            resp = test_client.get("/v1/debugger/demo/scenarios/missing")
        assert resp.status_code == 404


class TestHealth:
    def test_debugger_health_200(self, test_client):
        with patch(
            "service_app.api.routes.debugger.get_debugger_runtime_health",
            return_value={
                "status": "ok",
                "can_run": True,
                "checks": {"model": True},
            },
        ):
            resp = test_client.get("/v1/debugger/demo/health")
        assert resp.status_code == 200
        data = resp.json()
        assert "status" in data
        assert "can_run" in data
        assert "checks" in data


class TestValidateModel:
    def test_valid_request(self, test_client):
        with patch(
            "service_app.api.routes.debugger.validate_model_capabilities",
            return_value={"strategies": ["self_consistency"], "scorers": ["entropy"]},
        ):
            resp = test_client.post(
                "/v1/debugger/demo/validate-model",
                json={
                    "provider": "openrouter",
                    "model_id": "openai/gpt-4o-mini",
                    "api_key": "sk-test",
                },
            )
        assert resp.status_code == 200

    def test_missing_fields_422(self, test_client):
        resp = test_client.post(
            "/v1/debugger/demo/validate-model",
            json={"provider": "openrouter"},
        )
        assert resp.status_code == 422

    def test_empty_model_id_422(self, test_client):
        resp = test_client.post(
            "/v1/debugger/demo/validate-model",
            json={"model_id": "", "api_key": "sk-test"},
        )
        assert resp.status_code == 422

    def test_empty_api_key_422(self, test_client):
        resp = test_client.post(
            "/v1/debugger/demo/validate-model",
            json={"model_id": "openai/gpt-4o-mini", "api_key": ""},
        )
        assert resp.status_code == 422

    def test_value_error_400(self, test_client):
        with patch(
            "service_app.api.routes.debugger.validate_model_capabilities",
            side_effect=ValueError("bad model"),
        ):
            resp = test_client.post(
                "/v1/debugger/demo/validate-model",
                json={"model_id": "x", "api_key": "k"},
            )
        assert resp.status_code == 400

    def test_runtime_error_502(self, test_client):
        with patch(
            "service_app.api.routes.debugger.validate_model_capabilities",
            side_effect=RuntimeError("upstream failure"),
        ):
            resp = test_client.post(
                "/v1/debugger/demo/validate-model",
                json={"model_id": "x", "api_key": "k"},
            )
        assert resp.status_code == 502


class TestAdvancedConfig:
    def test_returns_config_and_yaml(self, test_client):
        template = {"config": {"key": "val"}, "config_yaml": "key: val\n"}
        with patch(
            "service_app.api.routes.debugger.get_advanced_config_template",
            return_value=template,
        ):
            resp = test_client.get(
                "/v1/debugger/demo/advanced-config/template?strategy_id=beam_search"
            )
        assert resp.status_code == 200
        data = resp.json()
        assert "config" in data
        assert "config_yaml" in data

    def test_with_scorer_id(self, test_client):
        template = {"config": {}, "config_yaml": "", "scorer": {"id": "prm"}}
        with patch(
            "service_app.api.routes.debugger.get_advanced_config_template",
            return_value=template,
        ):
            resp = test_client.get(
                "/v1/debugger/demo/advanced-config/template"
                "?strategy_id=beam_search&scorer_id=prm"
            )
        assert resp.status_code == 200
        assert "scorer" in resp.json()

    def test_invalid_strategy_400(self, test_client):
        with patch(
            "service_app.api.routes.debugger.get_advanced_config_template",
            side_effect=ValueError("unknown strategy"),
        ):
            resp = test_client.get(
                "/v1/debugger/demo/advanced-config/template?strategy_id=bad"
            )
        assert resp.status_code == 400

    def test_missing_strategy_422(self, test_client):
        resp = test_client.get("/v1/debugger/demo/advanced-config/template")
        assert resp.status_code == 422
