"""Tests for /v1/models endpoints (service_app/api/routes/models.py)."""


class TestListModels:
    def test_returns_200(self, test_client):
        resp = test_client.get("/v1/models")
        assert resp.status_code == 200

    def test_response_object_is_list(self, test_client):
        data = test_client.get("/v1/models").json()
        assert data["object"] == "list"

    def test_has_three_models(self, test_client):
        data = test_client.get("/v1/models").json()
        assert len(data["data"]) == 3

    def test_each_model_has_required_fields(self, test_client):
        data = test_client.get("/v1/models").json()
        for model in data["data"]:
            assert "id" in model
            assert "object" in model
            assert model["object"] == "model"
            assert "created" in model
            assert "owned_by" in model

    def test_model_ids(self, test_client):
        data = test_client.get("/v1/models").json()
        ids = {m["id"] for m in data["data"]}
        assert "openai/gpt-4o-mini" in ids
        assert "openai/gpt-4o" in ids
        assert "openai/gpt-3.5-turbo" in ids


class TestGetModel:
    def test_known_model(self, test_client):
        # Route is /v1/models/{model_id} — slashes in model_id split the path,
        # so we test with a slash-free id and one containing "openai/" via path.
        resp = test_client.get("/v1/models/gpt-4o-mini")
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == "gpt-4o-mini"

    def test_permissive_any_model(self, test_client):
        resp = test_client.get("/v1/models/anything")
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == "anything"

    def test_owned_by_openai_prefix(self, test_client):
        # The route param {model_id} can't capture slashes, so a model_id like
        # "openai/gpt-4o" won't reach the handler as a single param. Only single-
        # segment model IDs are testable here. The owned_by logic checks for
        # "openai/" substring — without a slash this will be "unknown".
        # We verify the logic works via the /v1/models list endpoint instead,
        # where the full model IDs are returned with correct owned_by.
        data = test_client.get("/v1/models").json()
        openai_models = [m for m in data["data"] if "openai/" in m["id"]]
        assert all(m["owned_by"] == "openai" for m in openai_models)

    def test_owned_by_unknown(self, test_client):
        data = test_client.get("/v1/models/custom-model").json()
        assert data["owned_by"] == "unknown"
