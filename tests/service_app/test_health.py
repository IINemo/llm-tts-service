"""Tests for health and root endpoints (service_app/main.py)."""


class TestHealthEndpoint:
    def test_health_returns_200(self, test_client):
        resp = test_client.get("/health")
        assert resp.status_code == 200

    def test_health_has_status_healthy(self, test_client):
        data = test_client.get("/health").json()
        assert data["status"] == "healthy"

    def test_health_has_version(self, test_client):
        data = test_client.get("/health").json()
        assert "version" in data
        assert isinstance(data["version"], str)

    def test_v1_health_returns_same(self, test_client):
        resp = test_client.get("/v1/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert "version" in data


class TestRootEndpoint:
    def test_root_returns_200(self, test_client):
        resp = test_client.get("/")
        assert resp.status_code == 200


class TestDeployEndpoint:
    def test_deploy_returns_200_or_404(self, test_client):
        resp = test_client.get("/deploy")
        assert resp.status_code in (200, 404)

    def test_deploy_trailing_slash(self, test_client):
        resp = test_client.get("/deploy/")
        assert resp.status_code in (200, 404)
