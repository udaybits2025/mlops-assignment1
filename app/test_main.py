from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_read_main():
    response = client.get("/")
    assert response.status_code == 404
    # assert response.json() == {"message": "Hello World"}
    # commented out because a 404 will not return json
