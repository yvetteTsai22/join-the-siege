from io import BytesIO

import pytest
from src.app import app, allowed_file
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    client = TestClient(app)
    yield client


@pytest.mark.parametrize(
    "filename, expected",
    [
        ("file.pdf", True),
        ("file.png", True),
        ("file.jpg", True),
        ("file.txt", False),
        ("file", False),
    ],
)
def test_allowed_file(filename, expected):
    assert allowed_file(filename) == expected


def test_no_selected_file(client):
    response = client.post("/classify_file")
    assert response.status_code == 400


def test_success(client, mocker):
    mocker.patch("src.app.classifier.predict", return_value="invoice")

    files = {"file": ("test.pdf", BytesIO(b"test file content"))}
    response = client.post("/classify_file", files=files)
    assert response.status_code == 200
    assert response.json() == {"file_class": "test_class"}
