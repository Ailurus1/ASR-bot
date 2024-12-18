import pytest
import pytest_asyncio
from httpx import AsyncClient
import os
from pathlib import Path
import asyncio
from inference_server.__main__ import app, batched_server
from inference_server.model import ASRModel
from inference_server.profiles import PROFILES

os.environ["BATCH_SIZE"] = "4"


@pytest_asyncio.fixture(scope="session")
def event_loop():
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture
async def test_client(event_loop):
    config = PROFILES["classical-tiny"]
    app.state.asr_model = ASRModel(config)

    queue_task = event_loop.create_task(batched_server.queue_processing())

    try:
        async with AsyncClient(app=app, base_url="http://test") as client:
            yield client
    finally:
        queue_task.cancel()
        try:
            await queue_task
        except asyncio.CancelledError:
            pass


@pytest.fixture
def test_audio_path():
    test_dir = Path(__file__).parent / "data"
    return test_dir / "test_audio.wav"


@pytest.mark.asyncio
async def test_server_up(test_client):
    response = await test_client.get("/")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_single_request(test_client, test_audio_path):
    with open(test_audio_path, "rb") as audio_file:
        files = {"audio_message": ("test_audio.wav", audio_file, "audio/wav")}
        response = await test_client.post("/asr/", files=files)

    assert response.status_code == 200
    assert "transcription" in response.json()
    assert isinstance(response.json()["transcription"], str)
    assert len(response.json()["transcription"]) > 0


@pytest.mark.asyncio
@pytest.mark.parametrize("num_requests", [2, 4])
async def test_concurrent_requests(test_client, test_audio_path, num_requests):
    with open(test_audio_path, "rb") as audio_file:
        file_content = audio_file.read()

    async def single_request():
        files = {"audio_message": ("test_audio.wav", file_content, "audio/wav")}
        return await test_client.post("/asr/", files=files)

    tasks = [single_request() for _ in range(num_requests)]
    responses = await asyncio.gather(*tasks)

    for response in responses:
        assert response.status_code == 200
        assert "transcription" in response.json()
