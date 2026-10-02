# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import json
import socket
import urllib.request

import pytest
from fastapi import FastAPI

from verl.workers.rollout.utils import run_uvicorn


def _app() -> FastAPI:
    app = FastAPI()

    @app.get("/health")
    async def health():
        return {"ok": True}

    return app


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


async def _serve_and_get(port: int | None) -> tuple[int, dict]:
    kwargs = {} if port is None else {"port": port}
    bound, task = await run_uvicorn(_app(), server_args=None, server_address="127.0.0.1", **kwargs)
    try:
        body = await asyncio.to_thread(
            lambda: urllib.request.urlopen(f"http://127.0.0.1:{bound}/health", timeout=5).read()
        )
        return bound, json.loads(body)
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


@pytest.mark.asyncio
async def test_run_uvicorn_default_picks_a_free_port():
    bound, body = await _serve_and_get(None)
    assert bound > 0
    assert body == {"ok": True}


@pytest.mark.asyncio
async def test_run_uvicorn_binds_the_requested_port():
    port = _free_port()
    bound, body = await _serve_and_get(port)
    assert bound == port
    assert body == {"ok": True}
