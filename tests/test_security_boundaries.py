"""Exercise HTTP rendering and filesystem boundaries without camera/GPU startup."""

import ast
import asyncio
import json
import os
import re
import subprocess
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace

import msgpack
import pytest
from fastapi.responses import HTMLResponse, JSONResponse

ROOT = Path(__file__).resolve().parents[1]


def load_definition(filename, name, namespace):
    tree = ast.parse((ROOT / filename).read_text(encoding="utf-8-sig"))
    node = next(node for node in tree.body if getattr(node, "name", None) == name)
    node.decorator_list = []
    module = ast.Module(body=[node], type_ignores=[])
    exec(compile(module, str(ROOT / filename), "exec"), namespace)  # noqa: S102 - trusted source under test
    return namespace[name]


def path_guard():
    # Before the fix, reproductions exercise the original unguarded functions.
    if not (ROOT / "safe_paths.py").exists():
        return {}
    from safe_paths import safe_child

    return {"safe_child": safe_child}


def test_webcam_never_discloses_configured_bearer():
    secret = "unit-test-only-bootstrap-credential"
    route = load_definition(
        "app/api/routes.py",
        "webcam_page",
        {
            "Request": object,
            "os": os,
            "json": json,
            "UI_HTML": "<html><head></head><body>camera</body></html>",
            "HTMLResponse": HTMLResponse,
            "_auth": SimpleNamespace(
                enabled=lambda: True, bootstrap_token_for_same_origin_ui=lambda: secret
            ),
        },
    )
    response = asyncio.run(route(SimpleNamespace(url=SimpleNamespace(query=""))))
    assert secret.encode() not in response.body
    assert b"camera" in response.body


@pytest.mark.parametrize(
    "session", ["..", ".", "C:escape", "../private", "..\\private"]
)
def test_camera_rejects_escaping_session_before_writing(tmp_path, monkeypatch, session):
    monkeypatch.chdir(tmp_path)
    route = load_definition(
        "app/api/routes.py",
        "calibration_record_multi_cam_frame",
        {
            "Request": object,
            "Path": Path,
            "JSONResponse": JSONResponse,
            **path_guard(),
            "dual_camera": SimpleNamespace(
                get_latest_jpeg=lambda: (None, 0), get_status=dict
            ),
        },
    )

    async def payload():
        return {"session_label": session, "cameras": []}

    response = asyncio.run(route(SimpleNamespace(json=payload)))
    assert isinstance(response, JSONResponse) and response.status_code == 400
    assert not list(tmp_path.rglob("manifest.json"))


def test_camera_preserves_valid_session_and_contains_role(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    route = load_definition(
        "app/api/routes.py",
        "calibration_record_multi_cam_frame",
        {
            "Request": object,
            "Path": Path,
            "JSONResponse": JSONResponse,
            **path_guard(),
            "dual_camera": SimpleNamespace(
                get_latest_jpeg=lambda: (None, 0), get_status=dict
            ),
        },
    )

    async def payload():
        return {
            "session_label": "operator glasses_2026-05-12",
            "cameras": [
                {"name": "Camera", "role": "browser_main", "jpeg_b64": "aW1hZ2U="}
            ],
        }

    response = asyncio.run(route(SimpleNamespace(json=payload)))
    assert response["ok"] and len(response["saved"]) == 1
    folder = tmp_path / "calibration/recorded_pairs/operator glasses_2026-05-12"
    assert len(list(folder.glob("*.jpg"))) == 1
    assert (folder / "manifest.json").exists()


def test_forensic_player_cannot_load_outside_session_root(tmp_path):
    sessions = tmp_path / "sessions"
    sessions.mkdir()
    private = tmp_path / "private"
    private.mkdir()
    (private / "session_data.json").write_text(
        '{"subject": "private", "affect_log": []}'
    )
    valid = sessions / "session_20260512_120000"
    valid.mkdir()
    (valid / "session_data.json").write_text('{"subject": "valid", "affect_log": []}')
    player = load_definition(
        "forensic.py",
        "SessionPlayer",
        {
            "SESSIONS_DIR": str(sessions),
            "os": os,
            "json": json,
            **path_guard(),
        },
    )()
    assert "error" in player.load("../private")
    assert not player.loaded
    assert player.load(valid.name)["subject"] == "valid"


def test_replay_rejects_windows_traversal_and_preserves_valid_stream(tmp_path):
    root = tmp_path / "replay"
    root.mkdir()
    blob = msgpack.packb({"private": True})
    data = len(blob).to_bytes(4, "big") + blob
    (tmp_path / "private.mpk").write_bytes(data)
    (root / "valid.mpk").write_bytes(data)
    stream = load_definition(
        "session_replay.py",
        "stream",
        {
            "Iterator": Iterator,
            "_ROOT": root,
            "_HAVE_MSGPACK": True,
            "msgpack": msgpack,
            "log": SimpleNamespace(warning=lambda *args: None),
            "session_path": lambda name: __import__("safe_paths").safe_child(
                root, name, suffix=".mpk"
            ),
            **path_guard(),
        },
    )
    with pytest.raises(ValueError):
        list(stream("..\\private"))
    assert list(stream("valid")) == [{"private": True}]


@pytest.mark.parametrize(
    "name", ["..", ".", "C:escape", "a/b", "a\\b", "CON", "NUL.txt", "name."]
)
def test_session_names_reject_windows_aliases(tmp_path, name):
    from safe_paths import safe_child

    with pytest.raises(ValueError):
        safe_child(tmp_path, name)


def test_qlora_rejects_escaping_session_and_preserves_consent(tmp_path):
    cfg = SimpleNamespace(SESSIONS_DIR=str(tmp_path / "sessions"))
    export = load_definition(
        "forensic_export.py",
        "export_qlora_jsonl",
        {
            "Path": Path,
            "cfg": cfg,
            **path_guard(),
        },
    )
    with pytest.raises(ValueError, match="consent"):
        export("valid", consent=False)
    with pytest.raises(ValueError, match="filename"):
        export("../private", consent=True)
    assert not list(tmp_path.iterdir())


def test_notifications_require_private_opt_in(monkeypatch):
    from pad import notify

    calls = []
    monkeypatch.delenv("DODO_NTFY_TOPIC", raising=False)
    monkeypatch.delenv("DODO_NTFY_TOKEN", raising=False)
    monkeypatch.setattr(
        notify.urllib.request,
        "urlopen",
        lambda request, **kwargs: (
            calls.append(request) or SimpleNamespace(read=lambda: b"")
        ),
    )
    assert notify.ntfy("test", "private") is False
    monkeypatch.setenv("DODO_NTFY_TOKEN", "test-notification-credential")
    monkeypatch.setenv("DODO_NTFY_TOPIC", "short-public-topic")
    assert notify.ntfy("test", "private") is False
    assert not calls
    topic = "unit-test-topic-" + "x" * 32
    monkeypatch.setenv("DODO_NTFY_TOPIC", topic)
    assert notify.ntfy("test", "private") is True
    assert len(calls) == 1 and calls[0].full_url == "https://ntfy.sh/" + topic
    assert calls[0].get_header("Authorization") == "Bearer test-notification-credential"


@pytest.mark.parametrize(
    "headers",
    [
        {"host": "rebound.example:8000"},
        {"host": "127.0.0.1:8000", "origin": "https://foreign.example"},
        {"host": "localhost:8000", "origin": "null"},
        {"host": "127.0.0.1:8000", "sec-fetch-site": "cross-site"},
    ],
)
def test_http_blocks_foreign_browsers_even_without_bearer(headers):
    from auth import browser_request_allowed

    middleware = load_definition(
        "main.py",
        "_dodo_http_bearer_middleware",
        {
            "Request": object,
            "_auth": None,
            "JSONResponse": JSONResponse,
            "browser_request_allowed": browser_request_allowed,
        },
    )
    calls = []

    async def next_handler(request):
        calls.append(request)
        return JSONResponse({"ok": True})

    request = SimpleNamespace(
        headers=headers, url=SimpleNamespace(scheme="http", path="/faces/clear")
    )
    response = asyncio.run(middleware(request, next_handler))
    assert response.status_code == 403 and not calls


@pytest.mark.parametrize(
    "scheme,host,origin",
    [
        ("http", "127.0.0.1:8000", None),
        ("https", "localhost:8000", "https://localhost:8000"),
        ("ws", "127.0.0.1:8000", "http://127.0.0.1:8000"),
        ("wss", "127.0.0.1:8000", "https://127.0.0.1:8000"),
    ],
)
def test_origin_guard_preserves_local_native_and_browser_clients(scheme, host, origin):
    from auth import browser_request_allowed

    headers = {"host": host}
    if origin is not None:
        headers["origin"] = origin
    assert browser_request_allowed(
        SimpleNamespace(headers=headers, url=SimpleNamespace(scheme=scheme))
    )


def test_qlora_valid_export_keeps_minimum_pairs_and_eeg_scrubbing(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    import forensic_export

    calibration = tmp_path / "calibration"
    calibration.mkdir(exist_ok=True)
    monkeypatch.setattr(forensic_export.cfg, "CALIBRATION_DIR", str(calibration))
    monkeypatch.setattr(forensic_export.cfg, "SESSIONS_DIR", str(tmp_path / "sessions"))
    pair = {
        "session_id": "valid",
        "affect_vector": {"hr": 72, "eeg_alpha": 1, "p300_peak": 2},
        "response": "test response",
    }
    (calibration / "llm_training_pairs.jsonl").write_text(json.dumps(pair) + "\n")
    with pytest.raises(ValueError, match="Insufficient"):
        forensic_export.export_qlora_jsonl("valid", consent=True, min_pairs=2)
    path = forensic_export.export_qlora_jsonl("valid", consent=True, min_pairs=1)
    assert path.parent == (tmp_path / "sessions/valid")
    assert json.loads(path.read_text())["affect_vector"] == {"hr": 72}


def test_browser_credential_stays_on_same_origin():
    route = load_definition(
        "app/api/routes.py",
        "webcam_page",
        {
            "Request": object,
            "os": os,
            "json": json,
            "UI_HTML": "<head></head>",
            "HTMLResponse": HTMLResponse,
            "_auth": SimpleNamespace(enabled=lambda: True),
        },
    )
    body = asyncio.run(
        route(SimpleNamespace(url=SimpleNamespace(query="")))
    ).body.decode()
    script = next(
        script
        for script in re.findall(r"<script>(.*?)</script>", body)
        if "__DODO_AUTH_TOKEN__" in script
    )
    harness = """
const fs = require('fs'), vm = require('vm'), assert = require('assert');
const calls = [], credential = 'entered-by-operator';
const window = {prompt: () => credential, fetch: (input, init) => calls.push([input, init])};
const context = {window, URL, Request, Headers, location: new URL('https://127.0.0.1:8000/webcam')};
vm.runInNewContext(fs.readFileSync(0, 'utf8'), context);
window.fetch('/faces');
assert.equal(calls[0][1].headers.get('Authorization'), 'Bearer ' + credential);
window.fetch('https://external.example/data');
assert.equal(calls[1][1].headers, undefined);
window.fetch('/faces', {headers: {Authorization: 'Bearer caller-provided'}});
assert.equal(calls[2][1].headers.get('Authorization'), 'Bearer caller-provided');
"""
    result = subprocess.run(
        ["node", "-e", harness],
        check=False,
        input=script,
        text=True,
        capture_output=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr


def test_replay_route_rejects_before_starting_stream(tmp_path, monkeypatch):
    import session_replay

    monkeypatch.setattr(session_replay, "_ROOT", tmp_path)
    route = load_definition(
        "app/api/routes.py",
        "replay_stream",
        {
            "_replay": session_replay,
            "JSONResponse": JSONResponse,
        },
    )
    response = asyncio.run(route("..\\private"))
    assert response.status_code == 400
    assert json.loads(response.body)["detail"]
    assert asyncio.run(route("valid")).status_code == 200


def test_bundle_sidecar_cannot_follow_outside_symlink(tmp_path):
    from typing import Any

    from safe_paths import safe_child

    exports = tmp_path / "exports"
    exports.mkdir()
    target = tmp_path / "private.json"
    target.write_text("sentinel")
    sidecar = exports / "forensic_all_fixed.json.manifest.json"
    try:
        sidecar.symlink_to(target)
    except OSError as error:
        pytest.skip(f"Windows symlink creation unavailable: {error}")
    write = load_definition(
        "forensic_export.py",
        "write_bundle",
        {
            "Optional": __import__("typing").Optional,
            "Dict": dict,
            "Any": Any,
            "EXPORTS_DIR": str(exports),
            "os": os,
            "json": json,
            "safe_child": safe_child,
            "build_bundle": lambda **kwargs: {"generated_iso": "test", "subject": {}},
            "datetime": SimpleNamespace(
                now=lambda: SimpleNamespace(strftime=lambda fmt: "fixed")
            ),
        },
    )
    with pytest.raises(ValueError, match="storage directory"):
        write()
    assert target.read_text() == "sentinel"


def test_overnight_sender_uses_private_notification_guard(monkeypatch):
    from pad import notify

    calls = []
    monkeypatch.setenv("NTFY_TOPIC", "old-public-topic")
    monkeypatch.delenv("DODO_NTFY_TOPIC", raising=False)
    monkeypatch.delenv("DODO_NTFY_TOKEN", raising=False)
    monkeypatch.setattr(
        notify.urllib.request, "urlopen", lambda *args, **kwargs: calls.append(args)
    )
    send = load_definition(
        "run_overnight.py", "ntfy", {"_ntfy": notify.ntfy, "log": lambda msg: None}
    )
    send("private research stage")
    assert not calls


@pytest.mark.parametrize(
    "headers",
    [
        {"host": "rebound.example:8000"},
        {"host": "127.0.0.1:8000", "origin": "https://foreign.example"},
        {"host": "127.0.0.1:8000", "origin": "null"},
    ],
)
def test_websocket_denies_foreign_browser_before_inference(headers):
    from fastapi import FastAPI, WebSocket
    from fastapi.testclient import TestClient
    from starlette.websockets import WebSocketDisconnect

    from auth import browser_request_allowed

    endpoint = load_definition(
        "app/api/ws.py",
        "ws_detect",
        {
            "WebSocket": WebSocket,
            "browser_request_allowed": browser_request_allowed,
        },
    )
    app = FastAPI()
    app.websocket("/ws/detect")(endpoint)
    with (
        TestClient(app, base_url="http://127.0.0.1:8000") as client,
        pytest.raises(WebSocketDisconnect) as denied,
        client.websocket_connect("/ws/detect", headers=headers),
    ):
        pytest.fail("foreign browser was accepted")
    assert denied.value.code == 1008
