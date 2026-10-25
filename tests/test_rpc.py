"""RPC serialization over temporary local sockets; never contact device workers."""
import json
import math
import pickle
import socketserver
import threading

import pytest

from astra_vla import rpc


@pytest.fixture
def local_server(tmp_path):
    servers = []

    def start(dispatch=None, packet=None):
        path = tmp_path / (str(len(servers)) + ".sock")
        if packet is None:
            handler = rpc.request_handler(dispatch)
        else:
            class PacketHandler(socketserver.BaseRequestHandler):
                def handle(self):
                    rpc.receive(self.request)
                    rpc.send(self.request, packet)
            handler = PacketHandler
        server = socketserver.ThreadingUnixStreamServer(str(path), handler)
        server.daemon_threads = True
        thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": .01}, daemon=True)
        servers.append((server, thread))
        thread.start()
        return path

    yield start
    for server, thread in servers:
        server.shutdown()
        thread.join(timeout=1)
        server.server_close()
        assert not thread.is_alive()


class WorkerError(ValueError):
    def __init__(self, message, diagnostics):
        super().__init__(message)
        self.stage = "ik_continuity"
        self.diagnostics = diagnostics


def test_structured_error_reaches_client_without_retry_or_text_change(local_server):
    requests = []
    diagnostics = {"max_joint_change_deg": 26.52, "local_attempt": {"position_residual_m": .003}}

    def dispatch(method, args):
        requests.append((method, args))
        raise WorkerError("discontinuous IK for MOVE P endpoint", diagnostics)

    path = local_server(dispatch)
    with pytest.raises(rpc.RPCError) as failure:
        rpc.call(path, "mock_action", {"target": "offline"}, timeout=1)
    error = failure.value
    assert isinstance(error, RuntimeError)
    assert str(error) == "discontinuous IK for MOVE P endpoint"
    assert error.stage == "ik_continuity" and error.diagnostics == diagnostics
    assert requests == [("mock_action", {"target": "offline"})]
    restored = pickle.loads(pickle.dumps(error))
    assert str(restored) == str(error)
    assert restored.stage == error.stage and restored.diagnostics == diagnostics


def test_plain_errors_keep_legacy_wire_contract(local_server):
    assert rpc.error_response(ValueError("legacy rejection")) == {"ok": False, "error": "legacy rejection"}

    def dispatch(method, args):
        raise ValueError("legacy rejection")

    with pytest.raises(RuntimeError, match="legacy rejection") as failure:
        rpc.call(local_server(dispatch), "offline", timeout=1)
    assert failure.value.stage is None and failure.value.diagnostics is None


@pytest.mark.parametrize("packet,message", [
    ({"ok": False, "error": "old worker error"}, "old worker error"),
    ({"ok": False}, "hardware operation failed"),
])
def test_client_accepts_old_worker_error_packets(local_server, packet, message):
    with pytest.raises(rpc.RPCError) as failure:
        rpc.call(local_server(packet=packet), "offline", timeout=1)
    assert str(failure.value) == message
    assert failure.value.stage is None and failure.value.diagnostics is None


def test_nonfinite_unsupported_and_cyclic_diagnostics_do_not_break_rpc(local_server):
    diagnostics = {"values": [float("nan"), float("inf"), -float("inf"), .003],
                   "unsupported": object(), "binary": b"not an image response"}
    diagnostics["cycle"] = diagnostics

    def dispatch(method, args):
        if method == "error":
            raise WorkerError("original error", diagnostics)
        return {"still_available": True, "echo": args}

    path = local_server(dispatch)
    with pytest.raises(rpc.RPCError, match="original error") as failure:
        rpc.call(path, "error", timeout=1)
    assert failure.value.stage == "ik_continuity"
    assert failure.value.diagnostics == {"values": [None, None, None, .003],
                                         "unsupported": None, "binary": None, "cycle": None}
    json.dumps(failure.value.diagnostics, allow_nan=False)
    assert math.isnan(diagnostics["values"][0]) and diagnostics["cycle"] is diagnostics
    # A failed request must leave the same simulated server usable.
    assert rpc.call(path, "healthy", {"value": 7}, timeout=1) == {
        "still_available": True, "echo": {"value": 7}}


def test_broken_optional_error_property_preserves_original_error(local_server):
    class BrokenProperty(ValueError):
        @property
        def diagnostics(self):
            raise RuntimeError("property cannot be read")

    def dispatch(method, args):
        raise BrokenProperty("original error")

    with pytest.raises(rpc.RPCError, match="original error") as failure:
        rpc.call(local_server(dispatch), "offline", timeout=1)
    assert failure.value.diagnostics is None


def test_oversized_optional_diagnostics_fall_back_to_original_text(local_server, monkeypatch):
    monkeypatch.setattr(rpc, "MAX_BYTES", 512)

    def dispatch(method, args):
        raise WorkerError("original error", {"large": "x" * 1024})

    with pytest.raises(rpc.RPCError, match="original error") as failure:
        rpc.call(local_server(dispatch), "offline", timeout=1)
    assert failure.value.diagnostics is None


def test_nonfinite_success_result_remains_a_failure(local_server):
    path = local_server(lambda method, args: {"invalid_success": float("nan")})
    with pytest.raises(rpc.RPCError, match="Out of range float"):
        rpc.call(path, "offline", timeout=1)


def test_shared_and_deep_diagnostics_are_safe_without_mutating_source():
    shared = {"count": 4}
    assert rpc.diagnostic_json([shared, shared]) == [{"count": 4}, {"count": 4}]
    nested = {"leaf": True}
    for _ in range(40):
        nested = {"next": nested}
    json.dumps(rpc.diagnostic_json(nested), allow_nan=False)
    assert "next" in nested
