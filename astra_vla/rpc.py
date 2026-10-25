"""Small, private Unix-socket RPC shared by Python 3.8/3.9 hardware workers."""
import json
import math
import numbers
import os
import socket
import socketserver
import struct

MAX_BYTES = 24 * 1024 * 1024


class RPCError(RuntimeError):
    """Preserve worker error text and optional evidence without implying execution."""

    def __init__(self, message, stage=None, diagnostics=None):
        super().__init__(message)
        self.stage = stage
        self.diagnostics = diagnostics


def diagnostic_json(value, ancestors=None, depth=0):
    """Errors must remain reportable even with nonfinite/unsupported evidence."""
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    if depth >= 32 or not isinstance(value, (dict, list, tuple)):
        return None
    ancestors = set() if ancestors is None else ancestors
    identity = id(value)
    if identity in ancestors:
        return None
    ancestors.add(identity)
    try:
        if isinstance(value, dict):
            return {str(key): diagnostic_json(item, ancestors, depth + 1)
                    for key, item in value.items() if isinstance(key, (str, numbers.Integral))}
        return [diagnostic_json(item, ancestors, depth + 1) for item in value]
    finally:
        ancestors.remove(identity)


def error_response(exc):
    result = {"ok": False, "error": str(exc)}
    for attribute in ("stage", "diagnostics"):
        try:
            value = getattr(exc, attribute, None)
            if attribute == "stage":
                if isinstance(value, str):
                    result[attribute] = value
            elif value is not None:
                result[attribute] = diagnostic_json(value)
        except Exception:
            # A broken optional property cannot hide the original error.
            continue
    return result


def read_exact(sock, size):
    result = bytearray()
    while len(result) < size:
        chunk = sock.recv(size - len(result))
        if not chunk:
            raise ConnectionError("hardware worker disconnected")
        result.extend(chunk)
    return bytes(result)


def receive(sock):
    size = struct.unpack("!I", read_exact(sock, 4))[0]
    if not 0 < size <= MAX_BYTES:
        raise ValueError("invalid RPC frame")
    return json.loads(read_exact(sock, size))


def send(sock, value):
    payload = json.dumps(value, ensure_ascii=False, allow_nan=False).encode()
    if len(payload) > MAX_BYTES:
        raise ValueError("RPC frame too large")
    sock.sendall(struct.pack("!I", len(payload)) + payload)


def call(path, method, args=None, timeout=75):
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout)
        sock.connect(str(path))
        send(sock, {"method": method, "args": args or {}})
        result = receive(sock)
        if not result.get("ok"):
            raise RPCError(result.get("error", "hardware operation failed"),
                           stage=result.get("stage") if isinstance(result.get("stage"), str) else None,
                           diagnostics=diagnostic_json(result.get("diagnostics")))
        return result["result"]


def request_handler(dispatch):
    class Handler(socketserver.BaseRequestHandler):
        def handle(self):
            try:
                request = receive(self.request)
                result = dispatch(request["method"], request.get("args", {}))
                send(self.request, {"ok": True, "result": result})
            except Exception as exc:
                try:
                    response = error_response(exc)
                    try:
                        send(self.request, response)
                    except (TypeError, ValueError, OverflowError, RecursionError):
                        # Oversized/invalid optional diagnostics fall back to the
                        # existing text-only error contract, never a success packet.
                        send(self.request, {"ok": False, "error": response["error"]})
                except (OSError, TypeError, ValueError, OverflowError, RecursionError):
                    pass
    return Handler


def serve(path, dispatch):
    path = str(path)
    if os.path.exists(path):
        try:
            with socket.socket(socket.AF_UNIX) as probe:
                probe.connect(path)
            raise RuntimeError("worker already owns " + path)
        except (ConnectionRefusedError, FileNotFoundError):
            os.unlink(path)

    class Server(socketserver.ThreadingUnixStreamServer):
        daemon_threads = True

    with Server(path, request_handler(dispatch)) as server:
        os.chmod(path, 0o600)
        server.serve_forever()
