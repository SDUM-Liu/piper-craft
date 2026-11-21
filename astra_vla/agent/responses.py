"""Public Responses streaming protocol for ChatGPT plan usage."""
from dataclasses import dataclass
import http.client
import json
import queue
import threading
import time

from .chatgpt import RESOURCE, ServiceError, connect, error_code


class Interrupted(RuntimeError):
    pass


def check_cancelled(cancelled):
    if cancelled():
        raise Interrupted('cancelled')


def sse_events(stream):
    """Decode complete SSE events. An unterminated final event is incomplete."""
    event, lines = None, []
    while line := stream.readline(4_000_001):
        if len(line) > 4_000_000:
            raise ServiceError('Responses stream', 'event_too_large')
        line = line.rstrip(b'\r\n')
        if not line:
            if lines:
                try:
                    value = json.loads(b'\n'.join(lines))
                    if not isinstance(value, dict) or event and value.get('type') != event:
                        raise ValueError('inconsistent SSE type')
                except (ValueError, UnicodeError):
                    raise ServiceError('Responses stream', 'invalid_event') from None
                yield value
            event, lines = None, []
        elif line.startswith(b'data:'):
            value = line[5:].lstrip(b' ')
            if value != b'[DONE]':
                lines.append(value)
                if sum(map(len, lines)) > 4_000_000:
                    raise ServiceError('Responses stream', 'event_too_large')
        elif line.startswith(b'event:'):
            event = line[6:].strip().decode('ascii', errors='replace')


@dataclass(frozen=True)
class ModelTurn:
    output: list
    usage: dict


def completed_turn(events, cancelled):
    try:
        for event in events:
            check_cancelled(cancelled)
            kind = event.get('type')
            if kind in ('error', 'response.failed', 'response.incomplete'):
                response = event.get('response') or event
                code = (error_code(response) if kind != 'response.incomplete'
                        else error_code({'error': (response.get('incomplete_details') or {}).get('reason')}))
                raise ServiceError('Responses '+kind, code)
            if kind == 'response.completed':
                response = event.get('response', {})
                output = response.get('output')
                if response.get('status') != 'completed' or not isinstance(output, list):
                    raise ServiceError('Responses stream', 'missing_completed_output')
                return ModelTurn(output, response.get('usage') or {})
    except (TimeoutError, OSError, http.client.HTTPException):
        check_cancelled(cancelled)
        raise ServiceError('Responses stream', 'connection_interrupted') from None
    check_cancelled(cancelled)
    raise ServiceError('Responses stream', 'missing_response_completed')


class Responses:
    def __init__(self, auth, account, model, *, open_stream=connect):
        self.auth, self.account, self.model, self.open_stream = auth, account, model, open_stream
        self.identity = {'provider': 'openai_siwc', 'model': model, 'account': account}

    def request_body(self, messages, instructions, tools):
        return dict(model=self.model, input=messages, instructions=instructions,
                    tools=tools, parallel_tool_calls=False, include=['reasoning.encrypted_content'],
                    store=False, stream=True)

    def complete(self, body, cancelled):
        check_cancelled(cancelled)
        stopped = threading.Event()
        results = queue.Queue(maxsize=1)

        def read():
            try:
                result = self._complete(body, lambda: stopped.is_set() or cancelled())
            except Exception as exc:
                results.put((None, exc))
            else:
                results.put((result, None))

        threading.Thread(target=read, daemon=True, name='piper-responses').start()
        deadline = time.monotonic()+180
        try:
            while True:
                check_cancelled(cancelled)
                if time.monotonic() >= deadline:
                    raise ServiceError('Responses stream', 'inference_deadline_exceeded')
                try:
                    result, failure = results.get(timeout=.05)
                except queue.Empty:
                    continue
                check_cancelled(cancelled)
                if failure is not None:
                    raise failure
                return result
        finally:
            stopped.set()

    def _complete(self, body, cancelled):
        check_cancelled(cancelled)
        token = self.auth.token(self.account)
        check_cancelled(cancelled)
        try:
            stream = self._open(body, token)
        except ServiceError as exc:
            if exc.status != 401 or exc.code not in ('invalid_token', 'token_expired'):
                raise
            check_cancelled(cancelled)
            stream = self._open(body, self.auth.token(self.account, rejected=token))
        with stream:
            # ChatGPT plan requests can deliver SSE as text/plain. The SSE
            # parser still requires a complete response.completed event.
            if stream.headers.get_content_type() not in ('text/event-stream', 'text/plain'):
                raise ServiceError('Responses stream', 'unexpected_content_type')
            return completed_turn(sse_events(stream), cancelled)

    def _open(self, body, token):
        return self.open_stream(RESOURCE+'/responses', operation='ChatGPT plan Responses',
                                token=token, body=body, timeout=35)
