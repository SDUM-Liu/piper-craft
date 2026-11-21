"""Piper Craft's public Sign in with ChatGPT OAuth client."""
import base64
from contextlib import contextmanager
import hashlib
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import os
from pathlib import Path
import re
import secrets
import stat
import tempfile
import time
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlencode, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener
import uuid
import webbrowser

import jwt

from .contracts import locked

ISSUER = 'https://auth.openai.com'
DISCOVERY = ISSUER+'/.well-known/openid-configuration'
AUTHORIZE = ISSUER+'/api/accounts/authorize'
TOKEN = ISSUER+'/api/accounts/oauth/token'
RESOURCE = 'https://api.openai.com/v1'
DYNAMIC = 'dynamic_agent_client'
SCOPES = 'openid profile email offline_access resource.invoke chatgpt.tokens.use.direct'
PLAN_SCOPES = {'resource.invoke', 'chatgpt.tokens.use.direct'}
CALLBACK = '/auth/callback'
INVALID_REFRESH_CODES = {'invalid_grant', 'invalid_refresh_token', 'token_expired', 'refresh_token_expired',
                         'refresh_token_invalidated', 'refresh_token_reused'}


class ServiceError(RuntimeError):
    def __init__(self, operation, code, status=None):
        self.code, self.status = code, status
        super().__init__(f'{operation}: {code}' + (f' (HTTP {status})' if status else ''))


def error_code(payload):
    error = payload.get('error', payload) if isinstance(payload, dict) else {}
    raw = error.get('code', error.get('type', 'unknown_error')) if isinstance(error, dict) else error
    return raw if isinstance(raw, str) and re.fullmatch(r'[a-zA-Z0-9_.-]{1,100}', raw) else 'unknown_error'


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def connect(url, *, operation, form=None, body=None, token=None, timeout=15):
    headers = {}
    if token:
        headers['Authorization'] = 'Bearer '+token
    if form is not None:
        headers['Content-Type'] = 'application/x-www-form-urlencoded'
        body = urlencode(form).encode()
    elif body is not None:
        headers.update({'Content-Type': 'application/json', 'Accept': 'text/event-stream'})
        body = json.dumps(body, ensure_ascii=False, allow_nan=False).encode()
    try:
        return build_opener(NoRedirect()).open(Request(url, data=body, headers=headers), timeout=timeout)
    except HTTPError as exc:
        with exc:
            try:
                code = error_code(json.loads(exc.read(65536)))
            except (UnicodeError, ValueError, OSError):
                code = 'http_error'
        raise ServiceError(operation, code, exc.code) from None
    except (URLError, TimeoutError, OSError):
        raise ServiceError(operation, 'network_error') from None


def json_request(url, *, operation, **kwargs):
    with connect(url, operation=operation, **kwargs) as response:
        try:
            return json.loads(response.read(2_000_000))
        except (UnicodeError, ValueError, OSError):
            raise ServiceError(operation, 'invalid_json') from None


def issuer_url(value):
    parsed = urlsplit(value)
    if parsed.scheme != 'https' or parsed.netloc != 'auth.openai.com' or parsed.fragment or parsed.query:
        raise ServiceError('OpenID discovery', 'unexpected_issuer_endpoint')
    return value


def private_json(path, data):
    fd, temp = tempfile.mkstemp(dir=path.parent, prefix='.credentials-')
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(data, stream, ensure_ascii=False, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        Path(temp).unlink(missing_ok=True)


class AccountStore:
    def __init__(self, directory=None):
        config = Path(os.environ.get('XDG_CONFIG_HOME') or Path.home()/'.config')
        self.directory = Path(directory) if directory is not None else config/'piper-craft'/'chatgpt'
        self.path = self.directory/'accounts.json'

    @contextmanager
    def edit(self):
        self.directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        self._check(self.directory, stat.S_ISDIR)
        with locked(self.directory):
            if self.path.exists() or self.path.is_symlink():
                self._check(self.path, stat.S_ISREG)
                data = json.loads(self.path.read_text())
            else:
                data = {'host_id': 'urn:uuid:'+str(uuid.uuid4()), 'active': None, 'accounts': {}}
                private_json(self.path, data)
            yield data

    @staticmethod
    def _check(path, file_type):
        info = path.lstat()
        if not file_type(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
            raise ServiceError('credential storage', 'owner_only_permissions_required')

    def save(self, data):
        private_json(self.path, data)

    def list(self):
        with self.edit() as data:
            return [{'account': label, 'email': row['email'], 'active': label == data['active'],
                     'signed_in': bool(row.get('refresh_token')),
                     'plan_enabled': PLAN_SCOPES <= set(row.get('scopes', ()))}
                    for label, row in sorted(data['accounts'].items())]

    def select(self, label):
        with self.edit() as data:
            self.account(data, label)
            data['active'] = label
            self.save(data)

    @staticmethod
    def account(data, label=None):
        label = label or data['active']
        row = data['accounts'].get(label)
        if row is None:
            raise ServiceError('Sign in with ChatGPT', 'no_account_selected')
        return row


def await_callback(params, *, timeout=180, browser=webbrowser.open):
    received = {}

    class Callback(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass  # callback URL contains a one-time authorization code

        def setup(self):
            self.request.settimeout(2)
            super().setup()

        def do_GET(self):
            uri = urlsplit(self.path)
            query = parse_qs(uri.query, keep_blank_values=True)
            valid = (uri.path == CALLBACK and len(query.get('state', [])) == 1
                     and secrets.compare_digest(query['state'][0].encode(), params['state'].encode()))
            if valid:
                received.update({k: v[0] for k, v in query.items() if len(v) == 1})
            self.send_response(200 if valid else 400)
            self.send_header('Content-Type', 'text/plain; charset=utf-8')
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(b'Return to Piper Craft.' if valid else b'Invalid callback.')

    with HTTPServer(('127.0.0.1', 0), Callback) as server:
        params['redirect_uri'] = f'http://127.0.0.1:{server.server_port}{CALLBACK}'
        if not browser(AUTHORIZE+'?'+urlencode(params)):
            raise ServiceError('Sign in with ChatGPT', 'system_browser_unavailable')
        end = time.monotonic()+timeout
        server.timeout = .25
        while not received and time.monotonic() < end:
            server.handle_request()
    if not received:
        raise ServiceError('Sign in with ChatGPT', 'callback_timeout')
    return received


class ChatGPT:
    def __init__(self, store=None, *, request=json_request, callback=await_callback, clock=time.time):
        self.store = store if store is not None else AccountStore()
        self.request, self.callback, self.clock = request, callback, clock

    def discovery(self):
        discovery = self.request(DISCOVERY, operation='OpenID discovery')
        if discovery.get('issuer') != ISSUER:
            raise ServiceError('OpenID discovery', 'unexpected_issuer')
        return discovery

    def validate_id(self, token, client_id, nonce):
        try:
            discovery = self.discovery()
            jwks = self.request(issuer_url(discovery['jwks_uri']), operation='OpenID signature keys')
            kid = jwt.get_unverified_header(token)['kid']
            key, = (key for key in jwks['keys'] if key.get('kid') == kid
                    and key.get('kty') == 'RSA' and key.get('use', 'sig') == 'sig')
            identity = jwt.decode(token, jwt.PyJWK.from_dict(key, algorithm='RS256').key,
                                  algorithms=['RS256'], issuer=ISSUER, audience=client_id,
                                  options={'require': ['exp', 'iat', 'iss', 'aud', 'sub', 'nonce']})
            if (not secrets.compare_digest(str(identity['nonce']).encode(), nonce.encode()) or
                    ('azp' in identity and identity['azp'] != client_id) or
                    (isinstance(identity['aud'], list) and len(identity['aud']) > 1
                     and identity.get('azp') != client_id) or not identity['sub']):
                raise ValueError('nonce/authorized party mismatch')
        except (jwt.PyJWTError, KeyError, TypeError, ValueError):
            raise ServiceError('Sign in with ChatGPT', 'invalid_id_token') from None
        return identity

    def sign_in(self, account=None):
        with self.store.edit() as data:
            existing = dict(self.store.account(data, account)) if account else None
            host = data['host_id']
        verifier = secrets.token_urlsafe(48)
        challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).decode().rstrip('=')
        params = dict(client_id=existing['client_id'] if existing else DYNAMIC, ext_agent_host_id=host,
                      response_type='code', resource=RESOURCE, scope=SCOPES, state=secrets.token_urlsafe(32),
                      nonce=secrets.token_urlsafe(32), code_challenge_method='S256', code_challenge=challenge)
        if not existing:
            params['agent_name_hint'] = 'Piper Craft'
        elif existing.get('email'):
            params['login_hint'] = existing['email']
        # Keep the browser account selector available, even during reauthorization.
        callback = self.callback(params)
        if not secrets.compare_digest(str(callback.get('state', '')).encode(), params['state'].encode()):
            raise ServiceError('Sign in with ChatGPT', 'callback_state_mismatch')
        if 'error' in callback:
            raise ServiceError('Sign in with ChatGPT', error_code(callback))
        client_id = callback.get('client_id', existing['client_id'] if existing else '')
        if (not re.fullmatch(r'oaiapp_[a-zA-Z0-9_-]+', client_id)
                or existing and existing['client_id'] != client_id or not callback.get('code')):
            raise ServiceError('Sign in with ChatGPT', 'registration_incomplete_or_client_mismatch')
        data = self.exchange(client_id, grant_type='authorization_code', code=callback['code'],
                             code_verifier=verifier, redirect_uri=params['redirect_uri'])
        identity = self.validate_id(data.get('id_token'), client_id, params['nonce'])
        if existing and identity['sub'] != existing['subject']:
            raise ServiceError('Sign in with ChatGPT', 'selected_account_mismatch')
        label = hashlib.sha256((client_id+'\0'+identity['sub']).encode()).hexdigest()[:16]
        row = dict(client_id=client_id, subject=identity['sub'], email=identity.get('email', ''), issuer=ISSUER)
        scope = data.get('scope', '')
        has_plan = isinstance(scope, str) and PLAN_SCOPES <= set(scope.split())
        if has_plan:
            row = self.credentials(data, row)
        with self.store.edit() as saved:
            if existing and saved['accounts'].get(account) != existing:
                raise ServiceError('Sign in with ChatGPT', 'account_changed_during_authorization')
            saved['accounts'][label] = row
            saved['active'] = label
            self.store.save(saved)
        self.require_plan(row)
        return {'account': label, 'email': row['email'], 'plan_enabled': True}

    def exchange(self, client_id, **form):
        return self.request(TOKEN, operation='ChatGPT OAuth token', form={**form, 'client_id': client_id,
                                                                        'resource': RESOURCE})

    def credentials(self, data, old):
        if not isinstance(data, dict) or data.get('token_type', '').lower() != 'bearer':
            raise ServiceError('ChatGPT OAuth token', 'invalid_token_response')
        expiry = data.get('expires_in')
        if (type(expiry) is not int or expiry <= 0 or not isinstance(data.get('access_token'), str)
                or not data['access_token'] or not isinstance(data.get('refresh_token'), str)
                or not data['refresh_token'] or not isinstance(data.get('scope', ''), str)):
            raise ServiceError('ChatGPT OAuth token', 'invalid_token_response')
        return {**old, 'id_token': old.get('id_token') or data.get('id_token'), 'access_token': data['access_token'],
                'refresh_token': data['refresh_token'], 'expires_at': self.clock()+expiry,
                'scopes': data['scope'].split() if 'scope' in data else old.get('scopes', [])}

    @staticmethod
    def require_plan(row):
        if not PLAN_SCOPES <= set(row.get('scopes', ())):
            raise ServiceError('ChatGPT plan usage', 'chatgpt_plan_permission_missing')

    def token(self, label, *, rejected=None):
        with self.store.edit() as data:
            row = self.store.account(data, label)
            if not row.get('refresh_token'):
                raise ServiceError('Sign in with ChatGPT', 'account_signed_out')
            self.require_plan(row)
            if self.clock()+60 < row['expires_at'] and rejected != row['access_token']:
                return row['access_token']
            try:
                refreshed = self.exchange(row['client_id'], grant_type='refresh_token',
                                          refresh_token=row['refresh_token'])
            except ServiceError as exc:
                if exc.code in INVALID_REFRESH_CODES:
                    self.clear(row)
                    self.store.save(data)
                raise
            updated = self.credentials(refreshed, row)
            data['accounts'][label] = updated
            self.store.save(data)
            self.require_plan(updated)
            return updated['access_token']

    @staticmethod
    def clear(row):
        for key in ('access_token', 'refresh_token', 'id_token', 'expires_at', 'scopes'):
            row.pop(key, None)

    def sign_out(self, label=None):
        with self.store.edit() as data:
            label = label or data['active']
            row = self.store.account(data, label)
            token = row.get('refresh_token')
            revoked = None
            try:
                if token:
                    revoked = False
                    for delay in (0, .5, 1.):
                        time.sleep(delay)
                        try:
                            url = issuer_url(self.discovery()['revocation_endpoint'])
                            with connect(url, operation='ChatGPT revoke', form=dict(
                                    token=token, token_type_hint='refresh_token', client_id=row['client_id'])) as reply:
                                revoked = reply.status == 200
                            break
                        except ServiceError as exc:
                            if exc.status and exc.status < 500:
                                break
            finally:
                self.clear(row)
                if data['active'] == label:
                    data['active'] = None
                self.store.save(data)
            return {'account': label, 'signed_out': True, 'remote_revocation_confirmed': revoked}

    def active_label(self, label=None):
        with self.store.edit() as data:
            label = label or data['active']
            self.store.account(data, label)
            return label

    def models(self, label):
        result = self.request(RESOURCE+'/models', operation='ChatGPT model list', token=self.token(label))
        return [dict(slug=row['slug'], display_name=row['display_name'])
                for row in result['models'] if row.get('visibility') == 'list']
