"""Real Git transport regressions against a local TLS Git HTTP backend."""

from __future__ import annotations

import base64
import getpass
import os
import socket
import ssl
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from coding_review_agent_loop import git_transport
from coding_review_agent_loop.errors import AgentLoopError
from coding_review_agent_loop.runner import Runner


def _git(path: Path, *args: str) -> str:
    return subprocess.check_output(("git", "-C", str(path), *args), text=True).strip()


@pytest.fixture
def https_git(tmp_path, monkeypatch):
    bare = tmp_path / "OWNER" / "REPO.git"
    bare.parent.mkdir()
    seed = tmp_path / "seed"
    checkout = tmp_path / "checkout"
    subprocess.run(("git", "init", "-q", "--bare", "-b", "main", str(bare)), check=True)
    subprocess.run(("git", "init", "-q", "-b", "main", str(seed)), check=True)
    (seed / "file.txt").write_text("transport fixture\n")
    _git(seed, "add", "file.txt")
    _git(seed, "-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "-qm", "fixture")
    _git(seed, "push", "-q", str(bare), "main")
    subprocess.run(("git", "clone", "-q", str(bare), str(checkout)), check=True)
    cert, key = tmp_path / "cert.pem", tmp_path / "key.pem"
    subprocess.run(("openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes",
                    "-keyout", str(key), "-out", str(cert), "-days", "1",
                    "-subj", "/CN=localhost", "-addext", "subjectAltName=DNS:localhost"),
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    requests: list[tuple[str, str | None]] = []
    required_auth = {"value": None}
    redirect = {"value": False}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_GET(self):
            self.serve_git()

        def do_POST(self):
            self.serve_git()

        def serve_git(self):
            auth = self.headers.get("Authorization")
            requests.append((self.path, auth))
            if redirect["value"]:
                self.send_response(302)
                self.send_header("Location", "https://127.0.0.1:1/redirected")
                self.end_headers()
                return
            if required_auth["value"] and (not auth or auth.lower() != required_auth["value"].lower()):
                self.send_response(401)
                self.send_header("WWW-Authenticate", 'Basic realm="Git"')
                self.end_headers()
                return
            body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            path, _, query = self.path.partition("?")
            env = dict(os.environ)
            env.update({"GIT_PROJECT_ROOT": str(tmp_path), "GIT_HTTP_EXPORT_ALL": "1",
                        "PATH_INFO": path, "QUERY_STRING": query,
                        "REQUEST_METHOD": self.command,
                        "CONTENT_TYPE": self.headers.get("Content-Type", ""),
                        "CONTENT_LENGTH": str(len(body))})
            result = subprocess.run(("git", "http-backend"), input=body, env=env,
                                    capture_output=True, check=True)
            header, payload = result.stdout.split(b"\r\n\r\n", 1)
            lines = header.decode("latin1").split("\r\n")
            status = 200
            for line in lines:
                if line.lower().startswith("status:"):
                    status = int(line.split(":", 1)[1].strip().split()[0])
            self.send_response(status)
            for line in lines:
                if ":" in line and not line.lower().startswith("status:"):
                    name, value = line.split(":", 1)
                    self.send_header(name, value.strip())
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(cert, key)
    server.socket = context.wrap_socket(server.socket, server_side=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"https://localhost:{server.server_port}/OWNER/REPO.git"
    repo = f"localhost:{server.server_port}/OWNER/REPO"
    monkeypatch.setenv("SSL_CERT_FILE", str(cert))
    original_env = git_transport._transport_env

    def fixture_env(endpoint, gh_cmd):
        env = original_env(endpoint, gh_cmd)
        env["GIT_SSL_CAINFO"] = str(cert)
        return env

    monkeypatch.setattr(git_transport, "_transport_env", fixture_env)
    _git(checkout, "config", "remote.origin.url", url)
    try:
        yield checkout, url, repo, requests, required_auth, redirect
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.mark.parametrize("mode", ["public", "private", "redirect", "rewrite"])
def test_real_https_transport_policy(https_git, monkeypatch, tmp_path, mode):
    checkout, url, repo, requests, required_auth, redirect = https_git
    marker = tmp_path / "helper-ran"
    _git(checkout, "config", "credential.helper", f"!touch {marker}; echo password=planted")
    if mode == "rewrite":
        _git(checkout, "config", "url.https://127.0.0.1:1/.insteadOf", url)
    if mode in {"private", "redirect"}:
        token = "fixture-secret-token"
        required_auth["value"] = "Basic " + base64.b64encode(f"x-access-token:{token}".encode()).decode()
        monkeypatch.setattr(git_transport, "_token_from_gh", lambda *_args: token)
    if mode == "redirect":
        redirect["value"] = True
    kwargs = {"repo": repo, "runner": Runner(), "gh_cmd": "gh" if mode in {"private", "redirect"} else None}
    if mode == "redirect":
        with pytest.raises(AgentLoopError, match="Trusted Git transport failed"):
            git_transport.import_ref(checkout, "refs/heads/main", "refs/remotes/origin/main", **kwargs)
        assert all(path != "/redirected" for path, _ in requests)
        assert all(path.startswith("/OWNER/REPO.git/") for path, _ in requests)
        assert any(auth and auth.lower() == required_auth["value"].lower() for _, auth in requests)
    else:
        sha = git_transport.import_ref(checkout, "refs/heads/main", "refs/remotes/origin/main", **kwargs)
        assert sha == _git(checkout, "rev-parse", "HEAD")
        assert _git(checkout, "cat-file", "-t", sha) == "commit"
        assert requests
        if mode == "private":
            assert any(auth and auth.lower() == required_auth["value"].lower() for _, auth in requests)
            assert all(path.startswith("/OWNER/REPO.git/") for path, _ in requests)
    assert not marker.exists()


@pytest.mark.skipif(os.name != "posix" or not Path("/usr/sbin/sshd").exists(),
                    reason="local OpenSSH server is unavailable")
def test_real_pinned_ssh_fetch(tmp_path, monkeypatch):
    bare = tmp_path / "repo.git"
    seed = tmp_path / "seed"
    checkout = tmp_path / "checkout"
    subprocess.run(("git", "init", "-q", "--bare", "-b", "main", str(bare)), check=True)
    subprocess.run(("git", "init", "-q", "-b", "main", str(seed)), check=True)
    (seed / "file.txt").write_text("ssh fixture\n")
    _git(seed, "add", "file.txt")
    _git(seed, "-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "-qm", "fixture")
    _git(seed, "push", "-q", str(bare), "main")
    subprocess.run(("git", "clone", "-q", str(bare), str(checkout)), check=True)
    key = tmp_path / "client_key"
    host_key = tmp_path / "host_key"
    for path in (key, host_key):
        subprocess.run(("ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-f", str(path)), check=True)
    authorized = tmp_path / "authorized_keys"
    authorized.write_bytes(key.with_suffix(".pub").read_bytes())
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    server_config = tmp_path / "sshd_config"
    server_config.write_text(f"Port {port}\nListenAddress 127.0.0.1\nHostKey {host_key}\n"
                             f"AuthorizedKeysFile {authorized}\nStrictModes no\nUsePAM no\n"
                             "PasswordAuthentication no\nPubkeyAuthentication yes\n"
                             "PermitRootLogin no\nLogLevel QUIET\n")
    client_config = tmp_path / "ssh_config"
    client_config.write_text(f"Host localhost\n  IdentityFile {key}\n  IdentitiesOnly yes\n"
                             "  StrictHostKeyChecking no\n  UserKnownHostsFile /dev/null\n"
                             "  LogLevel ERROR\n")
    client_config.chmod(0o600)
    server = subprocess.Popen(("/usr/sbin/sshd", "-D", "-e", "-f", str(server_config)),
                              stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    try:
        import time
        for _ in range(50):
            if server.poll() is not None:
                pytest.skip("local sshd could not start")
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.1):
                    break
            except OSError:
                time.sleep(0.02)
        else:
            pytest.skip("local sshd did not accept connections")
        url = f"ssh://{getpass.getuser()}@localhost:{port}{bare}"
        _git(checkout, "config", "remote.origin.url", url)
        original_env = git_transport._transport_env

        def fixture_env(endpoint, gh_cmd):
            env = original_env(endpoint, gh_cmd)
            assert env["GIT_SSH_COMMAND"] == "/usr/bin/ssh"
            env["GIT_SSH_COMMAND"] += f" -F {client_config}"
            return env

        monkeypatch.setattr(git_transport, "_transport_env", fixture_env)
        monkeypatch.setattr(git_transport, "trusted_url", lambda _repo, observed, **_kw: observed if observed == url else (_ for _ in ()).throw(AgentLoopError("endpoint mismatch")))
        sha = git_transport.import_ref(checkout, "refs/heads/main", "refs/remotes/origin/main",
                                       repo="OWNER/REPO", runner=Runner())
        assert sha == _git(checkout, "rev-parse", "HEAD")
    finally:
        server.terminate()
        server.communicate(timeout=5)
