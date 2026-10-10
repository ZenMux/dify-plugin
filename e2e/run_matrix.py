#!/usr/bin/env python3
"""Install the packaged plugin into real Dify releases and drive it through Dify's console API.

For each Dify version this starts the *official* docker-compose stack of that tag (only `api`,
`plugin_daemon` and their dependencies), then:
  1. installs the .difypkg (signature check disabled; the package is unsigned),
  2. validates credentials for both regions and rejects a bad key, then saves them,
  3. checks that every predefined LLM / embedding / rerank model is listed,
  4. runs a debug chat per protocol and checks Dify recorded tokens and price.
The stack and its volumes are removed afterwards unless --keep is given.

Usage:
  ZENMUX_E2E_API_KEY=sk-... .venv/bin/python e2e/run_matrix.py --pkg dist/zenmux-dify-plugin_0.0.9.difypkg \
      [--versions 1.6.0 1.13.1 1.14.2 1.17.1] [--keep]
"""

import argparse
import base64
import os
import secrets
import subprocess
import sys
import time
from pathlib import Path

import requests
import yaml

ROOT = Path(__file__).resolve().parent.parent
CACHE = Path(os.environ.get("ZM_E2E_CACHE", "/tmp/zm-e2e"))
DEFAULT_VERSIONS = ["1.6.0", "1.13.1", "1.14.2", "1.17.1"]
PROVIDER = "zenmux/zenmux-dify-plugin/zenmux"
EMAIL, PASSWORD = "e2e@example.com", "E2eTest123456"
CHAT_CASES = [
    ("openai/gpt-4.1-nano", {"max_tokens": 64}),
    ("anthropic/claude-haiku-4.5", {"max_tokens": 64}),
    ("google/gemini-2.5-flash-lite", {"max_output_tokens": 64}),
    ("moonshotai/kimi-k2.6", {"max_tokens": 512, "enable_thinking": True}),
]
ENV_OVERRIDES = {
    "FORCE_VERIFYING_SIGNATURE": "false",
    "COMPOSE_PROFILES": "postgresql",
    "DB_TYPE": "postgresql",
    # 1.13.x ships REDIS_MAX_CONNECTIONS= (empty) in .env.example, which the api fails to parse.
    "REDIS_MAX_CONNECTIONS": "100",
}
COMPOSE_OVERRIDE = """
services:
  api:
    ports: ["127.0.0.1:{port}:5001"]
{extra}
"""
# 1.17+ makes api depend on the agent backend; not needed to exercise a model plugin.
API_DEPENDS_OVERRIDE = """    depends_on: !override
      init_permissions: {condition: service_completed_successfully}
      db_postgres: {condition: service_healthy}
      redis: {condition: service_started}
"""


def log(msg):
    print(msg, flush=True)


# ── stack management ─────────────────────────────────────────────────────────

def docker_dir(version: str) -> Path:
    repo = CACHE / f"dify-{version}"
    if not (repo / "docker" / "docker-compose.yaml").exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "-q", "--depth", "1", "--branch", version, "--filter=blob:none",
                        "--sparse", "https://github.com/langgenius/dify.git", str(repo)], check=True)
        subprocess.run(["git", "sparse-checkout", "set", "docker"], cwd=repo, check=True)
    return repo / "docker"


def prepare(version: str, port: int) -> Path:
    d = docker_dir(version)
    # The official stack bind-mounts ./volumes/*; start every run from a clean slate
    # (only git-ignored runtime data is removed, tracked config stays).
    subprocess.run(["git", "clean", "-ffdqX", "--", "docker/volumes"], cwd=d.parent, check=True)
    env = (d / ".env.example").read_text()
    overrides = {**ENV_OVERRIDES, "SECRET_KEY": secrets.token_urlsafe(32)}
    if os.environ.get("ZM_E2E_PIP_MIRROR"):
        overrides["PIP_MIRROR_URL"] = os.environ["ZM_E2E_PIP_MIRROR"]
    (d / ".env").write_text(env + "\n# zenmux e2e overrides\n" + "".join(f"{k}={v}\n" for k, v in overrides.items()))
    services = yaml.safe_load((d / "docker-compose.yaml").read_text())["services"]
    extra = API_DEPENDS_OVERRIDE if "agent_backend" in (services["api"].get("depends_on") or {}) else ""
    (d / "docker-compose.zenmux-e2e.yaml").write_text(COMPOSE_OVERRIDE.format(port=port, extra=extra))
    return d


def compose(version: str, d: Path, *args, check=True):
    cmd = ["docker", "compose", "-p", f"zm-e2e-{version.replace('.', '-')}",
           "-f", "docker-compose.yaml", "-f", "docker-compose.zenmux-e2e.yaml", *args]
    return subprocess.run(cmd, cwd=d, check=check)


def wait_http(url: str, timeout=600):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if requests.get(url, timeout=5).status_code < 500:
                return
        except requests.RequestException:
            pass
        time.sleep(5)
    raise TimeoutError(f"{url} not ready after {timeout}s")


# ── console API client ───────────────────────────────────────────────────────

class Console:
    def __init__(self, base: str):
        self.base, self.s = base.rstrip("/"), requests.Session()

    def call(self, method, path, expect_ok=True, **kw):
        csrf = next((c.value for c in self.s.cookies if "csrf" in c.name.lower()), None)
        headers = {"X-CSRF-Token": csrf} if csrf else {}
        r = self.s.request(method, f"{self.base}/console/api{path}", headers=headers, timeout=300, **kw)
        if expect_ok and r.status_code >= 400:
            raise RuntimeError(f"{method} {path} -> {r.status_code}: {r.text[:500]}")
        return r

    def login(self):
        r = self.call("GET", "/setup", expect_ok=False)
        if r.ok and r.json().get("step") != "finished":
            self.call("POST", "/setup", json={"email": EMAIL, "name": "e2e", "password": PASSWORD})
        r = self.call("POST", "/login", json={"email": EMAIL, "password": PASSWORD, "remember_me": True}, expect_ok=False)
        if r.status_code == 401 and "encrypted" in r.text:  # >= 1.13 expects a base64-encoded password
            encoded = base64.b64encode(PASSWORD.encode()).decode()
            r = self.call("POST", "/login", json={"email": EMAIL, "password": encoded, "remember_me": True})
        elif not r.ok:
            raise RuntimeError(f"login -> {r.status_code}: {r.text[:300]}")
        token = (r.json().get("data") or {}).get("access_token") if isinstance(r.json().get("data"), dict) else None
        if token:  # <= 1.9 returns tokens in the body; newer versions use cookies + CSRF
            self.s.headers["Authorization"] = f"Bearer {token}"


def install_plugin(c: Console, pkg: Path):
    with pkg.open("rb") as f:
        uid = c.call("POST", "/workspaces/current/plugin/upload/pkg", files={"pkg": (pkg.name, f)}).json()["unique_identifier"]
    r = c.call("POST", "/workspaces/current/plugin/install/pkg", json={"plugin_unique_identifiers": [uid]}).json()
    if r.get("all_installed"):
        return
    deadline = time.time() + 900
    while time.time() < deadline:
        task = c.call("GET", f"/workspaces/current/plugin/tasks/{r['task_id']}").json()["task"]
        if task["status"] == "success":
            return
        if task["status"] == "failed":
            raise RuntimeError(f"plugin install failed: {task}")
        time.sleep(5)
    raise TimeoutError("plugin install did not finish")


def credentials_flow(c: Console, key: str):
    base = f"/workspaces/current/model-providers/{PROVIDER}"
    for region in ("global", "cn"):
        r = c.call("POST", f"{base}/credentials/validate", json={"credentials": {"api_key": key, "region": region}}).json()
        assert r.get("result") == "success", f"validate {region}: {r}"
    bad = c.call("POST", f"{base}/credentials/validate", json={"credentials": {"api_key": "sk-invalid"}}).json()
    assert bad.get("result") != "success", "invalid key was accepted"
    payload = {"credentials": {"api_key": key, "region": "global"}}
    r = c.call("POST", f"{base}/credentials", json=payload, expect_ok=False)
    # <= 1.6 saves on the provider resource itself; its <path:provider> route swallows "/credentials".
    if r.status_code in (404, 405) or "does not exist" in r.text:
        c.call("POST", base, json=payload)
    elif not r.ok:
        raise RuntimeError(f"save credentials -> {r.status_code}: {r.text[:300]}")


def expected_models(folder: str) -> int:
    return sum(1 for p in (ROOT / "models" / folder).glob("*.yaml") if not p.name.startswith("_"))


def check_model_lists(c: Console):
    for model_type, folder in (("llm", "llm"), ("text-embedding", "text_embedding"), ("rerank", "rerank")):
        data = c.call("GET", f"/workspaces/current/models/model-types/{model_type}").json()["data"]
        ours = next((p for p in data if p["provider"] == PROVIDER), None)
        assert ours, f"{model_type}: provider not listed"
        active = [m for m in ours["models"] if m.get("status", "active") == "active"]
        assert len(active) == expected_models(folder), f"{model_type}: {len(active)} active vs {expected_models(folder)} YAMLs"


def model_config(model, params):
    return {
        "pre_prompt": "", "prompt_type": "simple", "chat_prompt_config": {}, "completion_prompt_config": {},
        "user_input_form": [], "dataset_query_variable": "", "opening_statement": "", "suggested_questions": [],
        "suggested_questions_after_answer": {"enabled": False}, "speech_to_text": {"enabled": False},
        "text_to_speech": {"enabled": False}, "retriever_resource": {"enabled": False},
        "sensitive_word_avoidance": {"enabled": False, "type": "", "configs": []},
        "more_like_this": {"enabled": False}, "agent_mode": {"enabled": False, "tools": []},
        "dataset_configs": {"retrieval_model": "multiple", "datasets": {"datasets": []}},
        "file_upload": {"image": {"enabled": False, "number_limits": 3, "detail": "high", "transfer_methods": ["remote_url"]}},
        "model": {"provider": PROVIDER, "name": model, "mode": "chat", "completion_params": params},
    }


def chat_checks(c: Console):
    app = c.call("POST", "/apps", json={"name": "zenmux-e2e", "mode": "chat", "icon_type": "emoji",
                                         "icon": "robot", "icon_background": "#FFEAD5"}).json()
    results = []
    for model, params in CHAT_CASES:
        r = c.call("POST", f"/apps/{app['id']}/chat-messages", json={
            "inputs": {}, "query": "What is 17*23? Reply with just the number.",
            "response_mode": "blocking", "model_config": model_config(model, params),
        }, expect_ok=False)
        body = r.json() if r.headers.get("content-type", "").startswith("application/json") else {"raw": r.text[:300]}
        usage = (body.get("metadata") or {}).get("usage") or {}
        ok = r.ok and "391" in (body.get("answer") or "") and float(usage.get("total_price") or 0) > 0
        results.append((model, ok, usage.get("prompt_tokens"), usage.get("completion_tokens"), usage.get("total_price"),
                        None if ok else str(body)[:300]))
    return results


def run_version(version: str, pkg: Path, key: str, port: int, keep: bool):
    d = prepare(version, port)
    log(f"\n=== Dify {version}: starting stack on :{port}")
    try:
        compose(version, d, "up", "-d", "api", "plugin_daemon")
        wait_http(f"http://127.0.0.1:{port}/health")
        c = Console(f"http://127.0.0.1:{port}")
        c.login()
        log("  install plugin ..."); install_plugin(c, pkg)
        log("  credentials ..."); credentials_flow(c, key)
        log("  model lists ..."); check_model_lists(c)
        log("  chats ...")
        results = chat_checks(c)
        for model, ok, pt, ct, price, err in results:
            log(f"    {'PASS' if ok else 'FAIL'} {model:32} tokens={pt}/{ct} price={price} {err or ''}")
        return all(r[1] for r in results)
    except Exception as exc:
        log(f"  FAIL: {exc}")
        compose(version, d, "logs", "--tail", "60", "api", "plugin_daemon", check=False)
        return False
    finally:
        if not keep:
            compose(version, d, "down", "-v", "--remove-orphans", check=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pkg", required=True, type=Path)
    ap.add_argument("--versions", nargs="*", default=DEFAULT_VERSIONS)
    ap.add_argument("--port", type=int, default=15001)
    ap.add_argument("--keep", action="store_true", help="leave the stack running for manual inspection")
    args = ap.parse_args()
    key = os.environ.get("ZENMUX_E2E_API_KEY") or sys.exit("set ZENMUX_E2E_API_KEY")
    summary = {v: run_version(v, args.pkg.resolve(), key, args.port, args.keep) for v in args.versions}
    log("\n=== summary\n" + "\n".join(f"  Dify {v}: {'PASS' if ok else 'FAIL'}" for v, ok in summary.items()))
    sys.exit(0 if all(summary.values()) else 1)


if __name__ == "__main__":
    main()
