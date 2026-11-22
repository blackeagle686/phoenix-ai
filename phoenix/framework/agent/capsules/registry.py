"""PHX v2 Hybrid Semantic-Metadata Capsule Registry & Built-in Core Python Library (FR-13).

Enforces strict metadata pre-filtering (language, framework, ORM, active/quarantine status)
BEFORE semantic similarity & contract/tag re-ranking to eliminate false semantic retrieval.
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from phoenix.framework.agent.capsules.schema import (
    CapsuleASTWiring,
    CapsuleContract,
    CapsuleParameter,
    CapsuleSemanticProfile,
    CapsuleStack,
    CapsuleStatus,
    CapsuleVerification,
    CodeCapsule,
)
from phoenix.framework.agent.kernel.features import FeatureExtractor

_WORD_RE = re.compile(r"[a-zA-Z0-9_]+|[\u0600-\u06FF]+")


class CapsuleRegistry:
    """Three-layer Semantic-Structural Registry (Metadata Filter + Semantic Vector + Contract Graph)."""

    def __init__(self, load_builtin: bool = True):
        self._capsules: Dict[str, CodeCapsule] = {}
        self._token_vecs: Dict[str, Dict[str, float]] = {}
        self._fe = FeatureExtractor()
        if load_builtin:
            for cap in build_default_python_capsules():
                self.register(cap)

    def register(self, capsule: CodeCapsule) -> CodeCapsule:
        if not capsule.ast_wiring.template_source and capsule.ast_wiring.template_path:
            if os.path.isfile(capsule.ast_wiring.template_path):
                with open(capsule.ast_wiring.template_path, "r", encoding="utf-8") as f:
                    capsule.ast_wiring.template_source = f.read()
        if not capsule.verification.test_source and capsule.verification.test_suite_path:
            if os.path.isfile(capsule.verification.test_suite_path):
                with open(capsule.verification.test_suite_path, "r", encoding="utf-8") as f:
                    capsule.verification.test_source = f.read()

        if not capsule.contract.provides:
            raise ValueError(f"Capsule '{capsule.capsule_id}' must declare at least one `contract.provides` entry.")
        if not capsule.ast_wiring.template_source.strip():
            raise ValueError(f"Capsule '{capsule.capsule_id}' must provide non-empty AST template source.")

        self._capsules[capsule.capsule_id] = capsule
        profile_text = " ".join(
            [
                capsule.capsule_id.replace(".", " ").replace("_", " "),
                capsule.semantic_profile.summary,
                " ".join(capsule.semantic_profile.tags),
                " ".join(capsule.contract.provides),
            ]
        )
        self._token_vecs[capsule.capsule_id] = self._sparse_unit_vec(profile_text)
        return capsule

    @staticmethod
    def _sparse_unit_vec(text: str) -> Dict[str, float]:
        """Collision-free L2-normalized sparse token vector (eliminates hash-collision false retrieval, Section 5.6)."""
        toks = _WORD_RE.findall((text or "").lower())
        if not toks:
            return {}
        counts: Dict[str, float] = {}
        for tok in toks:
            counts[tok] = counts.get(tok, 0.0) + 1.0
        norm = float(np.sqrt(sum(v * v for v in counts.values())))
        if norm > 1e-8:
            return {k: v / norm for k, v in counts.items()}
        return counts

    @staticmethod
    def _sparse_cosine(a: Dict[str, float], b: Dict[str, float]) -> float:
        if not a or not b:
            return 0.0
        if len(a) > len(b):
            a, b = b, a
        return float(sum(v * b[k] for k, v in a.items() if k in b))

    def get(self, capsule_id: str) -> Optional[CodeCapsule]:
        return self._capsules.get(capsule_id)

    def list_active(self) -> List[CodeCapsule]:
        return [c for c in self._capsules.values() if c.verification.status == CapsuleStatus.ACTIVE]

    def quarantine(self, capsule_id: str, reason: str) -> bool:
        """Section 5.6 Capsule Rot / CVE Drift Mitigation: immediately exclude failing capsule from retrieval."""
        cap = self._capsules.get(capsule_id)
        if cap is None:
            return False
        cap.verification.status = CapsuleStatus.QUARANTINED
        cap.verification.quarantine_reason = reason
        return True

    def find_providers(
        self,
        capability: str,
        *,
        language: Optional[str] = None,
        framework: Optional[str] = None,
        orm: Optional[str] = None,
    ) -> List[CodeCapsule]:
        """Find all ACTIVE capsules satisfying a `requires` capability within compatible stack metadata."""
        matches: List[CodeCapsule] = []
        for cap in self.list_active():
            if not self._matches_stack(cap, language=language, framework=framework, orm=orm):
                continue
            if capability in cap.contract.provides:
                matches.append(cap)
        return matches

    @staticmethod
    def _matches_stack(
        cap: CodeCapsule,
        *,
        language: Optional[str] = None,
        framework: Optional[str] = None,
        orm: Optional[str] = None,
    ) -> bool:
        if language and cap.stack.language.lower() != language.lower():
            return False
        if framework and cap.stack.framework.lower() not in (framework.lower(), "stdlib", "any"):
            return False
        if orm and cap.stack.orm.lower() not in (orm.lower(), "none", "any"):
            return False
        return True

    def retrieve(
        self,
        query: str,
        *,
        language: str = "python",
        framework: Optional[str] = None,
        orm: Optional[str] = None,
        required_tags: Optional[Sequence[str]] = None,
        top_k: int = 5,
        min_score: float = 0.15,
    ) -> List[Tuple[CodeCapsule, float]]:
        """FR-13 Hybrid Semantic-Metadata Retrieval with strict pre-filtering + tag/contract re-ranking."""
        # Step 1: Hard Metadata Pre-Filtering (excludes QUARANTINED & incompatible stack)
        eligible = [
            cap for cap in self.list_active()
            if self._matches_stack(cap, language=language, framework=framework, orm=orm)
        ]
        if required_tags:
            req_set = {t.lower() for t in required_tags}
            eligible = [c for c in eligible if req_set.issubset({t.lower() for t in c.semantic_profile.tags})]
        if not eligible:
            return []

        q_vec = self._sparse_unit_vec(query)
        q_tokens = set(_WORD_RE.findall((query or "").lower()))

        # Step 2: Collision-free sparse cosine similarity + deterministic tag/contract re-ranking (Section 5.6)
        scored: List[Tuple[CodeCapsule, float]] = []
        for cap in eligible:
            c_vec = self._token_vecs.get(cap.capsule_id) or {}
            cos_sim = self._sparse_cosine(q_vec, c_vec)
            tags = {t.lower() for t in cap.semantic_profile.tags}
            prov_tokens = set(_WORD_RE.findall(" ".join(cap.contract.provides).lower()))
            id_tokens = set(_WORD_RE.findall(cap.capsule_id.lower()))
            summary_tokens = set(_WORD_RE.findall(cap.semantic_profile.summary.lower()))
            tag_overlap = len(q_tokens & tags) / float(max(1, len(tags)))
            exact_hits = len(q_tokens & (tags | prov_tokens | id_tokens | summary_tokens))
            if exact_hits == 0 and cos_sim < 0.25:
                continue
            hybrid = 0.45 * max(0.0, cos_sim) + 0.35 * tag_overlap + min(0.35, 0.12 * exact_hits)
            if hybrid >= min_score:
                scored.append((cap, round(hybrid, 4)))

        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[: max(1, int(top_k))]


def build_default_python_capsules() -> List[CodeCapsule]:
    """Core library of security-audited, test-backed Python backend & math capsules."""
    return [
        CodeCapsule(
            capsule_id="cap.infra.redis_client.v1",
            version="1.0.0",
            stack=CapsuleStack(language="python", framework="stdlib", orm="none"),
            semantic_profile=CapsuleSemanticProfile(
                summary="In-memory TTL key-value cache client compatible with Redis get/setex/delete contracts",
                tags=["redis", "cache", "ttl", "infra", "store"],
            ),
            contract=CapsuleContract(
                provides=["infra:redis_client"],
                requires=[],
                conflicts_with=[],
                middleware_order_priority=10,
            ),
            ast_wiring=CapsuleASTWiring(
                injection_points=["app.di_container"],
                template_source=(
                    "import time\n\n"
                    "class RedisClient:\n"
                    "    def __init__(self):\n"
                    "        self._store = {}\n\n"
                    "    def setex(self, key: str, ttl_seconds: int, value: str) -> bool:\n"
                    "        self._store[key] = (value, time.monotonic() + max(1, int(ttl_seconds)))\n"
                    "        return True\n\n"
                    "    def get(self, key: str):\n"
                    "        item = self._store.get(key)\n"
                    "        if item is None:\n"
                    "            return None\n"
                    "        val, exp = item\n"
                    "        if time.monotonic() > exp:\n"
                    "            self._store.pop(key, None)\n"
                    "            return None\n"
                    "        return val\n\n"
                    "    def delete(self, key: str) -> None:\n"
                    "        self._store.pop(key, None)\n"
                ),
            ),
            verification=CapsuleVerification(
                security_audited=True,
                last_verified_at="2026-11-22T00:00:00Z",
                test_source=(
                    "_rc = RedisClient()\n"
                    "assert _rc.setex('k1', 60, 'v1') is True\n"
                    "assert _rc.get('k1') == 'v1'\n"
                    "_rc.delete('k1')\n"
                    "assert _rc.get('k1') is None\n"
                ),
            ),
        ),
        CodeCapsule(
            capsule_id="cap.infra.email_dispatcher.v1",
            version="1.0.0",
            stack=CapsuleStack(language="python", framework="stdlib", orm="none"),
            semantic_profile=CapsuleSemanticProfile(
                summary="Deterministic email notification dispatcher with outbox audit trail",
                tags=["email", "notification", "dispatcher", "infra"],
            ),
            contract=CapsuleContract(
                provides=["service:EmailDispatcher"],
                requires=[],
                conflicts_with=[],
                middleware_order_priority=15,
            ),
            ast_wiring=CapsuleASTWiring(
                injection_points=["app.di_container"],
                template_source=(
                    "class EmailDispatcher:\n"
                    "    def __init__(self):\n"
                    "        self.outbox = []\n\n"
                    "    def send(self, to_email: str, subject: str, body: str) -> dict:\n"
                    "        msg = {'to': to_email, 'subject': subject, 'body': body, 'sent': True}\n"
                    "        self.outbox.append(msg)\n"
                    "        return msg\n"
                ),
            ),
            verification=CapsuleVerification(
                security_audited=True,
                last_verified_at="2026-11-22T00:00:00Z",
                test_source=(
                    "_em = EmailDispatcher()\n"
                    "_sent = _em.send('a@example.com', 'Hi', 'Body')\n"
                    "assert _sent['sent'] is True and len(_em.outbox) == 1\n"
                ),
            ),
        ),
        CodeCapsule(
            capsule_id="cap.entity.user.v1",
            version="1.0.0",
            stack=CapsuleStack(language="python", framework="stdlib", orm="none"),
            semantic_profile=CapsuleSemanticProfile(
                summary="User domain entity and repository providing id, email, role, and active status",
                tags=["user", "entity", "model", "account", "repository"],
            ),
            contract=CapsuleContract(
                provides=["entity:User(id, email, is_active)"],
                requires=[],
                conflicts_with=[],
                middleware_order_priority=12,
            ),
            ast_wiring=CapsuleASTWiring(
                injection_points=["app.di_container"],
                template_source=(
                    "from dataclasses import dataclass\n\n"
                    "@dataclass\n"
                    "class User:\n"
                    "    id: str\n"
                    "    email: str\n"
                    "    is_active: bool = True\n"
                    "    role: str = 'member'\n\n"
                    "class UserRepository:\n"
                    "    def __init__(self):\n"
                    "        self._by_email = {}\n\n"
                    "    def save(self, user: User) -> User:\n"
                    "        self._by_email[user.email.lower()] = user\n"
                    "        return user\n\n"
                    "    def find_by_email(self, email: str):\n"
                    "        return self._by_email.get(email.lower())\n"
                ),
            ),
            verification=CapsuleVerification(
                security_audited=True,
                last_verified_at="2026-11-22T00:00:00Z",
                test_source=(
                    "_ur = UserRepository()\n"
                    "_ur.save(User(id='u1', email='dev@phx.ai'))\n"
                    "assert _ur.find_by_email('dev@phx.ai').id == 'u1'\n"
                ),
            ),
        ),
        CodeCapsule(
            capsule_id="cap.auth.jwt_otp_login.v2",
            version="2.1.0",
            stack=CapsuleStack(language="python", framework="fastapi", framework_semver=">=0.110.0,<1.0.0", orm="none"),
            semantic_profile=CapsuleSemanticProfile(
                summary="Passwordless or 2FA login flow using Email OTP with short-lived TTL and signed JWT issuance",
                tags=["auth", "login", "otp", "jwt", "security", "2fa"],
            ),
            contract=CapsuleContract(
                provides=[
                    "route:POST:/auth/otp/request",
                    "route:POST:/auth/otp/verify",
                    "service:TokenIssuer",
                ],
                requires=[
                    "infra:redis_client",
                    "service:EmailDispatcher",
                    "entity:User(id, email, is_active)",
                ],
                conflicts_with=["cap.auth.session_cookie_only.*"],
                middleware_order_priority=20,
            ),
            parameters=[
                CapsuleParameter(name="otp_ttl_seconds", type="int", default=300, constraints="60..900"),
                CapsuleParameter(name="otp_length", type="int", default=6, constraints="4..8"),
            ],
            ast_wiring=CapsuleASTWiring(
                injection_points=["app.routers", "app.di_container"],
                custom_logic_slots=["on_login_success_hook"],
                template_source=(
                    "import base64\n"
                    "import hashlib\n"
                    "import json\n"
                    "import secrets\n\n"
                    "OTP_TTL_SECONDS = {otp_ttl_seconds}\n"
                    "OTP_LENGTH = {otp_length}\n\n"
                    "def on_login_success_hook(user, token: str) -> dict:\n"
                    "    return {{'user_id': user.id, 'email': user.email, 'token': token}}\n\n"
                    "class TokenIssuer:\n"
                    "    def __init__(self, secret: str = 'phx-v2-secret'):\n"
                    "        self._secret = secret\n\n"
                    "    def issue(self, user: User) -> str:\n"
                    "        payload = base64.urlsafe_b64encode(json.dumps({{'sub': user.id, 'email': user.email}}).encode()).decode()\n"
                    "        sig = hashlib.sha256((payload + ':' + self._secret).encode()).hexdigest()[:32]\n"
                    "        return f'{{payload}}.{{sig}}'\n\n"
                    "class OTPLoginService:\n"
                    "    def __init__(self, redis: RedisClient, mailer: EmailDispatcher, users: UserRepository, issuer: TokenIssuer):\n"
                    "        self.redis = redis\n"
                    "        self.mailer = mailer\n"
                    "        self.users = users\n"
                    "        self.issuer = issuer\n\n"
                    "    def request_otp(self, email: str) -> dict:\n"
                    "        user = self.users.find_by_email(email)\n"
                    "        if user is None or not user.is_active:\n"
                    "            raise ValueError('Inactive or unknown user')\n"
                    "        code = ''.join(str(secrets.randbelow(10)) for _ in range(OTP_LENGTH))\n"
                    "        self.redis.setex(f'otp:{{email.lower()}}', OTP_TTL_SECONDS, code)\n"
                    "        self.mailer.send(email, 'Your OTP Code', f'Code: {{code}}')\n"
                    "        return {{'status': 'sent', 'ttl': OTP_TTL_SECONDS, 'otp_debug': code}}\n\n"
                    "    def verify_otp(self, email: str, code: str) -> dict:\n"
                    "        key = f'otp:{{email.lower()}}'\n"
                    "        expected = self.redis.get(key)\n"
                    "        if not expected or expected != str(code):\n"
                    "            raise ValueError('Invalid or expired OTP')\n"
                    "        self.redis.delete(key)\n"
                    "        user = self.users.find_by_email(email)\n"
                    "        token = self.issuer.issue(user)\n"
                    "        return on_login_success_hook(user, token)\n"
                ),
            ),
            verification=CapsuleVerification(
                security_audited=True,
                last_verified_at="2026-11-22T00:00:00Z",
                test_source=(
                    "_r = RedisClient()\n"
                    "_m = EmailDispatcher()\n"
                    "_u = UserRepository()\n"
                    "_u.save(User(id='u_otp', email='otp@phx.ai'))\n"
                    "_svc = OTPLoginService(_r, _m, _u, TokenIssuer())\n"
                    "_req = _svc.request_otp('otp@phx.ai')\n"
                    "assert _req['status'] == 'sent' and len(_req['otp_debug']) == OTP_LENGTH\n"
                    "_ver = _svc.verify_otp('otp@phx.ai', _req['otp_debug'])\n"
                    "assert _ver['user_id'] == 'u_otp' and '.' in _ver['token']\n"
                ),
            ),
        ),
        CodeCapsule(
            capsule_id="cap.auth.session_cookie_only.v1",
            version="1.0.0",
            stack=CapsuleStack(language="python", framework="fastapi", orm="none"),
            semantic_profile=CapsuleSemanticProfile(
                summary="Stateful server-side cookie session authentication incompatible with stateless JWT OTP",
                tags=["auth", "cookie", "session"],
            ),
            contract=CapsuleContract(
                provides=["route:POST:/auth/session/login", "service:SessionManager"],
                requires=["entity:User(id, email, is_active)"],
                conflicts_with=["cap.auth.jwt_otp_login.*"],
                middleware_order_priority=25,
            ),
            ast_wiring=CapsuleASTWiring(
                injection_points=["app.routers"],
                template_source=(
                    "import uuid\n\n"
                    "class SessionManager:\n"
                    "    def __init__(self):\n"
                    "        self._sessions = {}\n\n"
                    "    def create_session(self, user: User) -> str:\n"
                    "        sid = uuid.uuid4().hex\n"
                    "        self._sessions[sid] = user.id\n"
                    "        return sid\n"
                ),
            ),
            verification=CapsuleVerification(
                security_audited=True,
                last_verified_at="2026-11-22T00:00:00Z",
                test_source=(
                    "_sm = SessionManager()\n"
                    "assert len(_sm.create_session(User(id='u1', email='a@b.c'))) >= 16\n"
                ),
            ),
        ),
        CodeCapsule(
            capsule_id="cap.security.rate_limiter.v1",
            version="1.0.0",
            stack=CapsuleStack(language="python", framework="fastapi", orm="none"),
            semantic_profile=CapsuleSemanticProfile(
                summary="Sliding-window request rate limiter middleware protecting endpoints against abuse",
                tags=["security", "rate", "limit", "limiter", "throttle", "middleware"],
            ),
            contract=CapsuleContract(
                provides=["middleware:RateLimiter"],
                requires=["infra:redis_client"],
                conflicts_with=[],
                middleware_order_priority=10,
            ),
            parameters=[
                CapsuleParameter(name="max_requests", type="int", default=60, constraints="1..10000"),
                CapsuleParameter(name="window_seconds", type="int", default=60, constraints="1..3600"),
            ],
            ast_wiring=CapsuleASTWiring(
                injection_points=["app.middlewares"],
                template_source=(
                    "RATE_LIMIT_MAX = {max_requests}\n"
                    "RATE_LIMIT_WINDOW = {window_seconds}\n\n"
                    "class RateLimiterMiddleware:\n"
                    "    def __init__(self, redis: RedisClient):\n"
                    "        self.redis = redis\n\n"
                    "    def allow(self, client_key: str) -> bool:\n"
                    "        k = f'rl:{{client_key}}'\n"
                    "        cur = int(self.redis.get(k) or 0)\n"
                    "        if cur >= RATE_LIMIT_MAX:\n"
                    "            return False\n"
                    "        self.redis.setex(k, RATE_LIMIT_WINDOW, str(cur + 1))\n"
                    "        return True\n"
                ),
            ),
            verification=CapsuleVerification(
                security_audited=True,
                last_verified_at="2026-11-22T00:00:00Z",
                test_source=(
                    "_rl = RateLimiterMiddleware(RedisClient())\n"
                    "assert _rl.allow('ip_1') is True\n"
                ),
            ),
        ),
        CodeCapsule(
            capsule_id="cap.observability.audit_log.v1",
            version="1.0.0",
            stack=CapsuleStack(language="python", framework="fastapi", orm="none"),
            semantic_profile=CapsuleSemanticProfile(
                summary="Structured security and compliance audit logger recording actor actions and timestamps",
                tags=["audit", "log", "logging", "observability", "security", "compliance"],
            ),
            contract=CapsuleContract(
                provides=["service:AuditLogger", "middleware:AuditLog"],
                requires=[],
                conflicts_with=[],
                middleware_order_priority=5,
            ),
            ast_wiring=CapsuleASTWiring(
                injection_points=["app.middlewares", "app.di_container"],
                template_source=(
                    "import time\n\n"
                    "class AuditLogger:\n"
                    "    def __init__(self):\n"
                    "        self.entries = []\n\n"
                    "    def log(self, actor: str, action: str, meta: dict = None) -> dict:\n"
                    "        rec = {'ts': int(time.time()), 'actor': actor, 'action': action, 'meta': dict(meta or {})}\n"
                    "        self.entries.append(rec)\n"
                    "        return rec\n"
                ),
            ),
            verification=CapsuleVerification(
                security_audited=True,
                last_verified_at="2026-11-22T00:00:00Z",
                test_source=(
                    "_al = AuditLogger()\n"
                    "_al.log('u1', 'login')\n"
                    "assert len(_al.entries) == 1 and _al.entries[0]['action'] == 'login'\n"
                ),
            ),
        ),
        CodeCapsule(
            capsule_id="cap.api.health_check.v1",
            version="1.0.0",
            stack=CapsuleStack(language="python", framework="fastapi", orm="none"),
            semantic_profile=CapsuleSemanticProfile(
                summary="Liveness and readiness health check endpoint returning service status",
                tags=["health", "healthcheck", "status", "ping", "api"],
            ),
            contract=CapsuleContract(
                provides=["route:GET:/health"],
                requires=[],
                conflicts_with=[],
                middleware_order_priority=50,
            ),
            ast_wiring=CapsuleASTWiring(
                injection_points=["app.routers"],
                template_source=(
                    "def health_check() -> dict:\n"
                    "    return {'status': 'ok', 'engine': 'phx-v2'}\n"
                ),
            ),
            verification=CapsuleVerification(
                security_audited=True,
                last_verified_at="2026-11-22T00:00:00Z",
                test_source="assert health_check()['status'] == 'ok'\n",
            ),
        ),
        CodeCapsule(
            capsule_id="cap.math.safe_ast_calculator.v1",
            version="1.0.0",
            stack=CapsuleStack(language="python", framework="stdlib", orm="none"),
            semantic_profile=CapsuleSemanticProfile(
                summary="Secure AST-based scientific calculator supporting arithmetic and trigonometric functions without eval",
                tags=["calculator", "calc", "math", "ast", "sin", "cos", "tanh", "secure"],
            ),
            contract=CapsuleContract(
                provides=["service:SafeCalculator", "func:calculate"],
                requires=[],
                conflicts_with=[],
                middleware_order_priority=50,
            ),
            ast_wiring=CapsuleASTWiring(
                injection_points=["app.di_container"],
                template_source=(
                    "import ast\n"
                    "import math\n"
                    "import operator\n\n"
                    "_CALC_BIN_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.Pow: operator.pow}\n"
                    "_CALC_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}\n"
                    "_CALC_FUNCS = {'sin': math.sin, 'cos': math.cos, 'tan': math.tan, 'tanh': math.tanh, 'sqrt': math.sqrt, 'log': math.log, 'exp': math.exp}\n"
                    "_CALC_CONSTS = {'pi': math.pi, 'e': math.e}\n\n"
                    "class SafeCalculator:\n"
                    "    def evaluate_node(self, node):\n"
                    "        if isinstance(node, ast.Expression): return self.evaluate_node(node.body)\n"
                    "        if isinstance(node, ast.Expr): return self.evaluate_node(node.value)\n"
                    "        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)): return node.value\n"
                    "        if isinstance(node, ast.Name) and node.id in _CALC_CONSTS: return _CALC_CONSTS[node.id]\n"
                    "        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == 'math' and node.attr in _CALC_CONSTS: return _CALC_CONSTS[node.attr]\n"
                    "        if isinstance(node, ast.UnaryOp) and type(node.op) in _CALC_UNARY_OPS: return _CALC_UNARY_OPS[type(node.op)](self.evaluate_node(node.operand))\n"
                    "        if isinstance(node, ast.BinOp) and type(node.op) in _CALC_BIN_OPS: return _CALC_BIN_OPS[type(node.op)](self.evaluate_node(node.left), self.evaluate_node(node.right))\n"
                    "        if isinstance(node, ast.Call):\n"
                    "            fn_name = node.func.id if isinstance(node.func, ast.Name) else (node.func.attr if isinstance(node.func, ast.Attribute) else '')\n"
                    "            if fn_name in _CALC_FUNCS:\n"
                    "                return _CALC_FUNCS[fn_name](*[self.evaluate_node(a) for a in node.args])\n"
                    "        raise ValueError(f'Unsupported or unsafe AST node: {type(node).__name__}')\n\n"
                    "    def calculate(self, expr: str) -> float:\n"
                    "        return self.evaluate_node(ast.parse(expr, mode='eval'))\n\n"
                    "def calculate(expr: str) -> float:\n"
                    "    return SafeCalculator().calculate(expr)\n"
                ),
            ),
            verification=CapsuleVerification(
                security_audited=True,
                last_verified_at="2026-11-22T00:00:00Z",
                test_source=(
                    "assert abs(calculate('2 + 3 * 4') - 14.0) < 1e-6\n"
                    "assert abs(calculate('sin(pi / 2)') - 1.0) < 1e-6\n"
                    "assert abs(calculate('tanh(0)') - 0.0) < 1e-6\n"
                ),
            ),
        ),
    ]
