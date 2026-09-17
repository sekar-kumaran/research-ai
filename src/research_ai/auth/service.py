from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
from datetime import datetime, timezone

from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from research_ai.database.models import SessionRecord, User


class AuthError(ValueError):
    pass


class AuthService:
    """Local password auth with PBKDF2 hashes and signed bearer tokens."""

    def __init__(self, secret_key: str | None = None) -> None:
        self.secret_key = (secret_key or os.getenv("SECRET_KEY") or "researchai-local-dev-secret").encode()

    def create_user(self, db: Session, email: str, username: str, password: str, full_name: str | None = None) -> User:
        email = email.strip().lower()
        username = username.strip()
        self._validate_password(password)
        exists = db.scalar(select(User).where(or_(User.email == email, User.username == username)))
        if exists:
            raise AuthError("Email or username is already registered.")
        user = User(email=email, username=username, password_hash=self.hash_password(password), full_name=full_name)
        db.add(user)
        db.commit()
        db.refresh(user)
        return user

    def authenticate(self, db: Session, login: str, password: str) -> tuple[User, str]:
        login = login.strip().lower()
        user = db.scalar(select(User).where(or_(User.email == login, User.username == login)))
        if not user or not user.is_active or not self.verify_password(password, user.password_hash):
            raise AuthError("Invalid login or password.")
        token = self.issue_token(user.id)
        db.add(SessionRecord(user_id=user.id, token_hash=self.token_hash(token)))
        db.commit()
        return user, token

    def user_from_token(self, db: Session, token: str) -> User | None:
        user_id = self.verify_token(token)
        if not user_id:
            return None
        token_hash = self.token_hash(token)
        session = db.scalar(
            select(SessionRecord).where(SessionRecord.token_hash == token_hash, SessionRecord.revoked_at.is_(None))
        )
        if not session:
            return None
        return db.get(User, user_id)

    def revoke(self, db: Session, token: str) -> None:
        session = db.scalar(select(SessionRecord).where(SessionRecord.token_hash == self.token_hash(token)))
        if session:
            session.revoked_at = datetime.now(timezone.utc)
            db.commit()

    @staticmethod
    def hash_password(password: str) -> str:
        salt = secrets.token_bytes(16)
        rounds = 260000
        digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, rounds)
        return f"pbkdf2_sha256${rounds}${base64.urlsafe_b64encode(salt).decode()}${base64.urlsafe_b64encode(digest).decode()}"

    @staticmethod
    def verify_password(password: str, encoded: str) -> bool:
        try:
            scheme, rounds_s, salt_s, digest_s = encoded.split("$", 3)
            if scheme != "pbkdf2_sha256":
                return False
            salt = base64.urlsafe_b64decode(salt_s.encode())
            expected = base64.urlsafe_b64decode(digest_s.encode())
            actual = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, int(rounds_s))
            return hmac.compare_digest(actual, expected)
        except Exception:
            return False

    def issue_token(self, user_id: str) -> str:
        payload = {
            "sub": user_id,
            "iat": int(datetime.now(timezone.utc).timestamp()),
            "nonce": secrets.token_urlsafe(12),
        }
        body = base64.urlsafe_b64encode(json.dumps(payload, separators=(",", ":")).encode()).rstrip(b"=").decode()
        sig = hmac.new(self.secret_key, body.encode(), hashlib.sha256).digest()
        return f"{body}.{base64.urlsafe_b64encode(sig).rstrip(b'=').decode()}"

    def verify_token(self, token: str) -> str | None:
        try:
            body, sig = token.split(".", 1)
            expected = hmac.new(self.secret_key, body.encode(), hashlib.sha256).digest()
            actual = base64.urlsafe_b64decode(sig + "=" * (-len(sig) % 4))
            if not hmac.compare_digest(actual, expected):
                return None
            payload = json.loads(base64.urlsafe_b64decode(body + "=" * (-len(body) % 4)))
            return str(payload.get("sub") or "") or None
        except Exception:
            return None

    @staticmethod
    def token_hash(token: str) -> str:
        return hashlib.sha256(token.encode()).hexdigest()

    @staticmethod
    def _validate_password(password: str) -> None:
        if len(password) < 8:
            raise AuthError("Password must be at least 8 characters.")
