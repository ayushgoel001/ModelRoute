from typing import Any

from sqlalchemy.engine import make_url
from sqlalchemy.exc import ArgumentError
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.db.models import Base


ASYNC_POSTGRESQL_SCHEME = "postgresql+asyncpg"
POSTGRESQL_SCHEMES = {"postgres", "postgresql", ASYNC_POSTGRESQL_SCHEME}
ASYNC_PG_SSL_MODES = {
    "allow",
    "disable",
    "prefer",
    "require",
    "verify-ca",
    "verify-full",
}
CHANNEL_BINDING_MODES = {"disable", "prefer", "require"}


def _single_query_value(value: Any, *, parameter: str) -> str | None:
    if value is None:
        return None
    if isinstance(value, tuple):
        if len(value) != 1:
            raise ValueError(f"Database URL parameter {parameter} must be singular")
        value = value[0]
    if not isinstance(value, str):
        raise ValueError(f"Database URL parameter {parameter} must be text")
    return value.casefold()


def prepare_database_connection(url: str) -> tuple[str, dict[str, str]]:
    """Adapt standard PostgreSQL URLs for SQLAlchemy's asyncpg dialect.

    SQLAlchemy parses URL query parameters before calling asyncpg. Consequently,
    asyncpg must receive its supported ``ssl`` keyword instead of the libpq-style
    ``sslmode`` query key. Neon may also include ``channel_binding``; asyncpg does
    not expose that keyword, so a recognized value is validated and removed.
    """
    try:
        parsed = make_url(url)
    except ArgumentError:
        raise ValueError("Invalid database URL") from None

    if parsed.drivername not in POSTGRESQL_SCHEMES:
        raise ValueError("Database URL must use a PostgreSQL scheme")

    parsed = parsed.set(drivername=ASYNC_POSTGRESQL_SCHEME)
    sslmode = _single_query_value(parsed.query.get("sslmode"), parameter="sslmode")
    ssl = _single_query_value(parsed.query.get("ssl"), parameter="ssl")
    channel_binding = _single_query_value(
        parsed.query.get("channel_binding"),
        parameter="channel_binding",
    )

    if sslmode is not None and ssl is not None:
        raise ValueError("Database URL cannot define both sslmode and ssl")
    if sslmode is not None and sslmode not in ASYNC_PG_SSL_MODES:
        raise ValueError("Database URL contains an unsupported SSL mode")
    if ssl is not None and ssl not in ASYNC_PG_SSL_MODES:
        raise ValueError("Database URL contains an unsupported SSL mode")
    if channel_binding is not None and channel_binding not in CHANNEL_BINDING_MODES:
        raise ValueError("Database URL contains an unsupported channel-binding mode")
    if channel_binding == "require" and sslmode is None and ssl is None:
        raise ValueError("channel_binding=require needs an explicit SSL mode")

    connect_args: dict[str, str] = {}
    removed_query_keys = ["channel_binding"]
    if sslmode is not None:
        connect_args["ssl"] = sslmode
        removed_query_keys.append("sslmode")

    parsed = parsed.difference_update_query(removed_query_keys)
    normalized_url = parsed.render_as_string(hide_password=False)
    return normalized_url, connect_args


class Database:
    """Owns one async engine and session factory for the application lifetime."""

    def __init__(self, url: str) -> None:
        normalized_url, connect_args = prepare_database_connection(url)
        engine_options: dict[str, Any] = {"pool_pre_ping": True}
        if connect_args:
            engine_options["connect_args"] = connect_args
        self.engine: AsyncEngine = create_async_engine(
            normalized_url,
            **engine_options,
        )
        self.session_factory = async_sessionmaker(
            self.engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )

    async def initialize(self) -> None:
        """Create missing tables without dropping or replacing existing schema."""
        async with self.engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)

    async def close(self) -> None:
        await self.engine.dispose()
