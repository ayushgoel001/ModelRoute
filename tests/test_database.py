import pytest
from sqlalchemy.engine import make_url

from app.db.database import prepare_database_connection


def test_asyncpg_database_url_remains_unchanged() -> None:
    supplied = "postgresql+asyncpg://user:fake-password@host.example/db"

    normalized, connect_args = prepare_database_connection(supplied)

    assert normalized == supplied
    assert connect_args == {}


@pytest.mark.parametrize("scheme", ["postgres", "postgresql"])
def test_standard_neon_url_is_normalized_for_asyncpg(scheme: str) -> None:
    supplied = (
        f"{scheme}://user:p%40ss@host.example:5432/db"
        "?sslmode=require&channel_binding=require&application_name=modelroute"
    )

    normalized, connect_args = prepare_database_connection(supplied)
    parsed = make_url(normalized)

    assert parsed.drivername == "postgresql+asyncpg"
    assert parsed.username == "user"
    assert parsed.password == "p@ss"
    assert parsed.host == "host.example"
    assert parsed.port == 5432
    assert parsed.database == "db"
    assert dict(parsed.query) == {"application_name": "modelroute"}
    assert connect_args == {"ssl": "require"}


def test_effective_asyncpg_arguments_exclude_unsupported_neon_keys() -> None:
    normalized, connect_args = prepare_database_connection(
        "postgresql://user:fake-password@host.example/db"
        "?sslmode=verify-full&channel_binding=require"
    )
    parsed = make_url(normalized)
    _, dialect_args = parsed.get_dialect()().create_connect_args(parsed)
    dialect_args.update(connect_args)

    assert dialect_args["ssl"] == "verify-full"
    assert "sslmode" not in dialect_args
    assert "channel_binding" not in dialect_args


def test_asyncpg_ssl_query_parameter_is_preserved() -> None:
    supplied = (
        "postgresql+asyncpg://user:fake-password@host.example/db?ssl=require"
    )

    normalized, connect_args = prepare_database_connection(supplied)

    assert normalized == supplied
    assert connect_args == {}


@pytest.mark.parametrize(
    "supplied",
    [
        "sqlite:///modelroute.db",
        "not-a-database-url",
        "postgresql://user:fake-password@host.example/db"
        "?ssl=require&sslmode=require",
        "postgresql://user:fake-password@host.example/db"
        "?channel_binding=require",
    ],
)
def test_invalid_or_incompatible_database_urls_are_rejected(supplied: str) -> None:
    with pytest.raises(ValueError):
        prepare_database_connection(supplied)
