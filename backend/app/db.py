from contextlib import contextmanager
from psycopg_pool import ConnectionPool
from pgvector.psycopg import register_vector
from .config import get_settings


def _url() -> str:
    return get_settings().database_url.replace("postgresql+psycopg://", "postgresql://")


pool = ConnectionPool(_url(), min_size=0, max_size=5, open=False, configure=register_vector)


def open_pool() -> None:
    pool.open()
    pool.wait()


def close_pool() -> None:
    pool.close()


@contextmanager
def connection():
    with pool.connection() as conn:
        yield conn

