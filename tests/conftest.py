import pytest

from oaiprobe import checks
from oaiprobe.http import Client
from oaiprobe.mock import DEFAULT_KEY, DEFAULT_MODEL, MockServer


@pytest.fixture
def probe():
    """probe(quirks, check_ids) -> {check_id: Result} against a fresh mock."""
    servers = []

    def _probe(quirks=(), ids=None, key=DEFAULT_KEY, model=DEFAULT_MODEL):
        srv = MockServer(quirks).start()
        servers.append(srv)
        ctx = checks.Context(Client(srv.url, key, timeout=10), model)
        selected = checks.select(only=list(ids) if ids else None)
        return {r.id: r for r in checks.run(ctx, selected)}

    yield _probe
    for s in servers:
        s.stop()
