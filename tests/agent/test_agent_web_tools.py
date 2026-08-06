from herd.services.agent import agent_fetch_url, agent_web_search


def test_agent_web_search(monkeypatch):
    """Verifies that agent_web_search parses search results from HTML response."""
    dummy_html = """
    <html>
      <body>
        <a class="result__a" href="https://example.com/fastapi">FastAPI Documentation</a>
        <a class="result__url" href="https://example.com/fastapi">example.com/fastapi</a>
        <a class="result__snippet">FastAPI framework, high performance, easy to learn.</a>
      </body>
    </html>
    """

    class DummyResponse:
        status_code = 200
        text = dummy_html

    monkeypatch.setattr("httpx.get", lambda url, **kwargs: DummyResponse())

    res = agent_web_search("fastapi documentation")
    assert "FastAPI Documentation" in res
    assert "https://example.com/fastapi" in res
    assert "Snippet: FastAPI framework" in res


def test_agent_fetch_url(monkeypatch):
    """Verifies that agent_fetch_url strips HTML tags and extracts text content."""
    dummy_page_html = """
    <html>
      <head><style>body { color: red; }</style></head>
      <body>
        <nav>Nav links</nav>
        <h1>Main Page Title</h1>
        <p>This is page body text explaining how the library works.</p>
        <footer>Footer content</footer>
      </body>
    </html>
    """

    class DummyResponse:
        status_code = 200
        text = dummy_page_html

    monkeypatch.setattr("httpx.get", lambda url, **kwargs: DummyResponse())

    res = agent_fetch_url("https://example.com/docs")
    assert "Main Page Title" in res
    assert "This is page body text explaining how the library works." in res
    assert "Nav links" not in res
    assert "Footer content" not in res
