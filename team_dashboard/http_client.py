"""HTTP helpers with robots.txt checks, random delays, and graceful fallbacks."""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import requests


DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0 Safari/537.36"
    ),
    "Accept": "text/html,application/json;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


@dataclass
class FetchResult:
    """A safe fetch result that never raises into callers."""

    ok: bool
    url: str
    status_code: int | None = None
    text: str | None = None
    json_data: Any = None
    error: str | None = None
    restricted: bool = False


@dataclass
class SafeHttpClient:
    """HTTP client that checks robots.txt and uses polite request delays."""

    delay_min: float = 1.5
    delay_max: float = 3.5
    timeout: int = 20
    verbose: bool = True
    _robots_cache: dict[str, RobotFileParser | None] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Create the underlying requests/cloudscraper session."""
        try:
            import cloudscraper  # type: ignore
            self.session = cloudscraper.create_scraper()
        except Exception:
            self.session = requests.Session()
        self.session.headers.update(DEFAULT_HEADERS)

    def sleep(self) -> None:
        """Sleep for a random polite delay."""
        time.sleep(random.uniform(self.delay_min, self.delay_max))

    def allowed(self, url: str) -> bool:
        """Return True when robots.txt allows fetching the URL."""
        parsed = urlparse(url)
        root = f"{parsed.scheme}://{parsed.netloc}"
        if root not in self._robots_cache:
            parser = RobotFileParser()
            robots_url = f"{root}/robots.txt"
            try:
                parser.set_url(robots_url)
                parser.read()
                self._robots_cache[root] = parser
            except Exception as exc:
                if self.verbose:
                    print(f"[robots] Could not read {robots_url}: {exc}")
                self._robots_cache[root] = None
        parser = self._robots_cache[root]
        if parser is None:
            return True
        try:
            return bool(parser.can_fetch(DEFAULT_HEADERS["User-Agent"], url))
        except Exception:
            return True

    def get(self, url: str, *, json_expected: bool = False) -> FetchResult:
        """Fetch a URL with robots checking and graceful error handling."""
        if not self.allowed(url):
            msg = f"robots.txt restricts endpoint: {url}"
            if self.verbose:
                print(f"[robots] {msg}")
            return FetchResult(False, url, restricted=True, error=msg)
        self.sleep()
        try:
            response = self.session.get(url, timeout=self.timeout)
            if response.status_code >= 400:
                return FetchResult(False, url, response.status_code, error=response.reason)
            if json_expected:
                try:
                    return FetchResult(True, url, response.status_code, json_data=response.json())
                except Exception as exc:
                    return FetchResult(False, url, response.status_code, response.text, error=str(exc))
            return FetchResult(True, url, response.status_code, text=response.text)
        except Exception as exc:
            return FetchResult(False, url, error=str(exc))

