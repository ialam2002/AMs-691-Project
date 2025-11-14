from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests


class WikipediaConnector:
    """
    Thin wrapper around the Wikipedia REST + MediaWiki APIs with on-disk caching.
    """

    SEARCH_URL = "https://en.wikipedia.org/w/rest.php/v1/search/page"
    PAGE_URL = "https://en.wikipedia.org/w/api.php"

    def __init__(self, cache_dir: Path, timeout: int = 10, max_retries: int = 3) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = requests.Session()

    def search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        if not query.strip():
            return []
        params = {"q": query, "limit": limit}
        try:
            resp = self.session.get(self.SEARCH_URL, params=params, timeout=self.timeout)
            resp.raise_for_status()
        except requests.RequestException:
            return []
        payload = resp.json() or {}
        return payload.get("pages", [])

    def fetch_page(self, title: str, force: bool = False) -> Optional[Dict[str, Any]]:
        if not title:
            return None
        cache_path = self.cache_dir / f"{self._slug(title)}.json"
        if cache_path.exists() and not force:
            with cache_path.open("r", encoding="utf-8") as fp:
                return json.load(fp)
        params = {
            "action": "query",
            "prop": "extracts",
            "explaintext": 1,
            "format": "json",
            "titles": title,
        }
        for _ in range(self.max_retries):
            try:
                resp = self.session.get(self.PAGE_URL, params=params, timeout=self.timeout)
                resp.raise_for_status()
            except requests.RequestException:
                continue
            payload = resp.json() or {}
            pages = payload.get("query", {}).get("pages", {})
            if not pages:
                continue
            page_data = next(iter(pages.values()))
            extract = page_data.get("extract", "")
            record = {
                "title": page_data.get("title", title),
                "pageid": page_data.get("pageid"),
                "extract": extract,
            }
            with cache_path.open("w", encoding="utf-8") as fp:
                json.dump(record, fp, ensure_ascii=False)
            return record
        return None

    def _slug(self, title: str) -> str:
        slug = re.sub(r"[^A-Za-z0-9]+", "_", title).strip("_")
        return slug.lower() or "page"

