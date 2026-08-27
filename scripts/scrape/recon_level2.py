"""Level-2 recon: follow submenu links, map every search interface.

Level-1 found that SearchMenu.aspx / GazetteDirectory.aspx exist. This
script drills into those pages to answer the remaining unknowns:

  - What search options exist (by date, ministry, gazette number, etc.)?
  - What form fields does each search require?
  - Where do submit actions post to?
  - What's the reasonable enumeration path (which search lets us walk
    ALL gazettes vs a filtered subset)?

Polite defaults: max 20 requests total, 2s delay each, no form submissions
(GETs only). We're mapping the site, not scraping it yet.
"""
from __future__ import annotations

import re
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
import urllib3
from bs4 import BeautifulSoup

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

BASE = "https://egazette.gov.in/"
UA = "Mozilla/5.0 EnclaveScribe-Recon/0.2"
POLITENESS_SEC = 2.0
MAX_REQUESTS = 20


class Recon2:
    def __init__(self):
        self.s = requests.Session()
        self.s.headers.update({"User-Agent": UA})
        self.s.verify = False
        self.n = 0
        self.out: list[str] = []
        self.forms_found: list[dict] = []

    def _log(self, msg: str) -> None:
        self.out.append(msg)

    def _get(self, url: str) -> requests.Response | None:
        if self.n >= MAX_REQUESTS:
            self._log(f"[STOP] hit max {MAX_REQUESTS} requests")
            return None
        try:
            t0 = time.time()
            r = self.s.get(url, timeout=30, allow_redirects=True)
            dt = time.time() - t0
            self.n += 1
            self._log(f"  GET {url}\n    -> {r.status_code}, {len(r.content):,}B, {dt:.1f}s (final={r.url})")
            time.sleep(POLITENESS_SEC)
            return r
        except Exception as e:
            self.n += 1
            self._log(f"  GET {url}\n    -> ERROR: {type(e).__name__}: {e}")
            return None

    def _describe_form(self, form, base_url: str) -> dict:
        action = urljoin(base_url, form.get("action", ""))
        method = form.get("method", "GET").upper()
        inputs = []
        for i in form.find_all(["input", "select", "textarea"]):
            inputs.append({
                "name":  i.get("name", ""),
                "type":  i.get("type", i.name),
                "value": i.get("value", "")[:80] if i.get("value") else "",
                "id":    i.get("id", ""),
            })
        return {"action": action, "method": method, "inputs": inputs}

    def analyze_page(self, url: str) -> dict:
        """Fetch a page, describe its forms and links, return a summary."""
        self._log(f"\n### Analyzing: {url}")
        r = self._get(url)
        if r is None:
            return {}

        soup = BeautifulSoup(r.text, "html.parser")

        # Forms
        forms_here = []
        for f in soup.find_all("form"):
            desc = self._describe_form(f, r.url)
            forms_here.append(desc)

        # Non-hidden inputs are the "real" search fields
        real_forms = []
        for f in forms_here:
            real_inputs = [
                i for i in f["inputs"]
                if i["type"] not in ("hidden",) and i["name"] not in ("", None)
            ]
            if real_inputs:
                real_forms.append({**f, "real_inputs": real_inputs})

        self._log(f"    forms: {len(forms_here)} total, {len(real_forms)} with visible inputs")
        for i, f in enumerate(real_forms):
            self._log(f"    form #{i}: {f['method']} → {f['action']}")
            for inp in f["real_inputs"][:12]:
                self._log(f"      {inp['type']:10s} name={inp['name']!r:30s} id={inp['id']!r}")

        # Links
        links = []
        for a in soup.find_all("a", href=True):
            url_abs = urljoin(r.url, a["href"])
            if urlparse(url_abs).netloc == urlparse(BASE).netloc:
                text = " ".join(a.get_text().split())[:60]
                links.append((url_abs, text))
        self._log(f"    internal links: {len(links)}")

        # Interesting links (ones that might lead to more forms or PDFs)
        keywords = re.compile(r"search|find|date|ministry|dept|gazette|browse|list|extraord|weekly", re.I)
        interesting = [(u, t) for u, t in links if keywords.search(u + " " + t)]
        for u, t in interesting[:8]:
            self._log(f"      → {t!r:<40s} {u}")

        self.forms_found.extend(real_forms)
        return {
            "url": r.url,
            "n_forms": len(forms_here),
            "n_real_forms": len(real_forms),
            "n_links": len(links),
            "interesting_links": interesting,
            "forms": real_forms,
        }

    def run(self) -> str:
        self._log(f"# egazette.gov.in Level-2 recon")
        self._log(f"Base: {BASE}\nMax requests: {MAX_REQUESTS}, delay: {POLITENESS_SEC}s")

        # Step 1: fetch homepage to get a fresh session-embedded URL
        home = self._get(BASE)
        if home is None:
            return "\n".join(self.out)
        session_prefix = home.url.rstrip("/") + "/"
        # Extract session token like (S(xxx))/
        m = re.search(r"/\(S\(([^)]+)\)\)/", home.url)
        sess = f"(S({m.group(1)}))" if m else ""
        self._log(f"Session token: {sess}")

        # Step 2: analyze the main menu pages
        candidates = []
        for name in ("SearchMenu.aspx", "GazetteDirectory.aspx", "Default.aspx"):
            candidates.append(f"{BASE}{sess}/{name}" if sess else urljoin(BASE, name))

        for url in candidates:
            summary = self.analyze_page(url)
            # Follow the top 1-2 "interesting" links from each menu page,
            # respecting the request budget
            for sub_url, _ in (summary.get("interesting_links") or [])[:2]:
                if self.n >= MAX_REQUESTS - 1:
                    break
                self.analyze_page(sub_url)

        # Summary
        self._log(f"\n## Summary")
        self._log(f"- Requests used: {self.n} / {MAX_REQUESTS}")
        self._log(f"- Total forms with visible inputs: {len(self.forms_found)}")
        actions_seen = sorted({f["action"] for f in self.forms_found})
        for a in actions_seen:
            self._log(f"    action target: {a}")

        return "\n".join(self.out)


def main():
    r = Recon2()
    report = r.run()
    print(report)
    Path("recon_level2_report.md").write_text(report, encoding="utf-8")
    print(f"\nSaved → recon_level2_report.md")


if __name__ == "__main__":
    main()
