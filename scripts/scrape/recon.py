"""Reconnaissance script for egazette.gov.in.

Probes the central eGazette portal and prints a technical report:
  - robots.txt policy
  - Homepage structure (links, forms, sessions)
  - Search / category pages found
  - Example PDF URL patterns
  - Session / cookie behavior
  - Rate-limit signals (headers, response times)

Output goes to stdout AND is saved as recon_report.md so we can iterate
on the scraper design without re-hitting the site.

Runs locally, no compute, no cost. Deliberately polite:
  - Max 20 requests total
  - 2s delay between requests
  - No JavaScript execution
  - No file downloads (HEAD only for PDFs)
"""
from __future__ import annotations

import json
import re
import ssl
import sys
import time
import urllib.parse
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
import urllib3
from bs4 import BeautifulSoup

# Their SSL cert is expired/self-signed — suppress warnings, don't verify
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

BASE = "https://egazette.gov.in/"
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) AppleWebKit/537.36 EnclaveScribe-Recon/0.1"
POLITENESS_SEC = 2.0
MAX_REQUESTS = 20


class Recon:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": UA})
        self.session.verify = False
        self.n_requests = 0
        self.findings: list[str] = []

    def _fetch(self, url: str, method: str = "GET") -> requests.Response | None:
        if self.n_requests >= MAX_REQUESTS:
            self.findings.append(f"[STOP] Reached max requests ({MAX_REQUESTS}); halting.")
            return None
        try:
            t0 = time.time()
            r = self.session.request(method, url, timeout=30, allow_redirects=True)
            dt = time.time() - t0
            self.n_requests += 1
            self.findings.append(
                f"  {method:4s} {url}\n"
                f"       -> HTTP {r.status_code}, {len(r.content):,} bytes, {dt:.1f}s"
            )
            time.sleep(POLITENESS_SEC)
            return r
        except Exception as e:
            self.findings.append(f"  {method} {url}\n       -> ERROR: {type(e).__name__}: {e}")
            self.n_requests += 1
            return None

    def check_robots(self) -> dict:
        self.findings.append("\n## robots.txt")
        r = self._fetch(urljoin(BASE, "/robots.txt"))
        if r is None or r.status_code != 200:
            return {"exists": False}
        text = r.text
        self.findings.append(f"```\n{text[:800]}\n```")
        # Parse for disallow rules
        disallows = re.findall(r"^Disallow:\s*(.+)$", text, re.MULTILINE)
        allows = re.findall(r"^Allow:\s*(.+)$", text, re.MULTILINE)
        crawl_delay = re.findall(r"^Crawl-delay:\s*(\d+)", text, re.MULTILINE | re.IGNORECASE)
        return {
            "exists": True,
            "disallows": disallows,
            "allows": allows,
            "crawl_delay": crawl_delay,
        }

    def check_homepage(self) -> dict:
        self.findings.append("\n## Homepage")
        r = self._fetch(BASE)
        if r is None:
            return {}
        # Detect ASP.NET
        is_aspnet = "__VIEWSTATE" in r.text or "aspx" in r.url.lower()
        # Session cookies
        cookies = dict(self.session.cookies)
        # Parse links & forms
        soup = BeautifulSoup(r.text, "html.parser")
        links = [urljoin(r.url, a["href"]) for a in soup.find_all("a", href=True)]
        forms = soup.find_all("form")

        self.findings.append(f"- ASP.NET detected: {is_aspnet}")
        self.findings.append(f"- Cookies set: {list(cookies.keys())}")
        self.findings.append(f"- Links found: {len(links)}")
        self.findings.append(f"- Forms found: {len(forms)}")

        # Categorize links
        internal = [l for l in links if urlparse(l).netloc == urlparse(BASE).netloc]
        pdfs = [l for l in internal if l.lower().endswith(".pdf")]
        search_pages = [l for l in internal if re.search(r"search|find|query|browse", l, re.I)]
        category_pages = [
            l for l in internal
            if re.search(r"category|dept|ministry|gazette|extraord|weekly|browse|list", l, re.I)
        ]

        self.findings.append(f"- PDF links on homepage: {len(pdfs)}")
        if pdfs:
            self.findings.append(f"  Example PDFs:")
            for p in pdfs[:3]:
                self.findings.append(f"    {p}")

        self.findings.append(f"- Likely search/category pages: {len(search_pages) + len(category_pages)}")
        candidates = list(dict.fromkeys(search_pages + category_pages))[:8]
        for c in candidates:
            self.findings.append(f"    {c}")

        # Forms detail
        for i, form in enumerate(forms[:3]):
            action = form.get("action", "(no action)")
            method = form.get("method", "GET").upper()
            inputs = [
                (i.get("name", "?"), i.get("type", "text"))
                for i in form.find_all("input")
            ]
            self.findings.append(f"- Form #{i}: {method} → {action}")
            for name, typ in inputs[:8]:
                self.findings.append(f"    input name={name!r} type={typ!r}")

        return {
            "is_aspnet": is_aspnet,
            "cookies": cookies,
            "n_links": len(links),
            "n_forms": len(forms),
            "n_pdfs": len(pdfs),
            "example_pdfs": pdfs[:3],
            "search_pages": candidates[:5],
        }

    def probe_pdf(self, pdf_url: str):
        self.findings.append(f"\n## Probe PDF URL: {pdf_url}")
        r = self._fetch(pdf_url, method="HEAD")
        if r is None:
            return
        headers = dict(r.headers)
        interesting = {k: v for k, v in headers.items() if k.lower() in (
            "content-type", "content-length", "content-disposition",
            "last-modified", "etag", "server", "x-frame-options", "cache-control",
            "x-ratelimit-remaining", "x-ratelimit-limit"
        )}
        for k, v in interesting.items():
            self.findings.append(f"  {k}: {v}")

    def probe_search_page(self, url: str):
        self.findings.append(f"\n## Probe search/category page: {url}")
        r = self._fetch(url)
        if r is None:
            return
        soup = BeautifulSoup(r.text, "html.parser")
        pdfs_here = [urljoin(r.url, a["href"]) for a in soup.find_all("a", href=True)
                     if a["href"].lower().endswith(".pdf")]
        self.findings.append(f"- PDFs linked from this page: {len(pdfs_here)}")
        for p in pdfs_here[:5]:
            self.findings.append(f"    {p}")
        # Look for pagination hints
        pagination = soup.find_all("a", href=re.compile(r"page|next|prev", re.I))
        if pagination:
            self.findings.append(f"- Pagination links: {len(pagination)}")

    def run(self) -> str:
        self.findings.append(f"# egazette.gov.in recon report")
        self.findings.append(f"Base URL: {BASE}")
        self.findings.append(f"User-Agent: {UA}")
        self.findings.append(f"Max requests: {MAX_REQUESTS}, delay: {POLITENESS_SEC}s\n")

        robots = self.check_robots()
        home = self.check_homepage()

        # Probe up to 3 discovered PDFs (HEAD only)
        for pdf in (home.get("example_pdfs") or [])[:3]:
            self.probe_pdf(pdf)

        # Probe up to 2 search/category pages
        for sp in (home.get("search_pages") or [])[:2]:
            self.probe_search_page(sp)

        # Summary
        self.findings.append(f"\n## Summary")
        self.findings.append(f"- robots.txt: {'exists' if robots.get('exists') else 'missing'}")
        if robots.get("disallows"):
            self.findings.append(f"  disallows: {robots['disallows']}")
        if robots.get("crawl_delay"):
            self.findings.append(f"  crawl-delay: {robots['crawl_delay']}")
        self.findings.append(f"- ASP.NET: {home.get('is_aspnet')}")
        self.findings.append(f"- Requests used: {self.n_requests} / {MAX_REQUESTS}")

        report = "\n".join(self.findings)
        return report


def main():
    recon = Recon()
    report = recon.run()
    print(report)
    Path("recon_report.md").write_text(report, encoding="utf-8")
    print(f"\n\nReport saved → recon_report.md")


if __name__ == "__main__":
    main()
