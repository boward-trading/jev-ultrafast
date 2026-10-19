"""The browser contract on a synchronous Playwright page.

Firefox and Camoufox have no CDP, so :mod:`.browser` cannot reach them. This backend keeps
the same observe / fresh / act / close surface and the same browser-side JavaScript, and
replaces the CDP calls with Playwright ones. The page is injected: whoever launched the
browser owns its lifetime, so :meth:`PlaywrightBrowser.close` never closes it.
"""

import base64
import json
import sys
import time

from playwright.sync_api import Error as PlaywrightError

from .browser import MARKER, READ_STATE, RESOLVE, SETTLE, StalePage, fingerprint

SELECT_ALL = "Meta+A" if sys.platform == "darwin" else "Control+A"


class PlaywrightBrowser:
    def __init__(self, page, url=None, *, ready_seconds=15):
        self.page = page
        self.after_input = None
        if url:
            page.goto(url, wait_until="domcontentloaded")
            deadline = time.monotonic() + ready_seconds
            while time.monotonic() < deadline:
                try:
                    if self.evaluate("document.readyState") == "complete":
                        break
                except StalePage:
                    pass
                time.sleep(0.02)

    def evaluate(self, expression):
        try:
            return self.page.evaluate(expression)
        except PlaywrightError:
            raise StalePage("Document changed during evaluation") from None

    def observe(self, screenshot=True):
        if self.after_input:
            action, self.after_input = self.after_input, None
            try:
                self.page.evaluate(SETTLE + "(" + json.dumps(action) + ")")
            except PlaywrightError:
                pass
        for attempt in range(10):
            try:
                return self._read(screenshot)
            except StalePage:
                if attempt == 9:
                    raise
                time.sleep(0.02)
        raise StalePage("Page did not settle")

    def _read(self, screenshot):
        info = self.evaluate(READ_STATE)
        if info is None:
            raise StalePage("Document is navigating")
        info["fingerprint"] = fingerprint(info)
        if screenshot:
            info["screenshot"] = base64.b64encode(self.page.screenshot(type="jpeg", quality=72)).decode()
        return info

    def fresh(self, page, action=None):
        if action is not None and action["kind"] in {"click", "select"}:
            node = action["node"]
            if type(node) is not int:
                return False
            current = self.evaluate(
                "(() => { const c=window.__jevFast; "
                f"return c ? [c.pageKey(),c.guard(c.nodes.get({node}))] : null; }})()"
            )
            return current == [page["page_key"], page["guards"].get(str(node))]
        return self.evaluate(MARKER) == page["marker"]

    def act(self, action, page, text=None):
        if not self.fresh(page, action):
            raise StalePage("Page changed since this decision. Observe again.")
        kind = action["kind"]
        if kind == "wait":
            time.sleep(0.1)
            self.after_input = None
            return {"executed": action["id"]}
        if kind == "scroll":
            self.page.mouse.move(page["w"] / 2, page["h"] * 0.8)
            self.page.mouse.wheel(0, action["delta"])
        else:
            if type(action["node"]) is not int:
                raise ValueError("Invalid observed node")
            target = self._resolve(action)
            if target is None:
                if kind == "select":
                    raise RuntimeError("Dropdown execution was not confirmed; inspect before retrying.")
                raise StalePage("Target changed or is covered. Observe again.")
            if kind != "select":
                self.page.mouse.click(target["x"], target["y"])
                if kind == "fill":
                    self.page.keyboard.press(SELECT_ALL)
                    self.page.keyboard.insert_text(text)
        self.after_input = action
        return {"executed": action["id"]}

    def _resolve(self, action):
        try:
            return self.page.evaluate(RESOLVE + "(" + json.dumps(action) + ")")
        except PlaywrightError:
            if action["kind"] == "select":
                raise RuntimeError("Dropdown execution was interrupted; inspect before retrying.") from None
            raise StalePage("Document changed during evaluation") from None

    def close(self):
        self.page = None
