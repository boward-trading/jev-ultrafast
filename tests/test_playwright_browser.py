"""The CDP browser's contract on a synchronous Playwright page. Offline: a fake page, no browser."""

import base64
import subprocess
import sys
from copy import deepcopy
from unittest.mock import Mock

import pytest

from jev_ultrafast import agent as loop
from jev_ultrafast import model
from jev_ultrafast import playwright_browser as pwb
from jev_ultrafast.browser import MARKER, READ_STATE, StalePage, fingerprint


def snapshot():
    return {
        "url": "https://example.test/",
        "title": "Search",
        "text": "Search",
        "w": 1000,
        "h": 700,
        "scroll": {"y": 0, "height": 1400},
        "actions": [
            {"id": "e1", "kind": "fill", "label": "Search", "role": "textbox", "value": "", "node": 10},
            {"id": "e2", "kind": "click", "label": "Open Search", "role": "textbox", "value": "", "node": 10},
            {"id": "e3", "kind": "click", "label": "Go", "role": "button", "value": "", "node": 20},
            {"id": "e4", "kind": "select", "label": "Size → Large", "role": "combobox", "value": "L",
             "current_value": "M", "node": 30},
            {"id": "scroll_down", "kind": "scroll", "label": "Scroll down", "delta": 560},
            {"id": "wait", "kind": "wait", "label": "Wait for the page to update"},
        ],
        "marker": ["m"],
        "page_key": ["k"],
        "guards": {"10": ["g10"], "20": ["g20"], "30": ["g30"]},
        "omitted_actions": 0,
    }


def action(page, action_id):
    return next(a for a in page["actions"] if a["id"] == action_id)


class FakePage:
    """Answers the expressions the backend evaluates; records every other Playwright call."""

    def __init__(self, state):
        self.state = state
        self.calls = []
        self.mouse = Mock()
        self.keyboard = Mock()
        self.failing_reads = 0
        self.marker_now = state["marker"]
        self.guard_now = None
        self.target = {"x": 5, "y": 6}

    def evaluate(self, expression):
        self.calls.append(expression)
        if expression == READ_STATE:
            if self.failing_reads:
                self.failing_reads -= 1
                raise pwb.PlaywrightError("Execution context was destroyed, most likely because of a navigation")
            return deepcopy(self.state)
        if expression == MARKER:
            return self.marker_now
        if expression.startswith(pwb.SETTLE):
            return None
        if expression.startswith(pwb.RESOLVE):
            if isinstance(self.target, Exception):
                raise self.target
            return self.target
        if "c.pageKey()" in expression:
            return self.guard_now
        if expression == "document.readyState":
            return "complete"
        raise AssertionError(f"unexpected evaluate: {expression[:80]}")

    def screenshot(self, **kwargs):
        self.calls.append(("screenshot", kwargs))
        return b"\xff\xd8jpeg"

    def goto(self, url, **kwargs):
        self.calls.append(("goto", url, kwargs))


@pytest.fixture
def fake():
    return FakePage(snapshot())


@pytest.fixture
def browser(fake):
    return pwb.PlaywrightBrowser(fake)


@pytest.fixture
def observed(browser):
    return browser.observe(screenshot=False)


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    sleeps = []
    monkeypatch.setattr(pwb.time, "sleep", sleeps.append)
    return sleeps


def test_observe_is_one_atomic_read_with_a_fingerprint(browser, fake):
    page = browser.observe(screenshot=False)
    assert page["actions"] == snapshot()["actions"]
    assert page["fingerprint"] == fingerprint(page)
    assert "screenshot" not in page
    assert fake.calls == [READ_STATE]


def test_observe_screenshot_is_a_base64_jpeg(browser, fake):
    page = browser.observe(screenshot=True)
    assert base64.b64decode(page["screenshot"]) == b"\xff\xd8jpeg"
    assert ("screenshot", {"type": "jpeg", "quality": 72}) in fake.calls


def test_observe_waits_for_the_page_to_settle_after_input(browser, fake):
    browser.after_input = {"id": "e1", "kind": "fill", "node": 10}
    browser.observe(screenshot=False)
    assert fake.calls[0].startswith(pwb.SETTLE) and '"node": 10' in fake.calls[0]
    assert fake.calls[1] == READ_STATE
    assert browser.after_input is None


def test_a_navigating_document_is_retried_then_reported_stale(browser, fake, no_sleep):
    fake.failing_reads = 2
    assert browser.observe(screenshot=False)["url"] == "https://example.test/"
    assert fake.calls.count(READ_STATE) == 3
    fake.calls.clear()
    fake.failing_reads = 10
    with pytest.raises(StalePage):
        browser.observe(screenshot=False)
    assert fake.calls.count(READ_STATE) == 10


def test_fresh_compares_the_marker_or_the_target_guard(browser, fake, observed):
    assert browser.fresh(observed)
    fake.marker_now = ["changed"]
    assert not browser.fresh(observed)
    click = action(observed, "e3")
    fake.guard_now = [["k"], ["g20"]]
    assert browser.fresh(observed, click)
    fake.guard_now = [["k"], ["moved"]]
    assert not browser.fresh(observed, click)
    assert not browser.fresh(observed, {**click, "node": "20"})


def test_click_resolves_the_target_then_clicks_its_centre(browser, fake, observed):
    fake.guard_now = [["k"], ["g20"]]
    clicked = action(observed, "e3")
    assert browser.act(clicked, observed) == {"executed": "e3"}
    fake.mouse.click.assert_called_once_with(5, 6)
    assert any(c.startswith(pwb.RESOLVE) and '"id": "e3"' in c for c in fake.calls if isinstance(c, str))
    assert browser.after_input == clicked


def test_fill_clicks_selects_all_then_inserts_the_text(browser, fake, observed):
    browser.act(action(observed, "e1"), observed, text="book")
    fake.mouse.click.assert_called_once_with(5, 6)
    fake.keyboard.press.assert_called_once_with("Meta+A" if sys.platform == "darwin" else "Control+A")
    fake.keyboard.insert_text.assert_called_once_with("book")


def test_select_mutates_through_evaluate_only(browser, fake, observed):
    fake.guard_now = [["k"], ["g30"]]
    browser.act(action(observed, "e4"), observed)
    fake.mouse.click.assert_not_called()
    fake.keyboard.insert_text.assert_not_called()


@pytest.mark.parametrize("outcome", [None, pwb.PlaywrightError("Execution context was destroyed")])
def test_an_unconfirmed_dropdown_is_not_retried_as_stale(browser, fake, observed, outcome):
    fake.guard_now = [["k"], ["g30"]]
    fake.target = outcome
    with pytest.raises(RuntimeError, match="Dropdown execution"):
        browser.act(action(observed, "e4"), observed)


def test_a_covered_or_vanished_target_is_stale(browser, fake, observed):
    fake.guard_now = [["k"], ["g20"]]
    fake.target = None
    with pytest.raises(StalePage):
        browser.act(action(observed, "e3"), observed)
    fake.mouse.click.assert_not_called()


def test_scroll_wheels_from_inside_the_viewport(browser, fake, observed):
    scroll = action(observed, "scroll_down")
    browser.act(scroll, observed)
    fake.mouse.move.assert_called_once_with(500, 560)
    fake.mouse.wheel.assert_called_once_with(0, 560)
    assert browser.after_input == scroll


def test_wait_sleeps_without_touching_the_page(browser, fake, observed, no_sleep):
    browser.act(action(observed, "wait"), observed)
    assert no_sleep == [0.1]
    assert fake.calls[-1] == MARKER
    fake.mouse.click.assert_not_called()
    assert browser.after_input is None


def test_a_stale_page_blocks_input(browser, fake, observed):
    fake.marker_now = ["changed"]
    with pytest.raises(StalePage):
        browser.act(action(observed, "wait"), observed)
    fake.guard_now = [["k"], ["moved"]]
    with pytest.raises(StalePage):
        browser.act(action(observed, "e3"), observed)
    fake.mouse.click.assert_not_called()


def test_constructor_navigates_and_waits_for_the_document(fake):
    pwb.PlaywrightBrowser(fake, "https://example.test/start")
    assert ("goto", "https://example.test/start", {"wait_until": "domcontentloaded"}) in fake.calls
    assert "document.readyState" in fake.calls


def test_close_leaves_the_injected_page_to_its_owner():
    page = Mock()
    pwb.PlaywrightBrowser(page).close()
    page.close.assert_not_called()
    page.context.close.assert_not_called()


def test_agent_uses_an_injected_browser_instead_of_cdp(monkeypatch):
    monkeypatch.setattr(loop, "Browser", Mock(side_effect=AssertionError("CDP browser must not be built")))
    observed = {**snapshot(), "fingerprint": "f"}
    injected = Mock(observe=Mock(return_value=observed), fresh=Mock(return_value=True))
    agent = loop.Agent(None, "Find a book", browser=injected)
    assert agent.state["browser"] is injected
    injected.observe.assert_called_once_with(screenshot=False)


def test_agent_run_on_an_injected_browser_reaches_done(monkeypatch):
    observed = {**snapshot(), "fingerprint": "f"}
    injected = Mock(observe=Mock(return_value=observed), fresh=Mock(return_value=True))
    monkeypatch.setattr(
        loop,
        "choose",
        Mock(return_value={"choice": "DONE", "operation": "DONE", "target": None, "confidence": 1.0,
                           "probabilities": {"DONE": 1.0}, "latency_ms": 1, "usage": {}}),
    )
    agent = loop.Agent(None, "Find a book", browser=injected)
    states = list(agent.run())
    assert states[-1]["status"] == "done"
    injected.act.assert_not_called()


def test_text_helper_sends_extra_headers(monkeypatch):
    monkeypatch.setenv("TEXT_MODEL_API_KEY", "k")
    monkeypatch.setenv("TEXT_MODEL_BASE_URL", "http://127.0.0.1:8000/v1")
    monkeypatch.setenv("TEXT_MODEL_HEADERS", '{"X-Claude-Tools": "", "X-Claude-Effort": "low"}')
    response = Mock(status_code=200, is_error=False)
    response.json.return_value = {"choices": [{"message": {"content": '{"text": "London"}'}}]}
    post = Mock(return_value=response)
    monkeypatch.setattr(model.CLIENT, "post", post)
    value, _ = model.field_text({"goal": "g", "field": {}, "page": {}, "recent_actions": []})
    assert value == "London"
    headers = post.call_args.kwargs["headers"]
    assert headers["Authorization"] == "Bearer k"
    assert headers["X-Claude-Tools"] == "" and headers["X-Claude-Effort"] == "low"


def test_malformed_extra_headers_are_rejected_before_any_request(monkeypatch):
    monkeypatch.setenv("TEXT_MODEL_API_KEY", "k")
    monkeypatch.setenv("TEXT_MODEL_HEADERS", "not json")
    post = Mock()
    monkeypatch.setattr(model.CLIENT, "post", post)
    with pytest.raises(ValueError, match="TEXT_MODEL_HEADERS"):
        model.field_text({"goal": "g", "field": {}, "page": {}, "recent_actions": []})
    post.assert_not_called()


def test_browser_module_imports_without_browser_harness():
    code = (
        "import sys; sys.modules['browser_harness'] = None; sys.modules['browser_harness.helpers'] = None; "
        "sys.modules['browser_harness.admin'] = None; import jev_ultrafast.browser"
    )
    assert subprocess.run([sys.executable, "-c", code], capture_output=True, text=True).returncode == 0
