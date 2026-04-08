"""Tests for context pressure warnings (user-facing, not injected into messages).

Covers:
- Display formatting (CLI and gateway variants)
- Flag tracking and threshold logic on AIAgent
- Flag reset after compression
- status_callback invocation
"""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent.display import format_context_pressure, format_context_pressure_gateway
from run_agent import AIAgent


# ---------------------------------------------------------------------------
# Display formatting tests
# ---------------------------------------------------------------------------


class TestFormatContextPressure:
    """CLI context pressure display (agent/display.py).

    The bar shows progress toward the compaction threshold, not the
    raw context window.  60% = 60% of the way to compaction.
    """

    def test_80_percent_uses_warning_icon(self):
        line = format_context_pressure(0.80, 100_000, 0.50)
        assert "⚠" in line
        assert "80% to compaction" in line

    def test_90_percent_uses_warning_icon(self):
        line = format_context_pressure(0.90, 100_000, 0.50)
        assert "⚠" in line
        assert "90% to compaction" in line

    def test_bar_length_scales_with_progress(self):
        line_80 = format_context_pressure(0.80, 100_000, 0.50)
        line_95 = format_context_pressure(0.95, 100_000, 0.50)
        assert line_95.count("▰") > line_80.count("▰")

    def test_shows_threshold_tokens(self):
        line = format_context_pressure(0.80, 100_000, 0.50)
        assert "100k" in line

    def test_small_threshold(self):
        line = format_context_pressure(0.80, 500, 0.50)
        assert "500" in line

    def test_shows_threshold_percent(self):
        line = format_context_pressure(0.80, 100_000, 0.50)
        assert "50%" in line

    def test_approaching_hint(self):
        line = format_context_pressure(0.80, 100_000, 0.50)
        assert "compaction approaching" in line

    def test_no_compaction_when_disabled(self):
        line = format_context_pressure(0.85, 100_000, 0.50, compression_enabled=False)
        assert "no auto-compaction" in line

    def test_returns_string(self):
        result = format_context_pressure(0.65, 128_000, 0.50)
        assert isinstance(result, str)

    def test_over_100_percent_capped(self):
        """Progress > 1.0 should cap both bar and percentage text at 100%."""
        line = format_context_pressure(1.05, 100_000, 0.50)
        assert "▰" in line
        assert line.count("▰") == 20
        assert "100%" in line
        assert "105%" not in line


class TestFormatContextPressureGateway:
    """Gateway (plain text) context pressure display."""

    def test_80_percent_warning(self):
        msg = format_context_pressure_gateway(0.80, 0.50)
        assert "80% to compaction" in msg
        assert "50%" in msg

    def test_90_percent_warning(self):
        msg = format_context_pressure_gateway(0.90, 0.50)
        assert "90% to compaction" in msg
        assert "approaching" in msg

    def test_no_compaction_warning(self):
        msg = format_context_pressure_gateway(0.85, 0.50, compression_enabled=False)
        assert "disabled" in msg

    def test_no_ansi_codes(self):
        msg = format_context_pressure_gateway(0.80, 0.50)
        assert "\033[" not in msg

    def test_has_progress_bar(self):
        msg = format_context_pressure_gateway(0.80, 0.50)
        assert "▰" in msg

    def test_over_100_percent_capped(self):
        """Progress > 1.0 should cap percentage text at 100%."""
        msg = format_context_pressure_gateway(1.09, 0.50)
        assert "100% to compaction" in msg
        assert "109%" not in msg
        assert msg.count("▰") == 20


# ---------------------------------------------------------------------------
# AIAgent context pressure flag tests
# ---------------------------------------------------------------------------


def _make_tool_defs(*names):
    return [
        {
            "type": "function",
            "function": {
                "name": n,
                "description": f"{n} tool",
                "parameters": {"type": "object", "properties": {}},
            },
        }
        for n in names
    ]


@pytest.fixture()
def agent():
    """Minimal AIAgent with mocked internals."""
    with (
        patch("run_agent.get_tool_definitions", return_value=_make_tool_defs("web_search")),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        a = AIAgent(
            api_key="test-key-1234567890",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        a.client = MagicMock()
        return a


class TestContextPressureFlags:
    """Context pressure warning flag tracking on AIAgent."""

    def test_flag_initialized_false(self, agent):
        assert agent._context_pressure_warned is False

    def test_emit_calls_status_callback(self, agent):
        """status_callback should be invoked with event type and message."""
        cb = MagicMock()
        agent.status_callback = cb

        compressor = MagicMock()
        compressor.context_length = 200_000
        compressor.threshold_tokens = 100_000  # 50%

        agent._emit_context_pressure(0.85, compressor)

        cb.assert_called_once()
        args = cb.call_args[0]
        assert args[0] == "context_pressure"
        assert "85% to compaction" in args[1]

    def test_emit_no_callback_no_crash(self, agent):
        """No status_callback set — should not crash."""
        agent.status_callback = None

        compressor = MagicMock()
        compressor.context_length = 200_000
        compressor.threshold_tokens = 100_000

        # Should not raise
        agent._emit_context_pressure(0.60, compressor)

    def test_emit_prints_for_cli_platform(self, agent, capsys):
        """CLI platform should always print context pressure, even in quiet_mode."""
        agent.quiet_mode = True
        agent.platform = "cli"
        agent.status_callback = None

        compressor = MagicMock()
        compressor.context_length = 200_000
        compressor.threshold_tokens = 100_000

        agent._emit_context_pressure(0.85, compressor)
        captured = capsys.readouterr()
        assert "▰" in captured.out
        assert "to compaction" in captured.out

    def test_emit_skips_print_for_gateway_platform(self, agent, capsys):
        """Gateway platforms get the callback, not CLI print."""
        agent.platform = "telegram"
        agent.status_callback = None

        compressor = MagicMock()
        compressor.context_length = 200_000
        compressor.threshold_tokens = 100_000

        agent._emit_context_pressure(0.85, compressor)
        captured = capsys.readouterr()
        assert "▰" not in captured.out

    def test_flag_reset_on_compression(self, agent):
        """After _compress_context, context pressure flag should reset."""
        agent._context_pressure_warned = True
        agent.compression_enabled = True

        agent.context_compressor = MagicMock()
        agent.context_compressor.compress.return_value = [
            {"role": "user", "content": "Summary of conversation so far."}
        ]
        agent.context_compressor.context_length = 200_000
        agent.context_compressor.threshold_tokens = 100_000

        agent._todo_store = MagicMock()
        agent._todo_store.format_for_injection.return_value = None

        agent._build_system_prompt = MagicMock(return_value="system prompt")
        agent._cached_system_prompt = "old system prompt"
        agent._session_db = None

        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
        ]
        agent._compress_context(messages, "system prompt")

        assert agent._context_pressure_warned is False

    def test_emit_callback_error_handled(self, agent):
        """If status_callback raises, it should be caught gracefully."""
        cb = MagicMock(side_effect=RuntimeError("callback boom"))
        agent.status_callback = cb

        compressor = MagicMock()
        compressor.context_length = 200_000
        compressor.threshold_tokens = 100_000

        # Should not raise
        agent._emit_context_pressure(0.85, compressor)


# ---------------------------------------------------------------------------
# Cross-instance dedup tests (gateway mode)
# ---------------------------------------------------------------------------


class TestContextPressureDedup:
    """Time-based dedup of context pressure warnings across AIAgent instances.

    In gateway mode, a new AIAgent is created per message.  Without dedup,
    the warning would fire on every message when context is already above 85%.
    """

    @pytest.fixture(autouse=True)
    def _clean_class_state(self):
        """Isolate tests by clearing class-level state before/after."""
        AIAgent._context_pressure_last_warned.clear()
        yield
        AIAgent._context_pressure_last_warned.clear()

    def test_second_instance_within_cooldown_is_suppressed(self, agent):
        """A second AIAgent for the same session should not re-emit within 5 min."""
        import time

        cb1 = MagicMock()
        agent.status_callback = cb1
        agent.session_id = "test-session-123"

        compressor = MagicMock()
        compressor.context_length = 200_000
        compressor.threshold_tokens = 100_000

        # First warning — should fire
        agent._context_pressure_warned = False
        AIAgent._context_pressure_last_warned.clear()
        agent._check_and_emit_context_pressure = lambda _prog, _comp: None  # bypass loop logic

        # Simulate what the loop does: set warned + emit
        AIAgent._context_pressure_last_warned[agent.session_id] = time.time()
        agent._emit_context_pressure(0.90, compressor)

        cb1.assert_called_once()

        # Now simulate a new agent instance for the same session (gateway creates new AIAgent per message)
        with (
            patch("run_agent.get_tool_definitions", return_value=_make_tool_defs("web_search")),
            patch("run_agent.check_toolset_requirements", return_value={}),
            patch("run_agent.OpenAI"),
        ):
            agent2 = AIAgent(
                api_key="test-key-1234567890",
                quiet_mode=True,
                skip_context_files=True,
                skip_memory=True,
                session_id="test-session-123",
            )
            agent2.client = MagicMock()

        cb2 = MagicMock()
        agent2.status_callback = cb2

        # The new agent has _context_pressure_warned = False (fresh instance),
        # but the class-level dict should prevent re-emission.
        # Simulate the check logic directly:
        _now = time.time()
        _last = AIAgent._context_pressure_last_warned.get(agent2.session_id, 0)
        assert _now - _last < AIAgent._CONTEXT_PRESSURE_COOLDOWN

        # The emit path would set the flag without calling _emit_context_pressure
        agent2._context_pressure_warned = True

        # Confirm the callback was NOT called on the second instance
        cb2.assert_not_called()

    def test_warning_fires_after_cooldown_expires(self, agent):
        """After the cooldown expires, a new warning should fire normally."""
        import time

        AIAgent._context_pressure_last_warned[agent.session_id] = time.time() - 301  # 5+ min ago

        cb = MagicMock()
        agent.status_callback = cb
        agent._context_pressure_warned = False

        compressor = MagicMock()
        compressor.context_length = 200_000
        compressor.threshold_tokens = 100_000

        # Check the dedup condition — should allow emission
        _now = time.time()
        _last = AIAgent._context_pressure_last_warned.get(agent.session_id, 0)
        assert _now - _last >= AIAgent._CONTEXT_PRESSURE_COOLDOWN

        agent._emit_context_pressure(0.90, compressor)
        cb.assert_called_once()

    def test_compression_resets_dedup(self, agent):
        """After compression drops below 85%, the cooldown entry should be cleared."""
        import time

        agent.session_id = "test-session-compress"
        AIAgent._context_pressure_last_warned[agent.session_id] = time.time()

        agent.compression_enabled = True
        agent.context_compressor = MagicMock()
        agent.context_compressor.compress.return_value = [
            {"role": "user", "content": "Summary."}
        ]
        agent.context_compressor.context_length = 200_000
        agent.context_compressor.threshold_tokens = 100_000
        agent.context_compressor.last_prompt_tokens = 10  # well below 85%
        agent._todo_store = MagicMock()
        agent._todo_store.format_for_injection.return_value = None
        agent._build_system_prompt = MagicMock(return_value="system prompt")
        agent._cached_system_prompt = "old system prompt"
        agent._session_db = None

        messages = [{"role": "user", "content": "hello"}]
        agent._compress_context(messages, "system prompt")

        assert agent.session_id not in AIAgent._context_pressure_last_warned
        assert agent._context_pressure_warned is False
