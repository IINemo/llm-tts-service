"""Telegram bot notifications for experiment lifecycle events."""

import logging
import os

import requests

log = logging.getLogger(__name__)


class TelegramNotifier:
    """Sends Telegram notifications for experiment start/finish/crash events.

    Reads TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID from environment variables.
    If either is missing, all notifications are silently disabled.
    """

    def __init__(self):
        self.token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
        self.chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")
        self.enabled = bool(self.token and self.chat_id)

    def _send(self, text: str) -> None:
        """Send a message via Telegram Bot API. Never raises."""
        if not self.enabled:
            return
        try:
            url = f"https://api.telegram.org/bot{self.token}/sendMessage"
            resp = requests.post(
                url,
                json={
                    "chat_id": self.chat_id,
                    "text": text,
                    "parse_mode": "HTML",
                    "disable_web_page_preview": True,
                },
                timeout=10,
            )
            if not resp.ok:
                log.warning(f"Telegram API returned {resp.status_code}: {resp.text}")
        except Exception as e:
            log.warning(f"Failed to send Telegram notification: {e}")

    def _wandb_links(self, wandb_url: str | None, wandb_group_url: str | None) -> str:
        """Format W&B links line."""
        parts = []
        if wandb_url:
            parts.append(f'<a href="{wandb_url}">W&amp;B Run</a>')
        if wandb_group_url:
            parts.append(f'<a href="{wandb_group_url}">W&amp;B Group</a>')
        if parts:
            return "🔗 " + "  |  ".join(parts)
        return "🔗 W&amp;B: disabled"

    def notify_started(
        self,
        run_name: str,
        strategy: str,
        model: str,
        dataset: str,
        scorer: str,
        machine: str,
        wandb_url: str | None = None,
        wandb_group_url: str | None = None,
    ) -> None:
        """Send 'started' notification."""
        lines = [
            "▶️ <b>Running</b>",
            "",
            f"Run: <code>{run_name}</code>",
            f"Strategy: {strategy}  |  Scorer: {scorer}",
            f"Model: {model}  |  Dataset: {dataset}",
            f"Machine: {machine}",
        ]
        lines.append(self._wandb_links(wandb_url, wandb_group_url))
        self._send("\n".join(lines))

    def notify_finished(
        self,
        run_name: str,
        strategy: str,
        model: str,
        dataset: str,
        scorer: str = "none",
        machine: str = "",
        metrics: dict | None = None,
        wandb_url: str | None = None,
        wandb_group_url: str | None = None,
    ) -> None:
        """Send 'finished' notification with key metrics."""
        lines = [
            "✅ <b>Finished</b>",
            "",
            f"Run: <code>{run_name}</code>",
            f"Strategy: {strategy}  |  Scorer: {scorer}",
            f"Model: {model}  |  Dataset: {dataset}",
        ]

        if metrics:
            lines.append("")
            lines.append("Results:")

            # Exact match accuracy
            em_acc = metrics.get("exact_match/accuracy")
            if em_acc is not None:
                lines.append(f"  Accuracy (EM): {em_acc * 100:.1f}%")

            # LLM judge accuracies
            for key, val in sorted(metrics.items()):
                if key.endswith("/accuracy") and key.startswith("llm_judge"):
                    label = key.replace("/accuracy", "").replace("_", " ").title()
                    lines.append(f"  {label}: {val * 100:.1f}%")

            # Compute metrics
            tflops = metrics.get("compute/total_tflops")
            if tflops is not None:
                lines.append(f"  TFLOPs: {tflops:,.1f}")

            tokens = metrics.get("compute/total_tokens")
            if tokens is not None:
                lines.append(f"  Tokens: {int(tokens):,}")

            samples = metrics.get("compute/num_samples")
            if samples is not None:
                lines.append(f"  Samples: {int(samples)}")

        lines.append(self._wandb_links(wandb_url, wandb_group_url))
        self._send("\n".join(lines))

    def notify_crashed(
        self,
        run_name: str,
        strategy: str,
        model: str,
        dataset: str,
        error: str,
        wandb_url: str | None = None,
    ) -> None:
        """Send 'crashed' notification with truncated error."""
        if len(error) > 500:
            error = error[:500] + "…"

        lines = [
            "❌ <b>Crashed</b>",
            "",
            f"Run: <code>{run_name}</code>",
            f"Strategy: {strategy}",
            f"Model: {model}  |  Dataset: {dataset}",
            "",
            "Error:",
            f"<pre>{error}</pre>",
        ]
        lines.append(self._wandb_links(wandb_url, None))
        self._send("\n".join(lines))
