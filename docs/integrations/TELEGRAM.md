# Telegram Notifications Setup

Experiment lifecycle notifications (start, finish, crash) are sent to your Telegram DM via [@thinkbooster_bot](https://t.me/thinkbooster_bot).

## Quick Start

1. Open [@thinkbooster_bot](https://t.me/thinkbooster_bot) in Telegram and send `/start`
2. The bot will reply with your **chat ID** — copy it
3. Get the **bot API token** from a colleague
4. Add both to your `.env`:
   ```
   TELEGRAM_BOT_TOKEN=<token from colleague>
   TELEGRAM_CHAT_ID=<your chat ID from step 2>
   ```
5. Run any experiment — you'll receive DM notifications on start and finish

## What You'll Receive

**On start:**
```
▶️ Running

Run: baseline_qwen25_math_seed0_12-30-00
Strategy: baseline  |  Scorer: entropy
Model: qwen25_math_7b_instruct  |  Dataset: math
Machine: MBZUAI-Artem-1
🔗 W&B Run  |  W&B Group
```

**On finish** — same info plus accuracy, TFLOPs, token count.

**On crash** — truncated error traceback.

If W&B is disabled (`report_to=none`), the last line shows `🔗 W&B: disabled`.

## Disabling

Remove or leave empty `TELEGRAM_BOT_TOKEN` / `TELEGRAM_CHAT_ID` in `.env`. Notifications are silently skipped when either is missing.

## Running the Bot Service

The `/start` handler only works while the bot service is running. To start it:

```bash
python -m llm_tts.utils.telegram_bot
```

It reads `TELEGRAM_BOT_TOKEN` from `.env` automatically. Keep it running on a shared machine so new team members can get their chat IDs.
