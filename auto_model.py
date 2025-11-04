# auto_model.py
# Runs simple_model.py, sends identical ASCII text to Telegram and via Email.

import os
import subprocess
import requests
from dotenv import load_dotenv
import datetime as dt
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

load_dotenv()

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
FRIEND_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID_FRIEND")  # optional

EMAIL_SENDER    = os.getenv("EMAIL_SENDER")
EMAIL_PASSWORD  = os.getenv("EMAIL_APP_PASSWORD")
EMAIL_RECIPIENT = os.getenv("EMAIL_RECIPIENT")


# ==========================
# EMAIL
# ==========================
def send_email(subject, body):
    if not (EMAIL_SENDER and EMAIL_PASSWORD and EMAIL_RECIPIENT):
        print("Email not configured; skipped.")
        return
    html_body = f"<pre style='font-family: monospace; white-space: pre-wrap'>{body}</pre>"
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = EMAIL_SENDER
    msg["To"] = EMAIL_RECIPIENT
    msg.attach(MIMEText(html_body, "html", "utf-8"))
    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as smtp:
            smtp.starttls()
            smtp.login(EMAIL_SENDER, EMAIL_PASSWORD)
            smtp.send_message(msg)
        print("Email sent.")
    except Exception as e:
        print(f"Email failed: {e}")


# ==========================
# TELEGRAM
# ==========================
def send_telegram(raw_text):
    if not (BOT_TOKEN and CHAT_ID):
        print("Telegram not configured; skipped.")
        return

    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    # Wrap message in Markdown code block to preserve formatting
    wrapped = f"```\n{raw_text.strip()}\n```"

    payload = {
        "chat_id": CHAT_ID,
        "text": wrapped[:3900],  # stay under Telegram limit
        "parse_mode": "MarkdownV2",
    }

    r = requests.post(url, json=payload)
    if r.status_code == 200:
        print("Telegram sent.")
    else:
        print(f"Telegram error {r.status_code}: {r.text}")


def send_telegram_friend(raw_text):
    if not (BOT_TOKEN and FRIEND_CHAT_ID):
        return

    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    wrapped = f"```\n{raw_text.strip()}\n```"

    payload = {
        "chat_id": FRIEND_CHAT_ID,
        "text": wrapped[:3900],
        "parse_mode": "MarkdownV2",
    }

    r = requests.post(url, json=payload)
    if r.status_code == 200:
        print("Telegram (friend) sent.")
    else:
        print(f"Telegram friend error {r.status_code}: {r.text}")


# ==========================
# RUN MODEL
# ==========================
def run_model_and_send():
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_model.py")
    try:
        result = subprocess.run(
            ["python", "-u", script],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=os.path.dirname(script),
            timeout=600,
        )
    except Exception as e:
        txt = f"Model run failed: {e}"
        print(txt)
        send_telegram(txt)
        send_email("Football Model Results - ERROR", txt)
        return

    stdout = (result.stdout or "").strip()
    stderr = (result.stderr or "").strip()

    with open("last_output.txt", "w", encoding="utf-8") as f:
        f.write(stdout)
    print(f"[DEBUG] Captured {len(stdout)} characters")

    if not stdout:
        msg = "(No stdout captured.)"
        if stderr:
            msg += f"\n\nSTDERR:\n{stderr}"
        print(msg)
        send_telegram(msg)
        send_telegram_friend(msg)
        send_email(f"Football Model Results - {dt.date.today()}", msg)
        return

    today = dt.date.today().strftime("%a %d %b")
    subject = f"Football Model Results - {today}"

    send_telegram(stdout)
    send_telegram_friend(stdout)
    send_email(subject, stdout)


if __name__ == "__main__":
    run_model_and_send()
