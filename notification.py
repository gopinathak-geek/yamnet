import requests

BOT_TOKEN = "8103484610:AAF04ln-Bl-pXE2Y1MbKo1vdN7YxDGJKIZc"
CHAT_ID = 1917333642

def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": CHAT_ID,
        "text": message
    }
    response = requests.post(url, data=payload)
    if response.status_code == 200:
        print("✅ Message sent to Telegram!")
    else:
        print("❌ Failed:", response.text)
