import requests

BOT_TOKEN = "8103484610:AAF04ln-Bl-pXE2Y1MbKo1vdN7YxDGJKIZc"
response = requests.get(f"https://api.telegram.org/bot{BOT_TOKEN}/getUpdates")
print(response.json())

