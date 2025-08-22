# app.py
from flask import Flask, request
import requests, os

app = Flask(__name__)

TELEGRAM_BOT_TOKEN = "<YOUR_TELEGRAM_BOT_TOKEN>"
TELEGRAM_CHAT_ID = "<YOUR_TELEGRAM_CHAT_ID>"

def send_telegram(text):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text}
    try:
        r = requests.post(url, json=payload, timeout=10)
        return r.status_code, r.text
    except Exception as e:
        return None, str(e)

@app.route("/webhook", methods=["POST"])
def webhook():
    data = request.get_json(force=True, silent=True)
    # Expect data like {"pair":"EURUSD","signal":"BUY","price":"1.085"}
    if not data:
        return {"status":"no-json"}, 400
    pair = data.get("pair","unknown")
    signal = data.get("signal","?")
    price = data.get("price","")
    text = f"📢 Signal: {pair} — {signal}  {price}"
    send_telegram(text)
    return {"status":"ok"}, 200

if __name__ == "__main__":
    app.run(port=5000, host="0.0.0.0")
