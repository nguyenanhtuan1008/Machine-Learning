import requests
import json
import os

data = {
    "text": "hi, this is a test"
}

webhook = 'https://hooks.slack.com/services/6ZHCF86A/B011ATP3S2J/ycN6xM2SkxgN99WJ1Zv1y5Nm'
# webhook = os.getenv("webhook_slack")

requests.post(webhook, json.dumps(data))