from sanic import json
import requests
from collections import Counter



url = 'http://0.0.0.0:5005/webhooks/rest/webhook'

payload = {
    "sender":"test_user",
    "message": "help me"
}


r = requests.post(url, json = payload)
#rget = requests.get('http://0.0.0.0:5005/domain')
#print(rget.content )
#print(type(rget.content))
message = []
for i in range(len(r.json())):
    text = r.json()[i]['text']
    message = {"answer": text}
    print(message)

print(r.status_code)

print(r.json())
print(len(r.json()))


