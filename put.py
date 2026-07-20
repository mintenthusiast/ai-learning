import requests

requests.put("http://127.0.0.1:8000/items/", data={"item_name" : "hi", "item_id": "1234"})