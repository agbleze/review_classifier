import requests
import json
#from typing import str


def request_prediction(URL: str, review_data: str):
    in_data = {'review': review_data}
    req = requests.post(url = URL, json=in_data)
    response = req.content
    prediction = json.loads(response)#['category']
    return prediction
    






