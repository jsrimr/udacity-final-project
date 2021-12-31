import ast
import base64
import hashlib
import hmac
import json
import time
import requests
import argparse
from GOPAX_utils import apikey, secret, cancel_orders, query_balance, current_price, place_order


# parser = argparse.ArgumentParser()

# parser.add_argument('--asset', type=int,
#                     help='BTCBULL-KRW 같은거')
# parser.add_argument('--amount', type=int,
#                     help='[0,10]')
# args = parser.parse_args()

action2decision = {0: 0.2, 1: 0.4, 2: 0.6, 3: 0.8, 4: 1, 6: 0.2, 7: 0.4, 8: 0.6, 9: 0.8, 10: 1, }

def place_order(asset, action):
    price = current_price(pair_name=f"{asset}-KRW")
    msg = "default"
    if action < 5:
        # buy  - 얼마에 몇 주 살 것인가
        side = "buy"
        if query_balance('KRW') < 1000:
            class res():
                def __init__(self):
                    self.ok = False

            msg = "out of KRW"
            return res(), msg
        n_stock = query_balance('KRW') * action2decision[action] / price

    elif action == 5:
        class res():
            def __init__(self):
                self.ok = False

        msg = "hold"
        return res(), msg
    else:
        # sell - 얼마에 몇 주 팔 것인가
        side = "sell"
        if query_balance(f'{asset}') < 0.00005:
            class res():
                def __init__(self):
                    self.ok = False

            msg = f"out of {asset}"
            return res(), msg
        n_stock = query_balance(f'{asset}') * action2decision[action]

    request_body = {
        "amount": n_stock,
        "price": price,
        "side": side,  # buy 또는 sell
        "tradingPairName": f"{asset}-KRW",
        "type": "limit"  # limit 또는 market
    }
    timestamp = str(int(time.time() * 1000))
    method = 'POST'
    request_path = '/orders'
    what = 't' + timestamp + method + request_path + json.dumps(request_body)
    # base64로 secret을 디코딩함
    key = base64.b64decode(secret)
    # hmac으로 필수 메시지에 서명하고
    signature = hmac.new(key, str(what).encode('utf-8'), hashlib.sha512)
    # 그 결과물을 base64로 인코딩함
    signature_b64 = base64.b64encode(signature.digest())

    custom_headers = {
        'API-Key': apikey,
        'Signature': signature_b64,
        'Timestamp': timestamp
    }
    req = requests.post(url='https://api.gopax.co.kr' + request_path, headers=custom_headers, json=request_body)
    return req, msg


#ETHBULL, BTCBULL, BTC, KRW
if __name__ == "__main__":
    # print(cancel_orders())

    for asset, action in zip(['BTC','BTCBULL'], [7,7]):
        res,msg = place_order(asset, action)
        print(res.json())