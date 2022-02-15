import json
import queue
import threading
from datetime import datetime
from threading import Thread

import numpy as np
import requests
from websocket import create_connection

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import sys

cacheQueue = queue.Queue()
# locker = threading.Lock()


            # print(np.sort(book.bids[-5:]), np.sort(book.asks[-5:]))


class Stream:
    def __init__(self, symbol):
        self.endpoint = "wss://stream.binance.com:9443/ws/" + symbol.lower() + "@depth"
        self.ws = create_connection(self.endpoint)

    def close(self):
        self.ws.close()

    def recieve(self):
        return Message(json.loads(self.ws.recv()))


class Message:
    def __init__(self, message):
        self.u, self.U = message["u"], message["U"]
        self.bids = np.array(message["b"])[:, 0].astype("float")
        self.bq = np.array(message["b"])[:, 1].astype("float")
        self.asks, self.aq = np.array(message["a"])[:, 0].astype("float"), np.array(
            message["a"]
        )[:, 1].astype("float")
        self.date = datetime.fromtimestamp(message["E"] / 1000).strftime(
            "%A, %B %d, %Y %I:%M:%S"
        )


class OrderBook:
    def __init__(self, symbol):
        self.endpoint = (
            "https://api.binance.com/api/v3/depth?symbol="
            + symbol.upper()
            + "&limit=1000"
        )
        self.snapshot = requests.get(self.endpoint).json()
        self.lastUpdateId = self.snapshot["lastUpdateId"]
        self.bids = np.array(self.snapshot["bids"])[:, 0].astype("float")
        self.bq = np.array(self.snapshot["bids"])[:, 1].astype("float")
        self.asks, self.aq = np.array(self.snapshot["asks"])[:, 0].astype(
            "float"
        ), np.array(self.snapshot["asks"])[:, 1].astype("float")

    def find_beg(self, message):
        if message.u <= self.lastUpdateId:
            return False
        elif message.U <= self.lastUpdateId + 1 and message.u >= self.lastUpdateId + 1:
            return True

    def update_bids(self, message):
        for num, i in enumerate(message.bids):
            ind = np.where(self.bids == i)
            if ind[0].any():
                if i == 0:
                    np.delete(self.bids, ind)
                else:
                    self.bq[ind] += message.bq[num]
            else:
                np.searchsorted(self.bids, i)

    def update_asks(self, message):
        for num, i in enumerate(message.asks):
            ind = np.where(self.asks == i)
            if ind[0].any():
                if i == 0:
                    np.delete(self.asks, ind)
                else:
                    self.aq[ind] += message.aq[num]
            else:
                np.searchsorted(self.asks, i)
                
                
def producer():
    for i in range(10):
        cacheQueue.put(s.recieve())


def consumer():
    while True:
        message = cacheQueue.get()
        a.update_bids(message)
        a.update_asks(message)
        print(sum(a.bids))

s = Stream("BTCUSDT")


t1 = Thread(target=producer)
t1.start()
time.sleep(5)
a = OrderBook("btcusdt")
t2 = Thread(target=consumer)
t2.start()
