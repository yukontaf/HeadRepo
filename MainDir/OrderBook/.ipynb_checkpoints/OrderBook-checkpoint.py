import asyncio
from binance import AsyncClient, BinanceSocketManager
import nest_asyncio
import pandas as pd


nest_asyncio.apply()

async def order_book(client, symbol, books):
    order_book = await client.get_order_book(symbol=symbol)
    books.append(order_book)
    print(books)

async def listener(client, books):
  bm = BinanceSocketManager(client)
  symbol = 'BNBBTC'
  cnt = 0
  async with bm.depth_socket(symbol=symbol) as stream:
    while True:
      res = await stream.recv()
      cnt += 1
      if cnt == 10:
        #print(res)
        loop.call_soon(asyncio.create_task, order_book(client, symbol, books))
        break
        
      

async def main():
    books = []
    client = await AsyncClient.create()
    await listener(client, books)


loop = asyncio.get_event_loop()
loop.run_until_complete(main())

