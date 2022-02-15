import asyncio
from binance import AsyncClient, BinanceSocketManager
import nest_asyncio


nest_asyncio.apply()

async def listener(client):
    bm = BinanceSocketManager(client)
    symbol = 'BNBBTC'
    async with client.get_order_book(symbol=symbol) as stream:
        while True:
            res = await stream.recv()
            print(res)


async def main():
    client = await AsyncClient.create()
    await listener(client)
    client.close_connection(bm)

loop = asyncio.get_event_loop()
loop.run_until_complete(main())