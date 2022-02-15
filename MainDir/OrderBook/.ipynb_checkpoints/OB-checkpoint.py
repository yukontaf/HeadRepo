import time

from binance import ThreadedWebsocketManager


def main():

    symbol = 'BNBBTC'

    twm = ThreadedWebsocketManager()
    twm.start()
    msgs = []
    def handle_socket_message(msg):
        msgs.append(msg)
    

    twm.start_depth_socket(callback=handle_socket_message, symbol=symbol, interval=100, depth=100)
    
    time.sleep(5)
    twm.stop()
    print(msgs[0])


if __name__ == "__main__":
    main()