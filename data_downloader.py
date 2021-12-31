import argparse
import datetime
import os
import sqlite3
import time

import ccxt
import pandas as pd


def download_binance_futures_data(market, db_path="binance_futures.db", symbols="all"):
    # DB 초기화
    db = sqlite3.connect(db_path)

    binance = ccxt.binance(
        {
            "options": {
                "defaultType": market
            },
            "enableRateLimit": True
        }
    )

    if symbols == "all":  # download all ticker
        symbols = [mkt["symbol"] for mkt in binance.fetch_markets()]
    
    else:  # download only specific ticker
        symbols = symbols.split(",")
    print(f"downloading data for {len(symbols)} symbols : {symbols}")

    for symbol in symbols:

        db.execute(f"""
        CREATE TABLE IF NOT EXISTS _{symbol.replace("/", "")} (
            timestamp int, 
            open float, 
            high float, 
            low float, 
            close float, 
            volume float
        )""")
        t = time.time()

        # make table and commit
        db.commit()

        prev_data = db.execute(f"SELECT * FROM _{symbol.replace('/', '')}").fetchall()

        # check previous data
        # new_data starts from (latest_previous_data + 1) timestamp       
        then = datetime.datetime(2021, 1, 1)
        timestamp = int(time.mktime(then.timetuple()) * 1000) if not prev_data else prev_data[-1][0] + 1

        downloaded = 0  # 로깅용

        while True:
            # download ohlcv data
            tohlcv = binance.fetch_ohlcv(
                symbol=symbol,
                timeframe=args.timeframe,
                params={"startTime": timestamp},
                limit=1500  
            )

            # no more data to download => break loop
            if not tohlcv:
                break

            # save to db
            for timestamp, open, high, low, close, volume in tohlcv:
                db.execute(f"""
                INSERT INTO _{symbol.replace('/', '')} VALUES (
                    {timestamp}, {open}, {high}, {low}, {close}, {volume}
                )""")

            db.commit()

            # prepare timestamp in the next loop
            timestamp = tohlcv[-1][0] + 1

            # amount of data downloaded
            downloaded += len(tohlcv)

            # time passed since start_time
            delta_t = time.time() - t

            # logging
            print(
                f"""downloaded {downloaded} rows for {symbol} in {round(delta_t)} seconds, download speed is {round(downloaded / delta_t)} row per second""",
                end="\r")
        print(
            f"""downloaded {downloaded} rows for {symbol} in {round(delta_t)} seconds, download speed is {round(downloaded / delta_t)} row per second""")

        db.commit()


def read_binance_futures_data(db_path, symbol, timeframe):
    
    symbol = symbol.replace("/", "")
    db = sqlite3.connect(db_path)

    # read all the data
    data = db.execute(f"SELECT * FROM _{symbol}").fetchall()

    # make dataframe from the raw data
    data = pd.DataFrame(
        data, columns=["timestamp", "open", "high", "low", "close", "volume"])

    # timestamp to datetime, and set as index
    data.index = pd.to_datetime(data["timestamp"] * 1000000)

    del data["timestamp"]

    return data


def export_data(db_path, symbols, timeframes, export_dir):
    timeframes = timeframes.split(",")

    # what symbol to export
    if symbols == "all":
        binance = ccxt.binance(
            {
                "options": {
                    "defaultType": "future"
                },
                "enableRateLimit": True
            }
        )
        symbols = [mkt["symbol"] for mkt in binance.fetch_markets()]

    else:
        symbols = symbols.split(",")

    # exporting loop
    for symbol in symbols:
        for timeframe in timeframes:
            # fetch data
            df = read_binance_futures_data(db_path)

            # export path: export_dir/symbol_timeframe.csv
            export_path = os.path.join(export_dir, f"{symbol.replace(' / ', '')}_{timeframe}.csv")

            # export in csv format
            df.to_csv(export_path)

            print(f"exported data to {export_path}")


if __name__ == "__main__":
    # cli 인터페이스

    # argparse
    parser = argparse.ArgumentParser()

    # argument #0 market "spot" or "future"
    parser.add_argument("--market", default="future", type=str)
    parser.add_argument("--db_path", default="binance_futures.db", type=str)

    # argument #2 symbols: "all" or specific ticker. ex) "BTC/USDT" 
    parser.add_argument("--symbols", default='all', type=str)

    # path to export data
    parser.add_argument("--export_dir", default=None, type=str)

    # timeframe to export
    parser.add_argument("--export_timeframes", default="1T", type=str)

    args = parser.parse_args()

    if not args.market in ["future", "spot"]:
        raise ValueError(f"market should be 'spot' or 'future', got {args.market}")

    # download
    if args.export_dir is None:
        download_binance_futures_data(args.market, args.db_path, args.symbols)
    # export
    else:
        export_data(args.db_path, args.symbols, args.export_timeframes, args.export_dir)
