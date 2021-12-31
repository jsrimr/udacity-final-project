import argparse
import datetime

def argparser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--agent_num', type=int, default=1,
                        help='which agent to load')

    parser.add_argument('--device_num', type=int, default=0,
                        help='cuda device num')

    parser.add_argument('--save_num', type=int, default=1,
                        help='folder name')

    parser.add_argument('--risk_aversion_multiplier', type=float, default=1.,
                        help='risk_aversion_level')

    parser.add_argument('--n_episodes', type=int, default=3000,
                        help='risk_aversion_level')

    parser.add_argument('--fee', type=float, default=.001,
                        help='fee percentage')
    
    parser.add_argument('--render', type=bool, default=False,
                        help='want to render?')
    
    parser.add_argument('--environment', type=str, default="default",
                        help='what environment to use')

    parser.add_argument('--save_location', type=str, default=f"saves/{datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}",
                        help='where to save?')

    parser.add_argument('--load_file', type=str, default="TradingGym_Rainbow_3000.pth",
                        help='which to load to save?')

    parser.add_argument('--data_path', type=str, default="binance_futures.db",
                        help='where to load source data')

    parser.add_argument('--symbol', type=str, default="BTCUSDT",
                        help='what symbol to read')
    
    parser.add_argument('--timeframe', type=str, default="1h",
                        help='resample period. ex : 10m, 1h, 4h, 1d')

    args = parser.parse_args()
    return args
