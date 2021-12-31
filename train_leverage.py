import os

import neptune.new as neptune
import numpy as np
import pandas as pd
import torch

import DQNTradingAgent.dqn_agent as dqn_agent
from arguments import argparser
from custom_hyperparameters import hyperparams
from leverage_trading_env import TradingEnv, action2position
from data_downloader import read_binance_futures_data

from config import API_TOKEN
# run = neptune.init(project='jeffrey/RL-trade',
#                    api_token=API_TOKEN,
#                    tags=['IQN'])

args = argparser()

torch.cuda.manual_seed_all(7)

device = torch.device("cuda:{}".format(args.device_num))
dqn_agent.set_device(device)

if not os.path.exists(args.save_location):
    os.makedirs(args.save_location)

save_interval = 100
print_interval = 1

n_episodes = args.n_episodes
sample_len = 480
obs_data_len = 192
step_len = 1

risk_aversion_multiplier = 0.5 + args.risk_aversion_multiplier / 2

n_action_intervals = 5

init_budget = 10000000

torch.save(hyperparams, os.path.join(args.save_location, "hyperparams.pth"))

df = read_binance_futures_data(args.data_path, args.symbol, args.timeframe)


def main():
    env = TradingEnv(custom_args=args, env_id='leverage_trading_env', obs_data_len=obs_data_len, step_len=step_len,
                     sample_len=sample_len,
                     df=df, fee=0.001, initial_budget=10000, n_action_intervals=n_action_intervals,
                     deal_col_name='close', sell_at_end=True,
                     feature_names=['open', 'high', 'low', 'close', 'volume', ])
    agent = dqn_agent.Agent(action_size=2 * n_action_intervals + 1, risk_averse_ratio=args.risk_aversion_multiplier, obs_len=obs_data_len,
                            num_features=env.observation_space[1], **hyperparams)

    # agent.qnetwork_local.load_state_dict(torch.load(args.load_file, map_location=device))
    # agent.qnetwork_local.to(device)

    beta = 0.4
    beta_inc = (1 - beta) / 1000
    agent.beta = beta

    scores_list = []

    for n_epi in range(1, n_episodes + 1):
        state, info = env.reset()
        score = 0.
        actions = []
        rewards = []
        price_list = []

        while True:
            action = int(agent.act(state, eps=0.))
            # if action > 5:
            #     action = 5
            actions.append(action)
            next_state, reward, done, info = env.step(action2position[action])
            price_list.append(info.cur_price)
            rewards.append(reward)
            score += reward

            # print(state[-1][3], f"r={reward:4f}, a={action}, asset={info.budget:.2f}, pos={info.position:.2f}, p.m={info.price_mean:.2f}")

            if reward < 0:
                reward *= risk_aversion_multiplier
            if done:
                action = 2 * n_action_intervals
            agent.step(state, action, reward, next_state, done)

            

            state = next_state
            if done:
                break
        else:
            agent.memory.reset_multisteps()  # todo : multi-step learning

        beta = min(1, beta + beta_inc)
        agent.beta = beta

        
        scores_list.append(score)
        # run["train/reward"].log(score)

        if n_epi % print_interval == 0 and n_epi != 0:
            # print_str = f"# of episode: {n_epi:d}, avg score: {sum(scores_list[-print_interval:]) / print_interval:.4f}, \
            #                 current_asset: {info.budget + info.cur_price * info.position}, Actions & next_price: {list(zip(actions, price_list))}"
            print_str = f"# of episode: {n_epi:d}, avg score: {sum(scores_list[-print_interval:]) / print_interval:.4f}, asset={info.budget:.2f}, \
                        action={np.array(actions)}"
            print(print_str)
            with open(os.path.join(args.save_location, "output_log.txt"), mode='a') as f:
                f.write(print_str + '\n')

        if n_epi % save_interval == 0:
            torch.save(agent.qnetwork_local.state_dict(),
                       os.path.join(args.save_location, f'IQN_leverage_{n_epi:d}.pth'))
            torch.save(scores_list, os.path.join(args.save_location, 'scores.pth'))

    del env


if __name__ == '__main__':
    # run['parameters'] = vars(args)
    main()
