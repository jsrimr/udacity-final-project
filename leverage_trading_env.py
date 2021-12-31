import logging

import numpy as np
from dataclasses import dataclass

@dataclass
class info:
    remain_time: int
    budget: float
    position: float
    price_mean: float
    cur_price: float


action2position = {
    0: 0.2,  # use 20% of budget to long
    1: 0.4,
    2: 0.6,
    3: 0.8,
    4: 1.0,  # full long
    5: 0,    # no position 
    6: -0.2, # use 20% of budget to short
    7: -0.4,
    8: -0.6,
    9: -0.8,
    10: -1.0, # full short
}

class TradingEnv:
    def __init__(self, env_id, df, sample_len, obs_data_len, step_len,
                 fee, initial_budget, deal_col_name='c',
                 feature_names=['c', 'v'], leverage=3, sell_at_end=True, *args, **kwargs):

        assert 0 <= fee <= 0.01, "fee must be between 0 and 1 (0% to 1%)"
        assert deal_col_name in df.columns, "deal_col not in Dataframe please define the correct column name of which column want to calculate the profit."
        for col in feature_names:
            assert col in df.columns, "feature name: {} not in Dataframe.".format(col)

        self.total_fee = 0

        logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s')
        self.logger = logging.getLogger(env_id)

        self.df = df
        self.sample_len = sample_len
        self.obs_len = obs_data_len
        self.step_len = step_len
        self.fee = fee
        self.initial_budget = initial_budget
        self.budget = initial_budget
        self.feature_len = len(feature_names)
        self.observation_space = np.array([obs_data_len, self.feature_len])
        self.using_feature = feature_names
        self.price_name = deal_col_name
        self.leverage = leverage  # 10x -> 0.1, 100x -> 0.01

    def _random_choice_section(self):
        begin_point = np.random.randint(len(self.df) - self.sample_len + 1)
        end_point = begin_point + self.sample_len
        df_section = self.df.iloc[begin_point: end_point]
        return df_section

    def reset(self):
        self.total_fee = 0
        self.df_sample = self._random_choice_section()
        self.step_st = 0
        # define the price to calculate the reward
        self.price = self.df_sample[self.price_name].to_numpy()

        self.long_liquidation_price = 0
        self.short_liquidation_price = np.Inf

        self.remain_time = (self.sample_len - self.obs_len) / self.step_len
        self.budget = self.initial_budget
        self.position = 0
        self.price_mean = 0
        self.margin = 0

        # define the observation feature
        self.obs_features = self.df_sample[self.using_feature].to_numpy()
        # maybe make market position feature in final feature, set as option

        # observation part : features + eps 남은 기간 + 현재 포지션 + budget (평단가는 고려안함)
        self.obs_state = self.obs_features[self.step_st: self.step_st + self.obs_len]
        # self.state = np.hstack(
        #     [self.obs_state, ])

        return self.obs_state, info(self.remain_time, self.budget, self.position, self.price_mean, self.obs_state[-1][3])

    @staticmethod
    def get_liq_price(price_mean, budget, position):
        if position >= 0:  # long_position
            liq_price = price_mean - budget/(position + 1e-9)
        else:
            liq_price = price_mean + budget/(position + 1e-9)
        return liq_price

    def test_state(self):
        self.total_fee = 0
        self.df_sample = self.df.iloc[-self.obs_len:]
        self.price = self.df_sample[self.price_name].to_numpy()

        self.remain_time = (self.sample_len - self.obs_len) / self.step_len
        self.budget = self.initial_budget
        self.position = 0
        self.price_mean = 0

        self.obs_state = self.df_sample[self.using_feature].to_numpy()

        return self.obs_state, info(self.remain_time, self.budget, self.position, self.price_mean, self.obs_state[-1][3])

    def step(self, action):  # action = 1[-1 , 1]
        """
        action : 다음 포지션 어떻게 가져갈 것인지
        ex) btc 1 개 들고 있는데 action 이 1.0 -> 가지고 있는 예산 다 털어 btc 추가매입
            btc 1 개 들고 있는데 action 이 -1.0 -> 가지고 있던 btc 팔고 남은 현금 전체로 short

        reward 는 _cover 에서 발생
        short, long 포지션 진입 시에는 reward 발생하지 않음

        # reward 를 자신이 결정할 수 있음. 현재 state 의 가격에 의해 reward 가 결정됨. 보통 mdp 는 reward 가 next_state 에 의해 결정되는데
        # unrealized PNL 은 고려하지 않음. -> 마지막에만 결정
        # regret 이 필요함


        :param action: [long, short] * n_interval + hold_action
        :return: next_state, reward, done,  [self.remain_time, self.budget, self.position, self.price_mean]
        """
        
        current_price = self.price[self.step_st + self.obs_len - 1]
        
        # next_price = self.price[self.step_st + self.obs_len - 1]
        pnl = (current_price - self.price_mean) * self.position
        self.budget += pnl

        liquidation_price = TradingEnv.get_liq_price(self.price_mean, self.budget, self.position)
        low = self.obs_state[-1][2]
        high = self.obs_state[-1][1]
        if (self.position > 0 and low < liquidation_price) or \
            (self.position < 0 and high > liquidation_price):  # long liquidation or short liquidation
            reward = pnl / self.initial_budget
            self.position = 0
            if self.budget < 10:  # if budget is lower than 10 due to liquidation : game ends
                done = True
                return self.obs_state, np.clip(reward, -1, 1), done, info(0, 0, 0, 0, self.obs_state[-1][3])

        current_price_mean = self.price_mean
        current_mkt_position = self.position
        current_asset = self.budget

        # observation part
        self.remain_time -= 1
        self.step_st += self.step_len
        self.obs_state = self.obs_features[self.step_st: self.step_st + self.obs_len]
        done = False
        reward = pnl / self.initial_budget

        if self.step_st + self.obs_len >= len(self.obs_features) and self.remain_time == 0:  # episode ends
            done = True
            reward = (current_price - current_price_mean) * current_mkt_position / self.initial_budget
            return self.obs_state, np.clip(reward, -1, 1), done, info(self.remain_time, self.budget, self.position,
                                                                  self.price_mean, current_price)

        if action == 0:  # Clear position
            self.price_mean = 0
            self.position = 0  
            return self.obs_state, np.clip(reward, -1, 1), done, info(self.remain_time, self.budget, self.position, self.price_mean, current_price)

        # 보유수량, 평단가, pnl 계산
        target_position = current_asset * action * self.leverage / current_price  # 같은 action 이여도 current_asset 이 줄면 position 에 변화를 주게 되는구나
        if action > 0:  # If target position is long

            if self.position < 0:  # short 포지션이었으면 일단 정리. # short_cover
                self.price_mean = 0
                self.position = 0

            if self.position > target_position:  # long 포지션 일부정리

                self.position = target_position

                # self.long_liquidation_price = 0  # todo : 크로스 마진 시에는 asset 에 dependent 하도록 바뀌어야

            else:  # long 포지션 늘리기

                self.price_mean = (current_price_mean * self.position + current_price * (
                        target_position - self.position)) / target_position
                self.position = target_position


        else:  # target position is short

            if self.position > 0:  # long 포지션이었으면 일단 정리. # short_cover

                self.price_mean = 0
                self.position = 0

            if self.position < target_position:  # short 포지션 일부정리

                self.position = target_position

            else:  # short 포지션 늘리기
                self.price_mean = (self.position * current_price_mean + current_price * (
                        target_position - self.position)) / target_position
                self.position = target_position

        return self.obs_state, np.clip(reward, -1, 1), done, info(self.remain_time, self.budget, self.position, self.price_mean, current_price)
