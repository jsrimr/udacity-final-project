import random
from collections import namedtuple, deque
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from .model import QNetwork
from .replay_buffer import ReplayBuffer, rp_set_device
from .default_hyperparameters import SEED, BUFFER_SIZE, BATCH_SIZE, START_SINCE,\
                                    GAMMA, T_UPDATE, TAU, LR, WEIGHT_DECAY, UPDATE_EVERY,\
                                    A, INIT_BETA, P_EPS, N_STEPS, CLIP, INIT_SIGMA, LINEAR, FACTORIZED

# device = torch.device("cpu")
device = "cuda" if torch.cuda.is_available() else "cpu"

def set_device(new_device):
    global device
    device = new_device
    rp_set_device(new_device)

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, action_size, obs_len, num_features=16, seed=SEED, batch_size=BATCH_SIZE,
                 buffer_size=BUFFER_SIZE, start_since=START_SINCE, gamma=GAMMA, target_update_every=T_UPDATE,
                 tau=TAU, lr=LR, weight_decay=WEIGHT_DECAY, update_every=UPDATE_EVERY, priority_eps=P_EPS,
                 a=A, initial_beta=INIT_BETA, n_multisteps=N_STEPS,
                 clip=CLIP, initial_sigma=INIT_SIGMA, linear_type=LINEAR, factorized=FACTORIZED, **kwds):
        """Initialize an Agent object.

        Params
        ======
            action_size (int): dimension of each action
            obs_len(int)
            num_features (int): number of features in the state
            seed (int): random seed
            batch_size (int): size of each sample batch
            buffer_size (int): size of the experience memory buffer
            start_since (int): number of steps to collect before start training
            gamma (float): discount factor
            target_update_every (int): how often to update the target network
            tau (float): target network soft-update parameter
            lr (float): learning rate
            weight_decay (float): weight decay for optimizer
            update_every (int): update(learning and target update) interval
            priority_eps (float): small base value for priorities
            a (float): priority exponent parameter
            initial_beta (float): initial importance-sampling weight
            n_multisteps (int): number of steps to consider for each experience
            v_min (float): minimum reward support value
            v_max (float): maximum reward support value
            clip (float): gradient norm clipping (`None` to disable)
            n_atoms (int): number of atoms in the discrete support distribution
            initial_sigma (float): initial noise parameter weights
            linear_type (str): one of ('linear', 'noisy'); type of linear layer to use
            factorized (bool): whether to use factorized gaussian noise in noisy layers
        """


        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        self.action_size         = action_size
        self.obs_len             = obs_len
        self.num_features        = num_features
        self.seed                = seed
        self.batch_size          = batch_size
        self.buffer_size         = buffer_size
        self.start_since         = start_since
        self.gamma               = gamma
        self.target_update_every = target_update_every
        self.tau                 = tau
        self.lr                  = lr
        self.weight_decay        = weight_decay
        self.update_every        = update_every
        self.priority_eps        = priority_eps
        self.a                   = a
        self.beta                = initial_beta
        self.n_multisteps        = n_multisteps

        self.clip                = clip
        self.initial_sigma       = initial_sigma
        self.linear_type         = linear_type.strip().lower()
        self.factorized          = factorized


        # Q-Network
        self.qnetwork_local  = QNetwork(action_size, obs_len, num_features, linear_type, initial_sigma, factorized).to(device)
        self.qnetwork_target = QNetwork(action_size, obs_len, num_features, linear_type, initial_sigma, factorized).to(device)
        self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())

        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=lr, weight_decay=weight_decay)

        # Replay memory
        self.memory = ReplayBuffer(buffer_size, batch_size, n_multisteps, gamma, a, False)
        # Initialize time step (for updating every UPDATE_EVERY steps and TARGET_UPDATE_EVERY steps)
        self.u_step = 0
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        #  experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps.
        self.u_step = (self.u_step + 1) % self.update_every
        if self.u_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) >= self.start_since:
                experiences, target_discount, is_weights, indices = self.memory.sample(self.beta)
                new_priorities = self.learn(experiences, is_weights, target_discount)
                self.memory.update_priorities(indices, new_priorities)

        # update the target network every TARGET_UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.target_update_every
        if self.t_step == 0:
            self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.

        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        if np.random.uniform() >= eps:
            # greedy case
            action_value, tau = self.qnetwork_local(state)  # (n, N_ACTIONS, N_QUANT)
            action_value = action_value.mean(dim=2)
            action = torch.argmax(action_value, dim=1).data.cpu().numpy()
        else:
            # random exploration case
            action = np.random.randint(0, self.action_size, (state.size(0)))
        return action

    def learn(self, experiences, is_weights, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            is_weights (torch.Tensor): tensor of importance-sampling weights
            gamma (float): discount factor for the target max-Q value

        Returns
        =======
            new_priorities (List[float]): list of new priority values for the given sample
        """
        b_s, b_a, b_r, b_s_, b_d = experiences

        # action value distribution prediction
        q_eval, q_eval_tau = self.qnetwork_local(b_s)  # (m, N_ACTIONS, N_QUANT), (N_QUANT, 1)
        q_next_eval, q_next_eval_tau = self.qnetwork_local(b_s_)
        best_actions = q_next_eval.mean(dim=2).argmax(dim=1)  # (m)

        mb_size = q_eval.size(0)
        q_eval = torch.stack([q_eval[i].index_select(0, b_a[i]) for i in range(mb_size)]).squeeze(1)
        # (m, N_QUANT)
        q_eval = q_eval.unsqueeze(2)  # (m, N_QUANT, 1)
        # note that dim 1 is for present quantile, dim 2 si for next quantile

        # get next state value
        q_next, q_next_tau = self.qnetwork_target(b_s_)  # (m, N_ACTIONS, N_QUANT), (N_QUANT, 1)

        # best_actions = q_next.mean(dim=2).argmax(dim=1)  # (m)
        q_next = torch.stack([q_next[i].index_select(0, best_actions[i]) for i in range(mb_size)]).squeeze(1)  # (m, N_QUANT)
        q_target = b_r + gamma * (1. - b_d) * q_next  # (m, N_QUANT)
        q_target = q_target.unsqueeze(1).detach()  # (m , 1, N_QUANT)

        # quantile Huber loss
        u = q_target.detach() - q_eval  # (m, N_QUANT, N_QUANT)
        tau = q_eval_tau.unsqueeze(0)  # (1, N_QUANT, 1)
        # note that tau is for present quantile
        weight = torch.abs(tau - u.le(0.).float())  # (m, N_QUANT, N_QUANT)
        loss = F.smooth_l1_loss(q_eval, q_target.detach(), reduction='none')
        # (m, N_QUANT, N_QUANT)
        loss = torch.mean(weight * loss, dim=1).mean(dim=1)

        new_priorities = loss.detach().add(self.priority_eps).cpu().numpy()
        loss = (loss * is_weights.squeeze(1)).mean()

        # calc importance weighted loss
        # b_w = torch.Tensor(b_w)
        # if USE_GPU:
        #     b_w = b_w.cuda()
        # loss = torch.mean(b_w * loss)

        # backprop loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return new_priorities


    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    def to(self, device):
        self.qnetwork_local  = self.qnetwork_local.to(device)
        self.qnetwork_target = self.qnetwork_target.to(device)
