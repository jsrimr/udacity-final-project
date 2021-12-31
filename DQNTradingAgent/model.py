import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .default_hyperparameters import N_QUANT, INIT_SIGMA, LINEAR, FACTORIZED, RISK_AVERSE


class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, initial_sigma=INIT_SIGMA, factorized=FACTORIZED):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.initial_sigma = initial_sigma
        self.factorized = factorized
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.noisy_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
            self.noisy_bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
            self.register_parameter('noisy_bias', None)
        self.reset_parameters()

        self.noise = True

    def reset_parameters(self):
        if self.factorized:
            sqrt_input_size = math.sqrt(self.weight.size(1))
            bound = 1 / sqrt_input_size
            nn.init.constant_(self.noisy_weight, self.initial_sigma / sqrt_input_size)
        else:
            bound = math.sqrt(3 / self.weight.size(1))
            nn.init.constant_(self.noisy_weight, self.initial_sigma)
        nn.init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -bound, bound)
            if self.factorized:
                nn.init.constant_(self.noisy_bias, self.initial_sigma / sqrt_input_size)
            else:
                nn.init.constant_(self.noisy_bias, self.initial_sigma)

    def forward(self, input):
        if self.noise:
            if self.factorized:
                input_noise = torch.randn(1, self.noisy_weight.size(1), device=self.noisy_weight.device)
                input_noise = input_noise.sign().mul(input_noise.abs().sqrt())
                output_noise = torch.randn(self.noisy_weight.size(0), device=self.noisy_weight.device)
                output_noise = output_noise.sign().mul(output_noise.abs().sqrt())
                weight_noise = input_noise.mul(output_noise.unsqueeze(1))
                bias_noise = output_noise
            else:
                weight_noise = torch.randn_like(self.noisy_weight)
                bias_noise = None if self.bias is None else torch.randn_like(self.noisy_bias)

            if self.bias is None:
                return F.linear(
                    input,
                    self.weight.add(self.noisy_weight.mul(weight_noise)),
                    None
                )
            else:
                return F.linear(
                    input,
                    self.weight.add(self.noisy_weight.mul(weight_noise)),
                    self.bias.add(self.noisy_bias.mul(bias_noise))
                )
        return F.linear(input, self.weight, self.bias)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}, initial_sigma={}, factorized={}'.format(
            self.in_features, self.out_features, self.bias is not None, self.initial_sigma, self.factorized
        )

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, action_size, obs_len, num_features=16, linear_type=LINEAR,
                 initial_sigma=INIT_SIGMA, factorized=FACTORIZED):
        """Initialize parameters and build model.
        Params
        ======
            action_size (int): Dimension of each action
            num_features (int): Number of features in the state
            n_atoms (int): number of support atoms
            linear_type (str): type of linear layers ('linear', 'noisy')
            initial_sigma (float): initial weight value for noise parameters
                when using noisy linear layers
        """
        super(QNetwork, self).__init__()
        self.action_size = action_size
        self.obs_len = obs_len
        self.num_features = num_features
        self.linear_type = linear_type.lower()
        self.factorized = bool(factorized)

        def noisy_layer(in_features, out_features):
            return NoisyLinear(in_features, out_features, True, initial_sigma, factorized)

        linear = {'linear': nn.Linear, 'noisy': noisy_layer}[self.linear_type]

        # Bottleneck idea from Google's MobileNetV2

        # N * obs_len * num_features
        # x.transpose(-1, -2).contiguous()
        # x = (N, L, C)
        self.norm = nn.InstanceNorm1d(self.num_features)
        self.embedding = nn.Linear(self.num_features, 512)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, dropout=0.1, activation='relu')

        self.phi = linear(1, 512)
        self.fc = linear(512, 64)
        self.fc_q = linear(64, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""

        # state = (N,L,C)
        x = state.transpose(-1, -2).contiguous()
        x = self.norm(x)
        x = x.transpose(-1, -2).contiguous()

        x = self.embedding(x)
        x = self.encoder_layer(x)  # (N, L, 512)
        x = x.mean(dim=1)  # (N, 512)

        tau = torch.rand(N_QUANT, 1) * RISK_AVERSE  # (N_QUANT, 1)
        quants = torch.arange(0, N_QUANT, 1.0)
        if torch.cuda.is_available():
            tau = tau.cuda()
            quants = quants.cuda()
        cos_trans = torch.cos(quants * tau * 3.141592).unsqueeze(2)  # (N_QUANT, N_QUANT, 1)
        rand_feat = F.relu(self.phi(cos_trans).mean(dim=1)).unsqueeze(0)  # (1, N_QUANT, 512)

        x = x.unsqueeze(1)  # (m, 1, 512)
        x = x * rand_feat  # (m, N_QUANT, 512)
        x = F.relu(self.fc(x))  # (m, N_QUANT, 64)

        # note that output of IQN is quantile values of value distribution
        action_value = self.fc_q(x).transpose(1, 2)  # (m, N_ACTIONS, N_QUANT)

        return action_value, tau

        # state_value = self.fc_s(x)  # (512, N_atom)
        #
        # advantage_values = self.fc_a(x)
        # advantage_values = advantage_values.view(
        #     advantage_values.size()[:-1] + (self.action_size, self.n_atoms))  # (N, L, action_size, N_atom)
        #
        # dist_weights = state_value.unsqueeze(dim=-2) + advantage_values - advantage_values.mean(dim=-2, keepdim=True)
        #
        # return dist_weights

    def noise(self, enable):
        enable = bool(enable)
        for m in self.children():
            if isinstance(m, NoisyLinear):
                m.noise = enable
