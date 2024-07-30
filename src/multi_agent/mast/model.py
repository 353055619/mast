import math

from einops import rearrange, repeat
from torch.distributions import Normal
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
from typing import Tuple

from safepo.utils.util import init, check

__all__ = ['init_',
           'SelfAttention',
           'EncodeBlock',
           'DecodeBlock',
           'ObsEncoder',
           'MultiAgentBaseModel',
           'MultiAgentModel',
           'MultiAgentModelDense']


def init_(m, gain=0.01, activate=False):
    if activate:
        gain = nn.init.calculate_gain("relu")
    return init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain=gain)


class SelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, n_agent, masked=False):
        super(SelfAttention, self).__init__()

        assert n_embd % n_head == 0
        self.masked = masked
        self.n_head = n_head
        # key, query, value projections for all heads
        self.key = init_(nn.Linear(n_embd, n_embd))
        self.query = init_(nn.Linear(n_embd, n_embd))
        self.value = init_(nn.Linear(n_embd, n_embd))
        # output projection
        self.proj = init_(nn.Linear(n_embd, n_embd))
        # if self.masked:
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(n_agent + 1, n_agent + 1)).view(
                1, 1, n_agent + 1, n_agent + 1
            ),
        )

        self.att_bp = None

    def forward(self, key, value, query):
        B, L, D = query.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = (
            self.key(key).view(B, L, self.n_head, D // self.n_head).transpose(1, 2)
        )  # (B, nh, L, hs)
        q = (
            self.query(query).view(B, L, self.n_head, D // self.n_head).transpose(1, 2)
        )  # (B, nh, L, hs)
        v = (
            self.value(value).view(B, L, self.n_head, D // self.n_head).transpose(1, 2)
        )  # (B, nh, L, hs)

        # causal attention: (B, nh, L, hs) x (B, nh, hs, L) -> (B, nh, L, L)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        # self.att_bp = F.softmax(att, dim=-1)

        if self.masked:
            att = att.masked_fill(self.mask[:, :, :L, :L] == 0, float("-inf"))

        att = F.softmax(att, dim=-1)

        y = att @ v  # (B, nh, L, L) x (B, nh, L, hs) -> (B, nh, L, hs)
        y = (
            y.transpose(1, 2).contiguous().view(B, L, D)
        )  # re-assemble all head outputs side by side

        # output projection
        y = self.proj(y)
        return y


class EncodeBlock(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, n_embd, n_head, n_agent):
        super(EncodeBlock, self).__init__()

        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn = SelfAttention(n_embd, n_head, n_agent, masked=False)
        self.mlp = nn.Sequential(
            init_(nn.Linear(n_embd, 1 * n_embd), activate=True),
            nn.GELU(),
            init_(nn.Linear(1 * n_embd, n_embd)),
        )

    def forward(self, x):
        x = self.ln1(x + self.attn(x, x, x))
        x = self.ln2(x + self.mlp(x))
        return x


class DecodeBlock(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, n_embd, n_head, n_agent):
        super(DecodeBlock, self).__init__()

        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ln3 = nn.LayerNorm(n_embd)
        self.attn1 = SelfAttention(n_embd, n_head, n_agent, masked=True)
        self.attn2 = SelfAttention(n_embd, n_head, n_agent, masked=True)
        self.mlp = nn.Sequential(
            init_(nn.Linear(n_embd, 1 * n_embd), activate=True),
            nn.GELU(),
            init_(nn.Linear(1 * n_embd, n_embd)),
        )

    def forward(self, x, rep_enc):
        x = self.ln1(x + self.attn1(x, x, x))
        x = self.ln2(rep_enc + self.attn2(key=x, value=x, query=rep_enc))
        x = self.ln3(x + self.mlp(x))
        return x


class ObsEncoder(nn.Module):  # Check
    def __init__(self, obs_dim, n_block, n_embd, n_head, n_agent):
        super(ObsEncoder, self).__init__()

        self.obs_dim = obs_dim
        self.n_embd = n_embd
        self.n_agent = n_agent

        self.obs_encoder = nn.Sequential(
            nn.LayerNorm(obs_dim),
            init_(nn.Linear(obs_dim, n_embd), activate=True),
            nn.GELU()
        )

        self.ln = nn.LayerNorm(n_embd)
        self.blocks = nn.Sequential(*[EncodeBlock(n_embd, n_head, n_agent) for _ in range(n_block)])

    def forward(self, obs):
        obs_embeddings = self.obs_encoder(obs)
        obs_embeddings = self.ln(obs_embeddings)
        obs_representation = self.blocks(obs_embeddings)
        return obs_representation


class TransformerDecoder(nn.Module):
    def __init__(
            self,
            act_dim,
            embed_dim,
            n_agent,
            n_block,
            n_head
    ):
        super(TransformerDecoder, self).__init__()

        self.act_dim = act_dim
        self.embed_dim = embed_dim

        self.action_encoder = nn.Sequential(
            init_(nn.Linear(act_dim, embed_dim), activate=True),
            nn.GELU(),
            nn.LayerNorm(embed_dim)
        )
        self.act_decoder = nn.Sequential(
            init_(nn.Linear(embed_dim, embed_dim), activate=True),
            nn.GELU(),
            nn.LayerNorm(embed_dim),
            init_(nn.Linear(embed_dim, act_dim)),
        )
        # attention blocks
        self.blocks = nn.Sequential(
            *[DecodeBlock(embed_dim, n_head, n_agent) for _ in range(n_block)]
        )

    # state, action, and return
    def forward(self, obs_rep, action):
        # action: (batch, n_agent, action_dim), one-hot/logits?
        # obs_rep: (batch, n_agent, n_embd)
        x = self.action_encoder(action)
        for block in self.blocks:
            x = block(x, obs_rep)
        logit = self.act_decoder(x)
        return logit


class RNNDecoder(nn.Module):  # check
    def __init__(self, act_dim, embed_dim, n_agent, n_block, rnn: str):
        super(RNNDecoder, self).__init__()
        self.act_encoder = nn.Sequential(  # gru的动作前置编码器, act -> act_gru, check
            init_(nn.Linear(act_dim, embed_dim), activate=True),
            nn.GELU(),
            nn.LayerNorm(embed_dim)
        )
        self.act_decoder = nn.Sequential(  # gru的后置解码器, gru_output -> act, check
            init_(nn.Linear(embed_dim, embed_dim), activate=True),
            nn.GELU(),
            nn.LayerNorm(embed_dim),
            init_(nn.Linear(embed_dim, act_dim))
        )

        self.obs_to_h_decoder = nn.Sequential(  # gru的状态前置编码器, obs_rep -> h, similar to decoder of critic
            init_(nn.Linear(n_agent * embed_dim, embed_dim), activate=True),
            nn.GELU(),
            nn.LayerNorm(embed_dim),
            init_(nn.Linear(embed_dim, embed_dim))
        )
        # gru blocks

        if rnn == 'gru':
            self.blocks = nn.GRU(input_size=embed_dim, hidden_size=embed_dim, num_layers=n_block, batch_first=True)
        elif rnn == 'lstm':
            self.blocks = nn.LSTM(input_size=embed_dim, hidden_size=embed_dim, num_layers=n_block, batch_first=True)
        elif rnn == 'rnn':
            self.blocks = nn.RNN(input_size=embed_dim, hidden_size=embed_dim, num_layers=n_block, batch_first=True)
        else:
            raise NotImplementedError

        self.embed_dim = embed_dim
        self.act_dim = act_dim
        self.n_agent = n_agent
        self.rnn_type = rnn

    def forward(self, obs_rep: Tensor, action):
        """
        Args:
            obs_rep: (b, n, n_embd)
            action: (b, n, act_dim)
        """
        obs_rep_dense = rearrange(obs_rep, 'b n embed_dim -> b (n embed_dim)')
        h = self.obs_to_h_decoder(obs_rep_dense)  # (b, embed_dim)
        h = repeat(h, 'b d -> n b d', n=self.blocks.num_layers)
        x = self.act_encoder(action)
        if self.rnn_type == 'lstm':
            h = (h, torch.zeros_like(h, dtype=obs_rep.dtype, device=obs_rep.device))

        x, _ = self.blocks(x, h)

        logit = self.act_decoder(x)
        return logit


# class Critic(nn.Module):  # Check
#     def __init__(self, obs_dim, n_block, embed_dim, n_head, n_agent):
#         super().__init__()
#         self.encoder = ObsEncoder(obs_dim, n_block, embed_dim, n_head, n_agent)
#         self.decoder = nn.Sequential(
#             init_(nn.Linear(embed_dim, embed_dim), activate=True),
#             nn.GELU(),
#             nn.LayerNorm(embed_dim),
#             init_(nn.Linear(embed_dim, 1))
#         )
#
#     def forward(self, obs):
#         obs_representation = self.encoder(obs)
#         value = self.decoder(obs_representation)
#         return value


# class Actor(nn.Module):  # check
#     """
#     Actor类结合了ObsEncoder和ActDecoder，完成从观测到动作的生成过程
#     """
#
#     def __init__(self,
#                  obs_dim,
#                  act_dim,
#                  n_agent,
#                  embed_dim,
#                  n_block,
#                  n_head,
#                  actor_decoder_type,
#                  device):
#         super(Actor, self).__init__()
#         self.encoder = ObsEncoder(obs_dim, n_block, embed_dim, n_head, n_agent)
#         if actor_decoder_type == 'transformer':
#             self.decoder = TransformerDecoder(act_dim, embed_dim, n_agent, n_block, n_head)
#         self.log_std = torch.nn.Parameter(torch.ones(act_dim))
#
#         self.act_dim = act_dim
#         self.n_agent = n_agent
#         self.tpdv = dict(dtype=torch.float32, device=device)
#
#     def act_autoregression(self, obs_rep: Tensor):
#         batch_size, _, _ = obs_rep.shape
#         shifted_action = torch.zeros((batch_size, self.n_agent, self.act_dim)).to(**self.tpdv)
#         output_action = torch.zeros((batch_size, self.n_agent, self.act_dim)).to(**self.tpdv)
#         output_action_log = torch.zeros_like(output_action)
#
#         for i in range(self.n_agent):
#             act_mean = self.decoder(obs_rep, shifted_action)[:, i, :]
#             action_std = torch.sigmoid(self.log_std) * 0.5
#
#             distri = Normal(act_mean, action_std)
#             action = distri.sample()
#             action_log = distri.log_prob(action)
#
#             output_action[:, i, :] = action
#             output_action_log[:, i, :] = action_log
#             if i + 1 < self.n_agent:
#                 shifted_action[:, i + 1, :] = action
#
#         return output_action, output_action_log
#
#     def act_parallel(self, obs_rep: Tensor, action: Tensor):
#         batch_size, _, _ = obs_rep.shape
#         shifted_action = torch.zeros((batch_size, self.n_agent, self.act_dim)).to(**self.tpdv)
#         shifted_action[:, 1:, :] = action[:, :-1, :]
#
#         act_mean = self.decoder(obs_rep, shifted_action)
#         action_std = torch.sigmoid(self.log_std) * 0.5
#         distri = Normal(act_mean, action_std)
#
#         action_log = distri.log_prob(action)
#         entropy = distri.entropy()
#         return action_log, entropy
#
#     def forward(self, obs: Tensor, action: Tensor):
#         """返回智能体的动作序列"""
#         # Encode the observations
#         obs_rep = self.encoder(obs)
#
#         # Decode the actions
#         act_log, entropy = self.act_parallel(obs_rep, action)
#
#         return act_log, entropy
#
#     def get_actions(self, obs):
#         obs_rep = self.encoder(obs)
#
#         # Decode the actions
#         act, act_log = self.act_autoregression(obs_rep)
#         return act, act_log


class MultiAgentBaseModel(nn.Module):
    def __init__(self, obs_dim, act_dim, n_agent, n_block, embed_dim, n_head, actor_decoder_type, device):
        super(MultiAgentBaseModel, self).__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.n_agent = n_agent
        self.tpdv = dict(dtype=torch.float32, device=device)

        self.log_std = torch.nn.Parameter(torch.ones(act_dim))

        self.encoder_obs = ObsEncoder(obs_dim, n_block, embed_dim, n_head, n_agent)

        self.decoder_v = nn.Sequential(
            init_(nn.Linear(embed_dim, embed_dim), activate=True),
            nn.GELU(),
            nn.LayerNorm(embed_dim),
            init_(nn.Linear(embed_dim, 1))
        )
        self.decoder_v_cost = nn.Sequential(
            init_(nn.Linear(embed_dim, embed_dim), activate=True),
            nn.GELU(),
            nn.LayerNorm(embed_dim),
            init_(nn.Linear(embed_dim, 1))
        )

        if actor_decoder_type == 'transformer':
            self.decoder = TransformerDecoder(act_dim, embed_dim, n_agent, n_block, n_head)
        elif actor_decoder_type == 'rnn':
            self.decoder = RNNDecoder(act_dim, embed_dim, n_agent, n_block, 'rnn')
        elif actor_decoder_type == 'gru':
            self.decoder = RNNDecoder(act_dim, embed_dim, n_agent, n_block, 'gru')
        elif actor_decoder_type == 'lstm':
            self.decoder = RNNDecoder(act_dim, embed_dim, n_agent, n_block, 'lstm')
        else:
            raise NotImplementedError

    def act_autoregression(self, obs_rep: Tensor):
        batch_size, _, _ = obs_rep.shape
        shifted_action = torch.zeros((batch_size, self.n_agent, self.act_dim)).to(**self.tpdv)
        output_action = torch.zeros((batch_size, self.n_agent, self.act_dim)).to(**self.tpdv)
        output_action_log = torch.zeros_like(output_action)

        for i in range(self.n_agent):
            act_mean = self.decoder(obs_rep, shifted_action)[:, i, :]
            action_std = torch.sigmoid(self.log_std) * 0.5

            distri = Normal(act_mean, action_std)
            action = distri.sample()
            action_log = distri.log_prob(action)

            output_action[:, i, :] = action
            output_action_log[:, i, :] = action_log
            if i + 1 < self.n_agent:
                shifted_action[:, i + 1, :] = action

        return output_action, output_action_log

    def act_parallel(self, obs_rep: Tensor, action: Tensor):
        batch_size, _, _ = obs_rep.shape
        shifted_action = torch.zeros((batch_size, self.n_agent, self.act_dim)).to(**self.tpdv)
        shifted_action[:, 1:, :] = action[:, :-1, :]

        act_mean = self.decoder(obs_rep, shifted_action)
        action_std = torch.sigmoid(self.log_std) * 0.5
        distri = Normal(act_mean, action_std)

        action_log = distri.log_prob(action)
        entropy = distri.entropy()
        return action_log, entropy

    def forward(self, obs: Tensor, act: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        obs = check(obs).to(**self.tpdv)
        act = check(act).to(**self.tpdv)

        obs_rep = self.encoder_obs(obs)
        act_log, entropy = self.act_parallel(obs_rep, act)

        v = self.decoder_v(obs_rep)
        v_cost = self.decoder_v_cost(obs_rep)
        return act_log, entropy, v, v_cost

    def get_actions(self, obs: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        obs = check(obs).to(**self.tpdv)
        obs_rep = self.encoder_obs(obs)

        act, act_log = self.act_autoregression(obs_rep)

        v = self.decoder_v(obs_rep)
        v_cost = self.decoder_v_cost(obs_rep)

        return act, act_log, v, v_cost

    def get_values(self, obs: Tensor) -> Tensor:
        obs = check(obs).to(**self.tpdv)
        obs_rep = self.encoder_obs(obs)
        v = self.decoder_v(obs_rep)
        return v

    def get_cost_values(self, obs: Tensor) -> Tensor:
        obs = check(obs).to(**self.tpdv)
        obs_rep = self.encoder_obs(obs)
        v_cost = self.decoder_v_cost(obs_rep)
        return v_cost


class MultiAgentModel(MultiAgentBaseModel):
    def __init__(self, obs_dim, act_dim, n_agent, n_block, embed_dim, n_head, actor_decoder_type, device):
        super(MultiAgentModel, self).__init__(obs_dim, act_dim, n_agent, n_block, embed_dim, n_head, actor_decoder_type,
                                              device)

        self.obs_encoder_actor = ObsEncoder(obs_dim, n_block, embed_dim, n_head, n_agent)
        self.obs_encoder_critic = ObsEncoder(obs_dim, n_block, embed_dim, n_head, n_agent)
        self.obs_encoder_critic_cost = ObsEncoder(obs_dim, n_block, embed_dim, n_head, n_agent)

    def forward(self, obs: Tensor, act: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        obs = check(obs).to(**self.tpdv)
        act = check(act).to(**self.tpdv)

        obs_rep_actor = self.obs_encoder_actor(obs)
        obs_rep_critic = self.obs_encoder_critic(obs)
        obs_rep_critic_cost = self.obs_encoder_critic_cost(obs)

        act_log, entropy = self.act_parallel(obs_rep_actor, act)

        v = self.decoder_v(obs_rep_critic)
        v_cost = self.decoder_v_cost(obs_rep_critic_cost)

        return act_log, entropy, v, v_cost

    def get_actions(self, obs: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        obs = check(obs).to(**self.tpdv)
        obs_rep_actor = self.obs_encoder_actor(obs)
        obs_rep_critic = self.obs_encoder_critic(obs)
        obs_rep_critic_cost = self.obs_encoder_critic_cost(obs)

        act, act_log = self.act_autoregression(obs_rep_actor)

        v = self.decoder_v(obs_rep_critic)
        v_cost = self.decoder_v_cost(obs_rep_critic_cost)

        return act, act_log, v, v_cost

    def get_values(self, obs: Tensor) -> Tensor:
        obs = check(obs).to(**self.tpdv)
        obs_rep_critic = self.obs_encoder_critic(obs)
        v = self.decoder_v(obs_rep_critic)
        return v

    def get_cost_values(self, obs: Tensor) -> Tensor:
        obs = check(obs).to(**self.tpdv)
        obs_rep_critic_cost = self.obs_encoder_critic_cost(obs)
        v_cost = self.decoder_v_cost(obs_rep_critic_cost)
        return v_cost


class MultiAgentModelDense(MultiAgentBaseModel):
    def __init__(self, obs_dim, act_dim, n_agent, n_block, embed_dim, n_head, actor_decoder_type, device):
        super().__init__(obs_dim, act_dim, n_agent, n_block, embed_dim, n_head, actor_decoder_type, device)
