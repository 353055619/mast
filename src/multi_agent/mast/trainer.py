import numpy as np
import torch
import torch.nn as nn
from einops import rearrange

from safepo.utils.util import check
from safepo.utils.valuenorm import ValueNorm
from safepo.utils.util import huber_loss
from .model import MultiAgentModel

__all__ = ['MASTrainer']


class MASTrainer:
    def __init__(self, n_agent: int, model, config):
        self.n_agent = n_agent
        self.clip_param = config["clip_param"]
        self.ppo_epoch = config["ppo_epoch"]
        self.num_mini_batch = config["num_mini_batch"]
        self.value_loss_coef = config["value_loss_coef"]
        self.entropy_coef = config["entropy_coef"]
        self.max_grad_norm = config["max_grad_norm"]
        self.huber_delta = config["huber_delta"]
        self.value_normalizer = ValueNorm(1, device=config['device'])
        self.num_updates = 0
        self.lamda_lagr = config["lamda_lagr"]

        self.model = model
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config['lr'],
            eps=config['opti_eps'],
            weight_decay=config["weight_decay"],
        )

        self.config = config
        self.device = config['device']
        self.tpdv = dict(dtype=torch.float32, device=config['device'])

    def cal_value_loss(self, values, value_preds_batch, return_batch):
        """
        Calculate value function loss.
        :param values: (torch.Tensor) value function predictions.
        :param value_preds_batch: (torch.Tensor) "old" value  predictions from data batch (used for value clip loss)
        :param return_batch: (torch.Tensor) reward to go returns.

        :return value_loss: (torch.Tensor) value function loss.
        """

        value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(
            -self.clip_param, self.clip_param
        )

        self.value_normalizer.update(return_batch)
        error_clipped = self.value_normalizer.normalize(return_batch) - value_pred_clipped
        error_original = self.value_normalizer.normalize(return_batch) - values

        value_loss_clipped = huber_loss(error_clipped, self.huber_delta)
        value_loss_original = huber_loss(error_original, self.huber_delta)

        value_loss = torch.max(value_loss_original, value_loss_clipped)

        return value_loss

    def ppo_update(self,
                   obs_batch,
                   actions_batch,
                   value_preds_batch,
                   return_batch,
                   old_action_log_probs_batch,
                   adv_targ,
                   cost_preds_batch,
                   cost_return_batch,
                   cost_adv_targ,
                   aver_episode_costs,
                   active_masks_batch):

        old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
        adv_targ = check(adv_targ).to(**self.tpdv)
        value_preds_batch = check(value_preds_batch).to(**self.tpdv)
        return_batch = check(return_batch).to(**self.tpdv)
        cost_returns_barch = check(cost_return_batch).to(**self.tpdv)
        cost_preds_batch = check(cost_preds_batch).to(**self.tpdv)
        cost_adv_targ = check(cost_adv_targ).to(**self.tpdv)

        # Reshape to do in a single forward pass for all steps
        action_log_probs, dist_entropy, values, cost_values = self.model(
            rearrange(obs_batch, '(b n) obs_dim -> b n obs_dim', n=self.n_agent),
            rearrange(actions_batch, '(b n) act_dim -> b n act_dim', n=self.n_agent)
        )
        # reshape
        action_log_probs = rearrange(action_log_probs, 'b n act_dim -> (b n) act_dim', n=self.n_agent)
        dist_entropy = rearrange(dist_entropy, 'b n act_dim -> (b n) act_dim', n=self.n_agent)
        values = rearrange(values, 'b n 1 -> (b n) 1', n=self.n_agent)
        cost_values = rearrange(cost_values, 'b n 1 -> (b n) 1', n=self.n_agent)

        adv_targ_hybrid = adv_targ - self.lamda_lagr * cost_adv_targ
        # actor update
        imp_weights = torch.exp(action_log_probs - old_action_log_probs_batch)

        surr1 = imp_weights * adv_targ_hybrid
        surr2 = (
                torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param)
                * adv_targ_hybrid
        )

        policy_loss = -torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True)

        # critic update
        value_loss = self.cal_value_loss(
            values, value_preds_batch, return_batch
        )
        # cost_critic update
        cost_loss = self.cal_value_loss(
            cost_values, cost_preds_batch, cost_returns_barch
        )
        if aver_episode_costs.mean() > self.config["safety_bound"] * self.n_agent:
            delta_lamda_lagr = -(
                    (aver_episode_costs.mean() - self.config["safety_bound"]) * (1 - self.config["gamma"]) + (
                    imp_weights * cost_adv_targ)).mean().detach()

            R_Relu = torch.nn.ReLU()
            new_lamda_lagr = R_Relu(self.lamda_lagr - (delta_lamda_lagr * self.config["lagrangian_coef_rate"]))
            self.lamda_lagr = new_lamda_lagr

        policy_loss = (policy_loss * active_masks_batch).sum() / active_masks_batch.sum()

        value_loss = (value_loss * active_masks_batch).sum() / active_masks_batch.sum()

        cost_loss = (cost_loss * active_masks_batch).sum() / active_masks_batch.sum()

        dist_entropy = (dist_entropy * active_masks_batch).sum() / active_masks_batch.sum()

        loss = (
                policy_loss
                - dist_entropy * self.entropy_coef
                + value_loss * self.value_loss_coef
                + cost_loss * self.value_loss_coef
        )

        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return (
            value_loss,
            grad_norm,
            policy_loss,
            dist_entropy,
            grad_norm,
            imp_weights,
            cost_loss,
            grad_norm
        )

    def train(self, buffer, logger):
        advantages = buffer.returns[:-1] - self.value_normalizer.denormalize(buffer.value_preds[:-1])
        advantages_copy = advantages.clone()
        advantages_copy[buffer.active_masks[:-1] == 0.0] = np.nan
        mean_advantages = torch.mean(advantages_copy)
        std_advantages = torch.std(advantages_copy)
        advantages = (advantages - mean_advantages) / (std_advantages + 1e-8)

        cost_adv = buffer.cost_returns[:-1] - self.value_normalizer.denormalize(buffer.cost_preds[:-1])
        cost_adv_copy = cost_adv.clone()
        cost_adv_copy[buffer.active_masks[:-1] == 0.0] = np.nan
        mean_cost_adv = torch.mean(cost_adv_copy)
        std_cost_adv = torch.std(cost_adv_copy)
        cost_adv = (cost_adv - mean_cost_adv) / (std_cost_adv + 1e-5)

        for k in range(self.ppo_epoch):
            data_generator = buffer.feed_forward_generator_transformer(
                advantages, self.num_mini_batch, cost_adv=cost_adv
            )

            for sample in data_generator:
                # Original MAT PPO update
                (
                    share_obs_batch,
                    obs_batch,
                    rnn_states_batch,
                    rnn_states_critic_batch,
                    actions_batch,
                    value_preds_batch,
                    return_batch,
                    masks_batch,
                    active_masks_batch,
                    old_action_log_probs_batch,
                    adv_targ,
                    available_actions_batch,
                    cost_preds_batch,
                    cost_return_batch,
                    rnn_states_cost_batch,
                    cost_adv_targ,
                    aver_episode_costs
                ) = sample
                (
                    value_loss,
                    critic_grad_norm,
                    policy_loss,
                    dist_entropy,
                    actor_grad_norm,
                    imp_weights,
                    cost_loss,
                    cost_grad_norm
                ) = self.ppo_update(
                    obs_batch,
                    actions_batch,
                    value_preds_batch,
                    return_batch,
                    old_action_log_probs_batch,
                    adv_targ,
                    cost_preds_batch,
                    cost_return_batch,
                    cost_adv_targ,
                    aver_episode_costs,
                    active_masks_batch
                )
                logger.store(
                    **{
                        "Loss/Loss_reward_critic": value_loss.item(),
                        "Loss/Loss_cost_critic": cost_loss.item(),
                        "Loss/Loss_actor": policy_loss.item(),
                        "Misc/Reward_critic_norm": critic_grad_norm.item(),
                        "Misc/Cost_critic_norm": cost_grad_norm.item(),
                        "Misc/Entropy": dist_entropy.item(),
                        "Misc/Ratio": imp_weights.detach().mean().item(),
                        "Lamda_lagr": self.lamda_lagr
                    }
                )

    def prep_training(self):
        self.model.train()

    def prep_rollout(self):
        self.model.eval()
