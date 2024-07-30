import torch
import os
import time

from safepo.common.env import make_ma_mujoco_env
from safepo.common.buffer import SharedReplayBuffer
from safepo.common.logger import EpochLogger
from safepo.utils.config import multi_agent_args, set_np_formatting, set_seed
from matlag import MASTrainer, MultiAgentTransformer

torch.autograd.set_detect_anomaly(True)
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.set_num_threads(4)  # 将 PyTorch 的计算操作限制为最多使用 4 个线程进行并行计算。
set_np_formatting()


class Runner:

    def __init__(self,
                 envs,
                 config,
                 model_dir=""):
        self.logger = EpochLogger(log_dir=config["log_dir"], seed=str(config["seed"]))
        self.save_dir = str(config["log_dir"] + '/models_seed{}'.format(config["seed"]))
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.logger.save_config(config)

        n_agent = envs.num_agents
        obs_dim = envs.observation_space[0].shape[0]
        act_dim = envs.action_space[0].shape[0]

        self.model = MultiAgentTransformer(
            obs_dim,
            act_dim,
            n_agent,
            n_block=config["n_block"],
            n_embd=config["n_embd"],
            n_head=config["n_head"],
            device=config['device'],
        )
        self.trainer = MASTrainer(config, self.model, config["device"])
        self.buffer = SharedReplayBuffer(config,
                                         envs.num_agents,
                                         envs.observation_space[0],
                                         envs.share_observation_space[0],
                                         envs.action_space[0])

        self.envs = envs
        self.config = config
        self.model_dir = model_dir
        self.num_agents = self.envs.num_agents
        self.logger.log(f'Training environment set up:\n'
                        f'- Number of agents: {self.num_agents}\n'
                        f'- Number of threads: {config["n_rollout_threads"]}\n'
                        f'- Number of episodes: {int(self.config["num_env_steps"]) // self.config["episode_length"]}\n'
                        f'- Observation space: {envs.observation_space[0]}\n'
                        f'- Action space: {envs.action_space[0]}\n'
                        f'- Log directory: {config["log_dir"]}\n'
                        f'- Random seed: {config["seed"]}\n'
                        f'- Device used: {config["device"]}\n')

    def run(self):
        self.warmup()

        start = time.time()
        episodes = int(self.config["num_env_steps"]) // self.config["episode_length"] // self.config[
            "n_rollout_threads"]

        train_episode_rewards = torch.zeros(1, self.config["n_rollout_threads"], device=self.config["device"])
        train_episode_costs = torch.zeros(1, self.config["n_rollout_threads"], device=self.config["device"])

        for episode in range(episodes):

            done_episodes_rewards = []
            done_episodes_costs = []

            for step in range(self.config["episode_length"]):
                # Sample actions
                values, actions, action_log_probs, cost_preds = self.collect(step)
                obs, share_obs, rewards, costs, dones, infos, _ = self.envs.step(actions)

                dones_env = torch.all(dones, dim=1)

                reward_env = torch.mean(rewards, dim=1).flatten()
                cost_env = torch.mean(costs, dim=1).flatten()

                train_episode_rewards += reward_env
                train_episode_costs += cost_env

                for t in range(self.config["n_rollout_threads"]):
                    if dones_env[t]:
                        done_episodes_rewards.append(train_episode_rewards[:, t].clone())
                        train_episode_rewards[:, t] = 0
                        done_episodes_costs.append(train_episode_costs[:, t].clone())
                        train_episode_costs[:, t] = 0

                done_episodes_costs_aver = train_episode_costs.mean()
                self.insert(obs,
                            share_obs,
                            rewards,
                            costs,
                            dones,
                            values,
                            actions,
                            action_log_probs,
                            self.buffer.rnn_states[step],  # for compatibility
                            self.buffer.rnn_states_critic[step],  # # for compatibility
                            cost_preds,
                            self.buffer.rnn_states_cost[step],  # # for compatibility
                            done_episodes_costs_aver)
            self.compute()
            self.train()

            total_num_steps = (episode + 1) * self.config["episode_length"] * self.config["n_rollout_threads"]

            end = time.time()

            if len(done_episodes_rewards) != 0:
                aver_episode_rewards = torch.stack(done_episodes_rewards).mean()
                aver_episode_costs = torch.stack(done_episodes_costs).mean()
                self.return_aver_cost(aver_episode_costs)
                self.logger.store(
                    **{
                        "Metrics/EpRet": aver_episode_rewards.item(),
                        "Metrics/EpCost": aver_episode_costs.item(),
                        "Eval/EpRet": 0.0,
                        "Eval/EpCost": 0.0,
                    }
                )

                self.logger.log_tabular("Metrics/EpRet", min_and_max=True, std=True)  #
                self.logger.log_tabular("Metrics/EpCost", min_and_max=True, std=True)
                self.logger.log_tabular("Eval/EpRet")
                self.logger.log_tabular("Eval/EpCost")
                self.logger.log_tabular("Train/Epoch", episode)
                self.logger.log_tabular("Train/TotalSteps", total_num_steps)
                self.logger.log_tabular("Loss/Loss_reward_critic")
                self.logger.log_tabular("Loss/Loss_cost_critic")
                self.logger.log_tabular("Loss/Loss_actor")
                self.logger.log_tabular("Misc/Reward_critic_norm")
                self.logger.log_tabular("Misc/Cost_critic_norm")
                self.logger.log_tabular("Misc/Entropy")
                self.logger.log_tabular("Misc/Ratio")
                self.logger.log_tabular("Time/Total", end - start)
                self.logger.log_tabular("Time/FPS", int(total_num_steps / (end - start)))
                self.logger.log_tabular("Lamda_lagr")
                self.logger.dump_tabular()

    def return_aver_cost(self, aver_episode_costs):
        self.buffer.return_aver_insert(aver_episode_costs)

    def warmup(self):
        # reset env
        obs, share_obs, _ = self.envs.reset()

        self.buffer.share_obs[0].copy_(share_obs)
        self.buffer.obs[0].copy_(obs)

    @torch.no_grad()
    def collect(self, step):
        self.trainer.prep_rollout()
        action, action_log_prob, value, cost_pred = self.model.get_actions(self.buffer.obs[step])
        return value.detach(), action.detach(), action_log_prob.detach(), cost_pred.detach()

    def insert(self,
               obs,
               share_obs,
               rewards,
               costs,
               dones,
               values,
               actions,
               action_log_probs,
               rnn_states,
               rnn_states_critic,
               cost_preds,
               rnn_states_cost,
               done_episodes_costs_aver,
               aver_episode_costs=0):  # check

        dones_env = torch.all(dones, dim=1)

        rnn_states[dones_env == True] = torch.zeros(
            (dones_env == True).sum(), self.num_agents, self.config["recurrent_N"], self.config["hidden_size"],
            device=self.config["device"])
        rnn_states_critic[dones_env == True] = torch.zeros(
            (dones_env == True).sum(), self.num_agents, *self.buffer.rnn_states_critic.shape[3:],
            device=self.config["device"])
        rnn_states_cost[dones_env == True] = torch.zeros(
            ((dones_env == True).sum(), self.num_agents, *self.buffer.rnn_states_cost.shape[3:]),
            device=self.config["device"])

        masks = torch.ones(self.config["n_rollout_threads"], self.num_agents, 1, device=self.config["device"])
        masks[dones_env == True] = torch.zeros((dones_env == True).sum(), self.num_agents, 1,
                                               device=self.config["device"])

        active_masks = torch.ones(self.config["n_rollout_threads"], self.num_agents, 1, device=self.config["device"])
        active_masks[dones == True] = torch.zeros((dones == True).sum(), 1, device=self.config["device"])
        active_masks[dones_env == True] = torch.ones((dones_env == True).sum(), self.num_agents, 1,
                                                     device=self.config["device"])

        obs_to_insert = obs
        self.buffer.insert(share_obs, obs_to_insert, rnn_states,
                           rnn_states_critic, actions,
                           action_log_probs,
                           values, rewards, masks, None,
                           active_masks, None, costs=costs,
                           cost_preds=cost_preds,
                           rnn_states_cost=rnn_states_cost,
                           done_episodes_costs_aver=done_episodes_costs_aver,
                           aver_episode_costs=aver_episode_costs)

    def train(self):
        self.trainer.prep_training()
        self.trainer.train(self.buffer, logger=self.logger)
        self.buffer.after_update()

    @torch.no_grad()
    def compute(self):
        self.trainer.prep_rollout()
        next_values = self.model.get_values(self.buffer.obs[-1])
        next_values = next_values.detach()
        self.buffer.compute_returns(next_values, self.trainer.value_normalizer)
        next_cost_values = self.model.get_cost_values(self.buffer.obs[-1])
        next_cost_values = next_cost_values.detach()
        self.buffer.compute_returns(next_cost_values, self.trainer.value_normalizer)


def main():
    args, cfg_env, cfg_train = multi_agent_args(algo="matlag")
    set_seed(cfg_train.get("seed", -1), cfg_train.get("torch_deterministic", False))
    env = make_ma_mujoco_env(
        scenario=args.scenario,
        agent_conf=args.agent_conf,
        seed=args.seed,
        cfg_train=cfg_train,
    )
    cfg_train["args"] = args
    cfg_train["num_agents"] = env.num_agents

    runner = Runner(env, cfg_train, args.model_dir)
    runner.run()


if __name__ == '__main__':
    main()
