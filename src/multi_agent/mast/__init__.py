from .trainer import *
from .model import MultiAgentModel, MultiAgentModelDense, MultiAgentBaseModel


def create_model(pattern: str,
                 obs_dim: int,
                 act_dim: int,
                 n_agent: int,
                 n_block: int,
                 embed_dim: int,
                 n_head: int,
                 device: str) -> MultiAgentBaseModel:
    model_type, actor_decoder_type = pattern.split('-')

    if model_type == 'dense':
        return MultiAgentModelDense(obs_dim, act_dim, n_agent, n_block, embed_dim, n_head, actor_decoder_type, device).to(device)
    elif model_type == 'sparse':
        return MultiAgentModel(obs_dim, act_dim, n_agent, n_block, embed_dim, n_head, actor_decoder_type, device).to(device)
    else:
        raise NotImplementedError
