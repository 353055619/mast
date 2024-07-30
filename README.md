## MAST: Multi-Agent Safe Transformer for Reinforcement Learning

This is an official implementation of "**MAST: Multi-Agent Safe Transformer for**

**Reinforcement Learning**" paper.

<div align="center">
  <img src="assets/MAST Framework.png" width="80%"/>
</div>

**Installation**

1. Create CONDA environement `mast`
2. Clone the repo
3.  Install Pytorch following Pytorch official [installation](https://pytorch.org/get-started/locally/) 
4.  `conda install numpy pandas matplotlib tqdm scikit-learn cython`
5. Install [safety-gymnasium](https://github.com/PKU-Alignment/safety-gymnasium)
6. Install [safe-policy-optimization](https://github.com/PKU-Alignment/Safe-Policy-Optimization) 
7. Install [mujoco](https://github.com/google-deepmind/mujoco)

**run**

MAST Results on 4 Safe Mujoco Environments:

```shell
python mast.py --env_name Safety2x4AntVelocity-v0  --config config
python mast.py --env_name Safety4x2AntVelocity-v0  --config config
python mast.py --env_name Safety2x3HalfCheetahVelocity-v0  --config config
python mast.py --env_name Safety6x1HalfCheetahVelocity-v0  --config config
```



