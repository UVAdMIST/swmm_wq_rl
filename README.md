# swmm_wq_rl
This repository contains python codes and SWMM models to develop real-time control policies that seek to mitigate flooding and reduce pollution for the stormwater system in the Hague area of Norfolk, Virginia.

## Required packages:
1. pyswmm
2. keras-rl (once installed, replace rl.core, rl.callbacks, and rl/agents.ddpg with modified files in the rl_wq/keras_rl_files folder)
3. openai-gym

## Control Scenarios
1. Passive (uncontrolled) SWMM simulations use code in the passive folder.
2. Rule-based Control SWMM simulations use code in the rbc folder.
3. Reinforcement Learning SWMM simulations will use code in the rl_wq folder.

## References
This work builds on https://github.com/UVAdMIST/swmm_rl which has been published in the Journal of Hydroinformatics and is available via open access at https://iwaponline.com/jh/article/doi/10.2166/hydro.2020.080/77759/Flood-mitigation-in-coastal-urban-catchments-using.
