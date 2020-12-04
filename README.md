# swmm_wq_rl
This repository contains python codes and SWMM models to develop real-time control policies that mitigate flooding and reduce pollution for the stormwater system in the Hague area of Norfolk, Virginia.

## Required packages:
1. pyswmm
2. keras-rl (once installed, replace rl.core with modified file: core.py in this repo)
3. openai-gym

## Control Scenarios
1. Passive (uncontrolled) SWMM simulations use code in the passive folder.
2. Rule-based Control SWMM simulations use code in the rbc folder.
3. Reinforcement Learning SWMM simulations will use code in the rl folder (in progress).

## References
This work builds on https://github.com/UVAdMIST/swmm_rl which has been published in the Journal of Hydroinformatics and is available via open access at https://iwaponline.com/jh/article/doi/10.2166/hydro.2020.080/77759/Flood-mitigation-in-coastal-urban-catchments-using.
