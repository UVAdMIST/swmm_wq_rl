# swmm_wq_rl
This repository contains python codes and SWMM models to develop real-time control policies that mitigate flooding and reduce pollution for the stormwater system in the Hague area of Norfolk, Virginia.

## Required packages:
1. pyswmm
2. keras-rl (once installed, replace rl.core with modified file: core.py in this repo)
3. openai-gym

Passive (uncontrolled) SWMM simulations use code in the passive folder.
Rule-based Control SWMM simulations use code in the rbc folder.
