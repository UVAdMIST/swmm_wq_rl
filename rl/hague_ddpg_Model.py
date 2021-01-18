# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 07:54:22 2019

@author: cw8xk

This script sets up a SWMM environment with forecasts of rain/tide as part of the state
"""
import pandas as pd
import numpy as np
from pyswmm import Simulation, Nodes, Links
from rl.core import Env
from gym import spaces


class BasicEnv(Env):
    def __init__(self, inp_file, fcst_file):
        # initialize simulation
        self.input_file = inp_file
        self.sim = Simulation(self.input_file)  # read input file
        self.fcst_file = fcst_file
        # self.fcst = np.genfromtxt(self.fcst_file, delimiter=',')  # read forecast file as array
        self.fcst = pd.read_csv(self.fcst_file, index_col="datetime", infer_datetime_format=True, parse_dates=True)
        self.control_time_step = 900  # control time step in seconds
        self.sim.step_advance(self.control_time_step)  # set control time step
        node_object = Nodes(self.sim)  # init node object
        self.St1 = node_object["st1"]
        self.St2 = node_object["st2"]
        self.St3 = node_object["F134101"]

        link_object = Links(self.sim)  # init link object
        self.R1 = link_object["R1"]
        self.R2 = link_object["R2"]
        self.R3 = link_object["R3"]
    
        self.sim.start()
        if self.sim.current_time == self.sim.start_time:
            self.R1.target_setting = 0.5
            self.R2.target_setting = 1
            self.R3.target_setting = 0.5
        sim_len = self.sim.end_time - self.sim.start_time
        self.T = int(sim_len.total_seconds()/self.control_time_step)
        self.t = 1

        current_fcst = self.fcst.iloc[self.fcst.index.get_loc(self.sim.current_time, method='nearest')]
        rain_fcst = sum(current_fcst[:int(len(current_fcst) / 2)])  # total rainfall in forecast
        tide_fcst = np.mean(current_fcst[int(len(current_fcst) / 2):])  # mean tide in forecast
        # self.state = np.concatenate([np.asarray([self.St1.depth, self.St3.depth, self.St1.flooding, self.St3.flooding,
        #                                          self.R1.current_setting, self.R3.current_setting]),
        #                              current_fcst])
        self.state = np.asarray([self.St1.depth, self.St3.depth, self.St1.flooding, self.St3.flooding,
                                 self.R1.current_setting, self.R3.current_setting, rain_fcst, tide_fcst])
        self.action_space = spaces.Box(low=np.array([0, 0]), high=np.array([1, 1]), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=1000, shape=(len(self.state),), dtype=np.float32)
        
    def step(self, action):
        self.R1.target_setting = action[0]
        self.R3.target_setting = action[1]
        self.sim.__next__()

        # self.state = np.concatenate([np.asarray([self.St1.depth, self.St3.depth, self.St1.flooding, self.St3.flooding,
        #                                          self.R1.current_setting, self.R3.current_setting]), self.fcst[self.t]])
        current_fcst = self.fcst.iloc[self.fcst.index.get_loc(self.sim.current_time, method='nearest')]
        rain_fcst = sum(current_fcst[:int(len(current_fcst) / 2)])  # total rainfall in forecast
        tide_fcst = np.mean(current_fcst[int(len(current_fcst) / 2):])  # mean tide in forecast
        self.state = np.asarray([self.St1.depth, self.St3.depth, self.St1.flooding, self.St3.flooding,
                                 self.R1.current_setting, self.R3.current_setting, rain_fcst, tide_fcst])

        # 1 flood and depth reward
        reward = - (self.St1.flooding + self.St3.flooding + abs(self.St1.depth - 7.5) + abs(self.St3.depth - 3.56))

        # # 2 Conditional reward, no pollutants
        # if np.sum(self.fcst[self.t, :-1]) > 0:  # check if rainfall forecast is positive
        #     reward = - (self.St1.flooding + self.St3.flooding)  # instead of flood, use abs difference from pond max?
        # else:
        #     reward = - (abs(self.St1.depth - 7.5) + abs(self.St3.depth - 3.56))
        #
        # # 3 flood and pollutant reward
        # reward = - (self.St1.flooding + self.St3.flooding + abs(self.St1.depth - 7.5) + abs(self.St3.depth - 3.56) +
        #             self.R1.pollut_quality + self.R3.pollut_quality)
        #
        # # 4 conditional with pollutants reward
        # if np.sum(self.fcst[self.t, :-1]) > 0:  # check if rainfall forecast is positive
        # reward = - (self.St1.flooding + self.St3.flooding + self.R1.pollut_quality + self.R3.pollut_quality)
        # else:
        #     reward = - (abs(self.St1.depth - 7.5) + abs(self.St3.depth - 3.56) +
        #                 self.R1.pollut_quality + self.R3.pollut_quality)

        if self.t < self.T-1:
            done = False
        else:
            done = True
        
        self.t += 1
        
        info = {}
        
        return self.state, reward, done, info       
    
    def reset(self):
        self.sim.close()
        self.sim = Simulation(self.input_file)
        # self.fcst = np.genfromtxt(self.fcst_file, delimiter=',')  # read forecast file as array
        self.fcst = pd.read_csv(self.fcst_file, index_col="datetime", infer_datetime_format=True, parse_dates=True)
        self.sim.step_advance(self.control_time_step)  # set control time step
        node_object = Nodes(self.sim)  # init node object
        self.St1 = node_object["st1"]
        self.St2 = node_object["st2"]
        self.St3 = node_object["F134101"]
        link_object = Links(self.sim)  # init link object
        self.R1 = link_object["R1"]
        self.R2 = link_object["R2"]
        self.R3 = link_object["R3"]
        self.sim.start()
        self.t = 1
        if self.sim.current_time == self.sim.start_time:
            self.R1.target_setting = 0.5
            self.R2.target_setting = 1
            self.R3.target_setting = 0.5

        # self.state = np.concatenate([np.asarray([self.St1.depth, self.St3.depth, self.St1.flooding, self.St3.flooding,
        #                                          self.R1.current_setting, self.R3.current_setting]), self.fcst[self.t]])
        current_fcst = self.fcst.iloc[self.fcst.index.get_loc(self.sim.current_time, method='nearest')]
        rain_fcst = sum(current_fcst[:int(len(current_fcst) / 2)])  # total rainfall in forecast
        tide_fcst = np.mean(current_fcst[int(len(current_fcst) / 2):])  # mean tide in forecast
        self.state = np.asarray([self.St1.depth, self.St3.depth, self.St1.flooding, self.St3.flooding,
                                 self.R1.current_setting, self.R3.current_setting, rain_fcst, tide_fcst])
        return self.state
    
    def close(self):
        self.sim.report()
        self.sim.close()
