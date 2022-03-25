# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 07:54:22 2019

@author: cw8xk

This script sets up a SWMM environment with forecasts of rain/tide as part of the state
"""
import pandas as pd
import numpy as np
import math
from swmm_rl_hague.swmm_utils import calc_dupuit
from pyswmm import Simulation, Nodes, Links, SystemStats
from rl.core import Env
from gym import spaces


class BasicEnv(Env):
    def __init__(self, inp_file, fcst_file, gwl_file):
        # initialize simulation
        self.input_file = inp_file
        self.sim = Simulation(self.input_file)  # read input file
        self.fcst_file = fcst_file
        self.gwl_file = gwl_file
        self.fcst = pd.read_csv(self.fcst_file, index_col="datetime", infer_datetime_format=True, parse_dates=True)
        self.gwl = pd.read_csv(self.gwl_file, index_col="Datetime", infer_datetime_format=True, parse_dates=True)
        self.control_time_step = 900  # control time step in seconds
        self.sim.step_advance(self.control_time_step)  # set control time step
        self.sys_stats = SystemStats(self.sim)

        node_object = Nodes(self.sim)  # init node object
        self.St1 = node_object["st1"]
        self.St2 = node_object["st2"]
        self.St3 = node_object["F134101"]
        self.E143361 = node_object["E143361"]
        self.E143250 = node_object["E143250"]

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

        # self.cum_fld = self.sys_stats.routing_stats['flooding']
        current_fcst = self.fcst.iloc[self.fcst.index.get_loc(self.sim.current_time, method='nearest')]
        rain_fcst = sum(current_fcst[:int(len(current_fcst) / 2)])  # total rainfall in forecast
        tide_fcst = np.mean(current_fcst[int(len(current_fcst) / 2):])  # mean tide in forecast

        # calculate radius (ft) of storage units for GWL calculations
        self.St1_rad = math.sqrt(100000 / math.pi)
        self.St3_rad = math.sqrt(50000 / math.pi)

        # calculate groundwater flow
        current_gwl = self.gwl.iloc[self.gwl.index.get_loc(self.sim.current_time, method='nearest')]["IDW"]
        St1_gwq = calc_dupuit(self.St1.depth, self.St1.invert_elevation, current_gwl, self.St1_rad)
        St3_gwq = calc_dupuit(self.St3.depth, self.St3.invert_elevation, current_gwl, self.St3_rad)

        # add (or subtract) gw flow to storage unit
        self.St1.generated_inflow(St1_gwq)
        self.St3.generated_inflow(St3_gwq)

        self.state = np.asarray([self.St1.depth, self.St3.depth,
                                 self.R1.current_setting, self.R3.current_setting,
                                 self.R1.flow, self.R3.flow,
                                 self.R1.pollut_quality['TSS'], self.R3.pollut_quality['TSS'],
                                 rain_fcst, tide_fcst])
        self.action_space = spaces.Box(low=np.array([0, 0]), high=np.array([1, 1]), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=1000, shape=(len(self.state),), dtype=np.float32)
        
    def step(self, action):
        r1_act_old = self.R1.current_setting
        r3_act_old = self.R3.current_setting
        self.R1.target_setting = np.float(action[0])  # swmm5 requires float, python float is C double
        self.R3.target_setting = np.float(action[1])
        cum_fld = self.sys_stats.routing_stats['flooding']  # get total sys flooding before step
        r1_cum_tss = self.R1.total_loading['TSS']  # total orifice loading before step
        r3_cum_tss = self.R3.total_loading['TSS']
        outfall_cum_tss = self.E143250.outfall_statistics['pollutant_loading']['TSS']  # total outfall loading before step
        past_fcst = self.fcst.iloc[self.fcst.index.get_loc(self.sim.current_time, method='nearest')]
        past_rain = sum(past_fcst[int(len(past_fcst) / 2) - 4: int(len(past_fcst) / 2)])  # past hour's rainfall

        self.sim.__next__()

        # self.state = np.concatenate([np.asarray([self.St1.depth, self.St3.depth, self.St1.flooding, self.St3.flooding,
        #                                          self.R1.current_setting, self.R3.current_setting]), self.fcst[self.t]])

        current_cum_fld = self.sys_stats.routing_stats['flooding']  # total flooding after step
        current_r1_cum_tss = self.R1.total_loading['TSS']  # total orifice loading after step
        current_r3_cum_tss = self.R3.total_loading['TSS']
        current_outfall_cum_tss = self.E143250.outfall_statistics['pollutant_loading']['TSS']  # total outfall loading after step
        r1_act_new = self.R1.current_setting
        r3_act_new = self.R3.current_setting
        current_fcst = self.fcst.iloc[self.fcst.index.get_loc(self.sim.current_time, method='nearest')]
        rain_fcst = sum(current_fcst[:int(len(current_fcst) / 2)])  # total rainfall in forecast
        tide_fcst = np.mean(current_fcst[int(len(current_fcst) / 2):])  # mean tide in forecast

        # calculate groundwater flow
        current_gwl = self.gwl.iloc[self.gwl.index.get_loc(self.sim.current_time, method='nearest')]["IDW"]
        St1_gwq = calc_dupuit(self.St1.depth, self.St1.invert_elevation, current_gwl, self.St1_rad)
        St3_gwq = calc_dupuit(self.St3.depth, self.St3.invert_elevation, current_gwl, self.St3_rad)

        # add (or subtract) gw flow to storage unit
        self.St1.generated_inflow(St1_gwq)
        self.St3.generated_inflow(St3_gwq)

        self.state = np.asarray([self.St1.depth, self.St3.depth,
                                 self.R1.current_setting, self.R3.current_setting,
                                 self.R1.flow, self.R3.flow,
                                 self.R1.pollut_quality['TSS'], self.R3.pollut_quality['TSS'],
                                 rain_fcst, tide_fcst])

        # 31 conditional with pollutants reward
        if self.St3.depth > 5.75:  # penalize st3 if above upstream flood threshold
            st3_depth_rwd = 1000
        else:
            st3_depth_rwd = 0

        if rain_fcst > 0.5:  # check if rainfall forecast is positive
            reward = - ((current_cum_fld - cum_fld) + self.St1.flooding * 1000 + st3_depth_rwd +
                        (current_r1_cum_tss - r1_cum_tss) + (current_r3_cum_tss - r3_cum_tss))
        else:
            reward = - (abs(self.St1.depth - 6.) + abs(self.St3.depth - 3.56) +
                        (current_r1_cum_tss - r1_cum_tss) + (current_r3_cum_tss - r3_cum_tss) +
                        (current_cum_fld - cum_fld)/35000)

        # # 39 conditional without pollutants reward
        # if self.St3.depth > 5.75:  # penalize st3 if above upstream flood threshold
        #     st3_depth_rwd = 1000
        # else:
        #     st3_depth_rwd = 0
        #
        # if rain_fcst > 0.5:  # check if rainfall forecast is positive
        #     reward = - ((current_cum_fld - cum_fld) + self.St1.flooding * 1000 + st3_depth_rwd)
        # else:
        #     reward = - (abs(self.St1.depth - 6.) + abs(self.St3.depth - 3.56))

        # # 47 conditional with pollutants reward
        # if self.St3.depth > 5.75:  # penalize st3 if above upstream flood threshold
        #     st3_depth_rwd = 1000
        # else:
        #     st3_depth_rwd = 0
        #
        # if rain_fcst > 0.5:  # check if rainfall forecast is positive
        #     reward = - ((current_cum_fld - cum_fld) + self.St1.flooding * 1000 + st3_depth_rwd)
        # else:
        #     reward = - (abs(self.St1.depth - 6.) + abs(self.St3.depth - 3.56) +
        #                 (current_r1_cum_tss - r1_cum_tss) + (current_r3_cum_tss - r3_cum_tss))

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
        self.fcst = pd.read_csv(self.fcst_file, index_col="datetime", infer_datetime_format=True, parse_dates=True)
        self.gwl = pd.read_csv(self.gwl_file, index_col="Datetime", infer_datetime_format=True, parse_dates=True)
        self.sim.step_advance(self.control_time_step)  # set control time step
        self.sys_stats = SystemStats(self.sim)
        node_object = Nodes(self.sim)  # init node object
        self.St1 = node_object["st1"]
        self.St2 = node_object["st2"]
        self.St3 = node_object["F134101"]
        self.E143361 = node_object["E143361"]
        self.E143250 = node_object["E143250"]

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

        current_fcst = self.fcst.iloc[self.fcst.index.get_loc(self.sim.current_time, method='nearest')]
        rain_fcst = sum(current_fcst[:int(len(current_fcst) / 2)])  # total rainfall in forecast
        tide_fcst = np.mean(current_fcst[int(len(current_fcst) / 2):])  # mean tide in forecast

        # calculate groundwater flow
        current_gwl = self.gwl.iloc[self.gwl.index.get_loc(self.sim.current_time, method='nearest')]["IDW"]
        St1_gwq = calc_dupuit(self.St1.depth, self.St1.invert_elevation, current_gwl, self.St1_rad)
        St3_gwq = calc_dupuit(self.St3.depth, self.St3.invert_elevation, current_gwl, self.St3_rad)

        # add (or subtract) gw flow to storage unit
        self.St1.generated_inflow(St1_gwq)
        self.St3.generated_inflow(St3_gwq)

        self.state = np.asarray([self.St1.depth, self.St3.depth,
                                 self.R1.current_setting, self.R3.current_setting,
                                 self.R1.flow, self.R3.flow,
                                 self.R1.pollut_quality['TSS'], self.R3.pollut_quality['TSS'],
                                 rain_fcst, tide_fcst])
        return self.state
    
    def close(self):
        self.sim.report()
        self.sim.close()
