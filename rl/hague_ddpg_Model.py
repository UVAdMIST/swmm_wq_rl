# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 07:54:22 2019

@author: cw8xk

This script sets up a SWMM environment with forecasts of rain/tide as part of the state
"""
import pandas as pd
import numpy as np
from pyswmm import Simulation, Nodes, Links, SystemStats
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
        # self.state = np.concatenate([np.asarray([self.St1.depth, self.St3.depth, self.St1.flooding, self.St3.flooding,
        #                                          self.R1.current_setting, self.R3.current_setting]),
        #                              current_fcst])
        # self.state = np.asarray([self.St1.depth, self.St3.depth, self.St1.flooding, self.St3.flooding,
        #                          self.R1.current_setting, self.R3.current_setting,
        #                          rain_fcst, tide_fcst])
        # self.state = np.asarray([self.St1.depth, self.St3.depth, self.St1.flooding, self.St3.flooding,
        #                          self.E143361.flooding,
        #                          self.R1.current_setting, self.R3.current_setting,
        #                          self.R1.pollut_quality['TSS'], self.R3.pollut_quality['TSS'], rain_fcst, tide_fcst])
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
        self.R1.target_setting = action[0]
        self.R3.target_setting = action[1]
        cum_fld = self.sys_stats.routing_stats['flooding']  # get total sys flooding before step
        r1_cum_tss = self.R1.total_loading['TSS']  # total orifice loading before step
        r3_cum_tss = self.R3.total_loading['TSS']
        outfall_cum_tss = self.E143250.outfall_statistics['pollutant_loading']['TSS']  # total outfall loading before step

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
        # self.state = np.asarray([self.St1.depth, self.St3.depth, self.St1.flooding, self.St3.flooding,
        #                          self.R1.current_setting, self.R3.current_setting,
        #                          rain_fcst, tide_fcst])
        # self.state = np.asarray([self.St1.depth, self.St3.depth, self.St1.flooding, self.St3.flooding,
        #                          self.E143361.flooding,
        #                          self.R1.current_setting, self.R3.current_setting,
        #                          self.R1.pollut_quality['TSS'], self.R3.pollut_quality['TSS'], rain_fcst, tide_fcst])
        self.state = np.asarray([self.St1.depth, self.St3.depth,
                                 self.R1.current_setting, self.R3.current_setting,
                                 self.R1.flow, self.R3.flow,
                                 self.R1.pollut_quality['TSS'], self.R3.pollut_quality['TSS'],
                                 rain_fcst, tide_fcst])

        # # 1 flood and depth reward
        # reward = - (self.St1.flooding + self.St3.flooding + abs(self.St1.depth - 7.5) + abs(self.St3.depth - 3.56))

        # # 2 Conditional reward, no pollutants
        # if rain_fcst > 0:  # check if rainfall forecast is positive
        #     reward = - (self.St1.flooding + self.St3.flooding)  # instead of flood, use abs difference from pond max?
        # else:
        #     reward = - (abs(self.St1.depth - 7.5) + abs(self.St3.depth - 3.56))

        # # 3 flood and pollutant reward
        # reward = - (self.St1.flooding + self.St3.flooding + abs(self.St1.depth - 7.5) + abs(self.St3.depth - 3.56) +
        #             self.R1.pollut_quality['TSS'] + self.R3.pollut_quality['TSS'])

        # # 4 flood and pollutant reward with downstream node E143361
        # reward = - (self.St1.flooding + self.St3.flooding + self.E143361.flooding +
        #             abs(self.St1.depth - 7.5) + abs(self.St3.depth - 3.56) +
        #             self.R1.pollut_quality['TSS'] + self.R3.pollut_quality['TSS'])

        # # 5 flood and pollutant reward with total flooding
        # reward = - (self.St1.flooding + self.St3.flooding + self.sys_stats.routing_stats['flooding'] +
        #             abs(self.St1.depth - 7.5) + abs(self.St3.depth - 3.56) +
        #             self.R1.pollut_quality['TSS'] + self.R3.pollut_quality['TSS'])

        # # 6 flood and weighted pollutant reward with downstream node E143361
        # reward = - (self.St1.flooding + self.St3.flooding + self.E143361.flooding +
        #             abs(self.St1.depth - 7.5) + abs(self.St3.depth - 3.56) +
        #             5 * self.R1.pollut_quality['TSS'] + 5 * self.R3.pollut_quality['TSS'])

        # # 7 flood and pollutant reward with downstream node E143361
        # reward = - (self.St1.flooding * 100 + self.St3.flooding * 100 + self.E143361.flooding +
        #             self.R1.pollut_quality['TSS'] + self.R3.pollut_quality['TSS'])

        # # 8 flood and pollutant reward with downstream node E143361
        # reward = - (self.St1.flooding * 100 + self.St3.flooding * 100 + self.E143361.flooding +
        #             abs(self.St1.depth - 7.5) + abs(self.St3.depth - 3.56) +
        #             self.R1.pollut_quality['TSS'] + self.R3.pollut_quality['TSS'])

        # # 9 weighted flood and pollutant reward with downstream node E143361
        # reward = - (self.St1.flooding * 100 + self.St3.flooding * 100 + self.E143361.flooding +
        #             self.R1.pollut_quality['TSS'] * 2 + self.R3.pollut_quality['TSS'] * 2)

        # # 10 flood and pollutant reward with total incremental flooding
        # reward = - (self.St1.flooding * 100 + self.St3.flooding * 100 + (current_cum_fld - self.cum_fld) +
        #             self.R1.pollut_quality['TSS'] + self.R3.pollut_quality['TSS'])

        # # 11 flood and pollutant reward with total incremental flooding
        # reward = - (self.St1.flooding * 100 + self.St3.flooding * 100 + (current_cum_fld - cum_fld) +
        #             self.R1.pollut_quality['TSS'] * 2 + self.R3.pollut_quality['TSS'] * 5)

        # # 12 flood and pollutant reward with total incremental flooding
        # reward = - (self.St1.flooding * 100 + self.St3.flooding * 100 +
        #             (current_cum_fld - cum_fld) + abs(self.St3.depth - 3.56) +
        #             self.R1.pollut_quality['TSS'] * 5 + self.R3.pollut_quality['TSS'] * 5)

        # # 13 flood and pollutant LOAD reward with total incremental flooding
        # reward = - (self.St1.flooding * 100 + self.St3.flooding * 100 +
        #             (current_cum_fld - cum_fld) + abs(self.St3.depth - 3.56) +
        #             (current_r1_cum_tss - r1_cum_tss) + (current_r3_cum_tss - r3_cum_tss))

        # # 14 system flood and pollutant LOAD reward
        # reward = - ((current_cum_fld - cum_fld) + (current_outfall_cum_tss - outfall_cum_tss))

        # # 15 system flood and pollutant load reward with action penalty
        # reward = - ((current_cum_fld - cum_fld) + (current_outfall_cum_tss - outfall_cum_tss) +
        #             abs(r1_act_old - r1_act_new) * 0.5 + abs(r3_act_old - r3_act_new) * 0.5)

        # # 16 system flood and pollutant load reward with small action penalty and st3 target
        # reward = - ((current_cum_fld - cum_fld) + (current_outfall_cum_tss - outfall_cum_tss) +
        #             abs(r1_act_old - r1_act_new) * 0.5 + abs(r3_act_old - r3_act_new) * 0.5 +
        #             abs(self.St3.depth - 3.56)*0.5)

        # # 17 system flood and pollutant load reward with small action penalty and st3 target
        # reward = - ((current_cum_fld - cum_fld) + (current_outfall_cum_tss - outfall_cum_tss) +
        #             abs(self.St3.depth - 3.56)*0.5)

        # # 18 system flood and pollutant load reward with small action penalty and st3 target
        # reward = - ((current_cum_fld - cum_fld)/50000 + (current_outfall_cum_tss - outfall_cum_tss))

        # # 19 system flood and pollutant load reward with small action penalty and st3 target
        # reward = - ((current_cum_fld - cum_fld) / 50000 + (current_outfall_cum_tss - outfall_cum_tss) +
        #             self.St1.flooding * 100 + self.St3.flooding * 100)

        # 20 system flood and pollutant load reward with small action penalty and st3 target
        reward = - ((current_cum_fld - cum_fld) / 40000 + (current_outfall_cum_tss - outfall_cum_tss) +
                    self.St1.flooding * 100 + self.St3.flooding * 100 + abs(self.St3.depth - 3.56) / 2)

        # # 21 system flood and pond depths reward
        # reward = - ((current_cum_fld - cum_fld) + abs(self.St3.depth - 3.56) + abs(self.St1.depth - 7.5))

        # # 22 system flood and pond depths reward
        # reward = - ((current_cum_fld - cum_fld)/50000 + (current_outfall_cum_tss - outfall_cum_tss) +
        #             abs(self.St3.depth - 3.56) * 0.5)

        # # 23 system flood and pond depths reward
        # reward = - ((current_cum_fld - cum_fld)/40000 + (current_outfall_cum_tss - outfall_cum_tss) +
        #             abs(self.St3.depth - 3.56) + abs(self.St1.depth - 7.5))

        # # 25 system flood and pollutant load reward with small action penalty and st3 target
        # reward = - ((current_cum_fld - cum_fld) / 40000 + (current_outfall_cum_tss - outfall_cum_tss) +
        #             self.St1.flooding * 100 + self.St3.flooding * 100 + abs(self.St3.depth - 3.56) / 2 +
        #             self.E143361.flooding * 100)

        # # conditional with pollutants reward
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

        # self.state = np.concatenate([np.asarray([self.St1.depth, self.St3.depth, self.St1.flooding, self.St3.flooding,
        #                                          self.R1.current_setting, self.R3.current_setting]), self.fcst[self.t]])
        current_fcst = self.fcst.iloc[self.fcst.index.get_loc(self.sim.current_time, method='nearest')]
        rain_fcst = sum(current_fcst[:int(len(current_fcst) / 2)])  # total rainfall in forecast
        tide_fcst = np.mean(current_fcst[int(len(current_fcst) / 2):])  # mean tide in forecast
        # self.state = np.asarray([self.St1.depth, self.St3.depth, self.St1.flooding, self.St3.flooding,
        #                          self.R1.current_setting, self.R3.current_setting,
        #                          rain_fcst, tide_fcst])
        # self.state = np.asarray([self.St1.depth, self.St3.depth, self.St1.flooding, self.St3.flooding,
        #                          self.E143361.flooding,
        #                          self.R1.current_setting, self.R3.current_setting,
        #                          self.R1.pollut_quality['TSS'], self.R3.pollut_quality['TSS'], rain_fcst, tide_fcst])
        self.state = np.asarray([self.St1.depth, self.St3.depth,
                                 self.R1.current_setting, self.R3.current_setting,
                                 self.R1.flow, self.R3.flow,
                                 self.R1.pollut_quality['TSS'], self.R3.pollut_quality['TSS'],
                                 rain_fcst, tide_fcst])
        return self.state
    
    def close(self):
        self.sim.report()
        self.sim.close()
