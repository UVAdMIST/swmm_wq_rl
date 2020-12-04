"""
Created by Benjamin Bowes, 11-19-2020
This script simulates stormwater system control based on OptiRTC's Continuous Monitoring and Control (CMAC) strategy.
Depth and flood values at each SWMM time step are recorded and plotted.

This script is for a simple Hague simulation where only St1 and St3 are controlled.
Pump is operated by default rules in inp file (on=1ft, off=0.5ft).
"""

import os
import math
import datetime
import subprocess
import json
import pandas as pd
import pyswmm
pyswmm.lib.use("swmm5_hs_520dev6")
from pyswmm import Simulation, Nodes, Links, Subcatchments
from swmm_rl_hague.gwl_utils import calc_gwl
from swmm_hague_rbc import control_rules
from swmm_hague_rbc import swmm_utils

control_time_step = 900  # control time step in seconds

project_dir = "C:/PycharmProjects/swmm_hague_rbc"
# inp_dir = "C:/PycharmProjects/LongTerm_SWMM/observed_data/obs_data_1month_all_controlled"
# fcst_dir = "C:/PycharmProjects/LongTerm_SWMM/observed_data/obs_data_96step_fcsts"

swmm_inp = "C:/PycharmProjects/swmm_rl_hague/hague_inp_files_pyswmm/hague_v17_simpleRBC.inp"
hs_inp = "C:/PycharmProjects/swmm_rl_hague/hague_inp_files_pyswmm/hague_v17_simpleRBC_hs.inp"
# swmm_inp = "C:/PycharmProjects/swmm_rl_hague/hague_inp_files_pyswmm/hague_v11_template_WQ_AllEnvData_withLC2_notreat.inp"
fcst_data = pd.read_csv("C:/Users/Ben Bowes/Documents/LongTerm_SWMM/hague_fcst_2010_2019.csv")
gwl_df = pd.read_csv("C:/Users/Ben Bowes/Documents/HRSD GIS/Site Data/Data_2010_2019/IDW_2010_2019.csv",
                     usecols=["Datetime", "IDW"], index_col="Datetime", infer_datetime_format=True,
                     parse_dates=True)

# # loop through input files
# for file in os.scandir(inp_dir):
#     if file.name.endswith('.inp'):
#         print(file.name)
#         swmm_inp = file.path  # swmm input file
#         fcst_data = pd.read_csv(os.path.join(fcst_dir, file.name.split('.')[0] + ".csv"))  # forecast data frame

time = []
St1_inflow, St2_inflow, St3_inflow = [], [], []
St1_gwflow, St2_gwflow, St3_gwflow = [], [], []
St1_depth, St2_depth, St3_depth = [], [], []
St1_flooding, St2_flooding, St3_flooding = [], [], []  # incremental flood rate
St1_fld_vol, St2_fld_vol, St3_fld_vol = [], [], []  # incremental flood volume
St1_full, St2_full, St3_full = [], [], []
St1_TSS, St2_TSS, St3_TSS = [], [], []
St1_TSS_mass, St2_TSS_mass, St3_TSS_mass = [], [], []
St1_TP_mass, St2_TP_mass, St3_TP_mass = [], [], []
St1_TN_mass, St2_TN_mass, St3_TN_mass = [], [], []
St1_TP, St2_TP, St3_TP = [], [], []
St1_TN, St2_TN, St3_TN = [], [], []
St1_flow, St2_flow, St3_flow = [], [], []
R1_TSS, R2_TSS, R3_TSS = [], [], []
R1_TP, R2_TP, R3_TP = [], [], []
R1_TN, R2_TN, R3_TN = [], [], []
R1_flow, R2_flow, R3_flow = [], [], []
R1_act, R2_act, R3_act = [], [], []

with Simulation(swmm_inp) as sim:  # loop through all steps in the simulation
    # sim.step_advance(control_time_step)
    sim.start_time = datetime.datetime(2019, 8, 1, 0, 0, 0)  # change start time here
    sim.end_time = datetime.datetime(2019, 9, 1, 0, 0, 0)  # change end time here
    previous_step = sim.start_time
    node_object = Nodes(sim)  # init node object
    St1 = node_object["st1"]
    St2 = node_object["st2"]
    St3 = node_object["F134101"]

    link_object = Links(sim)  # init link object
    R1 = link_object["R1"]
    R2 = link_object["R2"]
    R3 = link_object["R3"]
    #
    # subcatchment_object = Subcatchments(sim)
    # S1 = subcatchment_object["S1"]
    # S2 = subcatchment_object["S2"]

    # calculate radius (ft) of storage units for GWL calculations
    St1_rad = math.sqrt(100000 / math.pi)
    St2_rad = math.sqrt(20000 / math.pi)
    St3_rad = math.sqrt(50000 / math.pi)

    step_count = 1  # init counter for control step
    current_step = 0  # init counter for accessing forecast
    St1_drain_timer, St3_drain_timer = 0, 0  # init pre-storm drainage timers
    St1_retention_timer, St3_retention_timer = 0, 0  # init in-storm retention timers
    St1_drawdown_timer, St3_drawdown_timer = 0, 0  # init post-storm drawdown timers
    for step in sim:  # loop through all steps in the simulation

        # calculate GW exchange
        # current_gwl = gwl_df.iloc[gwl_df.index.get_loc(sim.current_time, method='nearest')]["IDW"]
        # # print(sim.current_time, current_gwl)
        # St1_gwq = calc_gwl(St1.depth - abs(St1.invert_elevation), current_gwl, St1_rad)
        # St2_gwq = calc_gwl(St2.depth - abs(St2.invert_elevation), current_gwl, St2_rad)
        # St3_gwq = calc_gwl(St3.depth - abs(St3.invert_elevation), current_gwl, St3_rad)
        # print(St1_gwq)

        # print("pregwl", St1.total_inflow)
        # add (or subtract) gw volume to storage unit
        # St1.generated_inflow(St1_gwq)
        # St2.generated_inflow(St2_gwq)
        # St3.generated_inflow(St3_gwq)
        # print("post gwl", St1.total_inflow)

        if step_count % control_time_step == 0:  # on control time step look at forecast
            print("control step:", sim.current_time)

            event_dict = control_rules.read_fcst("rain1", fcst_data, current_step)  # look at forecast
            print("forecast read")

            if event_dict['total'] > 0:  # if rain in forecast check for flooding
                # if St1_drain_timer <= 0 or St2_drain_timer <= 0:  # check if either valve control timer expired
                current_dt = sim.current_time
                print("rain in fcst at ", current_dt)

                # save hotstart file
                dt_hs_file = 'tmp_hsf.hsf'
                dt_hs_path = os.path.join("C:/PycharmProjects/swmm_hague_rbc/cmac_temp", dt_hs_file)
                sim.save_hotstart(dt_hs_path)
                print("hsf saved")

                # run sim for forecast period to get incoming volume
                temp_file = swmm_utils.write_ctl_dates(hs_inp, os.path.join(project_dir, "cmac_temp"), current_dt)
                print("running submodel")

                fcst_submodel = subprocess.check_output("C:/Anaconda2/envs/py36/python.exe C:/PycharmProjects/swmm_hague_rbc/run_hague_fcst.py".split())
                submodel_return = fcst_submodel.decode('utf-8')
                # flood_dict = json.loads(submodel_return)  # TODO fix reading json returned by subprocess
                flood_dict = json.loads(str('{') + submodel_return.split('{')[1].split('}')[0] + str('}'))
                # print(current_step, current_dt, flood_dict)
                print("submodel done")

                # check report file to see if storage units flooded and calculate new valve positions if needed
                if flood_dict["St1"] > 0:
                    St1_drain_steps = control_rules.drain_time(flood_dict["St1"], 100000, St1.head, St1.depth, diam=3.)
                    if St1_drain_steps > St1_drain_timer:
                        St1_drain_timer = St1_drain_steps  # update drain timer if needed
                        St1_retention_timer = 0  # reset retention timer
                        St1_drawdown_timer = 0  # reset drawdown timer
                    # print("St1 may flood, timer updated to:", St1_drain_timer)
                    R1.target_setting = 1.  # apply new valve positions

                if flood_dict["St3"] > 0:
                    St3_drain_steps = control_rules.drain_time(flood_dict["St3"], 50000, St3.head, St3.depth, diam=3.)
                    if St3_drain_steps > St3_drain_timer:
                        St3_drain_timer = St3_drain_steps
                        St3_retention_timer = 0
                        St3_drawdown_timer = 0
                    # print("St3 may flood, timer updated to:", St3_drain_timer)
                    R3.target_setting = 1.  # apply new valve positions

            # check drain timers to see if a pond is draining and decrement
            if St1_drain_timer > 0:
                if St1_drain_timer == 1:  # pond has drawn down before storm, start retention timer
                    St1_retention_timer = 97  # retain for 1 day (96 steps + 1 to account for first decrement)
                    # TODO retention timers could be based on when storm event ends
                St1_drain_timer -= 1
            if St3_drain_timer > 0:
                if St3_drain_timer == 1:
                    St3_retention_timer = 97
                St3_drain_timer -= 1

            # check retention timers
            if St1_retention_timer > 0:
                R1.target_setting = 0  # valve closed during retention period
                if St1_retention_timer == 1:  # pond has retained stormwater, start drawdown timer
                    St1_drawdown_timer = 97
                    R1.target_setting = control_rules.valve_position(St1.depth, 100000, target_depth=0., diam=3.)
                St1_retention_timer -= 1
            if St3_retention_timer > 0:
                R3.target_setting = 0
                if St3_retention_timer == 1:
                    St3_drawdown_timer = 97
                    R3.target_setting = control_rules.valve_position(St3.depth, 50000, target_depth=3., diam=3.)
                St3_retention_timer -= 1

            # check drawdown timers
            if St1_drawdown_timer > 0:
                St1_drawdown_timer -= 1
                if St1.depth <= -0.5:  # lower target depth
                    St1_drawdown_timer = 0
            if St3_drawdown_timer > 0:
                St3_drawdown_timer -= 1
                if St3.depth <= 2.5:
                    St3_drawdown_timer = 0

            # maintain target depth if no timers running
            if St1_drain_timer <= 0 and St1_retention_timer <= 0 and St1_drawdown_timer <= 0:
                if St1.depth > 0.5:  # upper target depth
                    R1.target_setting = 0.5
                if St1.depth < -0.5:  # lower target depth
                    R1.target_setting = 0
            if St3_drain_timer <= 0 and St3_retention_timer <= 0 and St3_drawdown_timer <= 0:
                if St3.depth > 3.5:
                    R3.target_setting = 0.5
                if St3.depth < 2.5:
                    R3.target_setting = 0

            # override all previous controls if ponds are flooding
            if St1.flooding > 0:
                R1.target_setting = 1.
            if St3.flooding > 0:
                R3.target_setting = 1.

            # print("St1 timers: ", St1_drain_timer, St1_retention_timer, St1_drawdown_timer, R1.current_setting)
            # print("St2 timers: ", St2_drain_timer, St2_retention_timer, St2_drawdown_timer, R2.current_setting)

            # print("cumulative J1 flooding: ", J1.statistics['flooding_volume'] * 7.481,
            #       "list: ", J1_fld_vol, "list sum: ", sum(J1_fld_vol),
            #       "incremental flooding: ", (J1.statistics['flooding_volume'] * 7.481 - sum(J1_fld_vol)))

            # record system state for current time step
            # St1_depth.append(St1.depth)
            # St2_depth.append(St2.depth)
            # J1_depth.append(J1.depth)
            # St1_flooding.append(St1.flooding)
            # St2_flooding.append(St2.flooding)
            # J1_flooding.append(J1.flooding)
            # St1_full.append(St1.full_depth)
            # St2_full.append(St2.full_depth)
            # J1_full.append(J1.full_depth)
            # R1_act.append(R1.current_setting)
            # R2_act.append(R2.current_setting)
            # St1_fld_vol.append((St1.statistics['flooding_volume'] * 7.481 - sum(St1_fld_vol)))  # incremental vol
            # St2_fld_vol.append((St2.statistics['flooding_volume'] * 7.481 - sum(St2_fld_vol)))
            # J1_fld_vol.append((J1.statistics['flooding_volume'] * 7.481 - sum(J1_fld_vol)))
            # total_flood.append((St1.statistics['flooding_volume'] + St2.statistics['flooding_volume'] +
            #                     J1.statistics['flooding_volume']) * 7.481 / 1e6)  # cumulative vol in 10^6 gallons
            time.append(sim.current_time)
            # St1_gwflow.append(St1_gwq), St2_gwflow.append(St2_gwq), St3_gwflow.append(St3_gwq),
            St1_depth.append(St1.depth - abs(St1.invert_elevation))
            St2_depth.append(St2.depth - abs(St2.invert_elevation))
            St3_depth.append(St3.depth - abs(St3.invert_elevation))
            St1_flooding.append(St1.flooding), St2_flooding.append(St2.flooding), St3_flooding.append(St3.flooding)
            St1_fld_vol.append((St1.statistics['flooding_volume'] * 7.481 - sum(St1_fld_vol)))  # incremental vol in gal
            St2_fld_vol.append((St2.statistics['flooding_volume'] * 7.481 - sum(St2_fld_vol)))
            St3_fld_vol.append((St3.statistics['flooding_volume'] * 7.481 - sum(St3_fld_vol)))
            St1_full.append(St1.full_depth - abs(St1.invert_elevation))
            St2_full.append(St2.full_depth - abs(St2.invert_elevation))
            St3_full.append(St3.full_depth - abs(St3.invert_elevation))
            St1_flow.append(St1.total_inflow * 28.3168)  # pond inflow, converted to L to calculate pollutant mass
            St2_flow.append(St2.total_inflow * 28.3168)
            St3_flow.append(St3.total_inflow * 28.3168)
            R1_flow.append(R1.flow), R2_flow.append(R2.flow), R3_flow.append(R3.flow)
            R1_act.append(R1.current_setting), R2_act.append(R2.current_setting), R3_act.append(R3.current_setting)
            St1_TSS_mass.append(St1.pollut_quality['TSS'] * St1.volume * 28.3168 / 453592)  # TSS mass in pond after treatment (lbs)
            St2_TSS_mass.append(St2.pollut_quality['TSS'] * St2.volume * 28.3168 / 453592)
            St3_TSS_mass.append(St3.pollut_quality['TSS'] * St3.volume * 28.3168 / 453592)
            St1_TP_mass.append(St1.pollut_quality['TP'] * St1.volume * 28.3168 / 453592)  # TP mass in pond after treatment (lbs)
            St2_TP_mass.append(St2.pollut_quality['TP'] * St2.volume * 28.3168 / 453592)
            St3_TP_mass.append(St3.pollut_quality['TP'] * St3.volume * 28.3168 / 453592)
            St1_TN_mass.append(St1.pollut_quality['TN'] * St1.volume * 28.3168 / 453592)  # TN mass in pond after treatment (lbs)
            St2_TN_mass.append(St2.pollut_quality['TN'] * St2.volume * 28.3168 / 453592)
            St3_TN_mass.append(St3.pollut_quality['TN'] * St3.volume * 28.3168 / 453592)
            St1_TSS.append(St1.pollut_quality['TSS'])  # TSS concentration after treatment (mg/L)
            St2_TSS.append(St2.pollut_quality['TSS'])
            St3_TSS.append(St3.pollut_quality['TSS'])
            St1_TP.append(St1.pollut_quality['TP'])  # TP concentration after treatment (mg/L)
            St2_TP.append(St2.pollut_quality['TP'])
            St3_TP.append(St3.pollut_quality['TP'])
            St1_TN.append(St1.pollut_quality['TN'])  # TN concentration after treatment (mg/L)
            St2_TN.append(St2.pollut_quality['TN'])
            St3_TN.append(St3.pollut_quality['TN'])
            R1_TSS.append(R1.total_loading['TSS'])  # pounds of TSS passing through orifice
            R2_TSS.append(R2.total_loading['TSS'])
            R3_TSS.append(R3.total_loading['TSS'])
            R1_TP.append(R1.total_loading['TP'])  # pounds of TP passing through orifice
            R2_TP.append(R2.total_loading['TP'])
            R3_TP.append(R3.total_loading['TP'])
            R1_TN.append(R1.total_loading['TN'])  # pounds of TN passing through orifice
            R2_TN.append(R2.total_loading['TN'])
            R3_TN.append(R3.total_loading['TN'])

            current_step += 1
        step_count += 1
    sim.close()

# # read rain and tide data from inp file
# df = swmm_utils.get_env_data(file.path)
#
# save results data
# result_list = [time, St1_depth, St2_depth, St3_depth, St1_flooding, St2_flooding, St3_flooding,
#              St1_TSS, St2_TSS, St3_TSS, St1_TSS_mass, St2_TSS_mass, St3_TSS_mass, R1_TSS, R2_TSS, R3_TSS,
#              St1_TP, St2_TP, St3_TP, R1_TP, R2_TP, R3_TP, St1_TP_mass, St2_TP_mass, St3_TP_mass,
#              St1_TN, St2_TN, St3_TN, R1_TN, R2_TN, R3_TN, St1_TN_mass, St2_TN_mass, St3_TN_mass,
#              St1_gwflow, St2_gwflow, St3_gwflow, St1_flow, St2_flow, St3_flow,
#              St1_full, St2_full, St3_full, R1_act, R2_act, R3_act]
# result_cols = ["Datetime", "St1_depth", "St2_depth", "St3_depth", "St1_flooding", "St2_flooding", "St3_flooding",
#                "St1_TSS", "St2_TSS", "St3_TSS", "St1_TSS_mass", "St2_TSS_mass", "St3_TSS_mass", "R1_TSS", "R2_TSS", "R3_TSS",
#                "St1_TP", "St2_TP", "St3_TP", "R1_TP", "R2_TP", "R3_TP", "St1_TP_mass", "St2_TP_mass", "St3_TP_mass",
#                "St1_TN", "St2_TN", "St3_TN", "R1_TN", "R2_TN", "R3_TN", "St1_TN_mass", "St2_TN_mass", "St3_TN_mass",
#                "St1_gwflow", "St2_gwflow", "St3_gwflow", "St1_flow", "St2_flow", "St3_flow",
#                "St1_full", "St2_full", "St3_full", "R1_act", "R2_act", "R3_act"]
# results_df = pd.DataFrame(result_list).transpose()
# results_df.columns = result_cols
# results_df = pd.concat([results_df, df], axis=1)
# results_df.to_csv(os.path.join(project_dir, "cmac_results/" + file.name.split('.')[0] + ".csv"), index=False)
#
# # put result lists in dictionary
# result_dict = {}
# for key, value in zip(result_cols, result_list):
#     result_dict[key] = value
#
# # plot results
# swmm_utils.plot_ctl_results(df, result_dict, file.name, os.path.join(project_dir, "cmac_results"))
