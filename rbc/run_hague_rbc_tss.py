"""
Created by Benjamin Bowes, 11-19-2020
This script simulates stormwater system control based on a TSS threshold similar to Sharior et al. (2019).
Depth and flood values at each SWMM time step are recorded and plotted.

This script is for a simple Hague simulation where only St1 and St3 are controlled.
Pump is operated by default rules in inp file (on=1ft, off=0.5ft).
"""

import os
import math
import datetime
import pandas as pd
from pyswmm import Simulation, Nodes, Links
from swmm_rl_hague.read_rpt import get_ele_df, get_file_contents, get_summary_df, get_total_flooding

control_time_step = 900  # control time step in seconds

project_dir = "C:/PycharmProjects/swmm_hague_rbc"

swmm_inp = "C:/PycharmProjects/swmm_rl_hague/hague_inp_files_pyswmm/hague_v22_passive2.inp"
swmm_inp2 = "C:/PycharmProjects/swmm_rl_hague/hague_inp_files_pyswmm/hague_v22_passive.inp"

gwl_df = pd.read_csv("C:/Users/Ben Bowes/Documents/HRSD GIS/Site Data/Data_2010_2019/IDW_2010_2019.csv",
                     usecols=["Datetime", "IDW"], index_col="Datetime", infer_datetime_format=True,
                     parse_dates=True)

run_start = datetime.datetime.now()

with Simulation(swmm_inp) as sim:  # run sim for rpt w/o time series
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

    # calculate radius (ft) of storage units for GWL calculations
    St1_rad = math.sqrt(100000 / math.pi)
    St2_rad = math.sqrt(20000 / math.pi)
    St3_rad = math.sqrt(50000 / math.pi)

    step_count = 1  # init counter for control step
    current_step = 0  # init counter for accessing forecast
    for step in sim:  # loop through all steps in the simulation
        if sim.current_time == sim.start_time:
            R1.target_setting = 1.
            R3.target_setting = 1.

        # calculate GW exchange
        # current_gwl = gwl_df.iloc[gwl_df.index.get_loc(sim.current_time, method='nearest')]["IDW"]
        # # print(sim.current_time, current_gwl)
        # St1_gwq = swmm_utils.calc_gwl(St1.depth - abs(St1.invert_elevation), current_gwl, St1_rad)
        # St2_gwq = swmm_utils.calc_gwl(St2.depth - abs(St2.invert_elevation), current_gwl, St2_rad)
        # St3_gwq = swmm_utils.calc_gwl(St3.depth - abs(St3.invert_elevation), current_gwl, St3_rad)
        # print(St1_gwq)

        # print("pregwl", St1.total_inflow)
        # add (or subtract) gw volume to storage unit
        # St1.generated_inflow(St1_gwq)
        # St2.generated_inflow(St2_gwq)
        # St3.generated_inflow(St3_gwq)
        # print("post gwl", St1.total_inflow)

        if step_count % control_time_step == 0:  # on control time step look at forecast
            print("control step:", sim.current_time)

            if St1.pollut_quality['TSS'] >= 1:  # threshold for valve operation
                R1.target_setting = 0
            else:
                R1.target_setting = 1.

            if St3.pollut_quality['TSS'] >= 1:  # threshold for valve operation
                R3.target_setting = 0
            else:
                R3.target_setting = 1.

            # set max allowed depth
            if St1.flooding > 0:
                R1.target_setting = 1.
            if St3.depth > 5.75:
                R3.target_setting = 1.

            # # override all previous controls if ponds are flooding
            # if St1.flooding > 0:
            #     R1.target_setting = 1.
            # if St3.flooding > 0:
            #     R3.target_setting = 1.

            current_step += 1
        step_count += 1
    sim.report()
sim.close()

with Simulation(swmm_inp2) as sim:  # run sim for rpt w/o time series
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

    # calculate radius (ft) of storage units for GWL calculations
    St1_rad = math.sqrt(100000 / math.pi)
    St2_rad = math.sqrt(20000 / math.pi)
    St3_rad = math.sqrt(50000 / math.pi)

    step_count = 1  # init counter for control step
    current_step = 0  # init counter for accessing forecast
    for step in sim:  # loop through all steps in the simulation
        if sim.current_time == sim.start_time:
            R1.target_setting = 1.
            R3.target_setting = 1.

        # calculate GW exchange
        # current_gwl = gwl_df.iloc[gwl_df.index.get_loc(sim.current_time, method='nearest')]["IDW"]
        # # print(sim.current_time, current_gwl)
        # St1_gwq = swmm_utils.calc_gwl(St1.depth - abs(St1.invert_elevation), current_gwl, St1_rad)
        # St2_gwq = swmm_utils.calc_gwl(St2.depth - abs(St2.invert_elevation), current_gwl, St2_rad)
        # St3_gwq = swmm_utils.calc_gwl(St3.depth - abs(St3.invert_elevation), current_gwl, St3_rad)
        # print(St1_gwq)

        # print("pregwl", St1.total_inflow)
        # add (or subtract) gw volume to storage unit
        # St1.generated_inflow(St1_gwq)
        # St2.generated_inflow(St2_gwq)
        # St3.generated_inflow(St3_gwq)
        # print("post gwl", St1.total_inflow)

        if step_count % control_time_step == 0:  # on control time step look at forecast
            print("control step:", sim.current_time)

            if St1.pollut_quality['TSS'] >= 1:  # threshold for valve operation
                R1.target_setting = 0
            else:
                R1.target_setting = 1.

            if St3.pollut_quality['TSS'] >= 1:  # threshold for valve operation
                R3.target_setting = 0
            else:
                R3.target_setting = 1.

            # set max allowed depth, overrides TSS controls
            if St1.flooding > 0:
                R1.target_setting = 1.
            if St3.depth > 5.75:
                R3.target_setting = 1.

            # # override all previous controls if ponds are flooding
            # if St1.flooding > 0:
            #     R1.target_setting = 1.
            # if St3.flooding > 0:
            #     R3.target_setting = 1.

            current_step += 1
        step_count += 1
    sim.report()
sim.close()

lines = get_file_contents(swmm_inp.split('.')[0] + ".rpt")  # rpt w/o time series
lines2 = get_file_contents(swmm_inp2.split('.')[0] + ".rpt")  # rpt with time series

# create dfs of time series
st1_df = get_ele_df("Node st1", lines2)
st1_df["St1_TSS_Load"] = st1_df["Inflow"] * st1_df["TSS"] * 900 * 28.317 / 453592  # est. load in lb/time step
st1_df.columns = ['Date', 'Time', 'St1_Inflow', 'St1_Flooding', 'St1_Depth',
                  'St1_Head', 'St1_TSS_Conc', "St1_TSS_Load"]
st1_df.drop(["Date", "Time"], axis=1, inplace=True)
st3_df = get_ele_df("Node F134101", lines2)
st3_df["St3_TSS_load"] = st3_df["Inflow"] * st3_df["TSS"] * 900 * 28.317 / 453592
st3_df.columns = ['Date', 'Time', 'St3_Inflow', 'St3_Flooding', 'St3_Depth',
                  'St3_Head', 'St3_TSS_Conc', "St3_TSS_Load"]
st3_df.drop(["Date", "Time"], axis=1, inplace=True)
r1_df = get_ele_df("Link R1", lines2)
# r1_df["R1_TSS_load"] = r1_df["Flow"] * r1_df["TSS"] * 900 * 28.317 / 453592
r1_df.columns = ['Date', 'Time', 'R1_Flow', 'R1_Velocity', 'R1_Depth', 'R1_act', 'R1_TSS_Conc']
r1_df.drop(["Date", "Time"], axis=1, inplace=True)
r3_df = get_ele_df("Link R3", lines2)
# r3_df["R3_TSS_load"] = r3_df["Flow"] * r3_df["TSS"] * 900 / 453592 / 28.317
r3_df.columns = ['Date', 'Time', 'R3_Flow', 'R3_Velocity', 'R3_Depth', 'R3_act', 'R3_TSS_Conc']
r3_df.drop(["Date", "Time"], axis=1, inplace=True)

# get data from summary rpt file
total_flood = get_total_flooding(lines)
flood_df = get_summary_df(lines, "Node Flooding Summary")
flood_df.columns = ["Hrs_Fld", "Max_Rate", "Max_Day", "Max_Time", "Total_Vol", "Max_Ponded"]
outfall_df = get_summary_df(lines, "Outfall Loading Summary")
outfall_df.columns = ["Flow_Pcnt", "Avg_Flow", "Max_Flow", "Total_Vol", "TSS"]
st_df = get_summary_df(lines, "Storage Volume Summary")
st_df.columns = ["Avg_Vol", "Avg_Pcnt_Full", "Evap_Pcnt", "Exfil_Pcnt", "Max_Vol",
                 "Max_Pcnt_Full", "Max_Day", "Max_Time", "Max_Outflow"]
pump_df = get_summary_df(lines, "Pumping Summary")
pump_df.columns = ["Percent_Utilized", "Start_Ups", "Min_Flow", "Avg_Flow",
                   "Max_Flow", "Total_Vol", "Power_Use", "Pcnt_Off_Low", "Pcnt_Off_High"]
pollut_df = get_summary_df(lines, "Link Pollutant Load Summary")
pollut_df.columns = ["TSS"]

# save results data
rpt_df = pd.concat([st1_df, st3_df, r1_df, r3_df], axis=1)

if rpt_df["St1_Flooding"].any() > 0:
    rpt_df["St1_Fld_Vol"] = flood_df.loc["st1"]["Total_Vol"]
if rpt_df["St3_Flooding"].any() > 0:
    rpt_df["St3_Fld_Vol"] = flood_df.loc["F134101"]["Total_Vol"]
rpt_df["Pump_Pcnt"] = pump_df.loc["P1"]["Percent_Utilized"]
rpt_df["Pump_Starts"] = pump_df.loc["P1"]["Start_Ups"]
rpt_df["St1_TSS_Load"] = rpt_df["St1_TSS_Load"].sum()
rpt_df["St3_TSS_Load"] = rpt_df["St3_TSS_Load"].sum()
rpt_df["R1_TSS_Load"] = pollut_df.loc["R1"]["TSS"]
rpt_df["R3_TSS_Load"] = pollut_df.loc["R3"]["TSS"]
rpt_df["Outfall_TSS_Load"] = outfall_df.loc["System"]["TSS"]
rpt_df["St1_Max"] = 10
rpt_df['St3_Max'] = 6.56
rpt_df['Total_Flood'] = total_flood

rpt_df.to_csv(os.path.join(project_dir, "cmac_results/082019_v22_tss1max575_rbc_withOutfall.csv"), index=True)

run_end = datetime.datetime.now()
print("\n run time (hr): ", (run_end - run_start))

# # put result lists in dictionary
# # read observed rain and tide data
# obs_rain_path = "C:/Users/Ben Bowes/Documents/LongTerm_SWMM/Clean_rain_data/rain_cleaned_combined.csv"
# obs_tide_path = "C:/Users/Ben Bowes/Documents/LongTerm_SWMM/Raw_tide_data/tide_15min_2010_2019.csv"
# obs_rain_df = pd.read_csv(obs_rain_path, index_col="datetime", infer_datetime_format=True, parse_dates=True)
# obs_tide_df = pd.read_csv(obs_tide_path, index_col="datetime", infer_datetime_format=True, parse_dates=True)
# # slice observed data to match simulation results
# # start = datetime.datetime(2019, 8, 1, 0, 0, 0)  # change start time here
# # end = datetime.datetime(2019, 9, 1, 0, 0, 0)  # change end time here
# start = time[0]  # change start time here
# end = time[-1]  # change end time here
#
# obs_rain_df = obs_rain_df.loc[start: end]
# obs_tide_df = obs_tide_df.loc[start: end]
# env_df = pd.concat([obs_rain_df, obs_tide_df], axis=1)
# env_df.reset_index(inplace=True)

# result_list = [time, St1_depth, St2_depth, St3_depth, St1_flooding, St2_flooding, St3_flooding,
#                St1_fld_vol, St2_fld_vol, St3_fld_vol, St1_TSS_Load, St2_TSS_Load, St3_TSS_Load,
#                R1_TSS_Load, R2_TSS_Load, R3_TSS_Load, R1_TSS_Conc, R2_TSS_Conc, R3_TSS_Conc,
#                St1_gwflow, St2_gwflow, St3_gwflow, St1_flow, St2_flow, St3_flow,
#                St1_full, St2_full, St3_full, R1_act, R2_act, R3_act]
# result_cols = ["Datetime", "St1_depth", "St2_depth", "St3_depth", "St1_flooding", "St2_flooding", "St3_flooding",
#                "St1_fld_vol", "St2_fld_vol", "St3_fld_vol", "St1_TSS_Load", "St2_TSS_Load", "St3_TSS_Load",
#                "R1_TSS_Load", "R2_TSS_Load", "R3_TSS_Load", "R1_TSS_Conc", "R2_TSS_Conc", "R3_TSS_Conc",
#                "St1_gwflow", "St2_gwflow", "St3_gwflow", "St1_flow", "St2_flow", "St3_flow",
#                "St1_full", "St2_full", "St3_full", "R1_act", "R2_act", "R3_act"]
# results_df = pd.DataFrame(result_list).transpose()
# results_df.columns = result_cols
# results_df = pd.concat([results_df, df], axis=1)

# result_dict = {}
# for key, value in zip(result_cols, result_list):
#     result_dict[key] = value
#
# # plot results TODO plot RBC results
# swmm_utils.plot_ctl_results(env_df, result_dict, "082019_v22_thrshld05", os.path.join(project_dir, "cmac_results"))
