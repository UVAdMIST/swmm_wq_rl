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
from swmm_hague_rbc import control_rules
from swmm_rl_hague import swmm_utils
from swmm_rl_hague.read_rpt import get_ele_df, get_file_contents, get_summary_df, get_total_flooding, get_mass_reacted
import pyswmm
pyswmm.lib.use("swmm5_hs_520dev6")  # use swmm with hot start functionality
from pyswmm import Simulation, Nodes, Links

control_time_step = 900  # control time step in seconds

project_dir = "C:/PycharmProjects/swmm_hague_rbc"

swmm_inp = "C:/PycharmProjects/swmm_rl_hague/hague_inp_files_pyswmm/hague_v22_simpleRBC.inp"
hs_inp = "C:/PycharmProjects/swmm_rl_hague/hague_inp_files_pyswmm/hague_v22_simpleRBC_hs.inp"
# swmm_inp = "C:/PycharmProjects/swmm_rl_hague/hague_inp_files_pyswmm/hague_v11_template_WQ_AllEnvData_withLC2_notreat.inp"
fcst_data = pd.read_csv("C:/Users/Ben Bowes/Documents/LongTerm_SWMM/hague_fcst_2010_2019_dated.csv",
                        index_col="datetime", infer_datetime_format=True, parse_dates=True)
gwl_df = pd.read_csv("C:/Users/Ben Bowes/Documents/HRSD GIS/Site Data/Data_2010_2019/IDW_2010_2019.csv",
                     usecols=["Datetime", "IDW"], index_col="Datetime", infer_datetime_format=True,
                     parse_dates=True)

start_time = datetime.datetime.now()

with Simulation(swmm_inp) as sim:  # loop through all steps in the simulation
    sim.step_advance(control_time_step)
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
    # St2_rad = math.sqrt(20000 / math.pi)
    St3_rad = math.sqrt(50000 / math.pi)

    step_count = 1  # init counter for control step
    current_step = 0  # init counter for accessing forecast
    St1_drain_timer, St3_drain_timer = 0, 0  # init pre-storm drainage timers
    St1_retention_timer, St3_retention_timer = 0, 0  # init in-storm retention timers
    St1_drawdown_timer, St3_drawdown_timer = 0, 0  # init post-storm drawdown timers
    for step in sim:  # loop through all steps in the simulation

        # calculate GW exchange
        # current_gwl = gwl_df.iloc[gwl_df.index.get_loc(sim.current_time, method='nearest')]["IDW"]
        # print(sim.current_time, current_gwl)
        # St1_gwq = swmm_utils.calc_dupuit(St1.depth, St1.invert_elevation, current_gwl, St1_rad)  #  - abs(St1.invert_elevation)
        # St2_gwq = calc_gwl(St2.depth - abs(St2.invert_elevation), current_gwl, St2_rad)
        # St3_gwq = swmm_utils.calc_dupuit(St3.depth, St3.invert_elevation, current_gwl, St3_rad)  #  - abs(St3.invert_elevation)

        # print("pregwl", St1.total_inflow)
        # add (or subtract) gw volume to storage unit
        # St1.generated_inflow(St1_gwq)
        # St2.generated_inflow(St2_gwq)
        # St3.generated_inflow(St3_gwq)
        # print("post gwl", St1.total_inflow)

        # if step_count % control_time_step == 0:  # on control time step look at forecast
        current_dt = sim.current_time
        print("control datetime:", current_dt)

        # event_dict = control_rules.read_fcst("rain1", fcst_data, current_step)  # look at forecast
        current_fcst = fcst_data.iloc[fcst_data.index.get_loc(current_dt, method='nearest')]  # ["IDW"]
        # print("forecast read")

        if sum(current_fcst[:int(len(current_fcst)/2)]) > 0.25:  # if storm in forecast check for flooding (1inch for study 3)
            # if event_dict['total'] > 0.5:  # if rain in forecast check for flooding
            # if St1_drain_timer <= 0 or St2_drain_timer <= 0:  # check if either valve control timer expired
            # current_dt = sim.current_time
            print(sum(current_fcst[:int(len(current_fcst)/2)]), " inches of rain in fcst at ", current_dt)

            # save hotstart file
            dt_hs_file = 'tmp_hsf.hsf'
            dt_hs_path = os.path.join("C:/PycharmProjects/swmm_hague_rbc/cmac_temp", dt_hs_file)
            sim.save_hotstart(dt_hs_path)
            # print("hsf saved")

            # run sim for forecast period to get incoming volume
            temp_file = swmm_utils.write_ctl_dates(hs_inp, os.path.join(project_dir, "cmac_temp"), current_dt)
            # print("running submodel")

            fcst_submodel = subprocess.check_output("C:/Anaconda2/envs/py36rbc/python.exe C:/PycharmProjects/swmm_hague_rbc/run_hague_fcst.py".split())
            # fcst_submodel = subprocess.run("C:/Anaconda2/envs/py36rbc/python.exe C:/PycharmProjects/swmm_hague_rbc/run_hague_fcst2.py")
            submodel_return = fcst_submodel.decode('utf-8')

            flood_dict = json.loads(str('{') + submodel_return.split('{')[1].split('}')[0] + str('}'))

            # check report file to see if storage units flooded and calculate new valve positions if needed
            if flood_dict["St1"] > 0:
                # if st1_fcst_vol > 0:
                St1_drain_steps = control_rules.drain_time(flood_dict["St1"], 100000, St1.head, St1.depth, diam=3.)
                # St1_drain_steps = control_rules.drain_time(st1_fcst_vol, 100000, St1.head, St1.depth, diam=3.)
                if St1_drain_steps > St1_drain_timer:
                    St1_drain_timer = St1_drain_steps  # update drain timer if needed
                    St1_retention_timer = 0  # reset retention timer
                    St1_drawdown_timer = 0  # reset drawdown timer
                # print("St1 may flood, timer updated to:", St1_drain_timer)
                R1.target_setting = 1.  # apply new valve positions
            else:
                St1_drain_timer = 0
                St1_retention_timer = 97
                St1_drawdown_timer = 0

            if flood_dict["St3"] > 0:
                # if st3_fcst_vol > 0:
                St3_drain_steps = control_rules.drain_time(flood_dict["St3"], 50000, St3.head, St3.depth, diam=3.)
                # St3_drain_steps = control_rules.drain_time(st3_fcst_vol, 50000, St3.head, St3.depth, diam=3.)
                if St3_drain_steps > St3_drain_timer:
                    St3_drain_timer = St3_drain_steps
                    St3_retention_timer = 0
                    St3_drawdown_timer = 0
                # print("St3 may flood, timer updated to:", St3_drain_timer)
                R3.target_setting = 1.  # apply new valve positions
            else:
                St3_drain_timer = 0
                St3_retention_timer = 97
                St3_drawdown_timer = 0

        # check drain timers to see if a pond is draining and decrement
        if St1_drain_timer > 0:
            if St1_drain_timer == 1:  # pond has drawn down before storm, start retention timer
                St1_retention_timer = 97  # retain for 1 day (96 steps + 1 to account for first decrement)
            St1_drain_timer -= 1
        # TODO retention timers need to operate when storm event ends with/without drain timer ending
        # if St1_drain_timer <= 0 and current_fcst[0] > 0.1:
        #     St1_retention_timer = 97

        if St3_drain_timer > 0:
            if St3_drain_timer == 1:
                St3_retention_timer = 97
            St3_drain_timer -= 1
        # if St3_drain_timer <= 0 and current_fcst[0] > 0.1:
        #     St3_retention_timer = 97

        # check retention timers
        if St1_retention_timer > 0:
            R1.target_setting = 0  # valve closed during retention period
            if St1_retention_timer == 1:  # pond has retained stormwater, start drawdown timer
                St1_drawdown_timer = 97
                R1.target_setting = control_rules.valve_position(St1.depth, 100000, target_depth=5.7, diam=3.)
            St1_retention_timer -= 1
        if St3_retention_timer > 0:
            if St3.depth > 5.75:  # threshold depth to limit US flooding
                R3.target_setting = 1
            else:
                R3.target_setting = 0
            if St3_retention_timer == 1:
                St3_drawdown_timer = 97
                R3.target_setting = control_rules.valve_position(St3.depth, 50000, target_depth=3.56, diam=3.)
            St3_retention_timer -= 1

        # check drawdown timers
        if St1_drawdown_timer > 0:
            St1_drawdown_timer -= 1
            if St1.depth <= 5.7:  # lower target depth referenced to storage invert
                St1_drawdown_timer = 0
        if St3_drawdown_timer > 0:
            St3_drawdown_timer -= 1
            if St3.depth <= 3.:
                St3_drawdown_timer = 0

        # maintain target depth if no timers running
        if St1_drain_timer <= 0 and St1_retention_timer <= 0 and St1_drawdown_timer <= 0:
            if St1.depth > 6.45:  # upper target depth referenced to storage invert
                R1.target_setting = 0.5
            if St1.depth < 5.7:  # lower target depth referenced to storage invert
                R1.target_setting = 0
        if St3_drain_timer <= 0 and St3_retention_timer <= 0 and St3_drawdown_timer <= 0:
            if St3.depth > 3.75:
                R3.target_setting = 0.5
            if St3.depth < 3.:
                R3.target_setting = 0

        # override all previous controls if ponds are flooding
        if St1.flooding > 0:
            R1.target_setting = 1.
        if St3.flooding > 0:
            R3.target_setting = 1.

            current_step += 1
        step_count += 1
    sim.report()
# sim.close()

# save results from rpt file
lines = get_file_contents(swmm_inp.split('.')[0] + ".rpt")

# create dfs of time series
st1_df = get_ele_df("Node st1", lines)
st1_df["St1_TSS_Load"] = st1_df["Inflow"] * st1_df["TSS"] * 900 * 28.317 / 453592  # est. load in lb/time step
st1_df.columns = ['Date', 'Time', 'St1_Inflow', 'St1_Flooding', 'St1_Depth',
                  'St1_Head', 'St1_TSS_Conc', "St1_TSS_Load"]
st1_df.drop(["Date", "Time"], axis=1, inplace=True)
st3_df = get_ele_df("Node F134101", lines)
st3_df["St3_TSS_load"] = st3_df["Inflow"] * st3_df["TSS"] * 900 * 28.317 / 453592
st3_df.columns = ['Date', 'Time', 'St3_Inflow', 'St3_Flooding', 'St3_Depth',
                  'St3_Head', 'St3_TSS_Conc', "St3_TSS_Load"]
st3_df.drop(["Date", "Time"], axis=1, inplace=True)
r1_df = get_ele_df("Link R1", lines)
# r1_df["R1_TSS_load"] = r1_df["Flow"] * r1_df["TSS"] * 900 * 28.317 / 453592
r1_df.columns = ['Date', 'Time', 'R1_Flow', 'R1_Velocity', 'R1_Depth', 'R1_act', 'R1_TSS_Conc']
r1_df.drop(["Date", "Time"], axis=1, inplace=True)
r3_df = get_ele_df("Link R3", lines)
# r3_df["R3_TSS_load"] = r3_df["Flow"] * r3_df["TSS"] * 900 / 453592 / 28.317
r3_df.columns = ['Date', 'Time', 'R3_Flow', 'R3_Velocity', 'R3_Depth', 'R3_act', 'R3_TSS_Conc']
r3_df.drop(["Date", "Time"], axis=1, inplace=True)

rpt_df = pd.concat([st1_df, st3_df, r1_df, r3_df], axis=1)

# get data from summary rpt file
total_flood = get_total_flooding(lines)
tss_reacted = get_mass_reacted(lines)
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
rpt_df['Mass_Reacted'] = tss_reacted

rpt_df.to_csv(os.path.join(project_dir, "cmac_results/082019_dntrbc_v22_mass_025thrshld.csv"), index=True)

end_time = datetime.datetime.now()
print("\n run time (hr): ", end_time - start_time)
