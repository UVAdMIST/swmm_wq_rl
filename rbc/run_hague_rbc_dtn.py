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
from swmm_hague_rbc import swmm_utils
from swmm_rl_hague.read_rpt import get_ele_df, get_file_contents, get_summary_df, get_total_flooding, get_mass_reacted
import pyswmm
pyswmm.lib.use("swmm5_hs_520dev6")
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
    sim.start_time = datetime.datetime(2017, 1, 1, 0, 0, 0)  # change start time here
    sim.end_time = datetime.datetime(2019, 11, 1, 0, 0, 0)  # change end time here
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

        # if step_count % control_time_step == 0:  # on control time step look at forecast
        current_dt = sim.current_time
        print("control datetime:", current_dt)

        # event_dict = control_rules.read_fcst("rain1", fcst_data, current_step)  # look at forecast
        current_fcst = fcst_data.iloc[fcst_data.index.get_loc(current_dt, method='nearest')]  # ["IDW"]
        # print("forecast read")

        if sum(current_fcst[:int(len(current_fcst)/2)]) > 1.:  # if storm in forecast check for flooding
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
            # st1_fcst_vol = 0
            # st3_fcst_vol = 0
            # fcst_rpt = get_file_contents("C:/PycharmProjects/swmm_hague_rbc/cmac_temp/temp_inp.rpt")
            # fcst_flood = get_total_flooding(fcst_rpt)
            # if fcst_flood > 0:
            #     fcst_df = get_summary_df(fcst_rpt, "Node Flooding Summary")
            #     if "st1" in fcst_df.index:
            #         st1_fcst_vol = fcst_df.loc["st1"]["Total_Vol"]
            #     if "st3" in fcst_df.index:
            #         st3_fcst_vol = fcst_df.loc["st3"]["Total_Vol"]
            # flood_dict = json.loads(submodel_return)  # TODO fix reading json returned by subprocess
            flood_dict = json.loads(str('{') + submodel_return.split('{')[1].split('}')[0] + str('}'))
            # print(current_step, current_dt, flood_dict)
            # print("submodel done")

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

            # print("St1 timers: ", St1_drain_timer, St1_retention_timer, St1_drawdown_timer, R1.current_setting)
            # print("St2 timers: ", St2_drain_timer, St2_retention_timer, St2_drawdown_timer, R2.current_setting)

            # record system state for current time step
            # total_flood.append((St1.statistics['flooding_volume'] + St2.statistics['flooding_volume'] +
            #                     J1.statistics['flooding_volume']) * 7.481 / 1e6)  # cumulative vol in 10^6 gallons
            # time.append(sim.current_time)
            # St1_gwflow.append(St1_gwq), St2_gwflow.append(St2_gwq), St3_gwflow.append(St3_gwq),
            # St1_depth.append(St1.depth), St2_depth.append(St2.depth), St3_depth.append(St3.depth)
            # St1_flooding.append(St1.flooding), St2_flooding.append(St2.flooding), St3_flooding.append(St3.flooding)
            # St1_fld_vol.append((St1.statistics['flooding_volume'] * 7.481 - sum(St1_fld_vol)))  # incremental vol in gal
            # St2_fld_vol.append((St2.statistics['flooding_volume'] * 7.481 - sum(St2_fld_vol)))
            # St3_fld_vol.append((St3.statistics['flooding_volume'] * 7.481 - sum(St3_fld_vol)))
            # St1_full.append(St1.full_depth), St2_full.append(St2.full_depth), St3_full.append(St3.full_depth)
            # St1_flow.append(St1.total_inflow * 28.3168)  # pond inflow, converted to L to calculate pollutant mass
            # St2_flow.append(St2.total_inflow * 28.3168)
            # St3_flow.append(St3.total_inflow * 28.3168)
            # R1_flow.append(R1.flow), R2_flow.append(R2.flow), R3_flow.append(R3.flow)
            # R1_act.append(R1.current_setting), R2_act.append(R2.current_setting), R3_act.append(R3.current_setting)
            # St1_TSS_Load.append(St1.pollut_quality['TSS'] * St1.volume * 28.3168 / 453592)  # TSS mass in pond after treatment (lbs)
            # St2_TSS_Load.append(St2.pollut_quality['TSS'] * St2.volume * 28.3168 / 453592)
            # St3_TSS_Load.append(St3.pollut_quality['TSS'] * St3.volume * 28.3168 / 453592)
            # St1_TP_Load.append(St1.pollut_quality['TP'] * St1.volume * 28.3168 / 453592)  # TP mass in pond after treatment (lbs)
            # St2_TP_Load.append(St2.pollut_quality['TP'] * St2.volume * 28.3168 / 453592)
            # St3_TP_Load.append(St3.pollut_quality['TP'] * St3.volume * 28.3168 / 453592)
            # St1_TN_Load.append(St1.pollut_quality['TN'] * St1.volume * 28.3168 / 453592)  # TN mass in pond after treatment (lbs)
            # St2_TN_Load.append(St2.pollut_quality['TN'] * St2.volume * 28.3168 / 453592)
            # St3_TN_Load.append(St3.pollut_quality['TN'] * St3.volume * 28.3168 / 453592)
            # St1_TSS_Conc.append(St1.pollut_quality['TSS'])  # TSS concentration after treatment (mg/L)
            # St2_TSS_Conc.append(St2.pollut_quality['TSS'])
            # St3_TSS_Conc.append(St3.pollut_quality['TSS'])
            # St1_TP_Conc.append(St1.pollut_quality['TP'])  # TP concentration after treatment (mg/L)
            # St2_TP_Conc.append(St2.pollut_quality['TP'])
            # St3_TP_Conc.append(St3.pollut_quality['TP'])
            # St1_TN_Conc.append(St1.pollut_quality['TN'])  # TN concentration after treatment (mg/L)
            # St2_TN_Conc.append(St2.pollut_quality['TN'])
            # St3_TN_Conc.append(St3.pollut_quality['TN'])
            # R1_TSS_Load.append(R1.total_loading['TSS'])  # pounds of TSS passing through orifice
            # R2_TSS_Load.append(R2.total_loading['TSS'])
            # R3_TSS_Load.append(R3.total_loading['TSS'])
            # R1_TSS_Conc.append(R1.pollut_quality['TSS'])  # TSS concentration passing through orifice
            # R2_TSS_Conc.append(R2.pollut_quality['TSS'])
            # R3_TSS_Conc.append(R3.pollut_quality['TSS'])
            # R1_TP_Load.append(R1.total_loading['TP'])  # pounds of TP passing through orifice
            # R2_TP_Load.append(R2.total_loading['TP'])
            # R3_TP_Load.append(R3.total_loading['TP'])
            # R1_TN_Load.append(R1.total_loading['TN'])  # pounds of TN passing through orifice
            # R2_TN_Load.append(R2.total_loading['TN'])
            # R3_TN_Load.append(R3.total_loading['TN'])
            # R1_TSS_inc.append(R1.total_loading['TSS'] - R1_TSS[-1])  # cumulative pounds of TSS passing through orifice
            # R2_TSS_inc.append(R2.total_loading['TSS'] - R2_TSS[-1])
            # R3_TSS_inc.append(R3.total_loading['TSS'] - R3_TSS[-1])

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

rpt_df.to_csv(os.path.join(project_dir, "cmac_results/all_data/2017_2019_dntrbc_v22_mass.csv"), index=True)

end_time = datetime.datetime.now()
print("\n run time (hr): ", end_time - start_time)

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
