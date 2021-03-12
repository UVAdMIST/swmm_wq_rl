"""
Created by Benjamin Bowes, 4-19-19
This script records depth and flood values at each swmm model time step and plots them.
"""

import datetime
import math
import pandas as pd
# from swmm_utils import calc_gwl
from pyswmm import Simulation, Nodes, Links
from processing.read_rpt import get_ele_df, get_file_contents, get_summary_df, get_total_flooding

start_time = datetime.datetime.now()
control_time_step = 900  # control time step in seconds
swmm_inp = "swmm_models/hague_v22_passive.inp"
# gwl_df = pd.read_csv("timeseries_data/GWL_2010_2019.csv",
#                      usecols=["Datetime", "IDW"], index_col="Datetime", infer_datetime_format=True, parse_dates=True)

with Simulation(swmm_inp) as sim:  # rpt with time series
    # sim.step_advance(control_time_step)
    sim.start_time = datetime.datetime(2010, 1, 1, 0, 0, 0)  # change start time here
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

    # calculate radius (ft) of storage units for GWL calculations
    St1_rad = math.sqrt(100000/math.pi)
    St2_rad = math.sqrt(20000/math.pi)
    St3_rad = math.sqrt(50000/math.pi)

    for step in sim:
        # print(step_count)
        if sim.percent_complete * 100 % 5 == 0:
            print(sim.percent_complete)

        if sim.current_time == sim.start_time:
            R1.target_setting = 1
            R2.target_setting = 1
            R3.target_setting = 1
    sim.report()

# read rpt file
lines = get_file_contents(swmm_inp.split('.')[0] + ".rpt")  # rpt with time series

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

rpt_df.to_csv("results_passive/alldata_passive_v22.csv", index=True)

print("\n run time: ", (datetime.datetime.now() - start_time))
