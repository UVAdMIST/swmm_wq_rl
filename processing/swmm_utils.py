"""
Benjamin Bowes, 02-28-2020
Script containing utility functions for working with SWMM simulation files
Utilities to extract information from SWMM report files modified from
https://github.com/UVAdMIST/swmm_mpc/blob/master/swmm_mpc/rpt_ele.py by jsadler2
"""

import os
import math
import pandas as pd
from datetime import timedelta
# import matplotlib.pyplot as plt
# from processing.read_rpt import get_ele_df, get_file_contents, get_summary_df, get_total_flooding


def get_file_contents(rpt_file):
    with open(rpt_file, 'r') as f:
        lines = f.readlines()
        return lines


def get_ele_df(ele, file_contents):
    # start_line_no, end_line_no = get_ele_lines(ele, file_contents)
    start_line_no = get_start_line("<<< {} >>>".format(ele.lower()), file_contents)
    end_line_no = get_end_line(start_line_no, file_contents)
    col_titles = file_contents[start_line_no + 3].strip().split()[:2]
    col_titles.extend(file_contents[start_line_no + 2].strip().split())
    content_start = start_line_no + 5
    content_end = end_line_no - 1
    content_list = []
    for i in range(content_start, content_end):
        content_list.append(file_contents[i].split())
    df = pd.DataFrame(content_list, columns=col_titles)
    df["datetime"] = pd.to_datetime(df["Date"] + " " + df["Time"])
    df["datetime"] = df["datetime"].dt.round('min')
    df.set_index("datetime", inplace=True)
    for c in df.columns:
        try:
            df[c] = pd.to_numeric(df[c])
        except:
            pass
    return df


def get_start_line(start_string, file_contents, start=0):
    for i in range(len(file_contents[start:])):
        line_no = i + start
        line_lower = file_contents[line_no].strip().lower()
        start_string_lower = start_string.lower().strip()

        if line_lower.startswith(start_string_lower):
            return i

    # raise error if start line of section not found
    raise KeyError('Start line for string {} not found'.format(start_string))


def get_end_line(start_line, file_contents):
    for i in range(len(file_contents[start_line:])):
        line_no = start_line + i
        if file_contents[line_no].strip() == "" and \
                file_contents[line_no + 1].strip() == "":
            return line_no
    # raise error if end line of section not found
    raise KeyError('Did not find end of section starting on line {}'.format(start_line))


def get_ele_lines(ele, file_contents):
    start_line = get_start_line("<<< {} >>>".format(ele.lower()), file_contents)
    end_line = get_end_line(start_line, file_contents)
    return start_line, end_line


def get_total_flooding(file_contents):
    fl_start_line = get_start_line("Flooding Loss", file_contents)
    return float(file_contents[fl_start_line].split()[-1])


def get_mass_reacted(file_contents):
    fl_start_line = get_start_line("Mass Reacted", file_contents)
    return float(file_contents[fl_start_line].split()[-1])


def get_summary_df(file_contents, heading):
    """
    heading: heading of summary table (e.g, "Node Flooding Summary")
    returns: a dataframe of the tabular data under the heading specified
    """
    summary_start = get_start_line(heading, file_contents, start=0)
    summary_end = get_end_line(summary_start, file_contents)
    lines = file_contents[summary_start:summary_end]
    # reverse the list of strings so data is on top. makes it easier to handle (less skipping)
    lines.reverse()
    first_row = True
    for i, l in enumerate(lines):
        if not l.strip().startswith('---'):
            # add as row to dataframe
            line = l.strip().split()
            if first_row:
                df = pd.DataFrame(columns=range(len(line)))
                first_row = False
            df.loc[i] = line
        else:
            df.set_index(0, inplace=True)
            return df


def get_result_df(rpt_file):
    # read rpt file
    lines = get_file_contents(rpt_file)

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

    # get summary data
    total_flood = get_total_flooding(lines)
    tss_reacted = get_mass_reacted(lines)
    if total_flood > 0.:
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
    rpt_df["Mass_Reacted"] = tss_reacted

    return rpt_df


def write_ctl_dates(inp_file, temp_dir, start_date, fcst_horizon=24):
    """forecast horizon is time in hours, default 24"""
    end_date = start_date + timedelta(minutes=fcst_horizon*60)

    with open(inp_file, 'r') as tmp_file:  # read inp file
        lines = tmp_file.readlines()

    new_lines = lines.copy()  # copy inp template and update times
    new_lines[13] = 'START_DATE           ' + start_date.date().strftime('%m/%d/%Y') + '\n'
    new_lines[14] = 'START_TIME           ' + start_date.time().strftime('%H:%M:%S') + '\n'
    new_lines[15] = 'REPORT_START_DATE    ' + start_date.date().strftime('%m/%d/%Y') + '\n'
    new_lines[16] = 'REPORT_START_TIME    ' + start_date.time().strftime('%H:%M:%S') + '\n'
    new_lines[17] = 'END_DATE             ' + end_date.date().strftime('%m/%d/%Y') + '\n'
    new_lines[18] = 'END_TIME             ' + end_date.time().strftime('%H:%M:%S') + '\n'

    with open(os.path.join(temp_dir, "temp_inp.inp"), 'w') as tmp_file:  # write temp inp file
        temporary_inp = tmp_file.writelines(new_lines)
    return temporary_inp


def calc_dupuit(pond_depth, pond_elv, gwl, radius, K=1.96/86400, L=1):
    """
    GW exchange is calculated as Q = (K/2L)(h1^2 - h2^2)A, assuming:
    1) Dupuit Assumptions: only horizontal flow through saturated unconfined aquifer, change in dh/dL is small
    2) pond geometry does not change with depth (storage units are right cylinders)
    3) there are two potential scenarios: pond gaining or losing to aquifer

    pond depth in feet, NAVD88
    gwl in feet, NAVD88
    radius in feet
    K converted from ft/day to ft/s
    L is distance from pond to well
    Q in ft^3/s
    """

    h1 = gwl - pond_elv
    h2 = pond_depth

    area = 2 * math.pi * radius * h2  # pond wetted area

    Q = (K/(2*L)) * (h1**2 - h2**2) * area  # negative gradient is water leaving pond

    return Q


def calc_gwl(pond_depth, pond_elv, gwl, radius, K=1.96/86400):
    """
    GW exchange is calculated as Q = KIA, assuming:
    1) Dupuit Assumptions: only horizontal flow through saturated unconfined aquifer
    2) pond geometry does not change with depth (storage units are right cylinders)
    3) there are two potential scenarios: pond gaining or losing to aquifer

    pond depth in feet, NAVD88
    gwl in feet, NAVD88
    radius in feet
    K converted from ft/day to ft/s
    Q in ft^3/s
    """

    gradient = gwl - pond_depth  # negative gradient is water leaving pond

    area = 2 * math.pi * radius * pond_depth

    Q = K * gradient * area

    return Q
