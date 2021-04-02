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

    return rpt_df


# def get_end_line(file_lines, start_line):
#     for i in range(len(file_lines[start_line:])):
#         line_no = start_line + i
#         if file_lines[line_no].strip() == "" and file_lines[line_no + 1].strip() == "":
#             return line_no
#     # raise error if end line of section not found
#     raise KeyError('Did not find end of section starting on line {}'.format(start_line))


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


# def read_rpt(rpt_path):
#     flood_dict = {}
#
#     # get total flood volume (10^6 gal) from .rpt file
#     total_flood = 0
#     with open(rpt_path, 'r') as rpt_file:
#         lines = rpt_file.readlines()
#     for i, l in enumerate(lines):
#         if l.startswith("  Node Flooding Summary"):  # find flooding section
#             start = i + 10
#             end = get_end_line(file_lines=lines, start_line=start)
#             for line in lines[start:end]:
#                 print(line)
#                 flood_dict[(line.strip().split()[0])] = float(line.strip().split()[-2])
#                 total_flood += float(line.strip().split()[-2])
#
#     return flood_dict


# def get_env_data(inp_path):
#     """get rain/tide data from inp file"""
#     rain1_str = []
#     rain2_str = []
#     tide_str = []
#     with open(inp_path, 'r') as tmp_file:
#         lines = tmp_file.readlines()
#         for i, l in enumerate(lines):
#             if l.startswith("[TIMESERIES]"):  # find time series section
#                 start = i + 3
#     for i, l in enumerate(lines[start:]):
#         if l.startswith('Rain1'):
#             rain1_str.append(l)
#         if l.startswith('Rain2'):
#             rain2_str.append(l)
#         if l.startswith('Tide1'):
#             tide_str.append(l)
#
#     rain1_data = []
#     rain1_time = []
#     rain2_data = []
#     rain2_time = []
#     tide_data = []
#     tide_time = []
#     for i in rain1_str:
#         rain1_data.append(i.split(' ')[3].rstrip())
#         rain1_time.append(i.split(' ')[1] + " " + i.split(' ')[2])
#
#     for i in rain2_str:
#         rain2_data.append(i.split(' ')[3].rstrip())
#         rain2_time.append(i.split(' ')[1] + " " + i.split(' ')[2])
#
#     for i in tide_str:
#         tide_data.append(i.split(' ')[3].rstrip())
#         tide_time.append(i.split(' ')[1] + " " + i.split(' ')[2])
#
#     rain1_df = pd.DataFrame([rain1_time, rain1_data]).transpose()
#     rain1_df.columns = ['datetime1', 'rain1']
#     rain1_df['datetime1'] = pd.to_datetime(rain1_df['datetime1'], infer_datetime_format=True)
#     rain1_df.set_index(pd.DatetimeIndex(rain1_df['datetime1']), inplace=True)
#     rain1_df['rain1'] = rain1_df['rain1'].astype('float')
#     rain1_df = rain1_df.resample('H').sum()
#
#     rain2_df = pd.DataFrame([rain2_time, rain2_data]).transpose()
#     rain2_df.columns = ['datetime2', 'rain2']
#     rain2_df['datetime2'] = pd.to_datetime(rain2_df['datetime2'], infer_datetime_format=True)
#     rain2_df.set_index(pd.DatetimeIndex(rain2_df['datetime2']), inplace=True)
#     rain2_df['rain2'] = rain2_df['rain2'].astype('float')
#     rain2_df = rain2_df.resample('H').sum()
#
#     tide_df = pd.DataFrame([tide_time, tide_data], dtype='float64').transpose()
#     tide_df.columns = ['datetime', 'tide']
#     tide_df['datetime'] = pd.to_datetime(tide_df['datetime'], infer_datetime_format=True)
#     tide_df.set_index(pd.DatetimeIndex(tide_df['datetime']), inplace=True)
#     tide_df['tide'] = tide_df['tide'].astype('float')
#
#     df = pd.concat([rain1_df['rain1'], rain2_df['rain2'], tide_df['tide']], axis=1)
#     df[['rain1', 'rain2']].fillna(0, inplace=True)
#     df.reset_index(inplace=True)
#
#     return df
#
#
# def plot_ctl_results(env_df, ctl_dict, file_name, out_dir):
#     # plot results
#     fig, axs = plt.subplots(4, sharey='none', sharex='none', figsize=(6, 8))
#     # first plot is tide and rainfall
#     ax = axs[0]
#     env_df["tide"].plot(ax=ax, color='c', legend=None)
#     ax2 = ax.twinx()
#     ax2.invert_yaxis()
#     env_df["rain1"].plot.bar(ax=ax2, color="b", legend=None)
#     ax2.set_xticks([])
#     ax.set_xticks([], minor=True)
#     ax.set_ylabel("Sea Level (ft.)")
#     ax2.set_ylabel("Rainfall (in.)")
#     ax.set_title('Inputs')
#     lines, labels = ax.get_legend_handles_labels()
#     lines2, labels2 = ax2.get_legend_handles_labels()
#     ax.legend(lines + lines2, ("Sea Level", "Incremental Rainfall"), bbox_to_anchor=(0.85, -0.05), ncol=2,
#               frameon=False)
#
#     # plot depths
#     axs[1].plot(ctl_dict["St1_depth"], label='St1')
#     axs[1].plot(ctl_dict["St2_depth"], label='St2')
#     axs[1].plot(ctl_dict["J1_depth"], label='J1', color='green')
#     axs[1].plot(ctl_dict["St1_full"], linestyle=':', color='k', label='Storage max')
#     axs[1].plot(ctl_dict["J1_full"], linestyle=':', color='grey', label='Pipe max')
#     axs[1].set_ylim(0, 6)
#     axs[1].set_title('Depths')
#     axs[1].set_ylabel("ft")
#     axs[1].legend(loc=4, bbox_to_anchor=(0.025, -.5, 1., .11), ncol=5,
#                   borderaxespad=0.1, frameon=False, columnspacing=.75)
#
#     # plot actions
#     axs[2].plot(ctl_dict["R1_act"], label='R1')
#     axs[2].plot(ctl_dict["R2_act"], ':', label='R2')
#     axs[2].set_ylim(0, 1.05)
#     axs[2].set_title('Policy')
#     axs[2].set_ylabel("Valve Position")
#     axs[2].legend()
#
#     # plot flooding
#     axs[3].plot(ctl_dict["St1_flooding"], label='St1')
#     axs[3].plot(ctl_dict["St2_flooding"], label='St2')
#     axs[3].plot(ctl_dict["J1_flooding"], label='J1')
#     axs[3].set_title('Flooding')
#     axs[3].set_ylabel("CFS")
#     axs[3].set_xlabel("time step")
#     axs[3].set_ylim(0)
#     flood_str = "Total Vol. = " + str(round(ctl_dict["total_flood"][-1], 3)) + "MG"
#     flood_max = max(max(ctl_dict["St1_flooding"]), max(ctl_dict["St2_flooding"]), max(ctl_dict["J1_flooding"])) * 0.9
#     axs[3].text(0, flood_max, flood_str)
#
#     plt.tight_layout()
#     plt.savefig(os.path.join(out_dir, file_name.split('.')[0] + ".png"), dpi=300)
#     plt.close()
#
#
# def read_saved_results(result_path, base=True):
#     raw_df = pd.read_csv(result_path)
#
#     if base:
#         result_df = raw_df[['St1_depth', 'St2_depth', 'J1_depth', 'St1_flooding', 'St2_flooding',
#                             'J1_flooding', 'total_flood', 'St1_full', 'St2_full', 'J1_full', 'R1_act', 'R2_act']]
#     else:
#         result_df = raw_df[['St1_depth', 'St2_depth', 'J1_depth', 'St1_flooding', 'St2_flooding',
#                             'J1_flooding', 'total_flood', 'St_full', 'J1_full', 'R1_act', 'R2_act']]
#
#     result_df['datetime'] = pd.date_range(start=raw_df["index"][0], periods=len(raw_df["St1_depth"]), freq='15T')
#     result_df.set_index(pd.DatetimeIndex(result_df['datetime']), inplace=True, drop=True)
#
#     data_df = raw_df[['index', 'rain1', 'rain2', 'tide']]
#     data_df.set_index(pd.DatetimeIndex(data_df['index']), inplace=True, drop=True)
#     data_df.dropna(subset=['tide'], inplace=True)
#     data_df.fillna(0, inplace=True)
#     # data_df.drop('index', axis=1, inplace=True)
#     # data_df.reset_index(inplace=True)
#     data_df = data_df.resample('15T').asfreq()  # 15min resample
#     data_df.fillna({'rain1': 0, 'rain2': 0, 'tide': data_df['tide'].interpolate()}, inplace=True)
#
#     df = pd.concat([result_df, data_df], axis=1)
#     df.reset_index(inplace=True)
#     return df  # result_df, data_df


def calc_gwl(pond_depth, gwl, radius, K=1.96/86400):
    """
    GWL exchange is calculated as Q = KIA, assuming:
    1) surface area does not change with depth (storage units are right cylinders)
    2) area for exchange = lateral surface area + bottom surface area

    pond depth in feet
    gwl in feet
    radius in feet
    K converted from ft/day to ft/s
    routing step = 1 second
    Q in ft^3/s
    """

    area = 2 * math.pi * radius * pond_depth + math.pi * radius**2  # x-sectional area

    gradient = gwl - pond_depth  # negative gradient is water leaving pond

    Q = K * gradient * area

    return Q
