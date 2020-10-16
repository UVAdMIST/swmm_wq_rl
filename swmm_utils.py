"""
Benjamin Bowes, 02-28-2020
Script containing utility functions for working with SWMM simulation files
"""

import os
import math
import pandas as pd
from datetime import timedelta
import matplotlib.pyplot as plt


def get_end_line(file_lines, start_line):
    for i in range(len(file_lines[start_line:])):
        line_no = start_line + i
        if file_lines[line_no].strip() == "" and file_lines[line_no + 1].strip() == "":
            return line_no
    # raise error if end line of section not found
    raise KeyError('Did not find end of section starting on line {}'.format(start_line))


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


def read_rpt(rpt_path):
    flood_dict = {}

    # get total flood volume (10^6 gal) from .rpt file
    total_flood = 0
    with open(rpt_path, 'r') as rpt_file:
        lines = rpt_file.readlines()
    for i, l in enumerate(lines):
        if l.startswith("  Node Flooding Summary"):  # find flooding section
            start = i + 10
            end = get_end_line(file_lines=lines, start_line=start)
            for line in lines[start:end]:
                print(line)
                flood_dict[(line.strip().split()[0])] = float(line.strip().split()[-2])
                total_flood += float(line.strip().split()[-2])

    return flood_dict


def get_env_data(inp_path):
    """get rain/tide data from inp file"""
    rain1_str = []
    rain2_str = []
    tide_str = []
    with open(inp_path, 'r') as tmp_file:
        lines = tmp_file.readlines()
        for i, l in enumerate(lines):
            if l.startswith("[TIMESERIES]"):  # find time series section
                start = i + 3
    for i, l in enumerate(lines[start:]):
        if l.startswith('Rain1'):
            rain1_str.append(l)
        if l.startswith('Rain2'):
            rain2_str.append(l)
        if l.startswith('Tide1'):
            tide_str.append(l)

    rain1_data = []
    rain1_time = []
    rain2_data = []
    rain2_time = []
    tide_data = []
    tide_time = []
    for i in rain1_str:
        rain1_data.append(i.split(' ')[3].rstrip())
        rain1_time.append(i.split(' ')[1] + " " + i.split(' ')[2])

    for i in rain2_str:
        rain2_data.append(i.split(' ')[3].rstrip())
        rain2_time.append(i.split(' ')[1] + " " + i.split(' ')[2])

    for i in tide_str:
        tide_data.append(i.split(' ')[3].rstrip())
        tide_time.append(i.split(' ')[1] + " " + i.split(' ')[2])

    rain1_df = pd.DataFrame([rain1_time, rain1_data]).transpose()
    rain1_df.columns = ['datetime1', 'rain1']
    rain1_df['datetime1'] = pd.to_datetime(rain1_df['datetime1'], infer_datetime_format=True)
    rain1_df.set_index(pd.DatetimeIndex(rain1_df['datetime1']), inplace=True)
    rain1_df['rain1'] = rain1_df['rain1'].astype('float')
    rain1_df = rain1_df.resample('H').sum()

    rain2_df = pd.DataFrame([rain2_time, rain2_data]).transpose()
    rain2_df.columns = ['datetime2', 'rain2']
    rain2_df['datetime2'] = pd.to_datetime(rain2_df['datetime2'], infer_datetime_format=True)
    rain2_df.set_index(pd.DatetimeIndex(rain2_df['datetime2']), inplace=True)
    rain2_df['rain2'] = rain2_df['rain2'].astype('float')
    rain2_df = rain2_df.resample('H').sum()

    tide_df = pd.DataFrame([tide_time, tide_data], dtype='float64').transpose()
    tide_df.columns = ['datetime', 'tide']
    tide_df['datetime'] = pd.to_datetime(tide_df['datetime'], infer_datetime_format=True)
    tide_df.set_index(pd.DatetimeIndex(tide_df['datetime']), inplace=True)
    tide_df['tide'] = tide_df['tide'].astype('float')

    df = pd.concat([rain1_df['rain1'], rain2_df['rain2'], tide_df['tide']], axis=1)
    df[['rain1', 'rain2']].fillna(0, inplace=True)
    df.reset_index(inplace=True)

    return df


def plot_ctl_results(env_df, ctl_dict, file_name, out_dir):
    # plot results
    fig, axs = plt.subplots(4, sharey='none', sharex='none', figsize=(6, 8))
    # first plot is tide and rainfall
    ax = axs[0]
    env_df["tide"].plot(ax=ax, color='c', legend=None)
    ax2 = ax.twinx()
    ax2.invert_yaxis()
    env_df["rain1"].plot.bar(ax=ax2, color="b", legend=None)
    ax2.set_xticks([])
    ax.set_xticks([], minor=True)
    ax.set_ylabel("Sea Level (ft.)")
    ax2.set_ylabel("Rainfall (in.)")
    ax.set_title('Inputs')
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, ("Sea Level", "Incremental Rainfall"), bbox_to_anchor=(0.85, -0.05), ncol=2,
              frameon=False)

    # plot depths
    axs[1].plot(ctl_dict["St1_depth"], label='St1')
    axs[1].plot(ctl_dict["St2_depth"], label='St2')
    axs[1].plot(ctl_dict["J1_depth"], label='J1', color='green')
    axs[1].plot(ctl_dict["St1_full"], linestyle=':', color='k', label='Storage max')
    axs[1].plot(ctl_dict["J1_full"], linestyle=':', color='grey', label='Pipe max')
    axs[1].set_ylim(0, 6)
    axs[1].set_title('Depths')
    axs[1].set_ylabel("ft")
    axs[1].legend(loc=4, bbox_to_anchor=(0.025, -.5, 1., .11), ncol=5,
                  borderaxespad=0.1, frameon=False, columnspacing=.75)

    # plot actions
    axs[2].plot(ctl_dict["R1_act"], label='R1')
    axs[2].plot(ctl_dict["R2_act"], ':', label='R2')
    axs[2].set_ylim(0, 1.05)
    axs[2].set_title('Policy')
    axs[2].set_ylabel("Valve Position")
    axs[2].legend()

    # plot flooding
    axs[3].plot(ctl_dict["St1_flooding"], label='St1')
    axs[3].plot(ctl_dict["St2_flooding"], label='St2')
    axs[3].plot(ctl_dict["J1_flooding"], label='J1')
    axs[3].set_title('Flooding')
    axs[3].set_ylabel("CFS")
    axs[3].set_xlabel("time step")
    axs[3].set_ylim(0)
    flood_str = "Total Vol. = " + str(round(ctl_dict["total_flood"][-1], 3)) + "MG"
    flood_max = max(max(ctl_dict["St1_flooding"]), max(ctl_dict["St2_flooding"]), max(ctl_dict["J1_flooding"])) * 0.9
    axs[3].text(0, flood_max, flood_str)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, file_name.split('.')[0] + ".png"), dpi=300)
    plt.close()


def read_saved_results(result_path, base=True):
    raw_df = pd.read_csv(result_path)

    if base:
        result_df = raw_df[['St1_depth', 'St2_depth', 'J1_depth', 'St1_flooding', 'St2_flooding',
                            'J1_flooding', 'total_flood', 'St1_full', 'St2_full', 'J1_full', 'R1_act', 'R2_act']]
    else:
        result_df = raw_df[['St1_depth', 'St2_depth', 'J1_depth', 'St1_flooding', 'St2_flooding',
                            'J1_flooding', 'total_flood', 'St_full', 'J1_full', 'R1_act', 'R2_act']]

    result_df['datetime'] = pd.date_range(start=raw_df["index"][0], periods=len(raw_df["St1_depth"]), freq='15T')
    result_df.set_index(pd.DatetimeIndex(result_df['datetime']), inplace=True, drop=True)

    data_df = raw_df[['index', 'rain1', 'rain2', 'tide']]
    data_df.set_index(pd.DatetimeIndex(data_df['index']), inplace=True, drop=True)
    data_df.dropna(subset=['tide'], inplace=True)
    data_df.fillna(0, inplace=True)
    # data_df.drop('index', axis=1, inplace=True)
    # data_df.reset_index(inplace=True)
    data_df = data_df.resample('15T').asfreq()  # 15min resample
    data_df.fillna({'rain1': 0, 'rain2': 0, 'tide': data_df['tide'].interpolate()}, inplace=True)

    df = pd.concat([result_df, data_df], axis=1)
    df.reset_index(inplace=True)
    return df  # result_df, data_df


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
