"""
Benjamin Bowes, 12-05-2020
Script to read saved results and plot for RL WQ paper
"""

import numpy as np
import pandas as pd
import datetime
import matplotlib
from matplotlib import lines as mpllines
import matplotlib.pyplot as plt
from swmm_hague_rbc.swmm_utils import read_saved_results

matplotlib.rcParams.update({'font.size': 8})

# directories
base_path = "C:/PycharmProjects/swmm_rl_hague/results_2010_2011/out_df_v18_notreat.csv"
# base_path = "C:/PycharmProjects/swmm_rl_hague/results_2010_2011/out_df_all_polluts_treated_082019_v17.csv"
# rbc_path = "C:/PycharmProjects/swmm_hague_rbc/cmac_results/082019_simple2_thrshld05.csv"
rbc_path = "C:/PycharmProjects/swmm_rl_hague/results_2010_2011/out_df_v18_sharior.csv"
# rbc_path = "C:/PycharmProjects/swmm_hague_rbc/cmac_results/082019_simple_retentionFixed.csv"
# TODO add RL details
obs_pond_path = "C:/Users/Ben Bowes/Documents/LongTerm_SWMM/elmwood_082019_gw.csv"
obs_rain_path = "C:/Users/Ben Bowes/Documents/LongTerm_SWMM/Clean_rain_data/rain_cleaned_combined.csv"
obs_tide_path = "C:/Users/Ben Bowes/Documents/LongTerm_SWMM/Raw_tide_data/tide_15min_2010_2019.csv"

# read data
# base_df = read_saved_results(base_path, base=True)
# rl_df = read_saved_results(rl_path, base=False)
base_df = pd.read_csv(base_path, index_col="Datetime", infer_datetime_format=True, parse_dates=True)
rbc_df = pd.read_csv(rbc_path, index_col="Datetime", infer_datetime_format=True, parse_dates=True)
obs_rain_df = pd.read_csv(obs_rain_path, index_col="datetime", infer_datetime_format=True, parse_dates=True)
obs_tide_df = pd.read_csv(obs_tide_path, index_col="datetime", infer_datetime_format=True, parse_dates=True)
obs_pond_df = pd.read_csv(obs_pond_path, index_col="Datetime", infer_datetime_format=True, parse_dates=True)

# slice observed data to match simulation results
start = datetime.datetime(2010, 1, 1, 0, 0, 0)  # change start time here
end = datetime.datetime(2011, 1, 1, 0, 0, 0)  # change end time here

base_df = base_df.loc[start: end]
obs_rain_df = obs_rain_df.loc[start: end]
obs_tide_df = obs_tide_df.loc[start: end]
env_df = pd.concat([obs_rain_df, obs_tide_df], axis=1)
env_df.reset_index(inplace=True)

# depth plots
fig, axs = plt.subplots(5, 1, sharex='none', figsize=(6.5, 8.5))

# first plot is tide and rainfall
ax = axs[0]
env_df["tide"].plot(ax=ax, color='c', legend=None)
ax2 = ax.twinx()
ax2.invert_yaxis()
env_df["combined"].plot.bar(ax=ax2, color="b", legend=None, width=8)
ax2.set_xticks([])
ax.set_xticks([], minor=True)
ax.set_ylabel("Sea Level (ft.)")
ax2.set_ylabel("Rainfall (in.)")
lines, labels = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines + lines2, ("Sea Level", "Incremental Rainfall"), bbox_to_anchor=(0.78, -0.03), ncol=2, frameon=False)

ax3 = axs[1]
ax3.plot(base_df["St1_depth"], label='Passive')
ax3.plot(rbc_df["St1_depth"], label='RBC')
ax3.plot(base_df["St1_full"], linestyle='--', color='k', label='max depth')
ax3.plot(obs_pond_df["Pond"], ':', label="Obs. Level")
ax3.set_ylabel("St1 Level (ft, NAVD88)")
ax3.set_ylim(-5)
ax3.set_xticklabels([])
ax3.legend(loc='lower center', bbox_to_anchor=(0.5, -0.25), frameon=False, ncol=4)

ax4 = axs[2]
ax4.plot(base_df["St2_depth"], label='Passive')
ax4.plot(rbc_df["St2_depth"], label='RBC')
ax4.plot(base_df["St2_full"], linestyle='--', color='k', label='max depth')
ax4.set_ylabel("St2 Level (ft, NAVD88)")
ax4.set_ylim(0)
ax4.set_xticklabels([])

ax5 = axs[3]
ax5.plot(base_df["St3_depth"], label='Passive')
ax5.plot(rbc_df["St3_depth"], label='RBC')
ax5.plot(base_df["St3_full"], linestyle='--', color='k', label='max depth')
ax5.set_ylabel("St3 Level (ft, NAVD88)")
ax5.set_ylim(-0.6)
ax5.set_xticklabels([])

ax6 = axs[4]
ax6.plot(rbc_df["R1_act"], ':', label='R1')
ax6.plot(rbc_df["R2_act"], '-', label='R2')
ax6.plot(rbc_df["R3_act"], '-.', label='R3')
ax6.set_ylim(0, 1.05)
ax6.set_ylabel("Valve Position")
ax6.legend()

fig.tight_layout()
fig.subplots_adjust(top=0.991, bottom=0.027, left=0.097, right=0.919, hspace=0.235, wspace=0.2)
fig.show()

# flooding plots
St1_fld_vol = pd.DataFrame([sum(base_df["St1_fld_vol"]), sum(rbc_df["St1_fld_vol"])], index=['Passive', 'RBC'], columns=['St1'])
St2_fld_vol = pd.DataFrame([sum(base_df["St2_fld_vol"]), sum(rbc_df["St2_fld_vol"])], index=['Passive', 'RBC'], columns=['St2'])
St3_fld_vol = pd.DataFrame([sum(base_df["St3_fld_vol"]), sum(rbc_df["St3_fld_vol"])], index=['Passive', 'RBC'], columns=['St3'])
fld_vol_df = pd.concat([St1_fld_vol, St2_fld_vol, St3_fld_vol], axis=1).transpose()

fig, axs = plt.subplots(4, 1, sharex='none', figsize=(6.5, 8))

ax = axs[0]
env_df["tide"].plot(ax=ax, color='c', legend=None)
ax2 = ax.twinx()
ax2.invert_yaxis()
env_df["combined"].plot.bar(ax=ax2, color="b", legend=None, width=8)
ax2.set_xticks([])
ax.set_xticks([], minor=True)
ax.set_ylabel("Sea Level (ft.)")
ax2.set_ylabel("Rainfall (in.)")
lines, labels = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines + lines2, ("Sea Level", "Incremental Rainfall"), bbox_to_anchor=(0.78, -0.03), ncol=2, frameon=False)

ax2 = axs[1]
ax2.plot(base_df[["St1_flooding", "St2_flooding", "St3_flooding"]])
ax2.set_ylabel('Passive Flood Rate (cfs)')

ax3 = axs[2]
ax3.plot(rbc_df[["St1_flooding", "St2_flooding", "St3_flooding"]])
ax3.set_ylabel('RBC Flood Rate (cfs)')

ax4 = axs[3]
fld_vol_df.plot.bar(ax=ax4, legend=False)
ax4.set_ylabel('Total Flood Volume (gal.)')
# lines, labels = ax3.get_legend_handles_labels()
# ax3.legend(lines, labels, bbox_to_anchor=(1, -0.1), ncol=2, frameon=False)

fig.tight_layout()
fig.show()

# pollutant plots
fig, axs = plt.subplots(4, 1, sharex='none', figsize=(6.5, 8.5))

ax = axs[0]
env_df["tide"].plot(ax=ax, color='c', legend=None)
ax2 = ax.twinx()
ax2.invert_yaxis()
env_df["combined"].plot.bar(ax=ax2, color="b", width=8, legend=None)
ax2.set_xticks([])
ax.set_xticks([], minor=True)
ax.set_ylabel("Sea Level (ft.)")
ax2.set_ylabel("Rainfall (in.)")
lines, labels = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines + lines2, ("Sea Level", "Incremental Rainfall"), bbox_to_anchor=(0.75, -0.05), ncol=2, frameon=False)

ax3 = axs[1]
ax3.plot(base_df["St1_TSS"], label='Untreated')
ax3.plot(rbc_df["St1_TSS"], label='Treated')
ax4 = ax3.twinx()
ax4.plot(base_df["R1_TSS"], ':', label='Untreated')
ax4.plot(rbc_df["R1_TSS"], ':', label='Treated')
ax3.set_ylabel("St1 TSS (mg/L)")
ax4.set_ylabel("Pollutant Outflow (lbs)")
# ax3.set_ylim(-5)
ax3.set_xticklabels([])
ax4.set_xticklabels([])

ax5 = axs[2]
ax5.plot(base_df["St2_TSS"], label='Untreated')
ax5.plot(rbc_df["St2_TSS"], label='Treated')
ax6 = ax5.twinx()
ax6.plot(base_df["R2_TSS"], ':', label='Untreated')
ax6.plot(rbc_df["R2_TSS"], ':', label='Treated')
ax5.set_ylabel("St2 TSS (mg/L)")
ax6.set_ylabel("Pollutant Outflow (lbs)")
ax5.set_xticklabels([])
ax6.set_xticklabels([])

ax7 = axs[3]
ax7.plot(base_df["St3_TSS"], label='Untreated Conc.')
ax7.plot(rbc_df["St3_TSS"], label='Treated Conc.')
ax8 = ax7.twinx()
ax8.plot(base_df["R3_TSS"], ':', label='Untreated Outflow')
ax8.plot(rbc_df["R3_TSS"], ':', label='Treated Outflow')
ax7.set_ylabel("St3 TSS (mg/L)")
ax8.set_ylabel("Pollutant Outflow (lbs)")
# ax7.set_xticklabels([])
# ax8.set_xticklabels([])
lines7, labels7 = ax7.get_legend_handles_labels()
lines8, labels8 = ax8.get_legend_handles_labels()
ax8.legend(lines7 + lines8, labels7 + labels8, bbox_to_anchor=(1, -0.1), ncol=4, frameon=False)

fig.tight_layout()
fig.subplots_adjust(top=0.991,
                    bottom=0.052,
                    left=0.112,
                    right=0.907,
                    hspace=0.235,
                    wspace=0.2)
fig.show()

# bar charts of pollutants
R1_TSS = pd.DataFrame([base_df["R1_TSS"][-1], rbc_df["R1_TSS"][-1]], index=['Passive', 'RBC'], columns=['R1'])
R2_TSS = pd.DataFrame([base_df["R2_TSS"][-1], rbc_df["R2_TSS"][-1]], index=['Passive', 'RBC'], columns=['R2'])
R3_TSS = pd.DataFrame([base_df["R3_TSS"][-1], rbc_df["R3_TSS"][-1]], index=['Passive', 'RBC'], columns=['R3'])
TSS_df = pd.concat([R1_TSS, R2_TSS, R3_TSS], axis=1).transpose()

R1_TP = pd.DataFrame([base_df["R1_TP"][-1], rbc_df["R1_TP"][-1]], index=['Passive', 'RBC'], columns=['R1'])
R2_TP = pd.DataFrame([base_df["R2_TP"][-1], rbc_df["R2_TP"][-1]], index=['Passive', 'RBC'], columns=['R2'])
R3_TP = pd.DataFrame([base_df["R3_TP"][-1], rbc_df["R3_TP"][-1]], index=['Passive', 'RBC'], columns=['R3'])
TP_df = pd.concat([R1_TP, R2_TP, R3_TP], axis=1).transpose()

R1_TN = pd.DataFrame([base_df["R1_TN"][-1], rbc_df["R1_TN"][-1]], index=['Passive', 'RBC'], columns=['R1'])
R2_TN = pd.DataFrame([base_df["R2_TN"][-1], rbc_df["R2_TN"][-1]], index=['Passive', 'RBC'], columns=['R2'])
R3_TN = pd.DataFrame([base_df["R3_TN"][-1], rbc_df["R3_TN"][-1]], index=['Passive', 'RBC'], columns=['R3'])
TN_df = pd.concat([R1_TN, R2_TN, R3_TN], axis=1).transpose()

fig, axs = plt.subplots(3, 1, sharex='all', figsize=(3, 5))

ax1 = axs[0]
TSS_df.plot.bar(ax=ax1)
ax1.set_ylabel('TSS (lbs)')

ax2 = axs[1]
TP_df.plot.bar(ax=ax2, legend=False)
ax2.set_ylabel('TP (lbs)')

ax3 = axs[2]
TN_df.plot.bar(ax=ax3, legend=False)
ax3.set_ylabel('TN (lbs)')
# lines, labels = ax3.get_legend_handles_labels()
# ax3.legend(lines, labels, bbox_to_anchor=(1, -0.1), ncol=2, frameon=False)

fig.tight_layout()
# fig.show()
fig.savefig("C:/PycharmProjects/swmm_rl_hague/model_comparison_plots/polluts_082019.png", dpi=300)

# # plot from paper 2
# fig, axs = plt.subplots(5, 1, sharey=False, sharex=False, figsize=(6.5, 6))
#
# ax1 = axs[0]
# ax1.plot(base_df['St1_depth']/3.281, color='k')
# ax1.plot(base_df['St1_full']/3.281, color='#636363', linestyle='--')
# ax1.plot(rl_df['St1_depth']/3.281, color='#bdbdbd', linestyle=':', lw=2.25)
# ax1.set_ylim(ymin=0., ymax=2.44)
# ax1.set_yticks([0, 0.5, 1.0, 1.5, 2])
# start, end = ax1.get_xlim()
# x_ticks = np.arange(0, end, 288)
# ax1.set_xticks(x_ticks)
# # ax1.set_xticklabels(base_df['datetime'][x_ticks].dt.strftime('%Y-%m-%d'), rotation='vertical')
# ax12 = ax1.twinx()
# x12_ticks_2 = np.arange(0, len(base_df['rain1']), 1)
# ax12.bar(x12_ticks_2, base_df['rain1']*25.4, color='k', width=8, label='Rainfall')
# ax12.set_xticklabels([])
# ax12.set_ylim(ymin=0.0, ymax=160)
# ax12.set_yticks([0, 20, 40, 60])
# plt.gca().invert_yaxis()
# ax1.set_ylabel("St1 Depth (m)")
# ax12.set_ylabel("Rain1 (mm)")
# # ax1.text(245, 1.65, 'GW7')
# # lines, labels = ax1.get_legend_handles_labels()
# lines, labels = ax12.get_legend_handles_labels()
# # legend = ax1.legend(lines+lines2, labels+labels2, bbox_to_anchor=(2.07, 0.1), ncol=3)
#
# ax1.tick_params(
#     axis='x',          # changes apply to the x-axis
#     which='minor',      # both major and minor ticks are affected
#     bottom=False,      # ticks along the bottom edge are off
#     top=False,         # ticks along the top edge are off
#     labelbottom=False)  # labels along the bottom edge are off
#
# ax2 = axs[1]
# base_df["R1_act"].plot(ax=ax2, color='k', legend=None)
# rl_df["R1_act"].plot(ax=ax2, color='#bdbdbd', linestyle=':', legend=None, lw=2)
# ax2.set_ylim(0., 1.05)
# ax2.set_yticks([0, 0.5, 1.])
# ax2.set_yticklabels([0, 50, 100])
# ax2.set_ylabel("R1 (% open)")
# ax2.set_xticks(x_ticks)
# ax2.tick_params(
#     axis='x',          # changes apply to the x-axis
#     which='both',      # both major and minor ticks are affected
#     bottom=True,      # ticks along the bottom edge are off
#     top=False,         # ticks along the top edge are off
#     labelbottom=False)  # labels along the bottom edge are off
#
# ax3 = axs[2]
# ax3.plot(base_df['St2_depth']/3.281, color='k')
# ax3.plot(base_df['St2_full']/3.281, color='#636363', linestyle='--')
# ax3.plot(rl_df['St2_depth']/3.281, color='#bdbdbd', linestyle=':', lw=2.25)
# ax3.set_xticks(x_ticks)
# ax3.set_ylim(ymin=0., ymax=2.44)
# ax3.set_yticks([0, 0.5, 1.0, 1.5, 2])
# # ax3.set_xticklabels(base_df['datetime'][x_ticks].dt.strftime('%Y-%m-%d'), rotation='vertical')
# ax32 = ax3.twinx()
# x32_ticks_2 = np.arange(0, len(base_df['rain2']), 1)
# ax32.bar(x32_ticks_2, base_df['rain2']*25.4, color='k', width=8)
# ax32.set_xticklabels([])
# ax32.set_ylim(ymin=0.0, ymax=160)
# ax32.set_yticks([0, 20, 40, 60])
# plt.gca().invert_yaxis()
# ax3.set_ylabel("St2 Depth (m)")
# ax32.set_ylabel("Rain2 (mm)")
# ax3.tick_params(
#     axis='x',          # changes apply to the x-axis
#     which='minor',      # both major and minor ticks are affected
#     bottom=False,      # ticks along the bottom edge are off
#     top=False,         # ticks along the top edge are off
#     labelbottom=False)  # labels along the bottom edge are off
#
# ax4 = axs[3]
# base_df["R2_act"].plot(ax=ax4, color='k', legend=None)
# rl_df["R2_act"].plot(ax=ax4, color='#bdbdbd', linestyle=':', legend=None, lw=2)
# ax4.set_ylim(0., 1.05)
# ax4.set_yticks([0, 0.5, 1.])
# ax4.set_yticklabels([0, 50, 100])
# ax4.set_ylabel("R2 (% open)")
# ax4.set_xticks(x_ticks)
# ax4.tick_params(
#     axis='x',          # changes apply to the x-axis
#     which='major',      # both major and minor ticks are affected
#     bottom=True,      # ticks along the bottom edge are off
#     top=False,         # ticks along the top edge are off
#     labelbottom=False)  # labels along the bottom edge are off
#
# ax5 = axs[4]
# ax5.plot(base_df['J1_depth']/3.281, 'k', label='RBC')
# ax5.plot(base_df['J1_full']/3.281, color='#636363', linestyle='--', label='Node max')
# ax5.plot(rl_df['J1_depth']/3.281, color='#bdbdbd', linestyle=':', label='RL', lw=2)
# ax5.set_ylim(ymin=0., ymax=1.22)
# ax5.set_yticks([0, 0.5, 1])
# ax5.set_xticks(x_ticks)
# ax5.set_xticklabels(base_df['datetime'][x_ticks].dt.strftime('%d'), rotation=0)
# ax52 = ax5.twinx()
# x52_ticks_2 = np.arange(0, len(base_df['tide']), 1)
# ax52.plot(base_df['tide']/3.281, 'k-.', label='Tide', lw=1.25)
# # ax52.set_xticklabels([])
# ax52.set_ylim(ymin=-2.44, ymax=1.22)
# ax52.set_yticks([-0.5, 0, 0.5, 1.0])
# ax5.set_ylabel("J1 Depth (m)")
# ax52.set_ylabel("Tide (m)")
# ax52.tick_params(
#     axis='x',          # changes apply to the x-axis
#     which='both',      # both major and minor ticks are affected
#     bottom=False,      # ticks along the bottom edge are off
#     top=False,         # ticks along the top edge are off
#     labelbottom=False)  # labels along the bottom edge are off
# ax5.tick_params(
#     axis='x',          # changes apply to the x-axis
#     which='major',      # both major and minor ticks are affected
#     bottom=True,      # ticks along the bottom edge are off
#     top=False,         # ticks along the top edge are off
#     labelbottom=True)  # labels along the bottom edge are off
#
# vertical_line = mpllines.Line2D([], [], color='k', marker='|', linestyle='None',
#                                 markersize=10, markeredgewidth=1.5, label='Rainfall')
# lines = [vertical_line]
# lines2, labels2 = ax5.get_legend_handles_labels()
# lines3, labels3 = ax52.get_legend_handles_labels()
# legend = ax5.legend(lines+lines2+lines3, labels+labels2+labels3, loc='lower center',
#                     bbox_to_anchor=(0.5, -0.5), frameon=True, ncol=5)
#
# plt.tight_layout()
# plt.subplots_adjust(hspace=.1)
# # plt.show()
# plt.savefig("C:/PycharmProjects/LongTerm_SWMM/observed_data/obs_test_data/comparison_plots/paper2_plots_si_R1/rl_vs_rbc_072010.png", dpi=300)
