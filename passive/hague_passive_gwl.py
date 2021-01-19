"""
Created by Benjamin Bowes, 4-19-19
This script records depth and flood values at each swmm model time step and plots them.
"""

import matplotlib.pyplot as plt
import datetime
import math
import pandas as pd
from pyswmm import Simulation, Nodes, Links, Subcatchments
from swmm_rl_hague.gwl_utils import calc_gwl
# from smart_stormwater_rl.pyswmm_utils import save_out

start_time = datetime.datetime.now()
control_time_step = 900  # control time step in seconds
swmm_inp = "C:/PycharmProjects/swmm_rl_hague/hague_inp_files_pyswmm/hague_v19_notreat.inp"
# swmm_inp = "C:/PycharmProjects/swmm_rl_hague/hague_inp_files_pyswmm/hague_v11_template_WQ_AllEnvData_withLC2_notreat.inp"
gwl_df = pd.read_csv("C:/Users/Ben Bowes/Documents/HRSD GIS/Site Data/Data_2010_2019/IDW_2010_2019.csv",
                     usecols=["Datetime", "IDW"], index_col="Datetime", infer_datetime_format=True, parse_dates=True)

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
R1_TSS_inc, R2_TSS_inc, R3_TSS_inc = [], [], []
R1_TP, R2_TP, R3_TP = [], [], []
R1_TN, R2_TN, R3_TN = [], [], []
R1_flow, R2_flow, R3_flow = [], [], []
R1_act, R2_act, R3_act = [], [], []

with Simulation(swmm_inp) as sim:  # loop through all steps in the simulation
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
    #
    # subcatchment_object = Subcatchments(sim)
    # S1 = subcatchment_object["S1"]
    # S2 = subcatchment_object["S2"]

    # calculate radius (ft) of storage units for GWL calculations
    St1_rad = math.sqrt(100000/math.pi)
    St2_rad = math.sqrt(20000/math.pi)
    St3_rad = math.sqrt(50000/math.pi)

    step_count = 1
    for step in sim:
        # print(step_count)
        # if int(sim.percent_complete * 100) % 5 == 0:
        #     print(sim.percent_complete)

        if sim.current_time == sim.start_time:
            R1.target_setting = 1
            R2.target_setting = 1
            R3.target_setting = 1

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

        # record variables on the 15min control time step
        # time_delta = sim.current_time - previous_step
        # if int(time_delta.total_seconds()) == control_time_step:
        if step_count % control_time_step == 0:
            print("control step:", sim.current_time)
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
            R1_TSS.append(R1.total_loading['TSS'])  # cumulative pounds of TSS passing through orifice
            R2_TSS.append(R2.total_loading['TSS'])
            R3_TSS.append(R3.total_loading['TSS'])
            R1_TSS_inc.append(R1.total_loading['TSS'] - R1_TSS[-1])  # cumulative pounds of TSS passing through orifice
            R2_TSS_inc.append(R2.total_loading['TSS'] - R2_TSS[-1])
            R3_TSS_inc.append(R3.total_loading['TSS'] - R3_TSS[-1])
            R1_TP.append(R1.total_loading['TP'])  # pounds of TP passing through orifice
            R2_TP.append(R2.total_loading['TP'])
            R3_TP.append(R3.total_loading['TP'])
            R1_TN.append(R1.total_loading['TN'])  # pounds of TN passing through orifice
            R2_TN.append(R2.total_loading['TN'])
            R3_TN.append(R3.total_loading['TN'])
            # print("current time: ", sim.current_time, "flooding: ", J3.flooding, "flood_vol: ",
            #       J3.statistics['flooding_volume'], "flood_rate: ", J3.statistics['peak_flooding_rate'])
            # previous_step = sim.current_time
        step_count += 1

out_lists = [time, St1_depth, St2_depth, St3_depth, St1_flooding, St2_flooding, St3_flooding,
             St1_fld_vol, St2_fld_vol, St3_fld_vol, St1_TSS, St2_TSS, St3_TSS,
             St1_TSS_mass, St2_TSS_mass, St3_TSS_mass, R1_TSS, R2_TSS, R3_TSS,
             R1_TSS_inc, R2_TSS_inc, R3_TSS_inc,
             St1_TP, St2_TP, St3_TP, R1_TP, R2_TP, R3_TP, St1_TP_mass, St2_TP_mass, St3_TP_mass,
             St1_TN, St2_TN, St3_TN, R1_TN, R2_TN, R3_TN, St1_TN_mass, St2_TN_mass, St3_TN_mass,
             St1_gwflow, St2_gwflow, St3_gwflow, St1_flow, St2_flow, St3_flow,
             St1_full, St2_full, St3_full, R1_act, R2_act, R3_act]

out_df = pd.DataFrame(out_lists).transpose()
out_df.columns = ["Datetime", "St1_depth", "St2_depth", "St3_depth", "St1_flooding", "St2_flooding", "St3_flooding",
                  "St1_fld_vol", "St2_fld_vol", "St3_fld_vol", "St1_TSS", "St2_TSS", "St3_TSS",
                  "St1_TSS_mass", "St2_TSS_mass", "St3_TSS_mass", "R1_TSS", "R2_TSS", "R3_TSS",
                  "R1_TSS_inc", "R2_TSS_inc", "R3_TSS_inc",
                  "St1_TP", "St2_TP", "St3_TP", "R1_TP", "R2_TP", "R3_TP", "St1_TP_mass", "St2_TP_mass", "St3_TP_mass",
                  "St1_TN", "St2_TN", "St3_TN", "R1_TN", "R2_TN", "R3_TN", "St1_TN_mass", "St2_TN_mass", "St3_TN_mass",
                  "St1_gwflow", "St2_gwflow", "St3_gwflow", "St1_flow", "St2_flow", "St3_flow",
                  "St1_full", "St2_full", "St3_full", "R1_act", "R2_act", "R3_act"]
out_df.to_csv("C:/PycharmProjects/swmm_rl_hague/results_2010_2011/out_df_v20_untreated.csv", index=False)

end_time = datetime.datetime.now()
print("start: ", start_time, "\n end: ", end_time)

# save_out(out_lists, "Uncontrolled")

# # plot results with GWL
# gwl_slice = gwl_df.loc['2019-08-01 00:00:00':'2019-09-01 00:00:00']
# gwl_slice = gwl_slice.resample('15T').ffill()
# # gwl_slice.reset_index(inplace=True)
# gwl_slice = gwl_slice[:-1]
#
# # read observed pond data
# obs_pond_df = pd.read_csv("C:/Users/Ben Bowes/Documents/LongTerm_SWMM/elmwood_082019_gw.csv", index_col="Datetime",
#                           infer_datetime_format=True, parse_dates=True)
# out_df_gw = pd.read_csv("C:/PycharmProjects/swmm_rl_hague/results_082019/out_df_gw.csv", index_col="Datetime",
#                         infer_datetime_format=True, parse_dates=True)
# out_df_no_gw = pd.read_csv("C:/PycharmProjects/swmm_rl_hague/results_082019/out_df_no_gw.csv", index_col="Datetime",
#                            infer_datetime_format=True, parse_dates=True)
#
# # compare obs and sim pond depth with and without gw
# plt.plot(out_df_no_gw["St1_depth"], label='St1 (w/o GW)')
# plt.plot(out_df_gw["St1_depth"], label='St1 (w/ GW)')
# plt.plot(out_df_gw["St1_full"], linestyle='--', color='k', label='max depth')
# plt.plot(obs_pond_df["Pond"], ':', label="Obs. Pond Level")
# plt.plot(gwl_slice["IDW"], label='GWL')
# plt.ylabel("St1/GW Level (ft)")
# plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.1), frameon=True, ncol=5)
# plt.tight_layout()
# plt.show()
#
# # plot all pond levels
# fig, axs = plt.subplots(3, 1, sharex='all', figsize=(6.5, 8.5))
#
# ax1 = axs[0]
# ax1.plot(out_df_no_gw["St1_depth"], label='St1 (w/o GW)')
# ax1.plot(out_df_gw["St1_depth"], label='St1 (w/ GW)')
# ax1.plot(out_df_gw["St1_full"], linestyle='--', color='k', label='max depth')
# ax1.plot(obs_pond_df["Pond"], ':', label="Obs. Pond Level")
# ax1.plot(gwl_slice["IDW"], label='GWL')
# ax1.set_ylabel("St1/GW Level (ft)")
# ax1.legend(loc='lower center', bbox_to_anchor=(0.5, -0.2), frameon=True, ncol=5)
#
# ax2 = axs[1]
# ax2.plot(out_df_no_gw["St2_depth"], label='St2 (w/o GW)')
# ax2.plot(out_df_gw["St2_depth"], label='St2 (w/ GW)')
# ax2.plot(out_df_gw["St2_full"], linestyle='--', color='k', label='max depth')
# ax2.plot(gwl_slice["IDW"], label='GWL')
# ax2.set_ylabel("St2/GW Level (ft)")
#
# ax3 = axs[2]
# ax3.plot(out_df_no_gw["St3_depth"], label='St3 (w/o GW)')
# ax3.plot(out_df_gw["St3_depth"], label='St3 (w/ GW)')
# ax3.plot(out_df_gw["St3_full"], linestyle='--', color='k', label='max depth')
# ax3.plot(gwl_slice["IDW"], label='GWL')
# ax3.set_ylabel("St3/GW Level (ft)")
#
# fig.tight_layout()
# fig.show()
#
# # depth-gw plot
# out_df_gw.reset_index(inplace=True)
# out_df_no_gw.reset_index(inplace=True)
# gwl_slice.reset_index(inplace=True)
# obs_pond_df.reset_index(inplace=True)
#
# fig, axs = plt.subplots(6, 1, sharex='all', figsize=(6.5, 8.5))
# width = 0.35  # the width of the bars
#
# ax1 = axs[0]
# ax1.plot(out_df_no_gw["St1_depth"], label='St1 (w/o GW)')
# ax1.plot(out_df_gw["St1_depth"], label='St1 (w/ GW)')
# ax1.plot(out_df_gw["St1_full"], linestyle='--', color='k', label='max depth')
# ax1.plot(obs_pond_df["Pond"], ':', label="Obs. Pond Level")
# ax1.plot(gwl_slice["IDW"], label='GWL')
# ax1.set_ylabel("St1/GW Level (ft)")
# ax1.legend(loc='lower center', bbox_to_anchor=(0.5, -0.5), frameon=True, ncol=5)
#
# ax2 = axs[1]
# ax2.bar(out_df_gw.index, out_df_gw["St1_gwflow"], color='g', label='GW Flow')
# ax22 = plt.twinx(ax2)
# ax22.bar(out_df_no_gw.index + width/2, out_df_no_gw["St1_flow"], width, label='St1 Flow (w/o GW)')
# ax22.bar(out_df_gw.index - width/2, out_df_gw["St1_flow"], width, label='St1 Flow (w/ GW)')
# ax2.set_ylabel("GW Flow (cfs)")
# ax22.set_ylabel("St1 Flow (cfs)")
# lines, labels = ax2.get_legend_handles_labels()  # depth legend
# lines2, labels2 = ax22.get_legend_handles_labels()  # action legend
# ax2.legend(lines+lines2, labels+labels2, loc='lower center', bbox_to_anchor=(0.5, -0.5), frameon=True, ncol=3)
#
# ax3 = axs[2]
# ax3.plot(out_df_no_gw["St2_depth"], label='St2 (w/o GW)')
# ax3.plot(out_df_gw["St2_depth"], label='St2 (w/ GW)')
# ax3.plot(out_df_gw["St2_full"], linestyle='--', color='k', label='max depth')
# ax3.plot(gwl_slice["IDW"], label='GWL')
# ax3.set_ylabel("St2/GW Level (ft)")
# # ax2.legend()
#
# ax4 = axs[3]
# ax4.bar(out_df_gw.index, out_df_gw["St2_gwflow"], color='g', label='GW Flow')
# ax42 = plt.twinx(ax4)
# ax42.bar(out_df_no_gw.index + width/2, out_df_no_gw["St2_flow"], width, label='St2 Flow (w/o GW)')
# ax42.bar(out_df_gw.index - width/2, out_df_gw["St2_flow"], width, label='St2 Flow (w/ GW)')
# ax4.set_ylabel("GW Flow (cfs)")
# ax42.set_ylabel("St2 Flow (cfs)")
# # lines, labels = ax2.get_legend_handles_labels()  # depth legend
# # lines4, labels4 = ax22.get_legend_handles_labels()  # action legend
# # ax4.legend(lines+lines2, labels+labels2)
#
# ax4 = axs[4]
# ax4.plot(out_df_no_gw["St3_depth"], label='St3 (w/o GW)')
# ax4.plot(out_df_gw["St3_depth"], label='St3 (w/ GW)')
# ax4.plot(out_df_gw["St3_full"], linestyle='--', color='k', label='Max Depth')
# ax4.plot(gwl_slice["IDW"], label='GWL')
# ax4.set_ylabel("St3/GW Level (ft)")
# # ax4.legend()
#
# ax5 = axs[5]
# ax5.bar(out_df_gw.index, out_df_gw["St3_gwflow"], color='g', label='GW Flow')
# ax52 = plt.twinx(ax5)
# ax52.bar(out_df_no_gw.index + width/2, out_df_no_gw["St3_flow"], width, label='St3 Flow (w/o GW)')
# ax52.bar(out_df_gw.index - width/2, out_df_gw["St3_flow"], width, label='St3 Flow (w/ GW)')
# ax5.set_ylabel("GW Flow (cfs)")
# ax52.set_ylabel("St3 Flow (cfs)")
# # lines, labels = ax2.get_legend_handles_labels()  # depth legend
# # lines4, labels4 = ax22.get_legend_handles_labels()  # action legend
# # ax4.legend(lines+lines2, labels+labels2)
#
# fig.tight_layout()
# fig.show()

# plot results with TSS

# plt.savefig("C:/Users/Ben Bowes/PycharmProjects/swmm_keras_rl/baseline_model_case3.png", dpi=300)
# plt.close()
