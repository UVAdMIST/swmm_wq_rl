"""
Benjamin Bowes, 12-03-2020

This script runs SWMM for a forecast period using initial conditions from a hot start file.
The total volume of flooding at each storage node is returned.
"""

import os
import json
import numpy as np
# import pyswmm
# pyswmm.lib.use("swmm5_hs_520dev6")
from pyswmm import Simulation, Nodes


def main():  # use inp with hsf file
    with Simulation(os.path.join(project_dir, "cmac_temp/temp_inp.inp")) as sim:
        sim.step_advance(control_time_step)
        node_object = Nodes(sim)  # init node object
        St1 = node_object["st1"]
        St2 = node_object["st2"]
        St3 = node_object["F134101"]

        # # set initial conditions
        # def init_conditions():
        #     St1.initial_depth = init_df[0]
        #     St2.initial_depth = init_df[1]
        #     J1.initial_depth = init_df[2]
        #     J2.initial_depth = init_df[3]
        #     Out1.initial_depth = init_df[4]
        #     C2.initial_flow = init_df[5]
        #     C3.initial_flow = init_df[6]
        #     R1.target_setting = init_df[7]
        #     R2.target_setting = init_df[8]
        #
        # sim.initial_conditions(init_conditions)

        for step in sim:
            pass
        flood_dict = {"St1": (St1.statistics['flooding_volume']),  # volume in ft^3, multiply by 7.481 for gallons
                      "St2": (St2.statistics['flooding_volume']),
                      "St3": (St3.statistics['flooding_volume']),
                      "current_dt": str(sim.current_time)}
        flood_json = json.dumps(flood_dict)

        sim.close()

    # save results data
    # result_list = [St1_flooding, St2_flooding, current_dt]
    # results_df = pd.DataFrame(result_list).transpose()
    # results_df.to_csv("C:/swmm_opti_cmac/cmac_temp/temp_flood.csv", index=False)

    return flood_json  # total flood volume during forecast period


project_dir = "C:/PycharmProjects/swmm_hague_rbc"
control_time_step = 900  # control time step in seconds

# read initial conditions from saved file
# init_df = np.genfromtxt(os.path.join(project_dir, "cmac_temp/temp_inp.csv"), delimiter=',', skip_header=True)

if __name__ == "__main__":
    print(main())
