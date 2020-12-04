"""
Benjamin Bowes, 12-04-2020

This script contains functions to implement rule-based control (RBC) modeled after
OptiRTC's Continuous Monitoring and Control (CMAC) based on data from a forecast file
and state information from a SWMM simulation.
"""

import numpy as np
import math


def find_nonzero_runs(a):
    # Create an array that is 1 where a is nonzero, and pad each end with an extra 0.
    isnonzero = np.concatenate(([0], (np.asarray(a) != 0).view(np.int8), [0]))
    absdiff = np.abs(np.diff(isnonzero))
    # Runs start and end where absdiff is 1.
    ranges = np.where(absdiff == 1)[0].reshape(-1, 2)
    return ranges


def read_fcst(rain_gages, fcst_data, timestep, fcst_len=97):
    # 1 determine number of events
    event_dict = {}  # dict to store number of events and volumes

    # get current forecast based on simulation timestep
    fcst_rain1 = fcst_data.iloc[timestep]
    # fcst_rain2 = fcst_data.iloc[timestep, fcst_len:-fcst_len]

    event_dict["total"] = fcst_rain1.sum()  # add 24hr totals to dict

    # find number of events for each rain gage
    for gage_idx, gage_name in enumerate(rain_gages):
        if gage_idx == 0:
            m = np.asarray(fcst_rain1 != 0)
        # elif gage_idx == 1:
        #     m = np.asarray(fcst_rain2 != 0)

        # combine events if within 6hr of each other
        for i in range(len(m)):
            if i < fcst_len - 1:
                if m[i]:  # if current value is true
                    if not m[i + 1]:  # if next value is false
                        if i + 25 > len(m):
                            if np.any(m[i + 1:]):
                                m[i + 1] = True
                        else:
                            if np.any(m[i+1:i+25]):
                                m[i+1] = True

        # 2 calculate per event precip. totals
        events = find_nonzero_runs(m)
        event_sums = []
        for event in events:
            if gage_idx == 0:
                event_sums.append(fcst_rain1[event[0]:event[1]].sum())
            # if gage_idx == 1:
            #     event_sums.append(fcst_rain2[event[0]:event[1]].sum())

        event_dict[gage_name] = event_sums

    # 3 track event start and end?

    return event_dict


def drain_time(flood_vol, stage_storage, storage_head, current_depth, diam=2., coeff=0.65, ctl_step=900):
    """
    takes in current valve positions and expected flood volume for pond
    calculates time to drain expected flood volume from pond, assuming valve fully open

    https://www.lmnoeng.com/Tank/TankTime.php
    https://www.mathopenref.com/segmentareaht.html
    """

    valve_area = round(math.pi * (diam/2)**2, 4)  # full area of circle
    if current_depth < 2.:  # partial area of circle
        r = (diam/2)**2
        h = current_depth
        # theta = 2 * math.acos(((diam / 2) - h) / (diam / 2))
        valve_area = round(r**2 * math.acos((r-h)/r) - (r-h) * math.sqrt(2*r*h-h**2), 4)

    req_vol = flood_vol * 1.2  # drain flood volume + 20% safety factor

    # find depth occupied by flood volume
    flood_depth = req_vol / stage_storage

    # find change in head required
    req_head = abs(storage_head - flood_depth)

    # find time required to drain to required head, valve fully open, assuming discharge to atm
    time = (stage_storage / (valve_area * coeff)) * (math.sqrt(storage_head) - math.sqrt(req_head)) * math.sqrt(2 / 32.2)  # assuming english units, time in seconds

    drain_steps = math.ceil(time / ctl_step) + 1  # number of time steps required to drain (rounded up)

    return drain_steps


def valve_position(current_depth, stage_storage, drawdown_time=86400, target_depth=2., diam=2., coeff=0.65):
    """
    Returns minimum valve opening to drain stormwater back to target depth over specified drawdown period
    Drawdown time is in seconds, defaults to 86400 (24hr)
    """
    if current_depth <= target_depth:  # close valve if current depth less than target depth
        return 0
    else:
        full_valve_area = round(math.pi * (diam / 2) ** 2, 4)  # full area of circle

        new_valve_area = stage_storage / ((drawdown_time * coeff) / (math.sqrt(current_depth) - math.sqrt(target_depth))
                                          * math.sqrt(2 / 32.2))  # assuming english units, time in seconds

        valve_opening = 1 - round(new_valve_area / full_valve_area, 4)

        return valve_opening
