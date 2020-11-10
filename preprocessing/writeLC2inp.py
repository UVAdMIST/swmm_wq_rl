"""
Benjamin Bowes, 28 Aug. 2020
This script reads a csv of subcatchment land cover data (generated in ArcGIS) and
writes it to a template SWMM inp file
"""

import pandas as pd

inp_file = "swmm_rl_hague/hague_inp_files/hague_v11_template_WQ.inp"
subcatch_raw = pd.read_csv("swmm_rl_hague/hague2arc_swmmr/hague_subs_withLC.csv",
                           usecols=["subcatchment", "val11_pct", "val21_pct", "val22_pct", "val42_pct", "val71_pc_1"])

# round all LC percentages to int and check they equal 100%
subcatch_df = subcatch_raw[["val11_pct", "val21_pct", "val22_pct", "val42_pct", "val71_pc_1"]] * 100
subcatch_df = subcatch_df.round(0)
subcatch_df["total"] = subcatch_df.sum(axis=1)
subcatch_df['Max'] = subcatch_df[["val11_pct", "val21_pct", "val22_pct", "val42_pct", "val71_pc_1"]].idxmax(axis=1)
subcatch_df['Min'] = subcatch_df[["val21_pct", "val22_pct", "val42_pct", "val71_pc_1"]].idxmin(axis=1)

for indx, row in subcatch_df.iterrows():
    if row.total > 100.:  # if row = 101%
        max_col = subcatch_df.iloc[indx]["Max"]
        new_value = subcatch_df.iloc[indx][max_col] - 1
        subcatch_df.at[indx, max_col] = new_value

    if row.total < 100.:  # if row = 99%
        min_col = subcatch_df.iloc[indx]["Min"]
        new_value = subcatch_df.iloc[indx][min_col] + 1
        subcatch_df.at[indx, min_col] = new_value

subcatch_df = pd.concat([subcatch_df, subcatch_raw["subcatchment"]], axis=1)

# find time series section
with open(inp_file, 'r') as tmp_file:
    lines = tmp_file.readlines()
    for i, l in enumerate(lines):
        if l.startswith("[COVERAGES]"):
            print(i, l)
            start = i + 3
    tmp_file.close()

# write data to inp file
with open(inp_file, 'w') as inpfile:
    new_lines = lines.copy()  # copy inp template
    for indx, row in subcatch_df.iterrows():  # write LC data into COVERAGES section
        new_line = str(row.subcatchment) + " Water " + str(row.val11_pct) + " Imperv " + str(row.val21_pct + row.val22_pct) + " Trees " + str(row.val42_pct) + " TurfGrass " + str(row.val71_pc_1)
        print(new_line)
        new_lines.insert(start + indx, new_line + '\n')
    inpfile.writelines(new_lines)
    inpfile.close()
