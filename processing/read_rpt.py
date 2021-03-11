"""
Benjamin Bowes, Jan 25, 2021

Modified from https://github.com/UVAdMIST/swmm_mpc/blob/master/swmm_mpc/rpt_ele.py by jsadler2
"""

import pandas as pd


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
