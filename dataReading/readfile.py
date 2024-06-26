# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 09:45:30 2024

@author: matth
"""

import numpy as np
import time
import ast
import pandas as pd
import math
import random
from gurobipy import *

with open(r"medium05.dat", 'r') as file:
    lines = [line.strip().split() for line in file]
    n_days = 28
    shift_types = []
    for shift in lines[4]:
        if shift != " ":
            shift_types.append(shift)

    n_shift_types = len(shift_types)
    n_contracts = int(lines[n_shift_types + 13][-1][-1]) + 1
    nurse_line = n_shift_types + n_contracts + 21
    line = lines[nurse_line + 1]
    while line[0][0] == 'N':
        nurse_line += 1
        line = lines[nurse_line + 1]
    n_nurses = int(lines[nurse_line][-1][1:]) + 1

    comp_shifts = np.zeros((n_nurses, n_shift_types))
    for k in range(n_nurses):
        idx = nurse_line + 4*(k+1)
        ns = int(lines[idx][1][4:-1])
        cs = lines[idx + 1]
        for i, shift_type in enumerate(shift_types):
            if shift_type in cs:
                comp_shifts[ns][i] = 1
    #comp_shifts = pd.DataFrame(comp_shifts)

    #comp_shifts.columns = shift_types
    curr_index = nurse_line + 4*n_nurses + 6
    shift_off_reqs = np.zeros((n_nurses, n_days, n_shift_types)) #n_days hard-coded here.
    while lines[curr_index][0][0] == '[':
        req = lines[curr_index]
        req_rel = ast.literal_eval(req[0].replace('[', "['").replace(',', "','").replace(']', "']"))
        shift_off_reqs
        for i, shift_type in enumerate(shift_types):
            if shift_type in req_rel[1]: #
                shift_off_reqs[int(req_rel[0][1:])][int(req_rel[2])-1][i] = int(req[1])
        curr_index += 1

    curr_index += 9
    wknds_by_c = np.zeros((n_contracts, n_days))
    while lines[curr_index]:
        for i in lines[curr_index + 1]:
            wknds_by_c[int(lines[curr_index][1][4:-1])][int(i)-1] = 1
        curr_index += 3

    while 1:
        if lines[curr_index]:
            if lines[curr_index][0] == 'param':
                if lines[curr_index][1] == 'r':
                    break
        curr_index += 1

    demand = np.zeros((n_days, n_shift_types))
    st2 = lines[curr_index+1][:n_shift_types]
    for day in range(28):
        dem = lines[curr_index+2+day][1:n_shift_types+1]
        value_dict = dict(zip(st2, dem))
        reorval = [value_dict[shift_type] for shift_type in shift_types]
        for st in range(n_shift_types):
            demand[day][st] = int(reorval[st])
    demand = pd.DataFrame(demand)
    demand.columns = shift_types

    curr_index += 34
    nurse_contracts = np.zeros((n_nurses))
    for nrs in range(n_nurses):
        nurse_contracts[int(lines[curr_index+nrs][0][1:])] = int(lines[curr_index+nrs][1][-1])

    curr_index += n_nurses + 2
    param_names = []
    contr_param = np.empty((14, n_contracts),dtype=object)
    contr_param.fill([0,0])
    for p in range(13):
        param_names.append(str(lines[curr_index + 2*p*(n_contracts + 4)]).replace(" ", "").replace(',','')[4:-4])
        for c in range(n_contracts):
            contr_param[p][c] = [int(lines[curr_index + 2*p*(n_contracts + 4) + 2 + c][1]), int(lines[curr_index + (2*p+1)*(n_contracts + 4) + 2 + c][1])]
    param_names.append('NoFridayOffIfSatSun')

    n_unw_pat = np.zeros((n_contracts))
    len_unw_pat = [[] for _ in range(n_contracts)]
    if lines[curr_index + 26*(n_contracts + 4) + 1 + c][0] != ';':
        for c in range(n_contracts):
            n_unw_pat[c] = int(lines[curr_index + 26*(n_contracts + 4) + 1 + c][1])
    curr_index = curr_index + 27*(n_contracts + 4)

    for i in range(n_contracts):
        for j in range(int(n_unw_pat[i])):
            len_unw_pat[i].append(int(lines[curr_index][2]))
            curr_index += 1

    curr_index += 3
    unw_pats = [[] for _ in range(n_contracts)]
    for i in range(n_contracts):
        unw_pats[i] = [[] for _ in range(len(len_unw_pat[i]))]
        for j in range(len(len_unw_pat[i])):
            pat_len = len_unw_pat[i][j]
            for k in range(pat_len):
                unw_pats[i][j].append(lines[curr_index][3])
                curr_index += 1

    curr_index += 3
    w_unw_pats = [[] for _ in range(n_contracts)]
    for i in range(n_contracts):
        for j in range(len(len_unw_pat[i])):
            w_unw_pats[i].append(int(lines[curr_index][2]))
            curr_index += 1

    curr_index += 3
    if lines[curr_index][0] == 'C0':
        for k in range(n_contracts):
            contr_param[12][k] = [np.sign(contr_param[12][k][1]), contr_param[12][k][1]]
            contr_param[13][k] = [1, int(lines[-4][2])]
    contr_param = pd.DataFrame(contr_param)
    param_names[11] = 'NoNightShiftBeforeFreeWeekend'
    param_names[12] = 'AlternativeSkills'
    contr_param.index = param_names

weekends_contract = np.empty((n_contracts, n_days), dtype=object)
weekends_contract.fill((0,[]))
for a in range(n_contracts):
    fday = 100
    lday = 0
    for day in range(n_days):
        if wknds_by_c[a][day] == 1:
            lday, fday = day, day
            while wknds_by_c[a][lday] == 1:
                lday += 1
            while wknds_by_c[a][fday] == 1:
                fday -= 1
            weekends_contract[a][day] = (1, np.array(range(fday+1, lday)))

print('shift types:', shift_types)
print('number of contracts types:', n_contracts)
print('n_nurses:', n_nurses)
print('\ncompatible shift types by nurse\n', comp_shifts)
print('\nshift-off request ([nurse][day][shift type] = weight)\n', shift_off_reqs)
#0 if not weekend, 1 if weekend. Array indicates indices that belong to same weekend
#Should help evaluate weekend-based violations
print('\nWeekend days by contract type:\n', weekends_contract)
print('\ndemand\n', demand)
print('\nnurse contract types:', nurse_contracts)
print('\nParameters by contract (value, weight):')
print(contr_param)
print('\nUnwanted patterns by contract:', unw_pats)
print('\nWeights unwanted patterns by contract:', w_unw_pats)

def createPar(shift_types, n_contracts, n_nurses, comp_shifts, shift_off_reqs, weekends_contract, demand, nurse_contracts, contr_param, unw_pats, w_unw_pats, n_days):

    print("")
    print("PARAMETERS\n")
    # Paramater creation in order of the article
    # A "#" means the parameter was already clearly identified by just reading in the data

    # The N parameter
    N = np.arange(1,n_nurses+1)
    print(f"N: {N}\n")

    # S parameter, clearly read in
    #shift_types
    print(f"S: {shift_types}\n")

    # S subsets
    S_a = [shift_types[0]]
    S_b = [i for i in shift_types if i != 'N']
    print(f"S subset(night): {S_a}")
    print(f"S subset(others): {S_b}\n")

    # D parameter
    D = np.arange(1, n_days+1)
    print(f"D: {D}\n")

    # r_ds parameter
    #demand
    print("r_ds:")
    print(demand)
    print("")

    # Pie parameter
    Pi = []
    for i in range(1, n_days+1):
        for j in range(i, n_days+1):
            Pi.append([i,j])
    print(f"Pie: {Pi}\n")

    # W_n parameter
    W_n = {}
    for i in range(1, n_nurses+1):
        W_n[i] = []
        w_count = 1
        count = 0
        for j in range(len(weekends_contract[int(nurse_contracts[i-1])])):
            if weekends_contract[int(nurse_contracts[i-1])][j][0] == 1:
                count += 1
                if count % 2 == 1:
                    W_n[i].append(w_count)
                    w_count +=1

    print(f"W_n: {W_n}\n")

    # D_in parameter
    temp = {}
    count = 0
    for i in range(1, n_contracts+1):
        temp[i] = []
        for j in range(len(weekends_contract[i-1])):
            if weekends_contract[i-1][j][0] == 1:
                count+=1
                if count % 2 == 1:
                    temp[i].append([j,j+1])

    D_in = {}

    for i in range(1, n_nurses+1):
        D_in[i] = temp[int(nurse_contracts[i-1])+1]

    print(f"D_in: {D_in}\n")

    #l_in parameter
    l_in = {}
    for i in N:
        l_in[i] = []
        for j in W_n[i]:
            l_in[i].append(D_in[i][j-1][0]-1)

    print(f"l_in: {l_in}\n")

    # P_n (shifts) parameter
    P_shifts = {}

    for i in range(1, n_nurses+1):
        P_shifts[i] = unw_pats[int(nurse_contracts[i-1])]

    print(f"P (unwanted shift patterns): {P_shifts}\n")

    # P_n (days) parameter
    # ???

    # y_lowerb_in, y_upperb_in, w_a_in, w_b_in", w_log_in parameters
    y_low_in = {} # [MinNumAssignments, MinConsecutiveFreeDays, MinConsecutiveWorkingDays, MinWorkingWeekendsInFourWeeks, MinConsecutiveWorkingWeekends]
    y_high_in = {} # same structure as y_low_in
    w_a_in = {} # weights for y_low_in; same structure
    w_b_in = {} # weights for y_high_in; same structure
    w_log_in = {}

    for i in range(1, n_nurses+1):
        y_low_in[i] = []
        y_high_in[i] = []
        w_a_in[i] = []
        w_b_in[i] = []
        w_log_in[i] = []

        y_low_in[i].append(contr_param.iloc[1][int(nurse_contracts[i-1])][0])
        y_high_in[i].append(contr_param.iloc[0][int(nurse_contracts[i-1])][0])
        w_a_in[i].append(contr_param.iloc[1][int(nurse_contracts[i-1])][1])
        w_b_in[i].append(contr_param.iloc[0][int(nurse_contracts[i-1])][1])
        w_log_in[i].append(contr_param.iloc[9][int(nurse_contracts[i-1])][1])

        y_low_in[i].append(contr_param.iloc[5][int(nurse_contracts[i-1])][0])
        y_high_in[i].append(contr_param.iloc[4][int(nurse_contracts[i-1])][0])
        w_a_in[i].append(contr_param.iloc[5][int(nurse_contracts[i-1])][1])
        w_b_in[i].append(contr_param.iloc[4][int(nurse_contracts[i-1])][1])
        w_log_in[i].append(contr_param.iloc[11][int(nurse_contracts[i-1])][1])

        y_low_in[i].append(contr_param.iloc[3][int(nurse_contracts[i-1])][0])
        y_high_in[i].append(contr_param.iloc[2][int(nurse_contracts[i-1])][0])
        w_a_in[i].append(contr_param.iloc[3][int(nurse_contracts[i-1])][1])
        w_b_in[i].append(contr_param.iloc[2][int(nurse_contracts[i-1])][1])
        w_log_in[i].append(contr_param.iloc[10][int(nurse_contracts[i-1])][1])

        y_low_in[i].append(0)
        y_high_in[i].append(contr_param.iloc[8][int(nurse_contracts[i-1])][0])
        w_a_in[i].append(0)
        w_b_in[i].append(contr_param.iloc[8][int(nurse_contracts[i-1])][1])
        w_log_in[i].append(contr_param.iloc[12][int(nurse_contracts[i-1])][1])

        y_low_in[i].append(contr_param.iloc[7][int(nurse_contracts[i-1])][0])
        y_high_in[i].append(contr_param.iloc[6][int(nurse_contracts[i-1])][0])
        w_a_in[i].append(contr_param.iloc[7][int(nurse_contracts[i-1])][1])
        w_b_in[i].append(contr_param.iloc[6][int(nurse_contracts[i-1])][1])
        w_log_in[i].append(w_unw_pats[int(nurse_contracts[i-1])])

        # Ranged soft constraint 6/#number of days of after night shift??
        # Logical soft constaint 7/day on/off request???

        w_log_in[i].append(shift_off_reqs[i-1])

    print(f"y_lowerb_in: {y_low_in}")
    print(f"y_upperb_in: {y_high_in}\n")

    print(f"w_a_in (lowerb): {w_a_in}")
    print(f"w_a_in (upperb): {w_b_in}\n")

    print(f"w_log_in:")
    print(w_log_in)
    print("")

    # Max number of complete weekends and corresponding weight
    max_comp_WE = {}
    w_max_comp_WE = {}

    for i in range(1, n_nurses + 1):
        max_comp_WE[i] = []
        w_max_comp_WE[i] = []

        max_comp_WE[i].append(contr_param.iloc[9][int(nurse_contracts[i-1])][0])
        w_max_comp_WE[i].append(contr_param.iloc[9][int(nurse_contracts[i-1])][1])

    # Number of identical shifts during complete weekends and corresponding weight
    max_ident_shifts_comp_WE = {}
    w_max_ident_shifts_comp_WE = {}

    for i in range(1, n_nurses + 1):
        max_ident_shifts_comp_WE[i] = []
        w_max_ident_shifts_comp_WE[i] = []

        max_ident_shifts_comp_WE[i].append(contr_param.iloc[10][int(nurse_contracts[i-1])][0])
        w_max_ident_shifts_comp_WE[i].append(contr_param.iloc[10][int(nurse_contracts[i-1])][1])

    # No night shift before free weekend

    NoNShiftBeforeFreeWE = {}
    for i in range(1, n_nurses + 1):
        NoNShiftBeforeFreeWE[i] = []
        NoNShiftBeforeFreeWE[i].append(contr_param.iloc[11][int(nurse_contracts[i-1])][0])
        NoNShiftBeforeFreeWE[i].append(contr_param.iloc[11][int(nurse_contracts[i-1])][1])

    # Alternative skills

    AltSkills = {}
    for i in range(1, n_nurses + 1):
        AltSkills[i] = []
        AltSkills[i].append(contr_param.iloc[12][int(nurse_contracts[i-1])][0])
        AltSkills[i].append(contr_param.iloc[12][int(nurse_contracts[i-1])][1])

    # No Friday off, if working on Sat and Sun

    NoFriOffIfSatSun = {}
    for i in range(1, n_nurses + 1):
        NoFriOffIfSatSun[i] = []
        NoFriOffIfSatSun[i].append(contr_param.iloc[13][int(nurse_contracts[i-1])][0])
        NoFriOffIfSatSun[i].append(contr_param.iloc[13][int(nurse_contracts[i-1])][1])

    # Shift-off requests
    #Already defined previously
    #shift_off_reqs

    # Unwanted patterns

    unwanted_patterns = {}
    w_unwanted_patterns = {}
    for i in range(1, n_nurses + 1):
        unwanted_patterns[i] = []
        w_unwanted_patterns[i] = []
        unwanted_patterns[i].append(unw_pats[int(nurse_contracts[i-1])])
        w_unwanted_patterns[i].append(w_unw_pats[int(nurse_contracts[i-1])])

    # Sigma parameters

    sigma = {}
    for i in range(1, n_nurses + 1):
        sigma[i] = []
        for pair in Pi:
            penalty = 0
            num_consec_days = pair[1] - pair[0]
            # Penalty for too little days of continous work
            penalty += w_a_in[i][2] * max(y_low_in[i][2] - num_consec_days, 0)
            # Penalty for too many days of continous work
            penalty += w_b_in[i][2] * max(num_consec_days - y_high_in[i][2], 0)
            # Penalty for incomplete weekends
            num_incomplete_weekends = 0
            for weekend in D_in[i]:
                sat = weekend[0]
                sun = weekend[1]
                if sat == pair[1] or sun == pair[0]:
                    num_incomplete_weekends += 1
            penalty += w_max_comp_WE[i][0] * num_incomplete_weekends

            sigma[i].append(penalty)

    # Tau parameters

    tau = {}
    for i in range(1, n_nurses + 1):
        tau[i] = []
        for pair in Pi:
            penalty = 0
            num_consec_days = pair[1] - pair[0]
            # Penalty for too little days of continous rest
            penalty += w_a_in[i][1] * max(y_low_in[i][1] - num_consec_days, 0)
            # Penalty for too many days of continous work
            penalty += w_b_in[i][1] * max(num_consec_days - y_high_in[i][1], 0)

            tau[i].append(penalty)

    # Nu parameters
    # We exclude soft constraints "alternative skill" for now

    nu = {}
    for i in range(1, n_nurses + 1):
        nu[i] = []
        for request_per_day in shift_off_reqs[i-1]:
            nu[i].append(request_per_day)

    # Omega parameters

    omega = {}
    for i in range(1, n_nurses + 1):
        omega[i] = {'omega1low': w_a_in[i][0],
                    'omega1high': w_b_in[i][0],
                    'omega4': w_b_in[i][3],
                    'omega8': NoNShiftBeforeFreeWE[i][1],
                    'omega9': w_max_ident_shifts_comp_WE[i][0],
                    'omega11': w_unwanted_patterns[i][0]}

    # Psi parameters

    WE_pairs = {}
    for n in range(1, n_nurses + 1):
        WE_pairs[n] = []
        for i in range(1, len(W_n[n]) + 1):
            for j in range(i, len(W_n[n]) + 1):
                WE_pairs[n].append([i,j])

    psi = {}
    for i in range(1, n_nurses + 1):
        psi[i] = []
        for pair in WE_pairs[i]:
            penalty = 0
            num_consec_WEs = pair[1] - pair[0]
            # Penalty for too little days of continous working weekends
            penalty += w_a_in[i][4] * max(y_low_in[i][4] - num_consec_WEs, 0)
            # Penalty for too many days of continous working weekends
            penalty += w_b_in[i][4] * max(num_consec_days - y_high_in[i][4], 0)

            psi[i].append(penalty)

    #return N, S_a, S_b, D, Pi, W_n, D_in, P_shifts, y_low_in, y_high_in, w_a_in, w_b_in, w_log_in
    return {
        'N': N,
        'S': shift_types,
        'S_a': S_a,
        'S_b': S_b,
        'D': D,
        'Pi': Pi,
        'W_n': W_n,
        'D_in': D_in,
        'l_in': l_in,
        'P_shifts': P_shifts,
        'y_low_in': y_low_in,
        'y_high_in': y_high_in,
        'w_a_in': w_a_in,
        'w_b_in': w_b_in,
        'w_log_in': w_log_in,
        'max_comp_WE': max_comp_WE,
        'w_max_comp_WE': w_max_comp_WE,
        'max_ident_shifts_comp_WE': max_ident_shifts_comp_WE,
        'w_max_ident_shifts_comp_WE': w_max_ident_shifts_comp_WE,
        'NoNShiftBeforeFreeWE': NoNShiftBeforeFreeWE,
        'AltSkills': AltSkills,
        'NoFriOffIfSatSun': NoFriOffIfSatSun,
        'shift_off_reqs': shift_off_reqs,
        'unwanted_patterns': unwanted_patterns,
        'w_unwanted_patterns': w_unwanted_patterns,
        'sigma': sigma,
        'tau': tau,
        'nu': nu,
        'omega': omega,
        'WE_pairs': WE_pairs,
        'psi': psi
    }


"""
In order to check the soft constraints related to consecutive days or weekends,
we use the function find_runs(x). Credit goes to Alistair Miles. The original
code can be found using the following link (last accessed on 30.05.2024):
https://gist.github.com/alimanfoo/c5977e87111abe8127453b21204c1065
"""

def find_runs(x):
    """Find runs of consecutive items in an array."""

    # ensure array
    x = np.asanyarray(x)
    if x.ndim != 1:
        raise ValueError('only 1D array supported')
    n = x.shape[0]

    # handle empty array
    if n == 0:
        return np.array([]), np.array([]), np.array([])

    else:
        # find run starts
        loc_run_start = np.empty(n, dtype=bool)
        loc_run_start[0] = True
        np.not_equal(x[:-1], x[1:], out=loc_run_start[1:])
        run_starts = np.nonzero(loc_run_start)[0]

        # find run values
        run_values = x[loc_run_start]

        # find run lengths
        run_lengths = np.diff(np.append(run_starts, n))

        return run_values, run_starts, run_lengths

##### Defining the objective function with the soft constraints #####

def penalty_per_nurse(solution, nurse_index, params):
    # Define nurse key
    nurse_key = nurse_index + 1

    # Initialise penalty to be 0
    penalty = 0

    # Maximum number of assignments
    if np.sum(solution[nurse_index,:,:]) > params['y_high_in'][nurse_key][0]:
        penalty += params['w_b_in'][nurse_key][0]

    # Minimum number of assignments
    if np.sum(solution[nurse_index,:,:]) < params['y_low_in'][nurse_key][0]:
        penalty += params['w_a_in'][nurse_key][0]

    # Maximum number of consecutive working days
    # Create array with days -> 1 if working, 0 otherwise
    working_days = np.zeros(np.amax(params['D']))
    for i in range(np.amax(params['D'])):
        if np.sum(solution[nurse_index,i,:]) > 0:
            working_days[i] = 1
    # Find the number of consecutive working days and compare it with the allowed value
    wdays_values, wdays_starts, wdays_length = find_runs(working_days)
    max_consec_wdays = params['y_high_in'][nurse_key][2]
    for value, start, length in zip(wdays_values, wdays_starts, wdays_length):
        if value == 1 and length > max_consec_wdays:
            penalty += (length - max_consec_wdays) * params['w_b_in'][nurse_key][2]

    # Minimum number of consecutive working days
    min_consec_wdays = params['y_low_in'][nurse_key][2]
    for value, start, length in zip(wdays_values, wdays_starts, wdays_length):
        if value == 1 and length < min_consec_wdays:
            penalty += (min_consec_wdays - length) * params['w_a_in'][nurse_key][2]

    # Maximum number of consecutive free days
    max_consec_fdays = params['y_high_in'][nurse_key][1]
    for value, start, length in zip(wdays_values, wdays_starts, wdays_length):
        if value == 0 and length > max_consec_fdays:
            penalty += (length - max_consec_fdays) * params['w_b_in'][nurse_key][1]

    # Minimum number of consecutive free days
    min_consec_fdays = params['y_low_in'][nurse_key][1]
    for value, start, length in zip(wdays_values, wdays_starts, wdays_length):
        if value == 0 and length < min_consec_fdays:
            penalty += (min_consec_fdays - length) * params['w_a_in'][nurse_key][1]

    # Maximum number of consecutive working weekends
    working_weekends = np.zeros(len(params['W_n'][nurse_key]))
    for i in range(len(params['W_n'][nurse_key])):
        weekend = params['D_in'][nurse_key][i]
        for weekendday in weekend:
            if working_days[weekendday] > 0:
                working_weekends[i] = 1
    weekends_values, weekends_starts, weekends_lenght = find_runs(working_weekends)
    max_consec_work_weekends = params['y_high_in'][nurse_key][4]
    for value, start, length in zip(weekends_values, weekends_starts, weekends_lenght):
        if value == 1 and length > max_consec_work_weekends:
            penalty += (length - max_consec_work_weekends) * params['w_b_in'][nurse_key][4]

    # Maximum number of weekends in four weeks
    if np.sum(working_weekends) > params['y_high_in'][nurse_key][3]:
        penalty += params['w_b_in'][nurse_key][3]

    # Complete weekends
    complete_weekends = np.zeros(len(params['W_n'][nurse_key]))
    for i in range(len(params['W_n'][nurse_key])):
        weekend = params['D_in'][nurse_key][i]
        working_days_WE_i = 0
        for weekendday in weekend:
            if working_days[weekendday] > 0:
                working_days_WE_i += 1
        if working_days_WE_i == 2:
            complete_weekends[i] = 1
    violation_complete_weekends = 0
    for i in range(len(complete_weekends)):
        if complete_weekends[i] == 0:
            weekend = params['D_in'][nurse_key][i]
            Sat = weekend[0]
            Sun = weekend[1]
            if np.sum(solution[nurse_index, Sat]) != np.sum(solution[nurse_index, Sun]):
                violation_complete_weekends += 1
    if violation_complete_weekends > params['max_comp_WE'][nurse_key][0]:
        penalty += params['w_max_comp_WE'][nurse_key][0]

    # Identical shifts during complete weekends
    num_unident_shifts_comp_WE = 0
    for i in range(len(complete_weekends)):
        if complete_weekends[i] == 1:
            weekend = params['D_in'][nurse_key][i]
            Sat = weekend[0]
            Sun = weekend[1]
            if not np.array_equal(solution[nurse_index, Sat], solution[nurse_index, Sun]):
                num_unident_shifts_comp_WE += 1
    if num_unident_shifts_comp_WE > params['max_ident_shifts_comp_WE'][nurse_key][0]:
        penalty += params['w_max_ident_shifts_comp_WE'][nurse_key][0]

    # No night shift before free weekend
    noNightShiftBeforeFreeWeekend = 0
    for i in range(len(complete_weekends)):
        if complete_weekends[i] == 0:
            weekend = params['D_in'][nurse_key][i]
            sat = weekend[0]
            sun = weekend[1]
            if sat > 0:
                fri = weekend[0] - 1
                if np.all(solution[nurse_index, sat] == 0) & np.all(solution[nurse_index, sun] == 0):
                    if solution[nurse_index, fri, 0] != 0: #zero index for night shift
                        noNightShiftBeforeFreeWeekend += 1
    if noNightShiftBeforeFreeWeekend > params['NoNShiftBeforeFreeWE'][nurse_key][0]:
        penalty += params['NoNShiftBeforeFreeWE'][nurse_key][1]

    # Alternative skill

    # No Friday off, if working on Sat and Sun
    nofridayOffIfWorkingSatAndSun = 0
    for i in range(len(complete_weekends)):
        if complete_weekends[i] == 1:
            weekend = params['D_in'][nurse_key][i]
            sat = weekend[0]
            if sat > 0:
                fri = weekend[0] - 1
                if np.all(solution[nurse_index, fri] == 0):
                    nofridayOffIfWorkingSatAndSun += 1
    if nofridayOffIfWorkingSatAndSun > params['NoFriOffIfSatSun'][nurse_key][0]:
        penalty += params['NoFriOffIfSatSun'][nurse_key][1]

    # Requested day on/off
    # Dealt with in 'Requested shift on/off'

    # Requested shift on/off

    for day in range(np.amax(params['D'])):
        # Since a nurse can at most be assigned to one shift per day, and a
        # nurse has a penalty of 0 if she is fine with having certain shift, it
        # is sufficient to take the sumproduct of both arrays as penalty.
        product = solution[nurse_index, day] * params['shift_off_reqs'][nurse_index][day]
        sumproduct = np.sum(product)
        penalty += sumproduct

    # Unwanted patterns
    # Shift order in solution: N E D L
    mapping = {'N': 0, 'E': 1, 'D': 2, 'L': 3}
    unw_pat_index = 0
    for unw_pat in params['unwanted_patterns'][nurse_key][0]:
        num_unw_pat = [mapping[element] for element in unw_pat]
        considered_days = len(unw_pat)
        for day1 in range(np.amax(params['D']) - (considered_days - 1)):
            all_days_match = True
            for day_offset in range(considered_days):
                if solution[nurse_index, day1 + day_offset, num_unw_pat[day_offset]] != 1:
                    all_days_match = False
                    break
            if all_days_match:
                penalty += params['w_unwanted_patterns'][nurse_key][0][unw_pat_index]
        unw_pat_index += 1

    # Return total penalty
    return penalty

##### Creating a good initial solution #####

# Solution assignment of the following form:
# [nurse][day][shift type] = 1 if shift is assigned to nurse; 0 otherwise

def greedy_initial_solution(demand, solution, params):
    for day in range(demand.shape[0]):
        for shift_index, shift in enumerate(demand.columns):
            required_demand = demand.loc[day][shift]
            assigned_nurses = np.sum(solution[:, day, shift_index])
            while assigned_nurses < required_demand:
                penalties = np.full(len(params['N']), np.inf)
                # Calculate the penalty score for every nurse if they were
                # assigned to that shift
                for nurse in range(len(params['N'])):
                    if not solution[nurse, day].any(): # Only consider nurses not already assigned to a shift that day
                        pen_before_assignm = penalty_per_nurse(solution, nurse, params)
                        solution[nurse, day, shift_index] = 1
                        pen_after_assignm = penalty_per_nurse(solution, nurse, params)
                        solution[nurse, day, shift_index] = 0 # reset because you only want to keep for smallest penalty
                        penalties[nurse] = pen_after_assignm - pen_before_assignm
                # Select the nurse with the smallest penalty
                # and assign her to the shift
                best_nurse = np.argmin(penalties)
                solution[best_nurse, day, shift_index] = 1
                assigned_nurses = np.sum(solution[:, day, shift_index])

    return solution

##### Creating the linear program #####

def lp_feasibility(N, D, shift_types, demand):

    model = Model('nrp')
    model.setParam('OutputFlag', False)

    # Defining Decision Variables (DVs)
    x = model.addVars(N, shift_types, D, vtype=GRB.BINARY, name='x')

    # Hard constraint I: demand
    for j in shift_types:
        for k in D:
            model.addConstr(quicksum(x[i, j, k] for i in N) == int(demand.loc[k-1][j]))

    # Hard constraint II: One shift per day
    for i in N:
        for k in D:
            model.addConstr(quicksum(x[i, j, k] for j in shift_types) <= 1)

    model.optimize()

    if model.SolCount >= 1:
        solution = {i: {k: None for k in D} for i in N}
        for i in N:
            for k in D:
                assigned_shifts = [j for j in shift_types if x[i, j, k].X > 0.5]
                if assigned_shifts:
                    solution[i][k] = random.choice(assigned_shifts)
        return solution
    else:
        return "No feasible solution found"

def lp_nrp(N, shift_types, S_a, S_b, D, Pi, W_n, D_in, l_in, P_shifts, y_low_in, y_high_in, w_a_in, w_b_in, w_log_in, sigma, tau, nu, omega, psi, WE_pairs, solution, timeWindow):

    model=Model('nrp')
    model.setParam('OutputFlag', True)

    # Defining DV's in order of the article
    # DV 1
    x = model.addVars(N, shift_types, D, vtype=GRB.BINARY, name = 'x')

    # DV 2
    y = model.addVars([(i,j) for i in N for j in W_n[i]], vtype=GRB.BINARY, name = 'y')

    # DV 3
    w = model.addVars([(i, j) for i in N for j in range(len(Pi))], vtype=GRB.BINARY, name = 'w')

    # DV 4
    r = model.addVars([(i, j) for i in N for j in range(len(Pi))], vtype=GRB.BINARY, name = 'r')

    # DV 5
    z = model.addVars([(i, j) for i in N for j in range(len(WE_pairs))], vtype=GRB.BINARY, name = 'z')

    # Fix variables
    for i in N:
        for j in shift_types:
            for k in D:
                if k not in timeWindow:
                    x[i, j, k].lb = solution[i-1][k-1][shift_types.index(j)]
                    x[i, j, k].ub = solution[i-1][k-1][shift_types.index(j)]


    # Slack variables
    alphaLower_1 = model.addVars([(i,1) for i in N], vtype=GRB.INTEGER, name = 'alphaLower_1')
    alphaUpper_1 = model.addVars([(i,1) for i in N], vtype=GRB.INTEGER, name = 'alphaUpper_1')

    alphaLower_4 = model.addVars([(i, j, 4) for i in N for j in W_n[i]], vtype=GRB.INTEGER, name = 'alphaLower_4')
    alphaUpper_4 = model.addVars([(i, j, 4) for i in N for j in W_n[i]], vtype=GRB.INTEGER, name = 'alphaUpper_4')

    alpha_8 = model.addVars([(i, j, 8) for i in N for j in W_n[i]], vtype=GRB.BINARY, name = 'alpha_8')

    alpha_9 = model.addVars([(i, j, k, 9) for i in N for j in D for k in D if k>=j], vtype=GRB.BINARY, name = 'alpha_9')

    alpha_11 = model.addVars([(i, j, k, 11) for i in N for j in range(1, len(P_shifts[i])+1) for k in range(1, len(D) - len(P_shifts[i][j-1]) + 2)], vtype=GRB.BINARY, name = 'alpha_11')

    # Define the objective
    objective = quicksum(
        quicksum(
            sigma[n][daypair] * w[n, daypair] +
            tau[n][daypair] * r[n, daypair]
            for daypair in range(len(Pi))
        ) +
        quicksum(
            nu[n][day - 1][shift_types.index(shift)] * x[n, shift, day]
            for day in D
            for shift in shift_types
        ) +
        omega[n]['omega1high'] * alphaUpper_1[n, 1] +
        omega[n]['omega1low'] * alphaLower_1[n, 1] +
        quicksum(
            omega[n]['omega4'] * alphaUpper_4[n, j, 4] +
            omega[n]['omega8'] * alpha_8[n, j, 8] +
            quicksum(
                quicksum(
                    omega[n]['omega9'] * alpha_9[n, i, k, 9]
                    for k in D if k>=i
                )
                for i in D
            )
            for j in W_n[n]
        ) +
        quicksum(
            psi[n][WEpair] * z[n, WEpair]
            for WEpair in range(len(WE_pairs[n]))
        ) +
        quicksum(
            quicksum(
                omega[n]['omega11'][pattern-1] * alpha_11[n, pattern, a, 11]
                for a in range(1, len(D) - len(P_shifts[n][pattern - 1]) + 2)
                    )
                for pattern in range(1, len(P_shifts[n]) + 1)
                )
            for n in N
    )

    # Set objective
    model.setObjective(objective, GRB.MINIMIZE)

    # Hard constraint I: demand
    for j in shift_types:
        for k in D:
            model.addConstr(quicksum(x[i, j, k] for i in N) == int(demand.loc[k-1][j]))

    # Hard constraint II: One shift per day
    for i in N:
        for k in D:
            model.addConstr(quicksum(x[i, j, k] for j in shift_types) <= 1)

    # Article constraint (3): Initialize working weekends DV
    for i in N:
        for v in W_n[i]:
            for k in D_in[i][v-1]:
                model.addConstr(y[i, v] >= (quicksum(x[i, j, k] for j in shift_types)))

    # Article constraint (4)
    for i in N:
        for v in W_n[i]:
            model.addConstr(y[i, v] <= (quicksum(x[i, j, k] for j in shift_types for k in D_in[i][v-1])))

    # Article constraint (5)
    for i in N:
        for k in D:
            model.addConstr(quicksum(x[i, j, k] for j in shift_types) == quicksum(w[i, q] for q in range(len(Pi)) if k >= Pi[q][0] and k <= Pi[q][1]))

    # Article constraint (6)
    for i in N:
        for k in D:
            model.addConstr(quicksum(x[i, j, k] for j in shift_types) == 1 - quicksum(r[i, q] for q in range(len(Pi)) if k >= Pi[q][0] and k <= Pi[q][1]))

    # Article constraint (7)
    for i in N:
        for k in D:
            model.addConstr(quicksum((r[i, q] + w[i, q]) for q in range(len(Pi)) if k >= Pi[q][0] and k <= Pi[q][1]) == 1)

    # Article constraint (8)
    for i in N:
        for k in D:
            model.addConstr(quicksum(w[i, q] for q in range(len(Pi)) if Pi[q][0] <= k and Pi[q][1] == k) + quicksum(w[i, t] for t in range(len(Pi)) if Pi[t][1] >= k+1 and Pi[t][0] == k+1) <= 1)

    # Article constraint (9)
    for i in N:
        for k in D:
            model.addConstr(quicksum(r[i, q] for q in range(len(Pi)) if Pi[q][0] <= k and Pi[q][1] == k) + quicksum(r[i, t] for t in range(len(Pi)) if Pi[t][1] >= k+1 and Pi[t][0] == k+1) <= 1)

    # Article constraint (10)
    for i in N:
        model.addConstr(y_low_in[i][0] - alphaLower_1[i,1] <= quicksum(x[i, j, k] for j in shift_types for k in D))
    for i in N:
        model.addConstr(quicksum(x[i, j, k] for j in shift_types for k in D) <= y_high_in[i][0] + alphaUpper_1[i,1])

    # Article constraint (11)
    for i in N:
        for v in range(1, W_n[i][-1]-3):
            model.addConstr(quicksum(y[i,t] for t in range(v, v+3)) <= y_high_in[i][3] + alphaUpper_4[i,v,4])

    # Article constraint (12)
    # Does not exist in our setting

    # Article constraint (13)
    for i in N:
        for v in W_n[i]:
            model.addConstr(x[i, 'N', l_in[i][v-1]+1] - y[i,v] <= alpha_8[i, v, 8])

    # Article constraint 14
    for i in N:
        for j in shift_types:
            for v in W_n[i]:
                model.addConstr(alpha_9[i, D_in[i][v-1][0], D_in[i][v-1][1], 9] >= x[i,j,D_in[i][v-1][0]] - x[i,j,D_in[i][v-1][1]])

    # Article constraint 15
    for i in N:
        for j in shift_types:
            for v in W_n[i]:
                model.addConstr(alpha_9[i, D_in[i][v-1][0], D_in[i][v-1][1], 9] >= x[i,j,D_in[i][v-1][1]] - x[i,j,D_in[i][v-1][0]])

    # Article constraint 16
    for i in N:
        for p in range(1, len(P_shifts[i]) + 1):
            for k in range(1, len(D)- len(P_shifts[i][p-1]) + 1):
                model.addConstr(quicksum(x[i, P_shifts[i][p-1][j], k+(j+1)-1] for j in range(len(P_shifts[i][p-1]))) <= len(P_shifts[i][p-1]) - 1 + alpha_11[i,  p,  k,11])

    # Article constraint 17
    # We don't have a set of unwanted working day patterns

    # Article constraint 18
    for i in N:
        for p in W_n[i]:
            model.addConstr(quicksum(z[i, q] for q in range(len(WE_pairs[i])) if WE_pairs[i][q][0] <= p and WE_pairs[i][q][1] >= p) <= 1)

    # Article constraint 19
    for i in N:
        for p in range(1, len(W_n[i])):
            model.addConstr(quicksum(z[i, q] for q in range(len(WE_pairs[i])) if WE_pairs[i][q][0] <= p  and WE_pairs[i][q][1] == p) + quicksum(z[i, t] for t in range(len(WE_pairs[i])) if WE_pairs[i][t][0] == p+1 and WE_pairs[i][t][1] >= p) <= 1)

    # Article constraint 20
    for i in N:
        for p in W_n[i]:
            model.addConstr(y[i, p] == quicksum(z[i, q]for q in range(len(WE_pairs[i])) if WE_pairs[i][q][0] <= p and WE_pairs[i][q][1] >= p))

    model.optimize()

    if model.SolCount >= 1:
        #for v in model.getVars():
            #print(str(v.varName) + " = " + str(v.x))
        solution = np.zeros([n_nurses, np.amax(params['D']), n_shift_types], dtype=int)
        for i in N:
            for j in shift_types:
                for k in D:
                    if x[i, j, k].X == 1:
                        solution[i-1][k-1][shift_types.index(j)] = 1
        return solution
    else:
        return "No feasible solution found"

##### Applying the functions #####

params = createPar(shift_types, n_contracts, n_nurses, comp_shifts, shift_off_reqs, weekends_contract, demand, nurse_contracts, contr_param, unw_pats, w_unw_pats, n_days)

# Random solution
initial_random_solution = lp_feasibility(params['N'],params['D'], shift_types, demand)

random_solution = np.zeros([n_nurses, np.amax(params['D']), n_shift_types], dtype=int)

for i in initial_random_solution:
    for j in initial_random_solution[i]:
        if initial_random_solution[i][j] != None:
            random_solution[i-1][j-1][shift_types.index(initial_random_solution[i][j])] = 1

# Optimization/time window
optTime_random = 7

# Needed to push the time window into the future
count_random = 1

# Max running time (in seconds)
maxTime_random = 60

# Max iteration
maxIt_random = 50

startTime_random = time.time()
endTime_random = time.time()

objValue_random = []

#while count_random < maxIt_random:
#while time_random <= n_days-optTime_random:
while (endTime_random- startTime_random) < maxTime_random:

    penalty_random = 0
    for i in range(n_nurses):
        penalty_random += penalty_per_nurse(random_solution, i, params)
    objValue_random.append(penalty_random)

    timeWindow_random = []
    for i in range(count_random, optTime_random + count_random):
        timeWindow_random.append(i)
    random_solution = lp_nrp(params['N'], params['S'], params['S_a'], params['S_b'], params['D'], params['Pi'], params['W_n'], params['D_in'], params['l_in'], params['P_shifts'], params['y_low_in'], params['y_high_in'], params['w_a_in'], params['w_b_in'], params['w_log_in'], params['sigma'], params['tau'], params['nu'], params['omega'], params['psi'], params['WE_pairs'], random_solution, timeWindow_random)
    count_random+=1
    if count_random > n_days-optTime_random:
        count_random = 1
    endTime_random = time.time()

#for i in range(len(final_solution)):
#print(final_solution[i])



# Greedy initial solution

solution = np.zeros([n_nurses, np.amax(params['D']), n_shift_types], dtype=int)

final_solution = greedy_initial_solution(demand, solution, params)

# Optimization/time window
optTime = 7

# Needed to push the time window into the future
count = 1

# Max running time (in seconds)
maxTime = 60

# Max iteration
maxIt = 50

startTime = time.time()
endTime = time.time()

objValue = []

#while count < maxIt:
#while time <= n_days-optTime:
while (endTime- startTime) < maxTime:

    penalty_total = 0
    for i in range(n_nurses):
        penalty_total += penalty_per_nurse(final_solution, i, params)
    objValue.append(penalty_total)

    timeWindow = []
    for i in range(count, optTime + count):
        timeWindow.append(i)
    final_solution = lp_nrp(params['N'], params['S'], params['S_a'], params['S_b'], params['D'], params['Pi'], params['W_n'], params['D_in'], params['l_in'], params['P_shifts'], params['y_low_in'], params['y_high_in'], params['w_a_in'], params['w_b_in'], params['w_log_in'], params['sigma'], params['tau'], params['nu'], params['omega'], params['psi'], params['WE_pairs'], final_solution, timeWindow)
    count+=1
    if count > n_days-optTime:
        count = 1
    endTime = time.time()

#for i in range(len(final_solution)):
    #print(final_solution[i])


# Penalty totals

objValue_random.sort(reverse=True)
print(f"Random obj. value: {objValue_random[-1]}")
#for i in range(len(objValue_random)):
    #print(int(objValue_random[i]))

objValue.sort(reverse=True)
print(f"Greedy obj. value: {objValue[-1]}")
#for i in range(len(objValue)):
    #print(int(objValue[i]))