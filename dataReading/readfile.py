# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 22:55:13 2024

@author: S_Schreuder
"""

import numpy as np
import ast
import pandas as pd
import math
from gurobipy import *

with open("sprint_hint03.dat", 'r', errors = 'ignore') as file:
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

    # P_n (shifts) parameter
    P_shifts = {}

    for i in range(1, n_nurses+1):
        P_shifts[i] = unw_pats[int(nurse_contracts[i-1])]

    print(f"P (unwanted shift patterns): {P_shifts}\n")

    # P_n (days) parameter
    # ???

    # y_lowerb_in, y_upperb_in, w_a_in, w_b_in", w_log_in parameters
    y_low_in = {}
    y_high_in = {}
    w_a_in = {}
    w_b_in = {}
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

    # Last four parameters??


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
        'P_shifts': P_shifts,
        'y_low_in': y_low_in,
        'y_high_in': y_high_in,
        'w_a_in': w_a_in,
        'w_b_in': w_b_in,
        'w_log_in': w_log_in
    }

def lp_nrp(N, shift_types, S_a, S_b, D, Pi, W_n, D_in, P_shifts, y_low_in, y_high_in, w_a_in, w_b_in, w_log_in):

    model=Model('nrp')
    model.setParam('OutputFlag', True)

    # Defining DV's in order of the article
    # DV 1
    x = model.addVars(N, shift_types, D, vtype=GRB.BINARY, name = 'x')

    # DV 2
    for i in N:
        y = model.addVars(N, W_n[i], vtype=GRB.BINARY, name = 'y')

    # DV 3
    w = model.addVars([(int(i), j, k) for i in N for j in D for k in D if k >= j], vtype=GRB.BINARY, name = 'w')

    # DV 4
    r = model.addVars([(int(i), j, k) for i in N for j in D for k in D if k >= j], vtype=GRB.BINARY, name = 'r')

    # DV 5
    z = model.addVars([(int(i), j, k) for i in N for j in W_n[i] for k in W_n[i] if k >= j], vtype=GRB.BINARY, name = 'z')

    # Random objective
    model.setObjective(quicksum(x[i,j,k] for i in N for j in shift_types for k in D), GRB.MINIMIZE)

    model.optimize()
    for v in model.getVars():
        print(str(v.varName) + " = " + str(v.x))


#N, S_a, S_b, D, Pi, W_n, D_in, P_shifts, y_low_in, y_high_in, w_a_in, w_b_in, w_log_in = createPar(shift_types, n_contracts, n_nurses, comp_shifts, shift_off_reqs, weekends_contract, demand, nurse_contracts, contr_param, unw_pats, w_unw_pats, n_days)
params = createPar(shift_types, n_contracts, n_nurses, comp_shifts, shift_off_reqs, weekends_contract, demand, nurse_contracts, contr_param, unw_pats, w_unw_pats, n_days)
lp_nrp(params['N'], params['S'], params['S_a'], params['S_b'], params['D'], params['Pi'], params['W_n'], params['D_in'], params['P_shifts'], params['y_low_in'], params['y_high_in'], params['w_a_in'], params['w_b_in'], params['w_log_in'])
