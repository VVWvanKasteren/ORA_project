# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 22:55:13 2024

@author: S_Schreuder
"""

import numpy as np 
import ast
import pandas as pd

with open("sprint_hint03.dat", 'r') as file:
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


















