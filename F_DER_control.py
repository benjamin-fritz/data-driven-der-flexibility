import json
import os
import re
import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyomo.environ as pyo

import tikzplotlib

from E_sensitivity_analysis import calculate_rel_sens_mat


def der_control(profile_constraining_dict=None, ev_ch_prob_dict=None, analyze_opt_results_dict=None):
    if profile_constraining_dict:
        compute_constrained_profiles(**profile_constraining_dict)
    if analyze_opt_results_dict:
        analyze_opt_results(**analyze_opt_results_dict)
    if ev_ch_prob_dict:
        compute_ev_charging_probabilities(**ev_ch_prob_dict)


def compute_constrained_profiles(sens_mat_path, voltages_path, phase_allocation_path, opt_config):
    """
    Computes (optimally) constrained power profiles for voltage correction

    :param sens_mat_path: paths where absolute sensitivities is located
    :param voltages_path: paths where voltage profiles are located
    :param phase_allocation_path: paths where tables for phase allocation are located
    :param opt_config: dict containing optimization settings
    :return:
    """
    print('\n\nCOMPUTE POWER PROFILES FOR VOLTAGE VIOLATION ELIMINATION:')
    for idx, _ in enumerate(sens_mat_path[:]):
        freq = re.search('sens_mat_(.*?).txt', sens_mat_path[idx]).group(1)
        grid_id = sens_mat_path[idx].split('/')[-2]
        results_dir = re.search('(.*?)results', voltages_path[idx]).group(1)
        print('\nGrid ID: {}'.format(grid_id))
        results_df = pd.read_csv(voltages_path[idx], sep='\t', skiprows=1).iloc[:, :-1]
        sens_mat, sens_mat_rel = calculate_rel_sens_mat(sens_mat_path[idx])
        # sens_mat_inv = -pd.DataFrame(np.linalg.pinv(-sens_mat.values), sens_mat.columns, sens_mat.index)
        ph_all_df = pd.read_csv(phase_allocation_path[idx], sep=None, engine='python')
        v_min, v_max = 400 / np.sqrt(3) * 0.95, 400 / np.sqrt(3) * 1.05
        v_df = results_df.iloc[:, 0:len(sens_mat)]
        p_df = results_df.iloc[:, len(sens_mat):2 * len(sens_mat)]
        # p_hh_df = pd.read_csv(results_dir + 'p_hh_only.csv', sep=';', index_col=[0])
        # p_ev_df = pd.read_csv(results_dir + 'p_ev_only.csv', sep=';', index_col=[0])
        # p_hp_df = pd.read_csv(results_dir + 'p_hp_only.csv', sep=';', index_col=[0])
        # p_pv_df = pd.read_csv(results_dir + 'p_pv_only.csv', sep=';', index_col=[0])
        v_under, v_over = (v_df - v_min).clip(upper=0), (v_df - v_max).clip(lower=0)

        # assigning proper datetime index and more readable column names
        dt_index = pd.to_datetime(
            pd.read_csv('timeseries/{}/combined/0.csv'.format(freq), sep=None, engine='python', usecols=[0]).Timestamp)
        p_df.index, p_df.columns = dt_index, sens_mat.columns
        v_df.index, v_df.columns = dt_index, sens_mat.index
        v_under.index, v_under.columns = dt_index, sens_mat.index
        v_over.index, v_over.columns = dt_index, sens_mat.index
        # p_hh_df.index = dt_index
        # p_ev_df.index = dt_index
        # p_hp_df.index = dt_index
        # p_pv_df.index = dt_index

        comb_profiles_num = len(
            [i for i in os.listdir('timeseries/{}/combined'.format(freq)) if i.split('.')[0].isnumeric()])
        cust_der_groups = pd.read_csv('timeseries/{}/combined/cust_der_groups.csv'.format(freq), index_col=[0],
                                      sep=None, engine='python')
        hh_ids = [int(i.split('.csv')[0]) for i in os.listdir('timeseries/{}/HH'.format(freq)) if
                  i.split('.csv')[0].isnumeric()]
        hh_ids = [i for i in hh_ids if i not in cust_der_groups['Household'].values]

        # p_under_delta_needed = v_viol[0] @ sens_mat_inv.values
        # p_over_delta_needed = v_viol[1] @ sens_mat_inv.values
        ph_appl_idx = pd.MultiIndex.from_product([[i.split('_')[-1] for i in sens_mat.columns], ['EV', 'HP', 'PV']],
                                                 names=['phase', 'appliance'])
        p_lim_df = pd.DataFrame(0, index=ph_appl_idx, columns=['P_min', 'P_max'])

        p_hh_df = pd.DataFrame(0, columns=p_df.columns, index=dt_index)
        p_ev_df = pd.DataFrame(0, columns=p_df.columns, index=dt_index)
        p_hp_df = pd.DataFrame(0, columns=p_df.columns, index=dt_index)
        p_pv_df = pd.DataFrame(0, columns=p_df.columns, index=dt_index)

        for node_id, node_df in ph_all_df.groupby('Node ID'):
            # node_df = ph_all_df.loc[ph_all_df['Node ID'] == node, :]

            for ph, phase_df in node_df.groupby(['Phase A', 'Phase B', 'Phase C']):
                ph_name = '{}{}'.format(node_id, [['a', 'b', 'c'][i] for i, p in enumerate(ph) if p][0])
                # flexible generators: PV
                p_lim_df.loc[(ph_name, 'PV'), 'P_min'] = -sum(phase_df.loc[phase_df['Appliance Name'] == 'PV', 'P_max'])
                p_lim_df.loc[(ph_name, 'PV'), 'P_max'] = p_pv_df.max()['P_' + ph_name]
                # flexible generators: EV, HP (HH is not!)
                p_lim_df.loc[(ph_name, 'EV'), 'P_max'] = sum(
                    phase_df.loc[phase_df['Appliance Name'].isin(['EV']), 'P_max'])
                p_lim_df.loc[(ph_name, 'HP'), 'P_max'] = sum(
                    phase_df.loc[phase_df['Appliance Name'].isin(['HP']), 'P_max'])

                cons_ts, gens_ts = pd.DataFrame(), pd.DataFrame()

                hh_ts = pd.DataFrame()
                ev_ts = pd.DataFrame()
                hp_ts = pd.DataFrame()
                pv_ts = pd.DataFrame()

                for cust_id, cust_df in phase_df.groupby('Customer ID'):
                    if cust_id < comb_profiles_num:
                        cust_ts = pd.read_csv('timeseries/{}/combined/{}.csv'.format(freq, cust_id), sep=None,
                                              engine='python', index_col=[0])
                    else:
                        cust_ts = pd.read_csv('timeseries/{}/HH/{}.csv'.format(freq, hh_ids[cust_id]), sep=None,
                                              engine='python', index_col=[0])
                    cust_ts.index = pd.to_datetime(cust_ts.index)
                    #     if not cust_ts.loc[:, cust_ts.columns.isin(['HH', 'EV', 'HP'])].empty:
                    #         cons_ts = pd.concat([cons_ts, cust_ts.loc[:, cust_ts.columns.isin(['HH', 'EV', 'HP'])]], axis=1)
                    #     if not cust_ts.loc[:, cust_ts.columns.isin(['PV'])].empty:
                    #         gens_ts = pd.concat([gens_ts, cust_ts.loc[:, cust_ts.columns.isin(['PV'])]], axis=1)
                    hh_ts = pd.concat([hh_ts, cust_ts.loc[:, cust_ts.columns.isin(['HH'])]], axis=1)
                    # hh_ts = pd.concat([hh_ts, cust_ts.loc[:, cust_ts.columns.isin(['PV'])].clip(lower=0)], axis=1)
                    ev_ts = pd.concat([ev_ts, cust_ts.loc[:, cust_ts.columns.isin(['EV'])]], axis=1)
                    hp_ts = pd.concat([hp_ts, cust_ts.loc[:, cust_ts.columns.isin(['HP'])]], axis=1)
                    pv_ts = pd.concat([pv_ts, cust_ts.loc[:, cust_ts.columns.isin(['PV'])]], axis=1)

                p_hh_df['P_' + ph_name] = hh_ts.sum(axis=1)
                p_ev_df['P_' + ph_name] = ev_ts.sum(axis=1)
                p_hp_df['P_' + ph_name] = hp_ts.sum(axis=1)
                p_pv_df['P_' + ph_name] = pv_ts.sum(axis=1).clip(upper=0)

        p_hh_df.to_csv(results_dir + 'p_hh_only.csv', sep=';')
        p_ev_df.to_csv(results_dir + 'p_ev_only.csv', sep=';')
        p_hp_df.to_csv(results_dir + 'p_hp_only.csv', sep=';')
        p_pv_df.to_csv(results_dir + 'p_pv_only.csv', sep=';')

        if opt_config:
            p_der_dfs = {
                'EV': p_ev_df,
                'HP': p_hp_df,
                'PV': p_pv_df
            }
            optimize_power_curtailments(p_df, p_hh_df, p_der_dfs, v_df, (v_under, v_over), p_lim_df, sens_mat,
                                        grid_id, freq, opt_config)

    pass


def optimize_power_curtailments(p_df, p_hh_df, p_der_dfs, v_df, v_viol, p_lim_df, sens_mat, grid_id, freq, opt_config):
    """
    Optimize power curtailments required for voltage correction based on linear programming (pyomo)

    :param p_df: (aggregated) active power profiles per node
    :param p_hh_df: household active power profiles per node
    :param p_der_dfs: DER active power profiles per node
    :param v_df: voltage profiles per node
    :param v_viol: under- and overvoltage profiles per node
    :param p_lim_df: active power limits per node
    :param sens_mat: sensitivity matrix
    :param grid_id: grid ID
    :param freq: sampling frequency
    :param opt_config: dict containing optimization settings
    :return:
    """
    print('\n\nOPTIMIZE POWER CURTAILMENTS:')
    t_lim = 6 * 24 * 1  # upper time limit up to which the profiles should be sliced
    v_under, v_over = v_viol[0], v_viol[1]
    t_segment = v_df.any(axis=1) | True
    t_segment = t_segment & (t_segment.reset_index().index < t_lim).T
    model = pyo.ConcreteModel('Sensitivity-Linearized OPF')

    # SETS
    model.nodes = range(p_df.shape[1])
    model.N = len(model.nodes)
    model.time = range(t_segment[t_segment].shape[0])
    model.T = len(model.time)
    model.DERs = range(len(opt_config['flex_DERs']))

    # PARAMETERS
    # sampling frequency
    model.freq = int(freq.split('min')[0])

    # voltage limits
    model.v_min = 400 / np.sqrt(3) * 0.95
    model.v_max = 400 / np.sqrt(3) * 1.05

    # active power limits
    model.p_min = p_lim_df['P_min']
    model.p_max = p_lim_df['P_max']

    # voltage timeseries
    model.v = v_df.loc[t_segment, :].values
    model.v_under = v_under.loc[t_segment, :].values
    model.v_over = v_over.loc[t_segment, :].values

    # total and household active power timeseries
    model.p_orig = p_df.loc[t_segment, :].values.round(3)
    model.p_hh = p_hh_df.loc[t_segment, :].values.round(3)

    # DER active power timeseries
    model.p_der_orig = {}
    for der in opt_config['flex_DERs']:
        model.p_der_orig[der] = p_der_dfs[der].loc[t_segment, :].values.round(3)

    # EV specific parameters
    ev_opt = opt_config['ev_opt']
    # ev_idx = opt_config['flex_DERs'].index('EV')
    model.E_ev_orig_day = p_der_dfs['EV'].loc[t_segment, :].groupby(by=p_der_dfs['EV'].index[t_segment].date).sum().mul(
        model.freq / 60)  # energy consumed per day by each EV
    model.days = range(model.E_ev_orig_day.shape[0])

    # option to consider the charging probability in the objective function
    if ev_opt['ch_probability']:
        # ch_prob CSV should have been created by running
        ch_prob = pd.read_csv('optimization/{}/ch_prob_{}.csv'.format(grid_id, freq), sep=';', index_col=[0])
        missing_ph = [ph[2:] for ph in p_df.columns if ph[2:] not in ch_prob.columns]
        ch_prob = pd.concat([ch_prob, pd.DataFrame(0.0, columns=missing_ph, index=ch_prob.index)], axis=1)
        ch_prob = ch_prob.reindex(columns=[ph[2:] for ph in p_df.columns])
        model.tot_days = p_df.shape[0] // ch_prob.shape[0]
        ch_prob = pd.concat([ch_prob] * model.tot_days)
        ch_prob.index = p_df.index
        model.ch_prob = ch_prob.loc[t_segment, :].values

    # sensitivity matrix
    model.S = sens_mat.values

    # cost of active power variation
    model.C_dp_up = np.ones(shape=(p_df.shape[1]))
    model.C_dp_down = -np.ones(shape=(p_df.shape[1]))

    # VARIABLES
    # DER active power variation
    model.dp_der_up = pyo.Var(model.DERs, model.time, model.nodes, domain=pyo.PositiveReals)
    model.dp_der_down = pyo.Var(model.DERs, model.time, model.nodes, domain=pyo.NegativeReals)
    model.p_der_new = pyo.Var(model.DERs, model.time, model.nodes, domain=pyo.Reals)

    # EV energy
    model.E_ev_opt_day = pyo.Var(model.time, model.nodes, domain=pyo.NonNegativeReals)

    # if voltage violation constraint should be relaxed, split voltage variation into two variables
    if opt_config['relax_v_con']:
        model.dv_up = pyo.Var(model.time, model.nodes, domain=pyo.NonNegativeReals)
        model.dv_down = pyo.Var(model.time, model.nodes, domain=pyo.NonPositiveReals)
    else:
        model.dv = pyo.Var(model.time, model.nodes, domain=pyo.Reals)

    # substation voltage variation
    model.dv_ss_up = pyo.Var(model.time, range(3), domain=pyo.NonNegativeReals)
    model.dv_ss_down = pyo.Var(model.time, range(3), domain=pyo.NonPositiveReals)

    t0 = time.time()

    # OBJECTIVE
    print('\tBuilding objective...')

    # cost of active power variations
    model.cost_dp = 0
    for der in model.DERs:
        # if 'PV' in opt_config['flex_DERs'] and der == opt_config['flex_DERs'].index('PV'):
        #     continue
        model.cost_dp += np.array(model.dp_der_up)[der, :, :].dot(model.C_dp_up).sum()
        model.cost_dp += np.array(model.dp_der_down)[der, :, :].dot(model.C_dp_down).sum()

    # cost of substation voltage variation
    model.cost_dv_ss = np.array(model.dv_ss_up).sum() - np.array(model.dv_ss_down).sum()

    # cost of charging at improbable times of the day
    if ev_opt['ch_probability']:
        model.cost_ev_ch = (
                    (1 - model.ch_prob) * np.array(model.p_der_new)[opt_config['flex_DERs'].index('EV'), :, :]).sum()
    else:
        model.cost_ev_ch = 0

    # cost of over- and undervoltages
    if opt_config['relax_v_con']:
        model.cost_vv = (
                + np.array(model.v_over)
                - np.array(model.v_under)
                + np.array(model.dv_down)
                - np.array(model.dv_up)
        ).sum()
    else:
        model.cost_vv = 0

    C_dp, C_ss, C_ev_ch, C_vv = opt_config['cost_factors'].values()

    model.cost = pyo.Objective(expr=
                               + C_dp * model.cost_dp
                               + C_ss * model.cost_dv_ss
                               + C_ev_ch * model.cost_ev_ch
                               + C_vv * model.cost_vv
                               , sense=pyo.minimize)

    t1 = time.time()
    print('\tTime passed for building objective: {:.2f}s'.format(t1 - t0))

    # CONSTRAINTS
    print('\tDefining constraints...')

    print('\t\tDER power limits...')

    # DER lower active power limits
    def p_der_lower_constraint(mdl, d, t, n):
        der_name = opt_config['flex_DERs'][d]
        if mdl.p_min[:, der_name][n] > mdl.p_der_orig[der_name][t, n]:
            raise
        return 1 * mdl.p_min[:, der_name][n] <= mdl.p_der_new[d, t, n]
        # return -100 <= mdl.p_der_new[d, t, n]

    # DER upper active power limits
    def p_der_upper_constraint(mdl, d, t, n):
        der_name = opt_config['flex_DERs'][d]
        if mdl.p_max[:, der_name][n] < mdl.p_der_orig[der_name][t, n]:
            raise
        return 1 * mdl.p_max[:, der_name][n] >= mdl.p_der_new[d, t, n]
        # return 100 >= mdl.p_der_new[d, t, n]

    def dp_pv_constraint(mdl, t, n):
        pv_idx = opt_config['flex_DERs'].index('PV')
        return mdl.dp_der_down[pv_idx, t, n] == 0

    def dp_ev_constraint(mdl, t, n):
        ev_idx = opt_config['flex_DERs'].index('EV')
        return mdl.dp_der_up[ev_idx, t, n] == 0

    # new active power value of DER after adding the variations
    def p_new_constraint(mdl, d, t, n):
        der_name = opt_config['flex_DERs'][d]
        if opt_config['rt_control']:
            if t == 0:
                return mdl.p_der_new[d, t, n] == mdl.p_der_orig[der_name][t, n]
            else:
                return mdl.p_der_new[d, t, n] == mdl.p_der_new[d, t - 1, n] + mdl.dp_der_up[d, t, n] + \
                       mdl.dp_der_down[d, t, n]
        else:
            return mdl.p_der_new[d, t, n] == mdl.p_der_orig[der_name][t, n] + mdl.dp_der_up[d, t, n] + \
                   mdl.dp_der_down[d, t, n]

    model.p_der_lower_con = pyo.Constraint(model.DERs, model.time, model.nodes, rule=p_der_lower_constraint)
    model.p_der_upper_con = pyo.Constraint(model.DERs, model.time, model.nodes, rule=p_der_upper_constraint)
    model.p_new_con = pyo.Constraint(model.DERs, model.time, model.nodes, rule=p_new_constraint)
    if 'PV' in opt_config['flex_DERs']:
        model.dp_pv_con = pyo.Constraint(model.time, model.nodes, rule=dp_pv_constraint)
    if 'EV' in opt_config['flex_DERs']:
        model.dp_ev_con = pyo.Constraint(model.time, model.nodes, rule=dp_ev_constraint)

    if ev_opt['E_deviation_con']:
        print('\t\tEV charging constraints...')

        p_opt = np.array(model.p_der_new)[opt_config['flex_DERs'].index('EV'), :, :]

        def E_ev_constraint(mdl, day, n):
            idx_day = p_df[t_segment].index.date == model.E_ev_orig_day.index[day]
            E_ev_opt_d = p_opt[idx_day, :].dot(model.freq / 60).sum(axis=0)
            return model.E_ev_opt_day[day, n] == E_ev_opt_d[n]

        def E_ev_deviation_lower_constraint(mdl, day, n):
            if mdl.E_ev_orig_day.iloc[day, n] == 0:
                return mdl.E_ev_orig_day.iloc[day, n] - mdl.E_ev_opt_day[day, n] == 0
            else:
                deviation = (mdl.E_ev_orig_day.iloc[day, n] - mdl.E_ev_opt_day[day, n]) / mdl.E_ev_orig_day.iloc[day, n]
                return deviation <= 0.02

        def E_ev_deviation_upper_constraint(mdl, day, n):
            if mdl.E_ev_orig_day.iloc[day, n] == 0:
                return mdl.E_ev_opt_day[day, n] - mdl.E_ev_orig_day.iloc[day, n] == 0
            else:
                deviation = (mdl.E_ev_opt_day[day, n] - mdl.E_ev_orig_day.iloc[day, n]) / mdl.E_ev_orig_day.iloc[day, n]
                return deviation <= 0.02

        # model.E_ev_con = pyo.Constraint(model.days, model.nodes, rule=E_ev_constraint)
        # model.E_ev_dev_lower_con = pyo.Constraint(model.days, model.nodes, rule=E_ev_deviation_lower_constraint)
        # model.E_ev_dev_upper_con = pyo.Constraint(model.days, model.nodes, rule=E_ev_deviation_upper_constraint)

    print('\t\tPower flow equations...')

    # voltage correction constraint (+ substation voltage control i.e. tap)
    def lin_pf_con(mdl, t, n):
        ph = n // (len(mdl.nodes) // 3)
        # ph = 0
        dv = mdl.dv_up[t, n] + mdl.dv_down[t, n] if opt_config['relax_v_con'] else mdl.dv[t, n]
        if n in [0, mdl.N // 3, mdl.N * 2 // 3]:  # substation voltage variation
            return mdl.dv[t, n] == mdl.dv_ss_up[t, ph] + mdl.dv_ss_down[t, ph]
        else:
            v_variation = sum(
                sum(mdl.p_der_new[d, t, m] - mdl.p_der_orig[opt_config['flex_DERs'][d]][t, m] for d in mdl.DERs
                    ) * mdl.S[n, m] for m in mdl.nodes)
            # v_variation = sum(
            #     sum(mdl.dp_der_up[d, t, m] + mdl.dp_der_down[d, t, m] for d in mdl.DERs
            #         ) * mdl.S[n, m] for m in mdl.nodes
            # )
            return mdl.dv[t, n] == v_variation + mdl.dv_ss_up[t, ph] + mdl.dv_ss_down[t, ph]
            # return dv == v_variation

    model.lin_pf_con = pyo.Constraint(model.time, model.nodes, rule=lin_pf_con)

    print('\t\tVoltage limits...')

    if not opt_config['relax_v_con']:
        def v_lower_constraint(mdl, t, n):  # lower voltage constraint
            return mdl.v_min <= mdl.v[t, n] + mdl.dv[t, n]

        def v_upper_constraint(mdl, t, n):  # upper voltage constraint
            return mdl.v[t, n] + mdl.dv[t, n] <= mdl.v_max
    else:
        def v_lower_constraint(mdl, t, n):  # lower voltage deviation constraint
            return -mdl.v_over[t, n] - (mdl.v_max - mdl.v_min) / 2 <= mdl.dv_down[t, n]

        def v_upper_constraint(mdl, t, n):  # upper voltage deviation constraint
            return mdl.dv_up[t, n] <= -mdl.v_under[t, n] + (mdl.v_max - mdl.v_min) / 2

    model.v_lower_con = pyo.Constraint(model.time, model.nodes, rule=v_lower_constraint)
    model.v_upper_con = pyo.Constraint(model.time, model.nodes, rule=v_upper_constraint)

    print('\tTime passed for constraint definition: {:.2f}s'.format(time.time() - t1))

    print('\tWriting to .lp input file...')
    # model.write(filename='optimization/{}/opt_input.lp'.format(grid_id))
    # model.display(filename='optimization/{}/opt_input.txt'.format(grid_id))

    print('\tSolving linear optimization problem...')
    ipopt_path = r'C:\Program Files\CoinOR\Ipopt\bin\ipopt.exe'
    glpk_path = r'C:\Solvers\glpk\glpsol.exe'
    cbc_path = r'C:\Solvers\cbc\bin\cbc.exe'
    solver = pyo.SolverFactory('ipopt', executable=ipopt_path)
    solver.options['max_iter'] = 3000
    results = solver.solve(model, tee=True)
    # results = pyo.SolverFactory('glpk', executable=glpk_path).solve(model, tee=True)
    # if results.solver.termination_condition != pyo.TerminationCondition.optimal:
    #     raise ValueError('Optimization infeasible!')

    # model.display('optimization/{}/opt_results.txt'.format(grid_id))

    print(
        '\tFinishing optimization...',
        '\n\t\tOptimization Objective Decomposition:',
        '\n\t\t\tCost Delta P: {:.2f}'.format(C_dp * pyo.value(model.cost_dp)),
        '\n\t\t\tCost Substation Voltage: {:.2f}'.format(C_ss * pyo.value(model.cost_dv_ss)),
        '\n\t\t\tCost EV Charging: {:.2f}'.format(C_ev_ch * pyo.value(model.cost_ev_ch)),
        '\n\t\t\tCost Voltage Violation: {:.2f}'.format(C_vv * pyo.value(model.cost_vv))
    )

    # POSTPROCESSING AND SAVING TO CSV FILES

    config_str = '-'.join(key for key, val in opt_config.items() if (val and key not in ['cost_factors', 'flex_DERs']))
    save_folder = 'optimization/{}/{}'.format(grid_id, datetime.now().strftime('%Y_%m_%d-%H_%M_%S-') + config_str)
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    if opt_config['relax_v_con']:
        dv_opt = (
                + np.reshape(pyo.value(model.dv_down[:, :]), (-1, model.N))
                + np.reshape(pyo.value(model.dv_up[:, :]), (-1, model.N))
        )
    else:
        dv_opt = np.reshape(pyo.value(model.dv[:, :]), (-1, model.N))
    v_corr = v_df.copy()
    v_corr.loc[t_segment, :] += dv_opt
    if not opt_config['ss_control']:
        dv_ss = (
                + np.repeat(np.reshape(pyo.value(model.dv_ss_down[:, :]), (-1, 3)), model.N // 3, axis=1)
                + np.repeat(np.reshape(pyo.value(model.dv_ss_up[:, :]), (-1, 3)), model.N // 3, axis=1)
        )
        v_corr.loc[t_segment, :] -= dv_ss
    v_df.to_csv('{}/v_orig.csv'.format(save_folder), sep=';')
    v_corr.to_csv('{}/v_corr.csv'.format(save_folder), sep=';')

    p_der_opt_dfs = {der: df.copy() for der, df in p_der_dfs.items()}
    p_opt = p_hh_df.copy()
    for der in p_der_dfs.keys():
        if der in opt_config['flex_DERs']:
            i = opt_config['flex_DERs'].index(der)
            dp_der_opt_up = np.reshape(pyo.value(model.dp_der_up[i, :, :]), (-1, model.N))
            dp_der_opt_down = np.reshape(pyo.value(model.dp_der_down[i, :, :]), (-1, model.N))
            p_der_new = np.reshape(pyo.value(model.p_der_new[i, :, :]), (-1, model.N))
            p_der_opt_dfs[der].loc[t_segment, :] = p_der_new
        p_opt += p_der_opt_dfs[der]
        p_der_opt_dfs[der].to_csv('{}/p_{}_corr.csv'.format(save_folder, der.lower()), sep=';')
        p_der_dfs[der].to_csv('{}/p_{}_orig.csv'.format(save_folder, der.lower()), sep=';')
    p_opt.to_csv('{}/p_corr.csv'.format(save_folder), sep=';')
    p_df.to_csv('{}/p_orig.csv'.format(save_folder), sep=';')

    with open('{}/opt_config.json'.format(save_folder), 'w+') as json_file:
        json.dump(opt_config, json_file)

    pass


def analyze_opt_results(opt_paths, kpi_df, plot_v_p=True, plot_der=True):
    """
    Analyzes the results of the optimization

    :param opt_paths: paths where optimization results are located
    :param plot_v_p: bool whether voltage and power comparison plots should be created or not
    :param plot_der: bool whether DER power plots should be created or not
    :return: [PNG] voltage and power comparison plots
    """
    print('\n\nANALYZE OPTIMIZATION RESULTS:')
    sl = 6 * 24 * 1  # upper time limit up to which the profiles should be sliced
    v_min = 400 / np.sqrt(3) * 0.95
    v_max = 400 / np.sqrt(3) * 1.05

    if kpi_df:
        print('\tCalculating KPIs...')
        kpi_df = pd.DataFrame(index=[re.findall(r'\d+', p)[0] for p in opt_paths])
        for idx, path in enumerate(opt_paths):
            if 'most_recent' in path:
                grid_id, path = find_most_recent_opt_results_dir(opt_paths[idx])
            else:
                grid_id = path.split('\\')[-1]
            freq = '10min'  # todo: generalize
            p_orig = pd.read_csv(path + '/p_orig.csv', sep=';', index_col=[0]).iloc[:sl, :]
            p_corr = pd.read_csv(path + '/p_corr.csv', sep=';', index_col=[0]).iloc[:sl, :]
            p_ev_orig = pd.read_csv(path + '/p_ev_orig.csv', sep=';', index_col=[0]).iloc[:sl, :]
            p_ev_corr = pd.read_csv(path + '/p_ev_corr.csv', sep=';', index_col=[0]).iloc[:sl, :]
            p_hp_orig = pd.read_csv(path + '/p_hp_orig.csv', sep=';', index_col=[0]).iloc[:sl, :]
            p_hp_corr = pd.read_csv(path + '/p_hp_corr.csv', sep=';', index_col=[0]).iloc[:sl, :]
            p_pv_orig = pd.read_csv(path + '/p_pv_orig.csv', sep=';', index_col=[0]).iloc[:sl, :]
            p_pv_corr = pd.read_csv(path + '/p_pv_corr.csv', sep=';', index_col=[0]).iloc[:sl, :]
            v_orig = pd.read_csv(path + '/v_orig.csv', sep=';', index_col=[0]).iloc[:sl, :]
            v_corr = pd.read_csv(path + '/v_corr.csv', sep=';', index_col=[0]).iloc[:sl, :]
            for df in [p_orig, p_corr, v_orig, v_corr]:
                df.index = pd.to_datetime(df.index)
            v_under_orig, v_over_orig = (v_orig - v_min).clip(upper=0), (v_orig - v_max).clip(lower=0)
            v_under_orig_sum = v_under_orig.sum().sum() * int(freq.split('min')[0]) / 60  # [Vh]
            v_over_orig_sum = v_over_orig.sum().sum() * int(freq.split('min')[0]) / 60  # [Vh]
            v_under_corr, v_over_corr = (v_corr - v_min).clip(upper=0), (v_corr - v_max).clip(lower=0)
            v_under_corr_sum = v_under_corr.sum().sum() * int(freq.split('min')[0]) / 60  # [Vh]
            v_over_corr_sum = v_over_corr.sum().sum() * int(freq.split('min')[0]) / 60  # [Vh]
            v_under_change = v_under_corr_sum / v_under_orig_sum
            v_over_change = v_over_corr_sum / v_over_orig_sum
            kWh_loss_ev = (p_ev_corr - p_ev_orig).sum().sum() * int(freq.split('min')[0]) / 60  # [kWh]
            kWh_loss_hp = (p_hp_corr - p_hp_orig).sum().sum() * int(freq.split('min')[0]) / 60  # [kWh]
            kWh_loss_pv = (p_pv_corr - p_pv_orig).sum().sum() * int(freq.split('min')[0]) / 60  # [kWh]
            kpi_df.loc[grid_id, 'v_under_orig'] = v_under_orig_sum
            kpi_df.loc[grid_id, 'v_under_corr'] = v_under_corr_sum
            kpi_df.loc[grid_id, 'v_under_change'] = v_under_change
            kpi_df.loc[grid_id, 'v_over_orig'] = v_over_orig_sum
            kpi_df.loc[grid_id, 'v_over_corr'] = v_over_corr_sum
            kpi_df.loc[grid_id, 'v_over_change'] = v_over_change
            kpi_df.loc[grid_id, 'kWh_loss_ev'] = kWh_loss_ev
            kpi_df.loc[grid_id, 'kWh_loss_hp'] = kWh_loss_hp
            kpi_df.loc[grid_id, 'kWh_loss_pv'] = kWh_loss_pv
            # kpi_df.round(2).to_csv(path + '/kpi.csv', sep=';')
        print(kpi_df.T.round(2).to_latex())

    if plot_v_p:
        print('\tPlotting nodal voltage/power profiles...')
        cm = plt.cm.Paired(range(4))
        for idx, path in enumerate(opt_paths):
            # config_str = '-'.join(
            # key for key, val in opt_config.items() if (val and key not in ['cost_factors', 'flex_DERs']))
            # save_folder = 'docu/optimization/{}/{}'.format(grid_id,
            #                                                datetime.now().strftime('%Y_%m_%d-%H_%M_%S-') + config_str)
            if 'most_recent' in path:
                grid_id, path = find_most_recent_opt_results_dir(opt_paths[idx])
            else:
                grid_id = path.split('\\')[-1]
            p_orig = pd.read_csv(path + '/p_orig.csv', sep=';', index_col=[0]).iloc[:sl, :]
            p_corr = pd.read_csv(path + '/p_corr.csv', sep=';', index_col=[0]).iloc[:sl, :]
            v_orig = pd.read_csv(path + '/v_orig.csv', sep=';', index_col=[0]).iloc[:sl, :]
            v_corr = pd.read_csv(path + '/v_corr.csv', sep=';', index_col=[0]).iloc[:sl, :]
            for df in [p_orig, p_corr, v_orig, v_corr]:
                df.index = pd.to_datetime(df.index)
            for n, ph in enumerate(v_orig):
                p_diff = ((p_corr.iloc[:, n] - p_orig.iloc[:, n]) ** 2).sum()
                v_diff = ((v_corr[ph] - v_orig[ph]) ** 2).sum()
                if p_diff > 3 or v_diff > 3:
                    node_str = str(int(re.findall(r'\d+', ph)[0])) + ph[-1]
                    fig, ax = plt.subplots(2, 1, figsize=(10, 5))
                    plt.suptitle('Power and voltage values after optimization, node {}'.format(node_str))

                    ax[0].set_xlabel('Time')
                    ax[0].set_ylabel('Power consumption [kW]')
                    ax[0].plot(p_orig.iloc[:, n], '--', c=cm[0])
                    ax[0].plot(p_corr.iloc[:, n], c=cm[1])
                    ax[0].legend(['Original', 'Optimized'])
                    ax[0].grid('both')

                    ax[1].set_xlabel('Time')
                    ax[1].set_ylabel('Voltage [V]')
                    ax[1].plot(v_orig.iloc[:, n], '--', c=cm[2])
                    ax[1].plot(v_corr.iloc[:, n], c=cm[3])
                    ax[1].axhline(y=v_min, color='tab:gray', linestyle='--')
                    ax[1].axhline(y=v_max, color='tab:gray', linestyle='--')
                    ax[1].legend(['Original', 'Optimized'])
                    ax[1].grid('both')

                    plt.tight_layout()

                    save_folder = 'docu/' + path

                    if not os.path.exists(save_folder):
                        os.mkdir(save_folder)
                    tikzplotlib.save('{}/{}.tex'.format(save_folder, node_str))
                    plt.savefig('{}/{}.png'.format(save_folder, node_str))
                    plt.close(fig)

                    # fig, ax = plt.subplots(2, 1, figsize=(10, 5))
                    # plt.suptitle('Power and voltage values after optimization, node {}'.format(node_str))
                    #
                    # ax[0].set_xlabel('Time')
                    # ax[0].set_ylabel('Power consumption [kW]')
                    # ax[0].plot(p_orig.sum(axis=1), '--', c=cm[0])
                    # ax[0].plot(p_corr.sum(axis=1), c=cm[1])
                    # ax[0].legend(['Original', 'Optimized'])
                    # ax[0].grid('both')
                    #
                    # ax[1].set_xlabel('Time')
                    # ax[1].set_ylabel('Voltage [V]')
                    # ax[1].plot(v_orig.mean(axis=1), '--', c=cm[2])
                    # ax[1].plot(v_corr.mean(axis=1), c=cm[3])
                    # ax[1].axhline(y=v_min, color='tab:gray', linestyle='--')
                    # ax[1].axhline(y=v_max, color='tab:gray', linestyle='--')
                    # ax[1].axhspan(v_max, ax[1].get_ylim()[1], ax[1].get_xlim()[0], ax[1].get_xlim()[0], color='red',
                    #               alpha=0.5)
                    # ax[1].axhspan(ax[1].get_ylim()[0], v_min, ax[1].get_xlim()[0], ax[1].get_xlim()[0], color='red',
                    #               alpha=0.5)
                    # ax[1].legend(['Original', 'Optimized'])
                    # ax[1].grid('both')
                    #
                    # plt.tight_layout()
                    #
                    # save_folder = 'docu/' + path
                    #
                    # if not os.path.exists(save_folder):
                    #     os.mkdir(save_folder)
                    # plt.show()

    if plot_der:
        print('\tPlotting DER profiles...')
        for path in opt_paths:
            if 'most_recent' in path:
                grid_id, path = find_most_recent_opt_results_dir(path)
            else:
                grid_id = path.split('\\')[-1]
            save_folder = 'docu/' + path
            if not os.path.exists(save_folder):
                os.mkdir(save_folder)
            cm = plt.cm.Paired.colors
            # ch_ev_opt = pd.read_csv(path + '/ch_ev.csv', sep=';', index_col=[0]).iloc[:sl, :]
            # sw_on = pd.read_csv(path + '/sw_on.csv', sep=';', index_col=[0]).iloc[:sl, :]
            # sw_off = pd.read_csv(path + '/sw_off.csv', sep=';', index_col=[0]).iloc[:sl, :]
            for i, der in enumerate(['EV', 'HP', 'PV']):
                p_orig = pd.read_csv(path + '/p_{}_orig.csv'.format(der.lower()), sep=';', index_col=[0]).iloc[:sl, :]
                p_corr = pd.read_csv(path + '/p_{}_corr.csv'.format(der.lower()), sep=';', index_col=[0]).iloc[:sl, :]
                for df in [p_orig, p_corr]:
                    df.index = pd.to_datetime(df.index)
                for n, ph in enumerate(p_corr):
                    p_diff = ((p_corr.iloc[:, n] - p_orig.iloc[:, n]) ** 2).sum()
                    if p_diff > 1:
                        node_str = str(int(re.findall(r'\d+', ph)[0])) + ph[-1]
                        fig = plt.figure(figsize=(10, 5))
                        if der == 'EV':
                            # ax = fig.add_subplot(311)
                            ax = fig.add_subplot(111)
                        else:
                            ax = fig.add_subplot(111)
                        plt.suptitle('{} power values after optimization, node {}'.format(der, node_str))

                        ax.set_xlabel('Time')
                        ax.set_ylabel('Power consumption [kW]')
                        ax.plot(p_orig.iloc[:, n], '--', c=cm[2 * i])
                        ax.plot(p_corr.iloc[:, n], c=cm[2 * i + 1])
                        ax.legend(['Original', 'Optimized'])
                        ax.grid('both')

                        # if der == 'EV':
                        #     ch_ev_orig = p_orig > 0.1
                        #     ch_ev_opt.index = ch_ev_orig.index
                        #     sw_on.index = ch_ev_orig.index
                        #     sw_off.index = ch_ev_orig.index
                        #     ax_ch = fig.add_subplot(312)
                        #     ax_ch.set_xlabel('Time')
                        #     ax_ch.set_ylabel('on/off')
                        #     ax_ch.plot(ch_ev_orig.iloc[:, n], '--', c='tab:grey')
                        #     ax_ch.plot(ch_ev_opt.iloc[:, n], c='k')
                        #     ax_ch.legend(['Original', 'Optimized'])
                        #
                        #     ax_sw = fig.add_subplot(313)
                        #     ax_sw.set_xlabel('Time')
                        #     ax_sw.set_ylabel('toggle')
                        #     ax_sw.plot(sw_on.iloc[:, n], c='g')
                        #     ax_sw.plot(sw_off.iloc[:, n], c='r')
                        #     ax_sw.legend(['On', 'Off'])

                        plt.tight_layout()
                        # plt.show()

                        if not os.path.exists('{}/{}'.format(save_folder, der)):
                            os.mkdir('{}/{}'.format(save_folder, der))
                        tikzplotlib.save('{}/{}/{}.tex'.format(save_folder, der, node_str))
                        plt.savefig('{}/{}/{}.png'.format(save_folder, der, node_str))
                        plt.close(fig)
    pass


def find_most_recent_opt_results_dir(opt_path):
    grid_id = re.findall(r'\d+', opt_path)[0]
    dirs = os.listdir('optimization/{}/'.format(grid_id))
    dirs = [d for d in dirs if (bool(re.search(r'\d', d)) and 'ch_prob' not in d)]
    dirs_dt = [datetime.strptime(d[:19], '%Y_%m_%d-%H_%M_%S') for d in dirs]
    dirs_td = [datetime.now() - dt for dt in dirs_dt]
    return grid_id, 'optimization/{}/'.format(grid_id) + dirs[dirs_td.index(min(dirs_td))]


def compute_ev_charging_probabilities(phase_allocation_path, ch_prob, plot, csv):
    """
    Computes the probability of every EV charging at specific times of the day based on the historical measurements

    :param phase_allocation_path:
    :param ch_prob:
    :param plot:
    :param csv:
    :return:
    """
    print('\n\nCOMPUTE EV CHARGING PROBABILITIES:')
    for idx, _ in enumerate(phase_allocation_path):
        freq = phase_allocation_path[idx].split('/')[-2]
        grid_id = phase_allocation_path[idx].split('/')[-3]
        ph_all_df = pd.read_csv(phase_allocation_path[idx], sep=None, engine='python')
        ph_all_df = ph_all_df[ph_all_df['Appliance Name'] == 'EV']
        ch_prob_df = pd.DataFrame()
        for node_id, node_df in ph_all_df.groupby('Node ID'):
            for ph, phase_df in node_df.groupby(['Phase A', 'Phase B', 'Phase C']):
                ph_name = '{}{}'.format(node_id, [['a', 'b', 'c'][i] for i, p in enumerate(ph) if p][0])
                for cust_id, _ in phase_df.groupby('Customer ID'):
                    ev_ts = pd.read_csv('timeseries/{}/combined/{}.csv'.format(freq, cust_id), sep=None,
                                        engine='python', index_col=[0])['EV']
                    ev_ts.index = pd.to_datetime(ev_ts.index)
                    n_days = len(np.unique([ev_ts.index.time]))
                    ev_ts_reshaped = ev_ts.values.reshape(-1, len(ev_ts) // n_days, order='F')
                    if ch_prob:
                        ev_ts_normalized = (ev_ts_reshaped > 0.01).sum(axis=1) / ev_ts_reshaped.shape[1]
                        ev_ts_normalized_sel = np.copy(ev_ts_normalized)
                        ev_ts_normalized_sel[ev_ts_normalized < ev_ts_normalized.mean()] = 0
                        ch_prob_df[ph_name] = ev_ts_normalized_sel
                    else:
                        ev_ts_normalized = ev_ts_reshaped.sum(axis=1) / ev_ts.sum()
                        ev_ts_normalized_sel = np.copy(ev_ts_normalized)
                        ev_ts_normalized_sel[ev_ts_normalized < ev_ts_normalized.mean()] = 0
                    if plot:
                        fig = plt.figure(figsize=(11, 5))
                        plt.bar(np.arange(0, 24, 24 / n_days), ev_ts_normalized)
                        # plt.fill_between(np.arange(0, 24, 24 / n_days), 0, ev_ts_normalized)
                        plt.bar(np.arange(0, 24, 24 / n_days), ev_ts_normalized_sel)
                        # plt.fill_between(np.arange(0, 24, 24 / n_days), 0, ev_ts_normalized_sel)
                        plt.title('Grid ID: {}, EV charging probability at node {}'.format(grid_id, ph_name))
                        plt.xlabel('Hours')
                        plt.ylabel('Probability')
                        if not os.path.exists('docu/ev_cons_hist/{}'.format(grid_id)):
                            os.makedirs('docu/ev_cons_hist/{}'.format(grid_id))
                        plt.savefig('docu/ev_cons_hist/{}/{}.png'.format(grid_id, ph_name))
                        plt.close(fig)
        if csv:
            if not os.path.exists('optimization/{}'.format(grid_id)):
                os.makedirs('optimization/{}'.format(grid_id))
            ch_prob_df.to_csv('optimization/{}/ch_prob_{}.csv'.format(grid_id, freq), sep=';')
