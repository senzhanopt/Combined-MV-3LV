import gurobipy as gp
from gurobipy import GRB, quicksum
import pandas as pd
import numpy as np
from numpy import inf
import math

nw_mv_rural_cable = pd.read_excel("mv_rural.xlsx", sheet_name="cable")
nw_lv_rural1_cable = pd.read_excel("lv_rural1.xlsx", sheet_name="cable")
nw_lv_rural2_cable = pd.read_excel("lv_rural2.xlsx", sheet_name="cable")
nw_lv_rural3_cable = pd.read_excel("lv_rural3.xlsx", sheet_name="cable")

non_linear = False
print_gurobi = False
print_status = False
lin_acce = True
newton_acce = False

def jain(x): # Jain's fairness index
    x = np.array(x)
    return (np.mean(x))**2/np.mean(x*x) # if np.mean(x*x) > 0.01 else 0

def opf_mv(dict_input_mv, dict_opt_lv1, dict_fea_lv1, dict_opt_lv2, dict_fea_lv2, dict_opt_lv3, dict_fea_lv3, dict_newton, objective):
    '''
    optimize MV nodes using cuts from LV grids
    '''
    #------------------------  Load Profiles ----------------------------------
    # sgen capacity
    cap_sgen_mv = dict_input_mv["cap_sgen_mv"]
    # load: P, Q
    arr_mv_p = dict_input_mv["arr_mv_p"]
    arr_mv_q = dict_input_mv["arr_mv_q"]
    # sgen: P
    arr_mv_sgen = dict_input_mv["arr_mv_sgen"]
    P_trafo_forecast = dict_input_mv["P_trafo_forecast"]
    #------------------------- Parameters -------------------------------------
    # grid topology
    n_bus_mv = 12 # 4~15
    n_cable_mv = 12 # 4~15
    n_trafo = 3
    # voltage
    v_mv = 20/np.sqrt(3)
    v_lb_mv = 0.965
    v_ub_mv = 1.055
    # power factor
    limit_pf = 0.95
    limit_tan_phi = np.sqrt(1-limit_pf**2)/limit_pf 
    #------------------------- Variables- -------------------------------------
    env = gp.Env(empty=True)
    env.setParam("OutputFlag",0)
    env.start()
    if print_gurobi:
        m = gp.Model("opf_mv")
    else:
        m = gp.Model("opf_mv", env = env) 
    m.Params.OptimalityTol = 1e-8
    # voltage magnitude squared
    v2_mv = m.addVars(n_bus_mv+1, lb = (v_mv*v_lb_mv)**2, ub = (v_mv*v_ub_mv)**2, name = "MV_bus_volt2") # of 13
    # sgen real power
    p_sgen_mv = m.addVars(n_bus_mv, lb = -GRB.INFINITY, name = "MV_P_sgen") # of 12
    # sgen reactive power
    q_sgen_mv = m.addVars(n_bus_mv, lb = -GRB.INFINITY, name = "MV_Q_sgen") # of 12
    # real power injection
    p_mv = m.addVars(n_bus_mv, lb = -GRB.INFINITY, name = "MV_bus_P_injection") # of 12
    # reactive power injection
    q_mv = m.addVars(n_bus_mv, lb = -GRB.INFINITY, name = "MV_bus_Q_injection") # of 12
    # real power flow
    P_mv = m.addVars(n_cable_mv, lb = -GRB.INFINITY, name = "MV_cable_Pflow") # length of 12
    # reactive power flow
    Q_mv = m.addVars(n_cable_mv, lb = -GRB.INFINITY, name = "MV_cable_Qflow") # length of 12
    # transformer power flow
    P_trafo_mv = m.addVars(n_trafo, lb = -GRB.INFINITY, name = "Trafo_Pflow_MV") # flow out MV
    Q_trafo_mv = m.addVars(n_trafo, lb = -GRB.INFINITY, name = "Trafo_Qflow_MV") # flow out MV    
    alpha_lv = m.addVars(n_trafo, name = "auxillaryVars") # auxillary variables for LV costs
    #------------------------- Constraints MV ---------------------------------
    m.addConstr(v2_mv[0] == v_mv**2) # slack bus, index = #bus - 3
    # node balance
    m.addConstrs(p_mv[i] == p_sgen_mv[i] - arr_mv_p[i] for i in range(n_bus_mv))
    m.addConstrs(q_mv[i] == q_sgen_mv[i] - arr_mv_q[i] for i in range(n_bus_mv))
    # sgen constraints
    m.addConstrs(p_sgen_mv[i] <= arr_mv_sgen[i] for i in range(n_bus_mv))
    m.addConstrs(q_sgen_mv[i] <=  limit_tan_phi * p_sgen_mv[i] for i in range(n_bus_mv))
    m.addConstrs(q_sgen_mv[i] >= -limit_tan_phi * p_sgen_mv[i] for i in range(n_bus_mv))
    if non_linear:
        m.addConstrs(1e3*p_sgen_mv[i]*p_sgen_mv[i]+1e3*q_sgen_mv[i]*q_sgen_mv[i] <= 1e3*(cap_sgen_mv[i])**2 for i in range(n_bus_mv))
    else:
        for i in range(n_bus_mv):
            n_pts = 36
            for p in range(n_pts):
                x0 = cap_sgen_mv[i] * math.cos(p*math.pi*2/n_pts)
                y0 = cap_sgen_mv[i] * math.sin(p*math.pi*2/n_pts)
                x1 = cap_sgen_mv[i] * math.cos((p+1)*math.pi*2/n_pts)
                y1 = cap_sgen_mv[i] * math.sin((p+1)*math.pi*2/n_pts)
                if -x0*y1 >= -x1*y0: # to decide which side (0,0) locates in
                    m.addConstr((y1-y0)*(p_sgen_mv[i]-x0) >= (x1-x0)*(q_sgen_mv[i]-y0))
                else:
                    m.addConstr((y1-y0)*(p_sgen_mv[i]-x0) <= (x1-x0)*(q_sgen_mv[i]-y0))
    # current law
    for b in range(4,16):
        if b not in [4,8,12]: # LV not attached
            P_out = 0
            Q_out = 0
            for l in range(n_cable_mv):
                if nw_mv_rural_cable.from_bus[l] == b:
                    P_out += P_mv[l]
                    Q_out += Q_mv[l]
            m.addConstr(P_mv[b-4] + p_mv[b-4] == P_out)
            m.addConstr(Q_mv[b-4] + q_mv[b-4] == Q_out)
        else: # LV attached
            P_out = 0
            Q_out = 0
            for l in range(n_cable_mv):
                if nw_mv_rural_cable.from_bus[l] == b:
                    P_out += P_mv[l]
                    Q_out += Q_mv[l]
            m.addConstr(P_mv[b-4] == P_out + P_trafo_mv[int(b/4-1)]) # happen to be 4,8,12
            m.addConstr(Q_mv[b-4] == Q_out + Q_trafo_mv[int(b/4-1)])  
    # voltage law
    for l in range(n_cable_mv):
        from_bus = max(nw_mv_rural_cable.from_bus[l]-3,0)
        to_bus = nw_mv_rural_cable.to_bus[l]-3
        r_ohm = nw_mv_rural_cable.length_km[l]*nw_mv_rural_cable.r_ohm_per_km[l]
        x_ohm = nw_mv_rural_cable.length_km[l]*nw_mv_rural_cable.x_ohm_per_km[l]
        m.addConstr(v2_mv[to_bus] == v2_mv[from_bus] - 2*(r_ohm*P_mv[l]+x_ohm*Q_mv[l]))
        s_max = nw_mv_rural_cable.max_i_ka[l] * v_mv
        # cable capacity limit
        if non_linear:
            m.addConstr(1e3*P_mv[l]*P_mv[l]+1e3*Q_mv[l]*Q_mv[l] <= 1e3*s_max**2)  
        else:
            n_pts = 36
            for p in range(n_pts):
                x0 = s_max * math.cos(p*math.pi*2/n_pts)
                y0 = s_max * math.sin(p*math.pi*2/n_pts)
                x1 = s_max * math.cos((p+1)*math.pi*2/n_pts)
                y1 = s_max * math.sin((p+1)*math.pi*2/n_pts)
                if -x0*y1 >= -x1*y0: # to decide which side (0,0) locates in
                    m.addConstr((y1-y0)*(P_mv[l]-x0) >= (x1-x0)*(Q_mv[l]-y0))
                else:
                    m.addConstr((y1-y0)*(P_mv[l]-x0) <= (x1-x0)*(Q_mv[l]-y0))
    # transformer capacity limit
    if non_linear:
        m.addConstrs(P_trafo_mv[t]*P_trafo_mv[t]+Q_trafo_mv[t]*Q_trafo_mv[t] <= [(0.16/3)**2,(0.63/3)**2,(0.63/3)**2][t] for t in range(n_trafo)) 
    else:
        for t in range(n_trafo):
            s_max = [(0.16/3),(0.63/3),(0.63/3)][t]
            n_pts = 36
            for p in range(n_pts):
                x0 = s_max * math.cos(p*math.pi*2/n_pts)
                y0 = s_max * math.sin(p*math.pi*2/n_pts)
                x1 = s_max * math.cos((p+1)*math.pi*2/n_pts)
                y1 = s_max * math.sin((p+1)*math.pi*2/n_pts)
                if -x0*y1 >= -x1*y0: # to decide which side (0,0) locates in
                    m.addConstr((y1-y0)*(P_trafo_mv[t]-x0) >= (x1-x0)*(Q_trafo_mv[t]-y0))
                else:
                    m.addConstr((y1-y0)*(P_trafo_mv[t]-x0) <= (x1-x0)*(Q_trafo_mv[t]-y0))
            
        
    # ----------------- optimality cuts --------------------------------------
    """
    For linear objective function, the relation between LV objective function and boundary variables are clearly known.
    """
    if objective == "obj_lin" and lin_acce == True:
        m.addConstrs(alpha_lv[i] == P_trafo_mv[i] - P_trafo_forecast[i] for i in range(n_trafo))
    else:
        # LV1
        for i in range(len(dict_opt_lv1["objVal"])):
            m.addConstr(alpha_lv[0] >= dict_opt_lv1["objVal"][i]
                        + (dict_opt_lv1["dual_P"][i]-dict_opt_lv1["dual_Pn"][i])*(dict_opt_lv1["P_trafo_lv"][i] - P_trafo_mv[0])
                        + (dict_opt_lv1["dual_Q"][i]-dict_opt_lv1["dual_Qn"][i])*(dict_opt_lv1["Q_trafo_lv"][i] - Q_trafo_mv[0])
                        + (dict_opt_lv1["dual_v2"][i]-dict_opt_lv1["dual_v2n"][i])*(dict_opt_lv1["v2_trafo"][i] - v2_mv[1])
                        )
        # LV2
        for i in range(len(dict_opt_lv2["objVal"])):
            m.addConstr(alpha_lv[1] >= dict_opt_lv2["objVal"][i]
                        + (dict_opt_lv2["dual_P"][i]-dict_opt_lv2["dual_Pn"][i])*(dict_opt_lv2["P_trafo_lv"][i] - P_trafo_mv[1])
                        + (dict_opt_lv2["dual_Q"][i]-dict_opt_lv2["dual_Qn"][i])*(dict_opt_lv2["Q_trafo_lv"][i] - Q_trafo_mv[1])
                        + (dict_opt_lv2["dual_v2"][i]-dict_opt_lv2["dual_v2n"][i])*(dict_opt_lv2["v2_trafo"][i] - v2_mv[5])
                        )
        # LV3
        for i in range(len(dict_opt_lv3["objVal"])):
            m.addConstr(alpha_lv[2] >= dict_opt_lv3["objVal"][i]
                        + (dict_opt_lv3["dual_P"][i]-dict_opt_lv3["dual_Pn"][i])*(dict_opt_lv3["P_trafo_lv"][i] - P_trafo_mv[2])
                        + (dict_opt_lv3["dual_Q"][i]-dict_opt_lv3["dual_Qn"][i])*(dict_opt_lv3["Q_trafo_lv"][i] - Q_trafo_mv[2])
                        + (dict_opt_lv3["dual_v2"][i]-dict_opt_lv3["dual_v2n"][i])*(dict_opt_lv3["v2_trafo"][i] - v2_mv[9])
                        )
    if objective == "obj_prop" and newton_acce == True:
        mat_B = dict_newton["mat_B"] # for three networks, 3 * (3*3)
        arr_c = dict_newton["arr_c"] # 3 * (3*1)
        num_q = dict_newton["num_q"] # 3 
        M = mat_B.copy()
        for i in range(3):
            w,v = np.linalg.eig(M[i])
            if w.min() <= 1e-5 and (M[i] != np.zeros((3,3))).any():
                M[i] = M[i] + (1e-5-w.min()) * np.identity(3) 
            # print(M[i])
            # print(w)
        m.addConstr(alpha_lv[0] >= 0.5 * quicksum(M[0][i,j]
                                        * [P_trafo_mv[0],Q_trafo_mv[0],v2_mv[1]][i]
                                        * [P_trafo_mv[0],Q_trafo_mv[0],v2_mv[1]][j]
                                            for i in range(3) for j in range(3))
                                        + quicksum(arr_c[0][i,0] * [P_trafo_mv[0],Q_trafo_mv[0],v2_mv[1]][i] for i in range(3))
                                        + num_q[0])
        m.addConstr(alpha_lv[1] >= 0.5 * quicksum(M[1][i,j]
                                        * [P_trafo_mv[1],Q_trafo_mv[1],v2_mv[5]][i]
                                        * [P_trafo_mv[1],Q_trafo_mv[1],v2_mv[5]][j]
                                            for i in range(3) for j in range(3))
                                        + quicksum(arr_c[1][i,0] * [P_trafo_mv[1],Q_trafo_mv[1],v2_mv[5]][i] for i in range(3))
                                        + num_q[1])
        m.addConstr(alpha_lv[2] >= 0.5 * quicksum(M[2][i,j]
                                        * [P_trafo_mv[2],Q_trafo_mv[2],v2_mv[9]][i]
                                        * [P_trafo_mv[2],Q_trafo_mv[2],v2_mv[9]][j]
                                            for i in range(3) for j in range(3))
                                        + quicksum(arr_c[2][i,0] * [P_trafo_mv[2],Q_trafo_mv[2],v2_mv[9]][i] for i in range(3))
                                        + num_q[2])
    # ----------------- feasibility cuts -------------------------------------
    for i in range(len(dict_fea_lv1["dual_P"])):
        m.addConstr((dict_fea_lv1["dual_P"][i]-dict_fea_lv1["dual_Pn"][i])*(dict_fea_lv1["P_trafo_lv"][i] - P_trafo_mv[0]) 
                    + (dict_fea_lv1["dual_Q"][i]-dict_fea_lv1["dual_Qn"][i])*(dict_fea_lv1["Q_trafo_lv"][i] - Q_trafo_mv[0])
                    + (dict_fea_lv1["dual_v2"][i]-dict_fea_lv1["dual_v2n"][i])*(dict_fea_lv1["v2_trafo"][i] - v2_mv[1]) <= 0
            )
    for i in range(len(dict_fea_lv2["dual_P"])):
        m.addConstr((dict_fea_lv2["dual_P"][i]-dict_fea_lv2["dual_Pn"][i])*(dict_fea_lv2["P_trafo_lv"][i] - P_trafo_mv[1]) 
                    + (dict_fea_lv2["dual_Q"][i]-dict_fea_lv2["dual_Qn"][i])*(dict_fea_lv2["Q_trafo_lv"][i] - Q_trafo_mv[1])
                    + (dict_fea_lv2["dual_v2"][i]-dict_fea_lv2["dual_v2n"][i])*(dict_fea_lv2["v2_trafo"][i] - v2_mv[5]) <= 0
            )
    for i in range(len(dict_fea_lv3["dual_P"])):
        m.addConstr((dict_fea_lv3["dual_P"][i]-dict_fea_lv3["dual_Pn"][i])*(dict_fea_lv3["P_trafo_lv"][i] - P_trafo_mv[2]) 
                    + (dict_fea_lv3["dual_Q"][i]-dict_fea_lv3["dual_Qn"][i])*(dict_fea_lv3["Q_trafo_lv"][i] - Q_trafo_mv[2])
                    + (dict_fea_lv3["dual_v2"][i]-dict_fea_lv3["dual_v2n"][i])*(dict_fea_lv3["v2_trafo"][i] - v2_mv[9]) <= 0
            )
        
    # ----------------- objective function -----------------------------------
    np.seterr(divide='ignore')
    arr_sgen_mv_inv = 1/arr_mv_sgen
    arr_sgen_mv_inv[arr_sgen_mv_inv == inf] = 0    
    obj_prop = quicksum((arr_mv_sgen[i]-p_sgen_mv[i])*(arr_mv_sgen[i]-p_sgen_mv[i])*(arr_sgen_mv_inv[i]) for i in range(n_bus_mv)) + quicksum(alpha_lv)
    obj_quad = quicksum((arr_mv_sgen[i]-p_sgen_mv[i])*(arr_mv_sgen[i]-p_sgen_mv[i]) for i in range(n_bus_mv)) + quicksum(alpha_lv)
    obj_lin = quicksum((arr_mv_sgen[i]-p_sgen_mv[i]) for i in range(n_bus_mv)) + quicksum(alpha_lv)
    if objective == "obj_lin":
        m.setObjective(obj_lin, GRB.MINIMIZE)
    elif objective == "obj_quad":
        m.setObjective(obj_quad, GRB.MINIMIZE)
    elif objective == "obj_prop":
        m.setObjective(obj_prop, GRB.MINIMIZE)
    else:
        print("Objective function error.")
        return 0
    m.optimize()
    dict_output = {}
    dict_output["status"] = m.status
    dict_output["objVal"] = m.objVal
    dict_output["objVal_mv"] = m.objVal - sum(alpha_lv[i].X for i in range(n_trafo))
    dict_output["P_trafo_mv"] = np.array([P_trafo_mv[i].X for i in range(n_trafo)])
    dict_output["Q_trafo_mv"] = np.array([Q_trafo_mv[i].X for i in range(n_trafo)])
    dict_output["v2_trafo_mv"] = np.array([v2_mv[1].X, v2_mv[5].X, v2_mv[9].X])
    dict_output["p_sgen_mv"] = np.array([p_sgen_mv[i].X for i in range(n_bus_mv)])
    dict_output["q_sgen_mv"] = np.array([q_sgen_mv[i].X for i in range(n_bus_mv)]) 
    dict_output["vm_pu_mv"] = np.array([np.sqrt(v2_mv[i].X)/v_mv for i in range(n_bus_mv+1)])
    dict_output["loading_line_mv"] = 100*np.array([np.sqrt((P_mv[l].X)**2+(Q_mv[l].X)**2)/(nw_mv_rural_cable.max_i_ka[l] * v_mv) for l in range(n_cable_mv)])
    return dict_output
    
def opf_lv1(dict_input_lv1, v2_trafo_mv, P_trafo_mv, Q_trafo_mv, objective):
    '''
    They can be solved together because of no mutual impact
    '''
    #------------------------  Load Profiles ----------------------------------
    # sgen capacity
    cap_sgen_lv1 = dict_input_lv1["cap_sgen_lv1"]
    # load: P, Q
    arr_lv1_p = dict_input_lv1["arr_lv1_p"]
    arr_lv1_q = dict_input_lv1["arr_lv1_q"]
    # sgen: P
    arr_lv1_sgen = dict_input_lv1["arr_lv1_sgen"]   
    #------------------------- Parameters -------------------------------------
    # grid topology
    n_bus_lv1_vec = 14
    # voltage
    v_mv = 20/np.sqrt(3)
    v_lv = 0.4/np.sqrt(3)
    v_lb_lv = 0.9
    v_ub_lv = 1.1
    # power factor
    limit_pf = 0.95
    limit_tan_phi = np.sqrt(1-limit_pf**2)/limit_pf 
    # transformer
    # https://pandapower.readthedocs.io/en/develop/elements/trafo.html
    turns_ratio = v_lv/v_mv
    r_ohm_160kva = 1.46875/100*1/0.16*0.4**2*1/0.16
    x_ohm_160kva = np.sqrt(4**2-1.46875**2)/100*1/0.16*0.4**2*1/0.16    
    #------------------------- Variables- -------------------------------------
    env = gp.Env(empty=True)
    env.setParam("OutputFlag",0)
    env.start()
    if print_gurobi:
        m = gp.Model("opf_lv1")
    else:
        m = gp.Model("opf_lv1", env = env)
    if non_linear == False:
        m.Params.OptimalityTol = 1E-8
    m.Params.numericFocus = 3
    # voltage magnitude squared
    v2_lv1 = m.addVars(n_bus_lv1_vec, lb = (v_lv*v_lb_lv)**2, ub = (v_lv*v_ub_lv)**2, name = "LV1_bus_volt2") # of 14
    # sgen real power
    p_sgen_lv1 = m.addVars(n_bus_lv1_vec, lb = -GRB.INFINITY, name = "LV1_P_sgen") # of 14
    # sgen reactive power
    q_sgen_lv1 = m.addVars(n_bus_lv1_vec, lb = -GRB.INFINITY, name = "LV1_Q_sgen") # of 14
    # real power injection
    p_lv1 = m.addVars(n_bus_lv1_vec, lb = -GRB.INFINITY, name = "LV1_bus_P_injection") # of 14
    # reactive power injection
    q_lv1 = m.addVars(n_bus_lv1_vec, lb = -GRB.INFINITY, name = "LV1_bus_Q_injection") # of 14
    # real power flow
    P_lv1 = m.addVars(n_bus_lv1_vec-1, lb = -GRB.INFINITY, name = "LV1_cable_Pflow") # length of 13
    # reactive power flow
    Q_lv1 = m.addVars(n_bus_lv1_vec-1, lb = -GRB.INFINITY, name = "LV1_cable_Qflow") # length of 137
    # transformer power flow
    P_trafo_lv = m.addVar(lb = -GRB.INFINITY, name = "Trafo_Pflow_LV") # flow in LV
    Q_trafo_lv = m.addVar(lb = -GRB.INFINITY, name = "Trafo_Qflow_LV") # flow in LV
    v2_trafo = m.addVar(lb = -GRB.INFINITY, name = "Trafo_volt2_mvside") # flow in LV    
    #------------------------- Constraints LV1 --------------------------------
    # node balance
    m.addConstrs(p_lv1[i] == p_sgen_lv1[i] - arr_lv1_p[i] for i in range(n_bus_lv1_vec))
    m.addConstrs(q_lv1[i] == q_sgen_lv1[i] - arr_lv1_q[i] for i in range(n_bus_lv1_vec))
    # sgen constraints
    m.addConstrs(p_sgen_lv1[i] <= arr_lv1_sgen[i] for i in range(n_bus_lv1_vec))
    m.addConstrs(q_sgen_lv1[i] <=  limit_tan_phi * p_sgen_lv1[i] for i in range(n_bus_lv1_vec))
    m.addConstrs(q_sgen_lv1[i] >= -limit_tan_phi * p_sgen_lv1[i] for i in range(n_bus_lv1_vec))  
    if non_linear:
        m.addConstrs(1e3*p_sgen_lv1[i]*p_sgen_lv1[i]+1e3*q_sgen_lv1[i]*q_sgen_lv1[i] <= 1e3*(cap_sgen_lv1[i])**2 for i in range(n_bus_lv1_vec))
    else:
        for i in range(n_bus_lv1_vec):
            n_pts = 36
            for p in range(n_pts):
                x0 = cap_sgen_lv1[i] * math.cos(p*math.pi*2/n_pts)
                y0 = cap_sgen_lv1[i] * math.sin(p*math.pi*2/n_pts)
                x1 = cap_sgen_lv1[i] * math.cos((p+1)*math.pi*2/n_pts)
                y1 = cap_sgen_lv1[i] * math.sin((p+1)*math.pi*2/n_pts)
                if -x0*y1 >= -x1*y0: # to decide which side (0,0) locates in
                    m.addConstr((y1-y0)*(p_sgen_lv1[i]-x0) >= (x1-x0)*(q_sgen_lv1[i]-y0))
                else:
                    m.addConstr((y1-y0)*(p_sgen_lv1[i]-x0) <= (x1-x0)*(q_sgen_lv1[i]-y0))
    # current law
    for b in range(n_bus_lv1_vec):
        if b == 3: # LV side of transformer
            P_out = 0
            Q_out = 0
            for l in range(n_bus_lv1_vec-1):
                if nw_lv_rural1_cable.from_bus[l] == b:
                    P_out += P_lv1[l]
                    Q_out += Q_lv1[l]
            m.addConstr(P_trafo_lv + p_lv1[b] == P_out)
            m.addConstr(Q_trafo_lv + q_lv1[b] == Q_out)
        else:
            P_out = 0
            Q_out = 0
            P_in = 0
            Q_in = 0
            for l in range(n_bus_lv1_vec-1):
                if nw_lv_rural1_cable.from_bus[l] == b:
                    P_out += P_lv1[l]
                    Q_out += Q_lv1[l]
                if nw_lv_rural1_cable.to_bus[l] == b:
                    P_in += P_lv1[l]
                    Q_in += Q_lv1[l]
            m.addConstr(P_in + p_lv1[b] == P_out)
            m.addConstr(Q_in + q_lv1[b] == Q_out)           
    # voltage law
    for l in range(n_bus_lv1_vec-1):
        from_bus = nw_lv_rural1_cable.from_bus[l]
        to_bus = nw_lv_rural1_cable.to_bus[l]
        r_ohm = nw_lv_rural1_cable.length_km[l]*nw_lv_rural1_cable.r_ohm_per_km[l]
        x_ohm = nw_lv_rural1_cable.length_km[l]*nw_lv_rural1_cable.x_ohm_per_km[l]
        m.addConstr(v2_lv1[to_bus] == v2_lv1[from_bus] - 2*(r_ohm*P_lv1[l]+x_ohm*Q_lv1[l]))
        s_max = nw_lv_rural1_cable.max_i_ka[l] * v_lv
        # cable capacity limit
        if non_linear:
            m.addConstr(1e6*P_lv1[l]*P_lv1[l]+1e6*Q_lv1[l]*Q_lv1[l] <= 1e6*s_max**2) 
        else:
            n_pts = 36
            for p in range(n_pts):
                x0 = s_max * math.cos(p*math.pi*2/n_pts)
                y0 = s_max * math.sin(p*math.pi*2/n_pts)
                x1 = s_max * math.cos((p+1)*math.pi*2/n_pts)
                y1 = s_max * math.sin((p+1)*math.pi*2/n_pts)
                if -x0*y1 >= -x1*y0: # to decide which side (0,0) locates in
                    m.addConstr((y1-y0)*(P_lv1[l]-x0) >= (x1-x0)*(Q_lv1[l]-y0))
                else:
                    m.addConstr((y1-y0)*(P_lv1[l]-x0) <= (x1-x0)*(Q_lv1[l]-y0))     
    #------------------------- Constraints Trafos -----------------------------
    # power flow balance
    # constraints named to retrive duals
    cons_P = m.addConstr(P_trafo_lv <= P_trafo_mv)
    cons_Pn = m.addConstr(-P_trafo_lv <= -P_trafo_mv)
    cons_Q = m.addConstr(Q_trafo_lv <= Q_trafo_mv)
    cons_Qn = m.addConstr(-Q_trafo_lv <= -Q_trafo_mv)
    cons_v2 = m.addConstr(v2_trafo <= v2_trafo_mv)
    cons_v2n = m.addConstr(-v2_trafo <= -v2_trafo_mv)  
    # voltage relations: for easier Benders decomposition here
    m.addConstr(v2_trafo*turns_ratio**2-v2_lv1[3] == 2*(r_ohm_160kva*P_trafo_lv+x_ohm_160kva*Q_trafo_lv))     
    # ----------------------- objective function -----------------------------  
    obj_quad = quicksum((arr_lv1_sgen[i]-p_sgen_lv1[i])*(arr_lv1_sgen[i]-p_sgen_lv1[i]) for i in range(n_bus_lv1_vec)) 
    obj_lin = quicksum((arr_lv1_sgen[i]-p_sgen_lv1[i]) for i in range(n_bus_lv1_vec)) 
    np.seterr(divide='ignore')
    arr_sgen_lv1_inv = 1/arr_lv1_sgen
    arr_sgen_lv1_inv[arr_sgen_lv1_inv == inf] = 0
    obj_prop = quicksum((arr_lv1_sgen[i]-p_sgen_lv1[i])*(arr_lv1_sgen[i]-p_sgen_lv1[i])*(arr_sgen_lv1_inv[i]) for i in range(n_bus_lv1_vec))
    if objective == "obj_lin":
        m.setObjective(obj_lin, GRB.MINIMIZE)
    elif objective == "obj_quad":
        m.setObjective(obj_quad, GRB.MINIMIZE)
    elif objective == "obj_prop":
        m.setObjective(obj_prop, GRB.MINIMIZE)
    else:
        print("Objective function error.")
        return 0
    m.optimize()
    # ------------------------ output ----------------------------------------
    dict_output = {}
    dict_output["status"] = m.status
    if m.status == 2: # 2 for optimal and 3 for infeasible
        dict_output["p_sgen_lv1"] = np.array([p_sgen_lv1[i].X for i in range(n_bus_lv1_vec)])
        dict_output["q_sgen_lv1"] = np.array([q_sgen_lv1[i].X for i in range(n_bus_lv1_vec)])  
        dict_output["vm_pu_lv1"] = np.array([np.sqrt(v2_lv1[i].X)/v_lv for i in range(n_bus_lv1_vec)])
        dict_output["P_trafo_lv"] = P_trafo_lv.X
        dict_output["Q_trafo_lv"] = Q_trafo_lv.X
        dict_output["v2_trafo"] = v2_trafo.X
        dict_output["loading_trafo"] = 100 * np.sqrt((P_trafo_lv.X)**2 + (Q_trafo_lv.X)**2)/(0.16/3)
        dict_output["loading_line_lv1"] = 100*np.array([np.sqrt((P_lv1[l].X)**2+(Q_lv1[l].X)**2)/(nw_lv_rural1_cable.max_i_ka[l] * v_lv) for l in range(n_bus_lv1_vec-1)])
        dict_output["dual_P"] = abs(cons_P.Pi)
        dict_output["dual_Pn"] = abs(cons_Pn.Pi)
        dict_output["dual_Q"] = abs(cons_Q.Pi)
        dict_output["dual_Qn"] = abs(cons_Qn.Pi)
        dict_output["dual_v2"] = abs(cons_v2.Pi)
        dict_output["dual_v2n"] = abs(cons_v2n.Pi)
        dict_output["objVal"] = m.objVal
    return dict_output

def opf_lv1_fea(dict_input_lv1, v2_trafo_mv, P_trafo_mv, Q_trafo_mv):
    '''
    They can be solved together because of no mutual impact
    '''
    #------------------------  Load Profiles ----------------------------------
    # sgen capacity
    cap_sgen_lv1 = dict_input_lv1["cap_sgen_lv1"]
    # load: P, Q
    arr_lv1_p = dict_input_lv1["arr_lv1_p"]
    arr_lv1_q = dict_input_lv1["arr_lv1_q"]
    # sgen: P
    arr_lv1_sgen = dict_input_lv1["arr_lv1_sgen"]   
    #------------------------- Parameters -------------------------------------
    # grid topology
    n_bus_lv1_vec = 14
    # voltage
    v_mv = 20/np.sqrt(3)
    v_lv = 0.4/np.sqrt(3)
    v_lb_lv = 0.9
    v_ub_lv = 1.1
    # power factor
    limit_pf = 0.95
    limit_tan_phi = np.sqrt(1-limit_pf**2)/limit_pf 
    # transformer
    # https://pandapower.readthedocs.io/en/develop/elements/trafo.html
    turns_ratio = v_lv/v_mv
    r_ohm_160kva = 1.46875/100*1/0.16*0.4**2*1/0.16
    x_ohm_160kva = np.sqrt(4**2-1.46875**2)/100*1/0.16*0.4**2*1/0.16    
    #------------------------- Variables- -------------------------------------
    env = gp.Env(empty=True)
    env.setParam("OutputFlag",0)
    env.start()
    if print_gurobi:
        m = gp.Model("opf_lv1")
    else:
        m = gp.Model("opf_lv1", env = env)
    if non_linear == False:
        m.Params.OptimalityTol = 1E-8
    m.Params.numericFocus = 3
    # voltage magnitude squared
    v2_lv1 = m.addVars(n_bus_lv1_vec, lb = (v_lv*v_lb_lv)**2, ub = (v_lv*v_ub_lv)**2, name = "LV1_bus_volt2") # of 14
    # sgen real power
    p_sgen_lv1 = m.addVars(n_bus_lv1_vec, lb = -GRB.INFINITY, name = "LV1_P_sgen") # of 14
    # sgen reactive power
    q_sgen_lv1 = m.addVars(n_bus_lv1_vec, lb = -GRB.INFINITY, name = "LV1_Q_sgen") # of 14
    # real power injection
    p_lv1 = m.addVars(n_bus_lv1_vec, lb = -GRB.INFINITY, name = "LV1_bus_P_injection") # of 14
    # reactive power injection
    q_lv1 = m.addVars(n_bus_lv1_vec, lb = -GRB.INFINITY, name = "LV1_bus_Q_injection") # of 14
    # real power flow
    P_lv1 = m.addVars(n_bus_lv1_vec-1, lb = -GRB.INFINITY, name = "LV1_cable_Pflow") # length of 13
    # reactive power flow
    Q_lv1 = m.addVars(n_bus_lv1_vec-1, lb = -GRB.INFINITY, name = "LV1_cable_Qflow") # length of 137
    # transformer power flow
    P_trafo_lv = m.addVar(lb = -GRB.INFINITY, name = "Trafo_Pflow_LV") # flow in LV
    Q_trafo_lv = m.addVar(lb = -GRB.INFINITY, name = "Trafo_Qflow_LV") # flow in LV
    v2_trafo = m.addVar(lb = -GRB.INFINITY, name = "Trafo_volt2_mvside") # flow in LV  
    beta = m.addVar(name = "beta")
    #------------------------- Constraints LV1 --------------------------------
    # node balance
    m.addConstrs(p_lv1[i] == p_sgen_lv1[i] - arr_lv1_p[i] for i in range(n_bus_lv1_vec))
    m.addConstrs(q_lv1[i] == q_sgen_lv1[i] - arr_lv1_q[i] for i in range(n_bus_lv1_vec))
    # sgen constraints
    m.addConstrs(p_sgen_lv1[i] <= arr_lv1_sgen[i] for i in range(n_bus_lv1_vec))
    m.addConstrs(q_sgen_lv1[i] <=  limit_tan_phi * p_sgen_lv1[i] for i in range(n_bus_lv1_vec))
    m.addConstrs(q_sgen_lv1[i] >= -limit_tan_phi * p_sgen_lv1[i] for i in range(n_bus_lv1_vec))  
    if non_linear:
        m.addConstrs(1e3*p_sgen_lv1[i]*p_sgen_lv1[i]+1e3*q_sgen_lv1[i]*q_sgen_lv1[i] <= 1e3*(cap_sgen_lv1[i])**2 for i in range(n_bus_lv1_vec))
    else:
        for i in range(n_bus_lv1_vec):
            n_pts = 36
            for p in range(n_pts):
                x0 = cap_sgen_lv1[i] * math.cos(p*math.pi*2/n_pts)
                y0 = cap_sgen_lv1[i] * math.sin(p*math.pi*2/n_pts)
                x1 = cap_sgen_lv1[i] * math.cos((p+1)*math.pi*2/n_pts)
                y1 = cap_sgen_lv1[i] * math.sin((p+1)*math.pi*2/n_pts)
                if -x0*y1 >= -x1*y0: # to decide which side (0,0) locates in
                    m.addConstr((y1-y0)*(p_sgen_lv1[i]-x0) >= (x1-x0)*(q_sgen_lv1[i]-y0))
                else:
                    m.addConstr((y1-y0)*(p_sgen_lv1[i]-x0) <= (x1-x0)*(q_sgen_lv1[i]-y0))  
    # current law
    for b in range(n_bus_lv1_vec):
        if b == 3: # LV side of transformer
            P_out = 0
            Q_out = 0
            for l in range(n_bus_lv1_vec-1):
                if nw_lv_rural1_cable.from_bus[l] == b:
                    P_out += P_lv1[l]
                    Q_out += Q_lv1[l]
            m.addConstr(P_trafo_lv + p_lv1[b] == P_out)
            m.addConstr(Q_trafo_lv + q_lv1[b] == Q_out)
        else:
            P_out = 0
            Q_out = 0
            P_in = 0
            Q_in = 0
            for l in range(n_bus_lv1_vec-1):
                if nw_lv_rural1_cable.from_bus[l] == b:
                    P_out += P_lv1[l]
                    Q_out += Q_lv1[l]
                if nw_lv_rural1_cable.to_bus[l] == b:
                    P_in += P_lv1[l]
                    Q_in += Q_lv1[l]
            m.addConstr(P_in + p_lv1[b] == P_out)
            m.addConstr(Q_in + q_lv1[b] == Q_out)           
    # voltage law
    for l in range(n_bus_lv1_vec-1):
        from_bus = nw_lv_rural1_cable.from_bus[l]
        to_bus = nw_lv_rural1_cable.to_bus[l]
        r_ohm = nw_lv_rural1_cable.length_km[l]*nw_lv_rural1_cable.r_ohm_per_km[l]
        x_ohm = nw_lv_rural1_cable.length_km[l]*nw_lv_rural1_cable.x_ohm_per_km[l]
        m.addConstr(v2_lv1[to_bus] == v2_lv1[from_bus] - 2*(r_ohm*P_lv1[l]+x_ohm*Q_lv1[l]))
        s_max = nw_lv_rural1_cable.max_i_ka[l] * v_lv
        # cable capacity limit
        if non_linear:
            m.addConstr(1e6*P_lv1[l]*P_lv1[l]+1e6*Q_lv1[l]*Q_lv1[l] <= 1e6*s_max**2) 
        else:
            n_pts = 36
            for p in range(n_pts):
                x0 = s_max * math.cos(p*math.pi*2/n_pts)
                y0 = s_max * math.sin(p*math.pi*2/n_pts)
                x1 = s_max * math.cos((p+1)*math.pi*2/n_pts)
                y1 = s_max * math.sin((p+1)*math.pi*2/n_pts)
                if -x0*y1 >= -x1*y0: # to decide which side (0,0) locates in
                    m.addConstr((y1-y0)*(P_lv1[l]-x0) >= (x1-x0)*(Q_lv1[l]-y0))
                else:
                    m.addConstr((y1-y0)*(P_lv1[l]-x0) <= (x1-x0)*(Q_lv1[l]-y0))    
    #------------------------- Constraints Trafos -----------------------------
    # power flow balance
    # constraints named to retrive duals
    cons_P = m.addConstr(P_trafo_lv <= P_trafo_mv + beta)
    cons_Pn = m.addConstr(-P_trafo_lv <= -P_trafo_mv + beta)
    cons_Q = m.addConstr(Q_trafo_lv <= Q_trafo_mv + beta)
    cons_Qn = m.addConstr(-Q_trafo_lv <= -Q_trafo_mv + beta)
    cons_v2 = m.addConstr(v2_trafo <= v2_trafo_mv + beta)
    cons_v2n = m.addConstr(-v2_trafo <= -v2_trafo_mv + beta)  
    # voltage relations: for easier Benders decomposition here
    m.addConstr(v2_trafo*turns_ratio**2-v2_lv1[3] == 2*(r_ohm_160kva*P_trafo_lv+x_ohm_160kva*Q_trafo_lv))     
    # ----------------------- objective function -----------------------------  
    m.setObjective(beta, GRB.MINIMIZE)
    m.optimize()
    # ------------------------ output ----------------------------------------
    dict_output = {}
    dict_output["status"] = m.status
    dict_output["objVal"] = m.objVal
    dict_output["p_sgen_lv1"] = np.array([p_sgen_lv1[i].X for i in range(n_bus_lv1_vec)])
    dict_output["q_sgen_lv1"] = np.array([q_sgen_lv1[i].X for i in range(n_bus_lv1_vec)])  
    dict_output["vm_pu_lv1"] = np.array([np.sqrt(v2_lv1[i].X)/v_lv for i in range(n_bus_lv1_vec)])
    dict_output["P_trafo_lv"] = P_trafo_lv.X
    dict_output["Q_trafo_lv"] = Q_trafo_lv.X
    dict_output["v2_trafo"] = v2_trafo.X
    dict_output["loading_trafo"] = 100 * np.sqrt((P_trafo_lv.X)**2 + (Q_trafo_lv.X)**2)/(0.16/3)
    dict_output["loading_line_lv1"] = 100*np.array([np.sqrt((P_lv1[l].X)**2+(Q_lv1[l].X)**2)/(nw_lv_rural1_cable.max_i_ka[l] * v_lv) for l in range(n_bus_lv1_vec-1)])
    dict_output["dual_P"] = abs(cons_P.Pi)
    dict_output["dual_Pn"] = abs(cons_Pn.Pi)
    dict_output["dual_Q"] = abs(cons_Q.Pi)
    dict_output["dual_Qn"] = abs(cons_Qn.Pi)
    dict_output["dual_v2"] = abs(cons_v2.Pi)
    dict_output["dual_v2n"] = abs(cons_v2n.Pi)
    return dict_output

def opf_lv2(dict_input_lv2, v2_trafo_mv, P_trafo_mv, Q_trafo_mv, objective):
    '''
    They can be solved together because of no mutual impact
    '''
    #------------------------  Load Profiles ----------------------------------
    # sgen capacity
    cap_sgen_lv2 = dict_input_lv2["cap_sgen_lv2"]
    # load: P, Q
    arr_lv2_p = dict_input_lv2["arr_lv2_p"]
    arr_lv2_q = dict_input_lv2["arr_lv2_q"]
    # sgen: P
    arr_lv2_sgen = dict_input_lv2["arr_lv2_sgen"]   
    #------------------------- Parameters -------------------------------------
    # grid topology
    n_bus_lv2_vec = 96
    # voltage
    v_mv = 20/np.sqrt(3)
    v_lv = 0.4/np.sqrt(3)
    v_lb_lv = 0.9
    v_ub_lv = 1.1
    # power factor
    limit_pf = 0.95
    limit_tan_phi = np.sqrt(1-limit_pf**2)/limit_pf 
    # transformer
    # https://pandapower.readthedocs.io/en/develop/elements/trafo.html
    turns_ratio = v_lv/v_mv
    r_ohm_630kva = 1.206/100*1/0.63*0.4**2*1/0.63
    x_ohm_630kva = np.sqrt(6**2-1.206**2)/100*1/0.63*0.4**2*1/0.63 
    #------------------------- Variables- -------------------------------------
    env = gp.Env(empty=True)
    env.setParam("OutputFlag",0)
    env.start()
    if print_gurobi:
        m = gp.Model("opf_lv2")
    else:
        m = gp.Model("opf_lv2", env = env)
    if non_linear == False:
        m.Params.OptimalityTol = 1E-8
    m.Params.numericFocus = 3
    # voltage magnitude squared
    v2_lv2 = m.addVars(n_bus_lv2_vec, lb = (v_lv*v_lb_lv)**2, ub = (v_lv*v_ub_lv)**2, name = "LV2_bus_volt2") # of 96
    # sgen real power
    p_sgen_lv2 = m.addVars(n_bus_lv2_vec, lb = -GRB.INFINITY, name = "LV2_P_sgen") # of 96
    # sgen reactive power
    q_sgen_lv2 = m.addVars(n_bus_lv2_vec, lb = -GRB.INFINITY, name = "LV2_Q_sgen") # of 96
    # real power injection
    p_lv2 = m.addVars(n_bus_lv2_vec, lb = -GRB.INFINITY, name = "LV2_bus_P_injection") # of 96
    # reactive power injection
    q_lv2 = m.addVars(n_bus_lv2_vec, lb = -GRB.INFINITY, name = "LV2_bus_Q_injection") # of 96
    # real power flow
    P_lv2 = m.addVars(n_bus_lv2_vec-1, lb = -GRB.INFINITY, name = "LV2_cable_Pflow") # length of 95
    # reactive power flow
    Q_lv2 = m.addVars(n_bus_lv2_vec-1, lb = -GRB.INFINITY, name = "LV2_cable_Qflow") # length of 95
    # transformer power flow
    P_trafo_lv = m.addVar(lb = -GRB.INFINITY, name = "Trafo_Pflow_LV") # flow in LV
    Q_trafo_lv = m.addVar(lb = -GRB.INFINITY, name = "Trafo_Qflow_LV") # flow in LV
    v2_trafo = m.addVar(lb = -GRB.INFINITY, name = "Trafo_volt2_mvside") # flow in LV       
    #------------------------- Constraints LV2 --------------------------------
    # node balance
    m.addConstrs(p_lv2[i] == p_sgen_lv2[i] - arr_lv2_p[i] for i in range(n_bus_lv2_vec))
    m.addConstrs(q_lv2[i] == q_sgen_lv2[i] - arr_lv2_q[i] for i in range(n_bus_lv2_vec))
    # sgen constraints
    m.addConstrs(p_sgen_lv2[i] <= arr_lv2_sgen[i] for i in range(n_bus_lv2_vec))
    m.addConstrs(q_sgen_lv2[i] <=  limit_tan_phi * p_sgen_lv2[i] for i in range(n_bus_lv2_vec))
    m.addConstrs(q_sgen_lv2[i] >= - limit_tan_phi * p_sgen_lv2[i] for i in range(n_bus_lv2_vec))  
    if non_linear:
        m.addConstrs(1e6*p_sgen_lv2[i]*p_sgen_lv2[i]+1e6*q_sgen_lv2[i]*q_sgen_lv2[i] <= 1e6*(cap_sgen_lv2[i])**2 for i in range(n_bus_lv2_vec))  
    else:
        for i in range(n_bus_lv2_vec):
            n_pts = 36
            for p in range(n_pts):
                x0 = cap_sgen_lv2[i] * math.cos(p*math.pi*2/n_pts)
                y0 = cap_sgen_lv2[i] * math.sin(p*math.pi*2/n_pts)
                x1 = cap_sgen_lv2[i] * math.cos((p+1)*math.pi*2/n_pts)
                y1 = cap_sgen_lv2[i] * math.sin((p+1)*math.pi*2/n_pts)
                if -x0*y1 >= -x1*y0: # to decide which side (0,0) locates in
                    m.addConstr((y1-y0)*(p_sgen_lv2[i]-x0) >= (x1-x0)*(q_sgen_lv2[i]-y0))
                else:
                    m.addConstr((y1-y0)*(p_sgen_lv2[i]-x0) <= (x1-x0)*(q_sgen_lv2[i]-y0))
    # current law
    for b in range(n_bus_lv2_vec):
        if b == 62: # LV side of transformer
            P_out = 0
            Q_out = 0
            for l in range(n_bus_lv2_vec-1):
                if nw_lv_rural2_cable.from_bus[l] == b:
                    P_out += P_lv2[l]
                    Q_out += Q_lv2[l]
            m.addConstr(P_trafo_lv + p_lv2[b] == P_out)
            m.addConstr(Q_trafo_lv + q_lv2[b] == Q_out)
        else:
            P_out = 0
            Q_out = 0
            P_in = 0
            Q_in = 0
            for l in range(n_bus_lv2_vec-1):
                if nw_lv_rural2_cable.from_bus[l] == b:
                    P_out += P_lv2[l]
                    Q_out += Q_lv2[l]
                if nw_lv_rural2_cable.to_bus[l] == b:
                    P_in += P_lv2[l]
                    Q_in += Q_lv2[l]
            m.addConstr(P_in + p_lv2[b] == P_out)
            m.addConstr(Q_in + q_lv2[b] == Q_out)           
    # voltage law
    for l in range(n_bus_lv2_vec-1):
        from_bus = nw_lv_rural2_cable.from_bus[l]
        to_bus = nw_lv_rural2_cable.to_bus[l]
        r_ohm = nw_lv_rural2_cable.length_km[l]*nw_lv_rural2_cable.r_ohm_per_km[l]
        x_ohm = nw_lv_rural2_cable.length_km[l]*nw_lv_rural2_cable.x_ohm_per_km[l]
        m.addConstr(1e2*v2_lv2[to_bus] == 1e2*v2_lv2[from_bus] - 1e2*2*(r_ohm*P_lv2[l]+x_ohm*Q_lv2[l]))
        s_max = nw_lv_rural2_cable.max_i_ka[l] * v_lv
        # cable capacity limit
        if non_linear:
            m.addConstr(1e6*P_lv2[l]*P_lv2[l]+1e6*Q_lv2[l]*Q_lv2[l] <= 1e6*s_max**2) 
        else:
            n_pts = 36
            for p in range(n_pts):
                x0 = s_max * math.cos(p*math.pi*2/n_pts)
                y0 = s_max * math.sin(p*math.pi*2/n_pts)
                x1 = s_max * math.cos((p+1)*math.pi*2/n_pts)
                y1 = s_max * math.sin((p+1)*math.pi*2/n_pts)
                if -x0*y1 >= -x1*y0: # to decide which side (0,0) locates in
                    m.addConstr((y1-y0)*(P_lv2[l]-x0) >= (x1-x0)*(Q_lv2[l]-y0))
                else:
                    m.addConstr((y1-y0)*(P_lv2[l]-x0) <= (x1-x0)*(Q_lv2[l]-y0))
            
    #------------------------- Constraints Trafos -----------------------------
    # power flow balance
    # constraints named to retrive duals
    cons_P = m.addConstr(P_trafo_lv <= P_trafo_mv)
    cons_Pn = m.addConstr(-P_trafo_lv <= -P_trafo_mv)
    cons_Q = m.addConstr(Q_trafo_lv <= Q_trafo_mv)
    cons_Qn = m.addConstr(-Q_trafo_lv <= -Q_trafo_mv)
    cons_v2 = m.addConstr(v2_trafo <= v2_trafo_mv)
    cons_v2n = m.addConstr(-v2_trafo <= -v2_trafo_mv) 
    # voltage relations: for easier Benders decomposition here
    m.addConstr(1e1*v2_trafo*turns_ratio**2-1e1*v2_lv2[62] ==1e1*2*(r_ohm_630kva*P_trafo_lv+x_ohm_630kva*Q_trafo_lv))       
    # ----------------------- objective function -----------------------------  
    obj_quad = quicksum((arr_lv2_sgen[i]-p_sgen_lv2[i])*(arr_lv2_sgen[i]-p_sgen_lv2[i]) for i in range(n_bus_lv2_vec))
    obj_lin = quicksum((arr_lv2_sgen[i]-p_sgen_lv2[i]) for i in range(n_bus_lv2_vec))
    np.seterr(divide='ignore')
    arr_sgen_lv2_inv = 1/arr_lv2_sgen
    arr_sgen_lv2_inv[arr_sgen_lv2_inv == inf] = 0
    obj_prop = quicksum((arr_lv2_sgen[i]-p_sgen_lv2[i])*(arr_lv2_sgen[i]-p_sgen_lv2[i])*(arr_sgen_lv2_inv[i]) for i in range(n_bus_lv2_vec))        
    if objective == "obj_lin":
        m.setObjective(obj_lin, GRB.MINIMIZE)
    elif objective == "obj_quad":
        m.setObjective(obj_quad, GRB.MINIMIZE)
    elif objective == "obj_prop":
        m.setObjective(obj_prop, GRB.MINIMIZE)
    else:
        print("Objective function error.")
        return 0
    m.optimize()
    # ------------------------ output ----------------------------------------
    dict_output = {}
    dict_output["status"] = m.status
    if m.status == 2:
        dict_output["p_sgen_lv2"] = np.array([p_sgen_lv2[i].X for i in range(n_bus_lv2_vec)])
        dict_output["q_sgen_lv2"] = np.array([q_sgen_lv2[i].X for i in range(n_bus_lv2_vec)]) 
        dict_output["vm_pu_lv2"] = np.array([np.sqrt(v2_lv2[i].X)/v_lv for i in range(n_bus_lv2_vec)])
        dict_output["P_trafo_lv"] = P_trafo_lv.X 
        dict_output["Q_trafo_lv"] = Q_trafo_lv.X
        dict_output["v2_trafo"] = v2_trafo.X
        dict_output["loading_trafo"] = 100*np.sqrt((P_trafo_lv.X)**2 + (Q_trafo_lv.X)**2)/0.21
        dict_output["loading_line_lv2"] = 100*np.array([np.sqrt((P_lv2[l].X)**2+(Q_lv2[l].X)**2)/(nw_lv_rural2_cable.max_i_ka[l] * v_lv) for l in range(n_bus_lv2_vec-1)])
        dict_output["dual_P"] = abs(cons_P.Pi)
        dict_output["dual_Pn"] = abs(cons_Pn.Pi)
        dict_output["dual_Q"] = abs(cons_Q.Pi)
        dict_output["dual_Qn"] = abs(cons_Qn.Pi)
        dict_output["dual_v2"] = abs(cons_v2.Pi)
        dict_output["dual_v2n"] = abs(cons_v2n.Pi)
        dict_output["objVal"] = m.objVal
    return dict_output

def opf_lv2_fea(dict_input_lv2, v2_trafo_mv, P_trafo_mv, Q_trafo_mv):
    '''
    They can be solved together because of no mutual impact
    '''
    #------------------------  Load Profiles ----------------------------------
    # sgen capacity
    cap_sgen_lv2 = dict_input_lv2["cap_sgen_lv2"]
    # load: P, Q
    arr_lv2_p = dict_input_lv2["arr_lv2_p"]
    arr_lv2_q = dict_input_lv2["arr_lv2_q"]
    # sgen: P
    arr_lv2_sgen = dict_input_lv2["arr_lv2_sgen"]   
    #------------------------- Parameters -------------------------------------
    # grid topology
    n_bus_lv2_vec = 96
    # voltage
    v_mv = 20/np.sqrt(3)
    v_lv = 0.4/np.sqrt(3)
    v_lb_lv = 0.9
    v_ub_lv = 1.1
    # power factor
    limit_pf = 0.95
    limit_tan_phi = np.sqrt(1-limit_pf**2)/limit_pf 
    # transformer
    # https://pandapower.readthedocs.io/en/develop/elements/trafo.html
    turns_ratio = v_lv/v_mv
    r_ohm_630kva = 1.206/100*1/0.63*0.4**2*1/0.63
    x_ohm_630kva = np.sqrt(6**2-1.206**2)/100*1/0.63*0.4**2*1/0.63 
    #------------------------- Variables- -------------------------------------
    env = gp.Env(empty=True)
    env.setParam("OutputFlag",0)
    env.start()
    if print_gurobi:
        m = gp.Model("opf_lv2")
    else:
        m = gp.Model("opf_lv2", env = env)
    if non_linear == False:
        m.Params.OptimalityTol = 1E-8
    m.Params.numericFocus = 3
    # voltage magnitude squared
    v2_lv2 = m.addVars(n_bus_lv2_vec, lb = (v_lv*v_lb_lv)**2, ub = (v_lv*v_ub_lv)**2, name = "LV2_bus_volt2") # of 96
    # sgen real power
    p_sgen_lv2 = m.addVars(n_bus_lv2_vec, lb = -GRB.INFINITY, name = "LV2_P_sgen") # of 96
    # sgen reactive power
    q_sgen_lv2 = m.addVars(n_bus_lv2_vec, lb = -GRB.INFINITY, name = "LV2_Q_sgen") # of 96
    # real power injection
    p_lv2 = m.addVars(n_bus_lv2_vec, lb = -GRB.INFINITY, name = "LV2_bus_P_injection") # of 96
    # reactive power injection
    q_lv2 = m.addVars(n_bus_lv2_vec, lb = -GRB.INFINITY, name = "LV2_bus_Q_injection") # of 96
    # real power flow
    P_lv2 = m.addVars(n_bus_lv2_vec-1, lb = -GRB.INFINITY, name = "LV2_cable_Pflow") # length of 95
    # reactive power flow
    Q_lv2 = m.addVars(n_bus_lv2_vec-1, lb = -GRB.INFINITY, name = "LV2_cable_Qflow") # length of 95
    # transformer power flow
    P_trafo_lv = m.addVar(lb = -GRB.INFINITY, name = "Trafo_Pflow_LV") # flow in LV
    Q_trafo_lv = m.addVar(lb = -GRB.INFINITY, name = "Trafo_Qflow_LV") # flow in LV
    v2_trafo = m.addVar(lb = -GRB.INFINITY, name = "Trafo_volt2_mvside") # flow in LV 
    beta = m.addVar(name = "beta")      
    #------------------------- Constraints LV2 --------------------------------
    # node balance
    m.addConstrs(p_lv2[i] == p_sgen_lv2[i] - arr_lv2_p[i] for i in range(n_bus_lv2_vec))
    m.addConstrs(q_lv2[i] == q_sgen_lv2[i] - arr_lv2_q[i] for i in range(n_bus_lv2_vec))
    # sgen constraints
    m.addConstrs(p_sgen_lv2[i] <= arr_lv2_sgen[i] for i in range(n_bus_lv2_vec))
    m.addConstrs(q_sgen_lv2[i] <=  limit_tan_phi * p_sgen_lv2[i] for i in range(n_bus_lv2_vec))
    m.addConstrs(q_sgen_lv2[i] >= -limit_tan_phi * p_sgen_lv2[i] for i in range(n_bus_lv2_vec))  
    if non_linear:
        m.addConstrs(1e6*p_sgen_lv2[i]*p_sgen_lv2[i]+1e6*q_sgen_lv2[i]*q_sgen_lv2[i] <= 1e6*(cap_sgen_lv2[i])**2 for i in range(n_bus_lv2_vec))  
    else:
        for i in range(n_bus_lv2_vec):
            n_pts = 36
            for p in range(n_pts):
                x0 = cap_sgen_lv2[i] * math.cos(p*math.pi*2/n_pts)
                y0 = cap_sgen_lv2[i] * math.sin(p*math.pi*2/n_pts)
                x1 = cap_sgen_lv2[i] * math.cos((p+1)*math.pi*2/n_pts)
                y1 = cap_sgen_lv2[i] * math.sin((p+1)*math.pi*2/n_pts)
                if -x0*y1 >= -x1*y0: # to decide which side (0,0) locates in
                    m.addConstr((y1-y0)*(p_sgen_lv2[i]-x0) >= (x1-x0)*(q_sgen_lv2[i]-y0))
                else:
                    m.addConstr((y1-y0)*(p_sgen_lv2[i]-x0) <= (x1-x0)*(q_sgen_lv2[i]-y0)) 
    # current law
    for b in range(n_bus_lv2_vec):
        if b == 62: # LV side of transformer
            P_out = 0
            Q_out = 0
            for l in range(n_bus_lv2_vec-1):
                if nw_lv_rural2_cable.from_bus[l] == b:
                    P_out += P_lv2[l]
                    Q_out += Q_lv2[l]
            m.addConstr(P_trafo_lv + p_lv2[b] == P_out)
            m.addConstr(Q_trafo_lv + q_lv2[b] == Q_out)
        else:
            P_out = 0
            Q_out = 0
            P_in = 0
            Q_in = 0
            for l in range(n_bus_lv2_vec-1):
                if nw_lv_rural2_cable.from_bus[l] == b:
                    P_out += P_lv2[l]
                    Q_out += Q_lv2[l]
                if nw_lv_rural2_cable.to_bus[l] == b:
                    P_in += P_lv2[l]
                    Q_in += Q_lv2[l]
            m.addConstr(P_in + p_lv2[b] == P_out)
            m.addConstr(Q_in + q_lv2[b] == Q_out)           
    # voltage law
    for l in range(n_bus_lv2_vec-1):
        from_bus = nw_lv_rural2_cable.from_bus[l]
        to_bus = nw_lv_rural2_cable.to_bus[l]
        r_ohm = nw_lv_rural2_cable.length_km[l]*nw_lv_rural2_cable.r_ohm_per_km[l]
        x_ohm = nw_lv_rural2_cable.length_km[l]*nw_lv_rural2_cable.x_ohm_per_km[l]
        m.addConstr(v2_lv2[to_bus] == v2_lv2[from_bus] - 2*(r_ohm*P_lv2[l]+x_ohm*Q_lv2[l]))
        s_max = nw_lv_rural2_cable.max_i_ka[l] * v_lv
        # cable capacity limit
        if non_linear:
            m.addConstr(1e6*P_lv2[l]*P_lv2[l]+1e6*Q_lv2[l]*Q_lv2[l] <= 1e6*s_max**2) 
        else:
            n_pts = 36
            for p in range(n_pts):
                x0 = s_max * math.cos(p*math.pi*2/n_pts)
                y0 = s_max * math.sin(p*math.pi*2/n_pts)
                x1 = s_max * math.cos((p+1)*math.pi*2/n_pts)
                y1 = s_max * math.sin((p+1)*math.pi*2/n_pts)
                if -x0*y1 >= -x1*y0: # to decide which side (0,0) locates in
                    m.addConstr((y1-y0)*(P_lv2[l]-x0) >= (x1-x0)*(Q_lv2[l]-y0))
                else:
                    m.addConstr((y1-y0)*(P_lv2[l]-x0) <= (x1-x0)*(Q_lv2[l]-y0))
    #------------------------- Constraints Trafos -----------------------------
    # power flow balance
    # constraints named to retrive duals
    cons_P = m.addConstr(P_trafo_lv <= P_trafo_mv+beta)
    cons_Pn = m.addConstr(-P_trafo_lv <= -P_trafo_mv+beta)
    cons_Q = m.addConstr(Q_trafo_lv <= Q_trafo_mv+beta)
    cons_Qn = m.addConstr(-Q_trafo_lv <= -Q_trafo_mv+beta)
    cons_v2 = m.addConstr(v2_trafo <= v2_trafo_mv+beta)
    cons_v2n = m.addConstr(-v2_trafo <= -v2_trafo_mv+beta) 
    # voltage relations: for easier Benders decomposition here
    m.addConstr(v2_trafo*turns_ratio**2-v2_lv2[62] ==2*(r_ohm_630kva*P_trafo_lv+x_ohm_630kva*Q_trafo_lv))       
    # ----------------------- objective function -----------------------------  
    m.setObjective(beta, GRB.MINIMIZE)
    m.optimize()
    # ------------------------ output ----------------------------------------
    dict_output = {}
    dict_output["status"] = m.status
    dict_output["objVal"] = m.objVal
    dict_output["p_sgen_lv2"] = np.array([p_sgen_lv2[i].X for i in range(n_bus_lv2_vec)])
    dict_output["q_sgen_lv2"] = np.array([q_sgen_lv2[i].X for i in range(n_bus_lv2_vec)]) 
    dict_output["vm_pu_lv2"] = np.array([np.sqrt(v2_lv2[i].X)/v_lv for i in range(n_bus_lv2_vec)])
    dict_output["P_trafo_lv"] = P_trafo_lv.X 
    dict_output["Q_trafo_lv"] = Q_trafo_lv.X
    dict_output["v2_trafo"] = v2_trafo.X
    dict_output["loading_trafo"] = 100*np.sqrt((P_trafo_lv.X)**2 + (Q_trafo_lv.X)**2)/0.21
    dict_output["loading_line_lv2"] = 100*np.array([np.sqrt((P_lv2[l].X)**2+(Q_lv2[l].X)**2)/(nw_lv_rural2_cable.max_i_ka[l] * v_lv) for l in range(n_bus_lv2_vec-1)])
    dict_output["dual_P"] = abs(cons_P.Pi)
    dict_output["dual_Pn"] = abs(cons_Pn.Pi)
    dict_output["dual_Q"] = abs(cons_Q.Pi)
    dict_output["dual_Qn"] = abs(cons_Qn.Pi)
    dict_output["dual_v2"] = abs(cons_v2.Pi)
    dict_output["dual_v2n"] = abs(cons_v2n.Pi)
    return dict_output

def opf_lv3(dict_input_lv3, v2_trafo_mv, P_trafo_mv, Q_trafo_mv, objective):
    '''
    They can be solved together because of no mutual impact
    '''
    #------------------------  Load Profiles ----------------------------------
    # sgen capacity
    cap_sgen_lv3 = dict_input_lv3["cap_sgen_lv3"]
    # load: P, Q
    arr_lv3_p = dict_input_lv3["arr_lv3_p"]
    arr_lv3_q = dict_input_lv3["arr_lv3_q"]
    # sgen: P
    arr_lv3_sgen = dict_input_lv3["arr_lv3_sgen"]
    
    #------------------------- Parameters -------------------------------------
    # grid topology
    n_bus_lv3_vec = 128
    # voltage
    v_mv = 20/np.sqrt(3)
    v_lv = 0.4/np.sqrt(3)
    v_lb_lv = 0.9
    v_ub_lv = 1.1
    # power factor
    limit_pf = 0.95
    limit_tan_phi = np.sqrt(1-limit_pf**2)/limit_pf 
    # transformer
    # https://pandapower.readthedocs.io/en/develop/elements/trafo.html
    turns_ratio = v_lv/v_mv
    r_ohm_630kva = 1.206/100*1/0.63*0.4**2*1/0.63
    x_ohm_630kva = np.sqrt(6**2-1.206**2)/100*1/0.63*0.4**2*1/0.63
    
    #------------------------- Variables- -------------------------------------
    env = gp.Env(empty=True)
    env.setParam("OutputFlag",0)
    env.start()
    if print_gurobi:
        m = gp.Model("opf_lv3")
    else:
        m = gp.Model("opf_lv3", env = env)
    if non_linear == False:
        m.Params.OptimalityTol = 1E-8
    m.Params.numericFocus = 3
    # voltage magnitude squared
    v2_lv3 = m.addVars(n_bus_lv3_vec, lb = (v_lv*v_lb_lv)**2, ub = (v_lv*v_ub_lv)**2, name = "LV3_bus_volt2") # of 128
    # sgen real power
    p_sgen_lv3 = m.addVars(n_bus_lv3_vec, lb = -GRB.INFINITY, name = "LV3_P_sgen") # of 128
    # sgen reactive power
    q_sgen_lv3 = m.addVars(n_bus_lv3_vec, lb = -GRB.INFINITY, name = "LV3_Q_sgen") # of 128
    # real power injection
    p_lv3 = m.addVars(n_bus_lv3_vec, lb = -GRB.INFINITY, name = "LV3_bus_P_injection") # of 128
    # reactive power injection
    q_lv3 = m.addVars(n_bus_lv3_vec, lb = -GRB.INFINITY, name = "LV3_bus_Q_injection") # of 128
    # real power flow
    P_lv3 = m.addVars(n_bus_lv3_vec-1, lb = -GRB.INFINITY, name = "LV3_cable_Pflow") # length of 127
    # reactive power flow
    Q_lv3 = m.addVars(n_bus_lv3_vec-1, lb = -GRB.INFINITY, name = "LV3_cable_Qflow") # length of 127
    # transformer power flow
    P_trafo_lv = m.addVar(lb = -GRB.INFINITY, name = "Trafo_Pflow_LV") # flow in LV
    Q_trafo_lv = m.addVar(lb = -GRB.INFINITY, name = "Trafo_Qflow_LV") # flow in LV
    v2_trafo = m.addVar(lb = -GRB.INFINITY, name = "Trafo_volt2_mvside") # flow in LV  
    #------------------------- Constraints LV3 --------------------------------
    # node balance
    m.addConstrs(p_lv3[i] == p_sgen_lv3[i] - arr_lv3_p[i] for i in range(n_bus_lv3_vec))
    m.addConstrs(q_lv3[i] == q_sgen_lv3[i] - arr_lv3_q[i] for i in range(n_bus_lv3_vec))
    # sgen constraints
    m.addConstrs(p_sgen_lv3[i] <= arr_lv3_sgen[i] for i in range(n_bus_lv3_vec))
    m.addConstrs(q_sgen_lv3[i] <=  limit_tan_phi * p_sgen_lv3[i] for i in range(n_bus_lv3_vec))
    m.addConstrs(q_sgen_lv3[i] >= -limit_tan_phi * p_sgen_lv3[i] for i in range(n_bus_lv3_vec))  
    if non_linear:
        m.addConstrs(1e6*p_sgen_lv3[i]*p_sgen_lv3[i]+1e6*q_sgen_lv3[i]*q_sgen_lv3[i] <= 1e6*(cap_sgen_lv3[i])**2 for i in range(n_bus_lv3_vec))  
    else:
        for i in range(n_bus_lv3_vec):
            n_pts = 36
            for p in range(n_pts):
                x0 = cap_sgen_lv3[i] * math.cos(p*math.pi*2/n_pts)
                y0 = cap_sgen_lv3[i] * math.sin(p*math.pi*2/n_pts)
                x1 = cap_sgen_lv3[i] * math.cos((p+1)*math.pi*2/n_pts)
                y1 = cap_sgen_lv3[i] * math.sin((p+1)*math.pi*2/n_pts)
                if -x0*y1 >= -x1*y0: # to decide which side (0,0) locates in
                    m.addConstr((y1-y0)*(p_sgen_lv3[i]-x0) >= (x1-x0)*(q_sgen_lv3[i]-y0))
                else:
                    m.addConstr((y1-y0)*(p_sgen_lv3[i]-x0) <= (x1-x0)*(q_sgen_lv3[i]-y0))   
    # current law
    for b in range(n_bus_lv3_vec):
        if b == 104: # LV side of transformer
            P_out = 0
            Q_out = 0
            for l in range(n_bus_lv3_vec-1):
                if nw_lv_rural3_cable.from_bus[l] == b:
                    P_out += P_lv3[l]
                    Q_out += Q_lv3[l]
            m.addConstr(P_trafo_lv + p_lv3[b]  == P_out)
            m.addConstr(Q_trafo_lv + q_lv3[b] == Q_out)
        else:
            P_out = 0
            Q_out = 0
            P_in = 0
            Q_in = 0
            for l in range(n_bus_lv3_vec-1):
                if nw_lv_rural3_cable.from_bus[l] == b:
                    P_out += P_lv3[l]
                    Q_out += Q_lv3[l]
                if nw_lv_rural3_cable.to_bus[l] == b:
                    P_in += P_lv3[l]
                    Q_in += Q_lv3[l]
            m.addConstr(P_in + p_lv3[b] == P_out)
            m.addConstr(Q_in + q_lv3[b] == Q_out)  
        
    # voltage law
    for l in range(n_bus_lv3_vec-1):
        from_bus = nw_lv_rural3_cable.from_bus[l]
        to_bus = nw_lv_rural3_cable.to_bus[l]
        r_ohm = nw_lv_rural3_cable.length_km[l]*nw_lv_rural3_cable.r_ohm_per_km[l]
        x_ohm = nw_lv_rural3_cable.length_km[l]*nw_lv_rural3_cable.x_ohm_per_km[l]
        m.addConstr(v2_lv3[to_bus] == v2_lv3[from_bus] - 2*(r_ohm*P_lv3[l]+x_ohm*Q_lv3[l]))
        s_max = nw_lv_rural3_cable.max_i_ka[l] * v_lv
        # cable capacity limit
        if non_linear:
            m.addConstr(1e6*P_lv3[l]*P_lv3[l]+1e6*Q_lv3[l]*Q_lv3[l] <= 1e6*s_max**2) 
        else:
            n_pts = 36
            for p in range(n_pts):
                x0 = s_max * math.cos(p*math.pi*2/n_pts)
                y0 = s_max * math.sin(p*math.pi*2/n_pts)
                x1 = s_max * math.cos((p+1)*math.pi*2/n_pts)
                y1 = s_max * math.sin((p+1)*math.pi*2/n_pts)
                if -x0*y1 >= -x1*y0: # to decide which side (0,0) locates in
                    m.addConstr((y1-y0)*(P_lv3[l]-x0) >= (x1-x0)*(Q_lv3[l]-y0))
                else:
                    m.addConstr((y1-y0)*(P_lv3[l]-x0) <= (x1-x0)*(Q_lv3[l]-y0))
    
    #------------------------- Constraints Trafos -----------------------------
    # power flow balance
    # constraints named to retrive duals
    cons_P = m.addConstr(P_trafo_lv <= P_trafo_mv)
    cons_Pn = m.addConstr(-P_trafo_lv <= -P_trafo_mv)
    cons_Q = m.addConstr(Q_trafo_lv <= Q_trafo_mv)
    cons_Qn = m.addConstr(-Q_trafo_lv <= -Q_trafo_mv)
    cons_v2 = m.addConstr(v2_trafo <= v2_trafo_mv)
    cons_v2n = m.addConstr(-v2_trafo <= -v2_trafo_mv) 
    
    # voltage relations: for easier Benders decomposition here
    m.addConstr(v2_trafo*turns_ratio**2-v2_lv3[104]==2*(r_ohm_630kva*P_trafo_lv+x_ohm_630kva*Q_trafo_lv))    
    
    # ----------------------- objective function -----------------------------  
    obj_quad = quicksum((arr_lv3_sgen[i]-p_sgen_lv3[i])*(arr_lv3_sgen[i]-p_sgen_lv3[i]) for i in range(n_bus_lv3_vec)) 
    obj_lin = quicksum((arr_lv3_sgen[i]-p_sgen_lv3[i]) for i in range(n_bus_lv3_vec)) 
    np.seterr(divide='ignore')
    arr_sgen_lv3_inv = 1/arr_lv3_sgen
    arr_sgen_lv3_inv[arr_sgen_lv3_inv == inf] = 0
    obj_prop = quicksum((arr_lv3_sgen[i]-p_sgen_lv3[i])*(arr_lv3_sgen[i]-p_sgen_lv3[i])*(arr_sgen_lv3_inv[i]) for i in range(n_bus_lv3_vec)) 
    if objective == "obj_lin":
        m.setObjective(obj_lin, GRB.MINIMIZE)
    elif objective == "obj_quad":
        m.setObjective(obj_quad, GRB.MINIMIZE)
    elif objective == "obj_prop":
        m.setObjective(obj_prop, GRB.MINIMIZE)
    else:
        print("Objective function error.")
        return 0
    m.optimize()
    # ------------------------ output ----------------------------------------
    dict_output = {}
    dict_output["status"] = m.status
    if m.status == 2:
        dict_output["p_sgen_lv3"] = np.array([p_sgen_lv3[i].X for i in range(n_bus_lv3_vec)])
        dict_output["q_sgen_lv3"] = np.array([q_sgen_lv3[i].X for i in range(n_bus_lv3_vec)])   
        dict_output["vm_pu_lv3"] = np.array([np.sqrt(v2_lv3[i].X)/v_lv for i in range(n_bus_lv3_vec)])
        dict_output["P_trafo_lv"] = P_trafo_lv.X 
        dict_output["Q_trafo_lv"] = Q_trafo_lv.X
        dict_output["v2_trafo"] = v2_trafo.X
        dict_output["loading_trafo"] = 100*np.sqrt((P_trafo_lv.X)**2 + (Q_trafo_lv.X)**2)/0.21
        dict_output["loading_line_lv3"] = 100*np.array([np.sqrt((P_lv3[l].X)**2+(Q_lv3[l].X)**2)/(nw_lv_rural3_cable.max_i_ka[l] * v_lv) for l in range(n_bus_lv3_vec-1)])
        dict_output["dual_P"] = abs(cons_P.Pi)
        dict_output["dual_Pn"] = abs(cons_Pn.Pi)
        dict_output["dual_Q"] = abs(cons_Q.Pi)
        dict_output["dual_Qn"] = abs(cons_Qn.Pi)
        dict_output["dual_v2"] = abs(cons_v2.Pi)
        dict_output["dual_v2n"] = abs(cons_v2n.Pi)
        dict_output["objVal"] = m.objVal
    return dict_output

def opf_lv3_fea(dict_input_lv3, v2_trafo_mv, P_trafo_mv, Q_trafo_mv):
    '''
    They can be solved together because of no mutual impact
    '''
    #------------------------  Load Profiles ----------------------------------
    # sgen capacity
    cap_sgen_lv3 = dict_input_lv3["cap_sgen_lv3"]
    # load: P, Q
    arr_lv3_p = dict_input_lv3["arr_lv3_p"]
    arr_lv3_q = dict_input_lv3["arr_lv3_q"]
    # sgen: P
    arr_lv3_sgen = dict_input_lv3["arr_lv3_sgen"]
    
    #------------------------- Parameters -------------------------------------
    # grid topology
    n_bus_lv3_vec = 128
    # voltage
    v_mv = 20/np.sqrt(3)
    v_lv = 0.4/np.sqrt(3)
    v_lb_lv = 0.9
    v_ub_lv = 1.1
    # power factor
    limit_pf = 0.95
    limit_tan_phi = np.sqrt(1-limit_pf**2)/limit_pf 
    # transformer
    # https://pandapower.readthedocs.io/en/develop/elements/trafo.html
    turns_ratio = v_lv/v_mv
    r_ohm_630kva = 1.206/100*1/0.63*0.4**2*1/0.63
    x_ohm_630kva = np.sqrt(6**2-1.206**2)/100*1/0.63*0.4**2*1/0.63
    
    #------------------------- Variables- -------------------------------------
    env = gp.Env(empty=True)
    env.setParam("OutputFlag",0)
    env.start()
    if print_gurobi:
        m = gp.Model("opf_lv3")
    else:
        m = gp.Model("opf_lv3", env = env)
    if non_linear == False:
        m.Params.OptimalityTol = 1E-8
    m.Params.numericFocus = 3
    # voltage magnitude squared
    v2_lv3 = m.addVars(n_bus_lv3_vec, lb = (v_lv*v_lb_lv)**2, ub = (v_lv*v_ub_lv)**2, name = "LV3_bus_volt2") # of 128
    # sgen real power
    p_sgen_lv3 = m.addVars(n_bus_lv3_vec, lb = -GRB.INFINITY, name = "LV3_P_sgen") # of 128
    # sgen reactive power
    q_sgen_lv3 = m.addVars(n_bus_lv3_vec, lb = -GRB.INFINITY, name = "LV3_Q_sgen") # of 128
    # real power injection
    p_lv3 = m.addVars(n_bus_lv3_vec, lb = -GRB.INFINITY, name = "LV3_bus_P_injection") # of 128
    # reactive power injection
    q_lv3 = m.addVars(n_bus_lv3_vec, lb = -GRB.INFINITY, name = "LV3_bus_Q_injection") # of 128
    # real power flow
    P_lv3 = m.addVars(n_bus_lv3_vec-1, lb = -GRB.INFINITY, name = "LV3_cable_Pflow") # length of 127
    # reactive power flow
    Q_lv3 = m.addVars(n_bus_lv3_vec-1, lb = -GRB.INFINITY, name = "LV3_cable_Qflow") # length of 127
    # transformer power flow
    P_trafo_lv = m.addVar(lb = -GRB.INFINITY, name = "Trafo_Pflow_LV") # flow in LV
    Q_trafo_lv = m.addVar(lb = -GRB.INFINITY, name = "Trafo_Qflow_LV") # flow in LV
    v2_trafo = m.addVar(lb = -GRB.INFINITY, name = "Trafo_volt2_mvside") # flow in LV
    beta = m.addVar(name = "beta")    
    #------------------------- Constraints LV3 --------------------------------
    # node balance
    m.addConstrs(p_lv3[i] == p_sgen_lv3[i] - arr_lv3_p[i] for i in range(n_bus_lv3_vec))
    m.addConstrs(q_lv3[i] == q_sgen_lv3[i] - arr_lv3_q[i] for i in range(n_bus_lv3_vec))
    # sgen constraints
    m.addConstrs(p_sgen_lv3[i] <= arr_lv3_sgen[i] for i in range(n_bus_lv3_vec))
    m.addConstrs(q_sgen_lv3[i] <=  limit_tan_phi * p_sgen_lv3[i] for i in range(n_bus_lv3_vec))
    m.addConstrs(q_sgen_lv3[i] >= -limit_tan_phi * p_sgen_lv3[i] for i in range(n_bus_lv3_vec))  
    if non_linear:
        m.addConstrs(1e6*p_sgen_lv3[i]*p_sgen_lv3[i]+1e6*q_sgen_lv3[i]*q_sgen_lv3[i] <= 1e6*(cap_sgen_lv3[i])**2 for i in range(n_bus_lv3_vec))  
    else:
        for i in range(n_bus_lv3_vec):
            n_pts = 36
            for p in range(n_pts):
                x0 = cap_sgen_lv3[i] * math.cos(p*math.pi*2/n_pts)
                y0 = cap_sgen_lv3[i] * math.sin(p*math.pi*2/n_pts)
                x1 = cap_sgen_lv3[i] * math.cos((p+1)*math.pi*2/n_pts)
                y1 = cap_sgen_lv3[i] * math.sin((p+1)*math.pi*2/n_pts)
                if -x0*y1 >= -x1*y0: # to decide which side (0,0) locates in
                    m.addConstr((y1-y0)*(p_sgen_lv3[i]-x0) >= (x1-x0)*(q_sgen_lv3[i]-y0))
                else:
                    m.addConstr((y1-y0)*(p_sgen_lv3[i]-x0) <= (x1-x0)*(q_sgen_lv3[i]-y0))   
    # current law
    for b in range(n_bus_lv3_vec):
        if b == 104: # LV side of transformer
            P_out = 0
            Q_out = 0
            for l in range(n_bus_lv3_vec-1):
                if nw_lv_rural3_cable.from_bus[l] == b:
                    P_out += P_lv3[l]
                    Q_out += Q_lv3[l]
            m.addConstr(P_trafo_lv + p_lv3[b] == P_out)
            m.addConstr(Q_trafo_lv + q_lv3[b] == Q_out)
        else:
            P_out = 0
            Q_out = 0
            P_in = 0
            Q_in = 0
            for l in range(n_bus_lv3_vec-1):
                if nw_lv_rural3_cable.from_bus[l] == b:
                    P_out += P_lv3[l]
                    Q_out += Q_lv3[l]
                if nw_lv_rural3_cable.to_bus[l] == b:
                    P_in += P_lv3[l]
                    Q_in += Q_lv3[l]
            m.addConstr(P_in + p_lv3[b] == P_out)
            m.addConstr(Q_in + q_lv3[b] == Q_out)  
        
    # voltage law
    for l in range(n_bus_lv3_vec-1):
        from_bus = nw_lv_rural3_cable.from_bus[l]
        to_bus = nw_lv_rural3_cable.to_bus[l]
        r_ohm = nw_lv_rural3_cable.length_km[l]*nw_lv_rural3_cable.r_ohm_per_km[l]
        x_ohm = nw_lv_rural3_cable.length_km[l]*nw_lv_rural3_cable.x_ohm_per_km[l]
        m.addConstr(v2_lv3[to_bus] == v2_lv3[from_bus] - 2*(r_ohm*P_lv3[l]+x_ohm*Q_lv3[l]))
        s_max = nw_lv_rural3_cable.max_i_ka[l] * v_lv
        # cable capacity limit
        if non_linear:
            m.addConstr(1e6*P_lv3[l]*P_lv3[l]+1e6*Q_lv3[l]*Q_lv3[l] <= 1e6*s_max**2) 
        else:
            n_pts = 36
            for p in range(n_pts):
                x0 = s_max * math.cos(p*math.pi*2/n_pts)
                y0 = s_max * math.sin(p*math.pi*2/n_pts)
                x1 = s_max * math.cos((p+1)*math.pi*2/n_pts)
                y1 = s_max * math.sin((p+1)*math.pi*2/n_pts)
                if -x0*y1 >= -x1*y0: # to decide which side (0,0) locates in
                    m.addConstr((y1-y0)*(P_lv3[l]-x0) >= (x1-x0)*(Q_lv3[l]-y0))
                else:
                    m.addConstr((y1-y0)*(P_lv3[l]-x0) <= (x1-x0)*(Q_lv3[l]-y0))
    
    #------------------------- Constraints Trafos -----------------------------
    # power flow balance
    # constraints named to retrive duals
    cons_P = m.addConstr(P_trafo_lv <= P_trafo_mv + beta)
    cons_Pn = m.addConstr(-P_trafo_lv <= -P_trafo_mv + beta)
    cons_Q = m.addConstr(Q_trafo_lv <= Q_trafo_mv + beta)
    cons_Qn = m.addConstr(-Q_trafo_lv <= -Q_trafo_mv + beta)
    cons_v2 = m.addConstr(v2_trafo <= v2_trafo_mv + beta)
    cons_v2n = m.addConstr(-v2_trafo <= -v2_trafo_mv + beta) 
    
    # voltage relations: for easier Benders decomposition here
    m.addConstr(v2_trafo*turns_ratio**2-v2_lv3[104]==2*(r_ohm_630kva*P_trafo_lv+x_ohm_630kva*Q_trafo_lv))    
    
    # ----------------------- objective function -----------------------------  
    m.setObjective(beta, GRB.MINIMIZE)
    m.optimize()
    # ------------------------ output ----------------------------------------
    dict_output = {}
    dict_output["status"] = m.status
    dict_output["objVal"] = m.objVal
    dict_output["p_sgen_lv3"] = np.array([p_sgen_lv3[i].X for i in range(n_bus_lv3_vec)])
    dict_output["q_sgen_lv3"] = np.array([q_sgen_lv3[i].X for i in range(n_bus_lv3_vec)])   
    dict_output["vm_pu_lv3"] = np.array([np.sqrt(v2_lv3[i].X)/v_lv for i in range(n_bus_lv3_vec)])
    dict_output["P_trafo_lv"] = P_trafo_lv.X 
    dict_output["Q_trafo_lv"] = Q_trafo_lv.X
    dict_output["v2_trafo"] = v2_trafo.X
    dict_output["loading_trafo"] = 100*np.sqrt((P_trafo_lv.X)**2 + (Q_trafo_lv.X)**2)/0.21
    dict_output["loading_line_lv3"] = 100*np.array([np.sqrt((P_lv3[l].X)**2+(Q_lv3[l].X)**2)/(nw_lv_rural3_cable.max_i_ka[l] * v_lv) for l in range(n_bus_lv3_vec-1)])
    dict_output["dual_P"] = abs(cons_P.Pi)
    dict_output["dual_Pn"] = abs(cons_Pn.Pi)
    dict_output["dual_Q"] = abs(cons_Q.Pi)
    dict_output["dual_Qn"] = abs(cons_Qn.Pi)
    dict_output["dual_v2"] = abs(cons_v2.Pi)
    dict_output["dual_v2n"] = abs(cons_v2n.Pi)
    return dict_output

def opf_mv_lv(dict_input, objective, epsilon = 1E-4):
    '''
    unit in kv, kA, ohm, mva,
    decision variables are sgen output (real & reactive power),
    this program optimizes MV first,
    LV optimized according to MV signals
    '''  
    # grid topology
    n_bus_mv = 12 # 4~15
    n_bus_lv1_vec = 14
    n_bus_lv2_vec = 96
    n_bus_lv3_vec = 128
    #------------------------  Optimize MV ----------------------------------
    # sgen capacity
    cap_sgen_mv = dict_input["cap_sgen_mv"]
    cap_sgen_lv1 = dict_input["cap_sgen_lv1"]
    cap_sgen_lv2 = dict_input["cap_sgen_lv2"]
    cap_sgen_lv3 = dict_input["cap_sgen_lv3"]
    # load: P, Q
    arr_mv_p = dict_input["arr_mv_p"]
    arr_mv_q = dict_input["arr_mv_q"]
    arr_lv1_p = dict_input["arr_lv1_p"]
    arr_lv1_q = dict_input["arr_lv1_q"]
    arr_lv2_p = dict_input["arr_lv2_p"]
    arr_lv2_q = dict_input["arr_lv2_q"]
    arr_lv3_p = dict_input["arr_lv3_p"]
    arr_lv3_q = dict_input["arr_lv3_q"]
    # sgen: P
    arr_mv_sgen = dict_input["arr_mv_sgen"]
    arr_lv1_sgen = dict_input["arr_lv1_sgen"]
    arr_lv2_sgen = dict_input["arr_lv2_sgen"]
    arr_lv3_sgen = dict_input["arr_lv3_sgen"]  
    # read MV input
    dict_input_mv = {}
    dict_input_mv["cap_sgen_mv"] = cap_sgen_mv
    dict_input_mv["arr_mv_p"] = arr_mv_p
    dict_input_mv["arr_mv_q"] = arr_mv_q
    dict_input_mv["arr_mv_sgen"] = arr_mv_sgen
    # trafo real power flow
    dict_input_mv["P_trafo_forecast"] = np.array([arr_lv1_p.sum()-arr_lv1_sgen.sum(), 
                                    arr_lv2_p.sum()-arr_lv2_sgen.sum(), arr_lv3_p.sum()-arr_lv3_sgen.sum()])
    # LV parameters
    dict_input_lv1 = {}
    dict_input_lv2 = {}
    dict_input_lv3 = {}
    dict_input_lv1["cap_sgen_lv1"] = cap_sgen_lv1
    dict_input_lv2["cap_sgen_lv2"] = cap_sgen_lv2 
    dict_input_lv3["cap_sgen_lv3"] = cap_sgen_lv3
    # load: P, Q
    dict_input_lv1["arr_lv1_p"] = arr_lv1_p 
    dict_input_lv1["arr_lv1_q"] = arr_lv1_q 
    dict_input_lv2["arr_lv2_p"] = arr_lv2_p
    dict_input_lv2["arr_lv2_q"] = arr_lv2_q
    dict_input_lv3["arr_lv3_p"] = arr_lv3_p
    dict_input_lv3["arr_lv3_q"] = arr_lv3_q
    # sgen: P
    dict_input_lv1["arr_lv1_sgen"] = arr_lv1_sgen 
    dict_input_lv2["arr_lv2_sgen"] = arr_lv2_sgen 
    dict_input_lv3["arr_lv3_sgen"] = arr_lv3_sgen 
    # cut parameters
    dict_opt_lv1 = {"objVal":[],
                    "dual_P":[],
                    "dual_Pn":[],
                    "dual_Q":[],
                    "dual_Qn":[],
                    "dual_v2":[],
                    "dual_v2n":[],
                    "P_trafo_lv":[],
                    "Q_trafo_lv":[],
                    "v2_trafo":[]
                    }
    dict_fea_lv1 = {"dual_P":[],
                    "dual_Pn":[],
                    "dual_Q":[],
                    "dual_Qn":[],
                    "dual_v2":[],
                    "dual_v2n":[],
                    "P_trafo_lv":[],
                    "Q_trafo_lv":[],
                    "v2_trafo":[]
                    }
    dict_opt_lv2 = {"objVal":[],
                    "dual_P":[],
                    "dual_Pn":[],
                    "dual_Q":[],
                    "dual_Qn":[],
                    "dual_v2":[],
                    "dual_v2n":[],
                    "P_trafo_lv":[],
                    "Q_trafo_lv":[],
                    "v2_trafo":[]
                    }
    dict_fea_lv2 = {"dual_P":[],
                    "dual_Pn":[],
                    "dual_Q":[],
                    "dual_Qn":[],
                    "dual_v2":[],
                    "dual_v2n":[],
                    "P_trafo_lv":[],
                    "Q_trafo_lv":[],
                    "v2_trafo":[]
                    }
    dict_opt_lv3 = {"objVal":[],
                    "dual_P":[],
                    "dual_Pn":[],
                    "dual_Q":[],
                    "dual_Qn":[],
                    "dual_v2":[],
                    "dual_v2n":[],
                    "P_trafo_lv":[],
                    "Q_trafo_lv":[],
                    "v2_trafo":[]
                    }
    dict_fea_lv3 = {"dual_P":[],
                    "dual_Pn":[],
                    "dual_Q":[],
                    "dual_Qn":[],
                    "dual_v2":[],
                    "dual_v2n":[],
                    "P_trafo_lv":[],
                    "Q_trafo_lv":[],
                    "v2_trafo":[]
                    }
    dict_newton = {"mat_B":[np.identity(3),np.identity(3),np.identity(3)],
                   "arr_c":[np.zeros((3,1)),np.zeros((3,1)),np.zeros((3,1))],
                   "num_q":[0,0,0]
                   }
    # --------------- benders iterations -------------------------------------
    lb = 0
    ub = 1E2
    list_lb = [lb]
    list_ub = [ub]
    n_iterations = 0
    max_iterations=200
    add_newton = True
    while (ub - lb > epsilon or add_newton) and n_iterations < max_iterations:
        n_iterations += 1
        print("Iteration {}".format(n_iterations))
        if n_iterations >= 0 and n_iterations % 5 == 0:
            add_newton = True
        else:
            add_newton = False
        bin_ub = True # decide if upper bound is derivale
        # solve MV , derive LB
        if add_newton:
            dict_output_opf_mv = opf_mv(dict_input_mv, dict_opt_lv1, dict_fea_lv1,
                                        dict_opt_lv2, dict_fea_lv2, dict_opt_lv3, dict_fea_lv3, dict_newton, objective = objective)
        else:
            dict_newton_null = {"mat_B":[np.zeros((3,3)),np.zeros((3,3)),np.zeros((3,3))],
                   "arr_c":[np.zeros((3,1)),np.zeros((3,1)),np.zeros((3,1))],
                   "num_q":[0,0,0]
                   }
            dict_output_opf_mv = opf_mv(dict_input_mv, dict_opt_lv1, dict_fea_lv1,
                                        dict_opt_lv2, dict_fea_lv2, dict_opt_lv3, dict_fea_lv3, dict_newton_null, objective = objective)
        if print_status:
            print("status :{}".format(dict_output_opf_mv["status"]))
        if dict_output_opf_mv["status"] == 2:
            lb = dict_output_opf_mv["objVal"]
        if add_newton: # lower bound not updated
            list_lb.append(list_lb[-1])
        else:
            list_lb.append(lb)
        # return boundary variables
        P_trafo_mv = dict_output_opf_mv["P_trafo_mv"]
        Q_trafo_mv = dict_output_opf_mv["Q_trafo_mv"]
        v2_trafo_mv = dict_output_opf_mv["v2_trafo_mv"]     
        # solve LV 1 2 3
        dict_output_opf_lv1 = opf_lv1(dict_input_lv1, v2_trafo_mv[0], P_trafo_mv[0], Q_trafo_mv[0], objective = objective)
        if print_status:
            print("status :{}".format(dict_output_opf_lv1["status"]))
        if dict_output_opf_lv1["status"] == 2:
            # generate optimality cuts
            dict_opt_lv1["objVal"].append(dict_output_opf_lv1["objVal"]),
            dict_opt_lv1["dual_P"].append(dict_output_opf_lv1["dual_P"]),
            dict_opt_lv1["dual_Pn"].append(dict_output_opf_lv1["dual_Pn"]),
            dict_opt_lv1["dual_Q"].append(dict_output_opf_lv1["dual_Q"]),
            dict_opt_lv1["dual_Qn"].append(dict_output_opf_lv1["dual_Qn"]),
            dict_opt_lv1["dual_v2"].append(dict_output_opf_lv1["dual_v2"]),
            dict_opt_lv1["dual_v2n"].append(dict_output_opf_lv1["dual_v2n"]),
            dict_opt_lv1["P_trafo_lv"].append(dict_output_opf_lv1["P_trafo_lv"]),
            dict_opt_lv1["Q_trafo_lv"].append(dict_output_opf_lv1["Q_trafo_lv"]),
            dict_opt_lv1["v2_trafo"].append(dict_output_opf_lv1["v2_trafo"])
            if len(dict_opt_lv1["objVal"]) >= 2:
                Bk = dict_newton["mat_B"][0]
                xk_next = np.array([ dict_opt_lv1["P_trafo_lv"][-1],
                                dict_opt_lv1["Q_trafo_lv"][-1],
                                dict_opt_lv1["v2_trafo"][-1]
                               ]).reshape(3,1)
                xk = np.array([ dict_opt_lv1["P_trafo_lv"][-2],
                                dict_opt_lv1["Q_trafo_lv"][-2],
                                dict_opt_lv1["v2_trafo"][-2]
                               ]).reshape(3,1)             
                sk = xk_next - xk
                gradk_next = np.array([ dict_opt_lv1["dual_Pn"][-1] - dict_opt_lv1["dual_P"][-1],
                                dict_opt_lv1["dual_Qn"][-1] - dict_opt_lv1["dual_Q"][-1],
                                dict_opt_lv1["dual_v2n"][-1] - dict_opt_lv1["dual_v2"][-1]
                               ]).reshape(3,1)
                gradk = np.array([ dict_opt_lv1["dual_Pn"][-2] - dict_opt_lv1["dual_P"][-2],
                               dict_opt_lv1["dual_Qn"][-2] - dict_opt_lv1["dual_Q"][-2],
                               dict_opt_lv1["dual_v2n"][-2] - dict_opt_lv1["dual_v2"][-2]
                               ]).reshape(3,1)
                yk = gradk_next - gradk
                if yk.T@sk >= 0.2*sk.T@Bk@sk:
                    theta_k = 1
                else:
                    theta_k = 0.8*sk.T@Bk@sk / (sk.T@Bk@sk - yk.T@sk)
                    theta_k = theta_k[0,0]
                rk = theta_k * yk + (1-theta_k) * Bk@sk
                Bk_next = Bk - Bk@sk@sk.T@Bk/(sk.T@Bk@sk) + rk@rk.T/(sk.T@rk)
                dict_newton["mat_B"][0] = Bk_next
                dict_newton["arr_c"][0] = gradk_next - Bk_next @ xk_next
                dict_newton["num_q"][0] = dict_output_opf_lv1["objVal"] - 0.5*(xk_next.T@Bk_next@xk_next)[0,0] - ((gradk_next - Bk_next @ xk_next).T@xk_next)[0,0]
        else:
            bin_ub = False
            dict_output_opf_lv1 = opf_lv1_fea(dict_input_lv1, v2_trafo_mv[0], P_trafo_mv[0], Q_trafo_mv[0])
            if print_status:
                print("status :{}".format(dict_output_opf_lv1["status"]))
                print("beta :{}".format(dict_output_opf_lv1["objVal"]))
            # generate optimality cuts
            dict_fea_lv1["dual_P"].append(dict_output_opf_lv1["dual_P"]),
            dict_fea_lv1["dual_Pn"].append(dict_output_opf_lv1["dual_Pn"]),
            dict_fea_lv1["dual_Q"].append(dict_output_opf_lv1["dual_Q"]),
            dict_fea_lv1["dual_Qn"].append(dict_output_opf_lv1["dual_Qn"]),
            dict_fea_lv1["dual_v2"].append(dict_output_opf_lv1["dual_v2"]),
            dict_fea_lv1["dual_v2n"].append(dict_output_opf_lv1["dual_v2n"]),
            dict_fea_lv1["P_trafo_lv"].append(dict_output_opf_lv1["P_trafo_lv"]),
            dict_fea_lv1["Q_trafo_lv"].append(dict_output_opf_lv1["Q_trafo_lv"]),
            dict_fea_lv1["v2_trafo"].append(dict_output_opf_lv1["v2_trafo"])
            
        dict_output_opf_lv2 = opf_lv2(dict_input_lv2, v2_trafo_mv[1], P_trafo_mv[1], Q_trafo_mv[1], objective = objective)
        if print_status:
            print("status :{}".format(dict_output_opf_lv2["status"]))
        if dict_output_opf_lv2["status"] == 2:
            # generate optimality cuts
            dict_opt_lv2["objVal"].append(dict_output_opf_lv2["objVal"]),
            dict_opt_lv2["dual_P"].append(dict_output_opf_lv2["dual_P"]),
            dict_opt_lv2["dual_Pn"].append(dict_output_opf_lv2["dual_Pn"]),
            dict_opt_lv2["dual_Q"].append(dict_output_opf_lv2["dual_Q"]),
            dict_opt_lv2["dual_Qn"].append(dict_output_opf_lv2["dual_Qn"]),
            dict_opt_lv2["dual_v2"].append(dict_output_opf_lv2["dual_v2"]),
            dict_opt_lv2["dual_v2n"].append(dict_output_opf_lv2["dual_v2n"]),
            dict_opt_lv2["P_trafo_lv"].append(dict_output_opf_lv2["P_trafo_lv"]),
            dict_opt_lv2["Q_trafo_lv"].append(dict_output_opf_lv2["Q_trafo_lv"]),
            dict_opt_lv2["v2_trafo"].append(dict_output_opf_lv2["v2_trafo"])
            if len(dict_opt_lv2["objVal"]) >= 2:
                Bk = dict_newton["mat_B"][1]
                xk_next = np.array([ dict_opt_lv2["P_trafo_lv"][-1],
                                dict_opt_lv2["Q_trafo_lv"][-1],
                                dict_opt_lv2["v2_trafo"][-1]
                               ]).reshape(3,1)
                xk = np.array([ dict_opt_lv2["P_trafo_lv"][-2],
                                dict_opt_lv2["Q_trafo_lv"][-2],
                                dict_opt_lv2["v2_trafo"][-2]
                               ]).reshape(3,1)               
                sk = xk_next - xk
                gradk_next = np.array([ dict_opt_lv2["dual_Pn"][-1] - dict_opt_lv2["dual_P"][-1],
                                dict_opt_lv2["dual_Qn"][-1] - dict_opt_lv2["dual_Q"][-1],
                                dict_opt_lv2["dual_v2n"][-1] - dict_opt_lv2["dual_v2"][-1]
                               ]).reshape(3,1)
                gradk = np.array([ dict_opt_lv2["dual_Pn"][-2] - dict_opt_lv2["dual_P"][-2],
                               dict_opt_lv2["dual_Qn"][-2] - dict_opt_lv2["dual_Q"][-2],
                               dict_opt_lv2["dual_v2n"][-2] - dict_opt_lv2["dual_v2"][-2]
                               ]).reshape(3,1)
                yk = gradk_next - gradk
                if yk.T@sk >= 0.2*sk.T@Bk@sk:
                    theta_k = 1
                else:
                    theta_k = 0.8*sk.T@Bk@sk / (sk.T@Bk@sk - yk.T@sk)
                    theta_k = theta_k[0,0]
                rk = theta_k * yk + (1-theta_k) * Bk@sk
                Bk_next = Bk - Bk@sk@sk.T@Bk/(sk.T@Bk@sk) + rk@rk.T/(sk.T@rk)
                dict_newton["mat_B"][1] = Bk_next
                dict_newton["arr_c"][1] = gradk_next - Bk_next @ xk_next
                dict_newton["num_q"][1] = dict_output_opf_lv2["objVal"] - 0.5*(xk_next.T@Bk_next@xk_next)[0,0] - ((gradk_next - Bk_next @ xk_next).T@xk_next)[0,0]
        else:
            bin_ub = False
            dict_output_opf_lv2 = opf_lv2_fea(dict_input_lv2, v2_trafo_mv[1], P_trafo_mv[1], Q_trafo_mv[1])
            if print_status:
                print("status :{}".format(dict_output_opf_lv2["status"]))
                print("beta :{}".format(dict_output_opf_lv2["objVal"]))
            # generate optimality cuts
            dict_fea_lv2["dual_P"].append(dict_output_opf_lv2["dual_P"]),
            dict_fea_lv2["dual_Pn"].append(dict_output_opf_lv2["dual_Pn"]),
            dict_fea_lv2["dual_Q"].append(dict_output_opf_lv2["dual_Q"]),
            dict_fea_lv2["dual_Qn"].append(dict_output_opf_lv2["dual_Qn"]),
            dict_fea_lv2["dual_v2"].append(dict_output_opf_lv2["dual_v2"]),
            dict_fea_lv2["dual_v2n"].append(dict_output_opf_lv2["dual_v2n"]),
            dict_fea_lv2["P_trafo_lv"].append(dict_output_opf_lv2["P_trafo_lv"]),
            dict_fea_lv2["Q_trafo_lv"].append(dict_output_opf_lv2["Q_trafo_lv"]),
            dict_fea_lv2["v2_trafo"].append(dict_output_opf_lv2["v2_trafo"])
            
        dict_output_opf_lv3 = opf_lv3(dict_input_lv3, v2_trafo_mv[2], P_trafo_mv[2], Q_trafo_mv[2], objective = objective)
        if print_status:
            print("status :{}".format(dict_output_opf_lv3["status"]))
        if dict_output_opf_lv3["status"] == 2:
            # generate optimality cuts
            dict_opt_lv3["objVal"].append(dict_output_opf_lv3["objVal"]),
            dict_opt_lv3["dual_P"].append(dict_output_opf_lv3["dual_P"]),
            dict_opt_lv3["dual_Pn"].append(dict_output_opf_lv3["dual_Pn"]),
            dict_opt_lv3["dual_Q"].append(dict_output_opf_lv3["dual_Q"]),
            dict_opt_lv3["dual_Qn"].append(dict_output_opf_lv3["dual_Qn"]),
            dict_opt_lv3["dual_v2"].append(dict_output_opf_lv3["dual_v2"]),
            dict_opt_lv3["dual_v2n"].append(dict_output_opf_lv3["dual_v2n"]),
            dict_opt_lv3["P_trafo_lv"].append(dict_output_opf_lv3["P_trafo_lv"]),
            dict_opt_lv3["Q_trafo_lv"].append(dict_output_opf_lv3["Q_trafo_lv"]),
            dict_opt_lv3["v2_trafo"].append(dict_output_opf_lv3["v2_trafo"])
            if len(dict_opt_lv3["objVal"]) >= 2:
                Bk = dict_newton["mat_B"][2]
                xk_next = np.array([ dict_opt_lv3["P_trafo_lv"][-1],
                                dict_opt_lv3["Q_trafo_lv"][-1],
                                dict_opt_lv3["v2_trafo"][-1]
                               ]).reshape(3,1)
                xk = np.array([ dict_opt_lv3["P_trafo_lv"][-2],
                                dict_opt_lv3["Q_trafo_lv"][-2],
                                dict_opt_lv3["v2_trafo"][-2]
                               ]).reshape(3,1)              
                sk = xk_next - xk
                gradk_next = np.array([ dict_opt_lv3["dual_Pn"][-1] - dict_opt_lv3["dual_P"][-1],
                                dict_opt_lv3["dual_Qn"][-1] - dict_opt_lv3["dual_Q"][-1],
                                dict_opt_lv3["dual_v2n"][-1] - dict_opt_lv3["dual_v2"][-1]
                               ]).reshape(3,1)
                gradk = np.array([ dict_opt_lv3["dual_Pn"][-2] - dict_opt_lv3["dual_P"][-2],
                               dict_opt_lv3["dual_Qn"][-2] - dict_opt_lv3["dual_Q"][-2],
                               dict_opt_lv3["dual_v2n"][-2] - dict_opt_lv3["dual_v2"][-2]
                               ]).reshape(3,1)
                yk = gradk_next - gradk
                if yk.T@sk >= 0.2*sk.T@Bk@sk:
                    theta_k = 1
                else:
                    theta_k = 0.8*sk.T@Bk@sk / (sk.T@Bk@sk - yk.T@sk)
                    theta_k = theta_k[0,0]
                rk = theta_k * yk + (1-theta_k) * Bk@sk
                Bk_next = Bk - Bk@sk@sk.T@Bk/(sk.T@Bk@sk) + rk@rk.T/(sk.T@rk)
                dict_newton["mat_B"][2] = Bk_next
                dict_newton["arr_c"][2] = gradk_next - Bk_next @ xk_next
                dict_newton["num_q"][2] = dict_output_opf_lv3["objVal"] - 0.5*(xk_next.T@Bk_next@xk_next)[0,0] - ((gradk_next - Bk_next @ xk_next).T@xk_next)[0,0]
        else:
            bin_ub = False
            dict_output_opf_lv3 = opf_lv3_fea(dict_input_lv3, v2_trafo_mv[2], P_trafo_mv[2], Q_trafo_mv[2])
            if print_status:            
                print("status :{}".format(dict_output_opf_lv3["status"]))
                print("beta :{}".format(dict_output_opf_lv3["objVal"]))
            # generate optimality cuts
            dict_fea_lv3["dual_P"].append(dict_output_opf_lv3["dual_P"]),
            dict_fea_lv3["dual_Pn"].append(dict_output_opf_lv3["dual_Pn"]),
            dict_fea_lv3["dual_Q"].append(dict_output_opf_lv3["dual_Q"]),
            dict_fea_lv3["dual_Qn"].append(dict_output_opf_lv3["dual_Qn"]),
            dict_fea_lv3["dual_v2"].append(dict_output_opf_lv3["dual_v2"]),
            dict_fea_lv3["dual_v2n"].append(dict_output_opf_lv3["dual_v2n"]),
            dict_fea_lv3["P_trafo_lv"].append(dict_output_opf_lv3["P_trafo_lv"]),
            dict_fea_lv3["Q_trafo_lv"].append(dict_output_opf_lv3["Q_trafo_lv"]),
            dict_fea_lv3["v2_trafo"].append(dict_output_opf_lv3["v2_trafo"])
            
        # derive UB
        if bin_ub == True: # only when all LV problems are optimal
            ub_current = dict_output_opf_mv["objVal_mv"] + dict_output_opf_lv1["objVal"] + dict_output_opf_lv2["objVal"] + dict_output_opf_lv3["objVal"]
            if ub_current <= ub:
                ub = ub_current
                dict_output_opf_mv_best = dict_output_opf_mv
                dict_output_opf_lv1_best = dict_output_opf_lv1
                dict_output_opf_lv2_best = dict_output_opf_lv2
                dict_output_opf_lv3_best = dict_output_opf_lv3
        list_ub.append(ub)
        print("Gap is {}.".format(ub - lb))
        
    dict_output = {}
    dict_output["resid"] = ub-lb
    dict_output["n_iter"] = n_iterations
    dict_output["lb"] = np.array(list_lb)
    dict_output["ub"] = np.array(list_ub)
    dict_output["p_sgen_mv"] = dict_output_opf_mv_best["p_sgen_mv"]
    dict_output["p_sgen_lv1"] = dict_output_opf_lv1_best["p_sgen_lv1"]
    dict_output["p_sgen_lv2"] = dict_output_opf_lv2_best["p_sgen_lv2"]
    dict_output["p_sgen_lv3"] = dict_output_opf_lv3_best["p_sgen_lv3"]
    dict_output["q_sgen_mv"] = dict_output_opf_mv_best["q_sgen_mv"]
    dict_output["q_sgen_lv1"] = dict_output_opf_lv1_best["q_sgen_lv1"]
    dict_output["q_sgen_lv2"] = dict_output_opf_lv2_best["q_sgen_lv2"]
    dict_output["q_sgen_lv3"] = dict_output_opf_lv3_best["q_sgen_lv3"]
    dict_output["vm_pu_mv"] = dict_output_opf_mv_best["vm_pu_mv"]
    dict_output["vm_pu_lv1"] = dict_output_opf_lv1_best["vm_pu_lv1"]
    dict_output["vm_pu_lv2"] = dict_output_opf_lv2_best["vm_pu_lv2"]
    dict_output["vm_pu_lv3"] = dict_output_opf_lv3_best["vm_pu_lv3"]
    dict_output["P_trafo_lv"] = np.array([dict_output_opf_lv1_best["P_trafo_lv"],dict_output_opf_lv2_best["P_trafo_lv"],dict_output_opf_lv3_best["P_trafo_lv"]])
    dict_output["Q_trafo_lv"] = np.array([dict_output_opf_lv1_best["Q_trafo_lv"],dict_output_opf_lv2_best["Q_trafo_lv"],dict_output_opf_lv3_best["Q_trafo_lv"]])
    dict_output["loading_trafo"] = np.array([dict_output_opf_lv1_best["loading_trafo"],dict_output_opf_lv2_best["loading_trafo"],dict_output_opf_lv3_best["loading_trafo"]])
    dict_output["loading_line_mv"] = dict_output_opf_mv_best["loading_line_mv"]
    dict_output["loading_line_lv1"] = dict_output_opf_lv1_best["loading_line_lv1"]
    dict_output["loading_line_lv2"] = dict_output_opf_lv2_best["loading_line_lv2"]
    dict_output["loading_line_lv3"] = dict_output_opf_lv3_best["loading_line_lv3"]
    dict_output["p_curt_mv"] = np.array([arr_mv_sgen[i]-dict_output_opf_mv_best["p_sgen_mv"][i] for i in range(n_bus_mv)])
    dict_output["p_curt_lv1"] = np.array([arr_lv1_sgen[i]-dict_output_opf_lv1_best["p_sgen_lv1"][i] for i in range(n_bus_lv1_vec)])
    dict_output["p_curt_lv2"] = np.array([arr_lv2_sgen[i]-dict_output_opf_lv2_best["p_sgen_lv2"][i] for i in range(n_bus_lv2_vec)])
    dict_output["p_curt_lv3"] = np.array([arr_lv3_sgen[i]-dict_output_opf_lv3_best["p_sgen_lv3"][i] for i in range(n_bus_lv3_vec)])
    arr_sgen_mv_inv = 1/arr_mv_sgen
    arr_sgen_mv_inv[arr_sgen_mv_inv == inf] = 0
    arr_sgen_lv1_inv = 1/arr_lv1_sgen
    arr_sgen_lv1_inv[arr_sgen_lv1_inv == inf] = 0
    arr_sgen_lv2_inv = 1/arr_lv2_sgen
    arr_sgen_lv2_inv[arr_sgen_lv2_inv == inf] = 0
    arr_sgen_lv3_inv = 1/arr_lv3_sgen
    arr_sgen_lv3_inv[arr_sgen_lv3_inv == inf] = 0
    dict_output["jain_idx"] = jain(np.concatenate(((dict_output["p_curt_mv"]*arr_sgen_mv_inv)[cap_sgen_mv!=0],
                                                    (dict_output["p_curt_lv1"]*arr_sgen_lv1_inv)[cap_sgen_lv1!=0],
                                                    (dict_output["p_curt_lv2"]*arr_sgen_lv2_inv)[cap_sgen_lv2!=0],
                                                    (dict_output["p_curt_lv3"]*arr_sgen_lv3_inv)[cap_sgen_lv3!=0])))  
    dict_output["std"] = np.std(np.concatenate(((dict_output["p_curt_mv"]*arr_sgen_mv_inv)[cap_sgen_mv!=0],
                                                    (dict_output["p_curt_lv1"]*arr_sgen_lv1_inv)[cap_sgen_lv1!=0],
                                                    (dict_output["p_curt_lv2"]*arr_sgen_lv2_inv)[cap_sgen_lv2!=0],
                                                    (dict_output["p_curt_lv3"]*arr_sgen_lv3_inv)[cap_sgen_lv3!=0])))
    dict_output["mean"] = np.mean(np.concatenate(((dict_output["p_curt_mv"]*arr_sgen_mv_inv)[cap_sgen_mv!=0],
                                                    (dict_output["p_curt_lv1"]*arr_sgen_lv1_inv)[cap_sgen_lv1!=0],
                                                    (dict_output["p_curt_lv2"]*arr_sgen_lv2_inv)[cap_sgen_lv2!=0],
                                                    (dict_output["p_curt_lv3"]*arr_sgen_lv3_inv)[cap_sgen_lv3!=0])))
    dict_output["curt_prop"] = np.concatenate(((dict_output["p_curt_mv"]*arr_sgen_mv_inv)[cap_sgen_mv!=0],
                                                    (dict_output["p_curt_lv1"]*arr_sgen_lv1_inv)[cap_sgen_lv1!=0],
                                                    (dict_output["p_curt_lv2"]*arr_sgen_lv2_inv)[cap_sgen_lv2!=0],
                                                    (dict_output["p_curt_lv3"]*arr_sgen_lv3_inv)[cap_sgen_lv3!=0]))
    dict_output["curt_prop_full"] = np.concatenate(((dict_output["p_curt_mv"]*arr_sgen_mv_inv),
                                                    (dict_output["p_curt_lv1"]*arr_sgen_lv1_inv),
                                                    (dict_output["p_curt_lv2"]*arr_sgen_lv2_inv),
                                                    (dict_output["p_curt_lv3"]*arr_sgen_lv3_inv)))    
    return dict_output

