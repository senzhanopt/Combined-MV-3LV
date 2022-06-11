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
print_status = True

def jain(x): # Jain's fairness index
    x = np.array(x)
    return (np.mean(x))**2/np.mean(x*x) # if np.mean(x*x) > 0.01 else 0

def opf_mv(dict_input_mv, lambda1, lambda2, lambda3, var_bd_mv, var_bd_lv1, var_bd_lv2, var_bd_lv3, c, arr_b, objective): 
    '''
    lambda is np array of [lambda_P, lambda_Q, lambda_v2]
    '''
    #------------------------  Load Profiles ----------------------------------
    # sgen capacity
    cap_sgen_mv = dict_input_mv["cap_sgen_mv"]
    # load: P, Q
    arr_mv_p = dict_input_mv["arr_mv_p"]
    arr_mv_q = dict_input_mv["arr_mv_q"]
    # sgen: P
    arr_mv_sgen = dict_input_mv["arr_mv_sgen"]
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

    # --------------- dd obj modification ------------------------------------
    modification = lambda1[0]*P_trafo_mv[0] + lambda1[1]*Q_trafo_mv[0] + lambda1[2]*v2_mv[1]\
        + lambda2[0]*P_trafo_mv[1] + lambda2[1]*Q_trafo_mv[1] + lambda2[2]*v2_mv[5]\
            + lambda3[0]*P_trafo_mv[2] + lambda3[1]*Q_trafo_mv[2] + lambda3[2]*v2_mv[9]\
                + c[0] * (var_bd_mv[0]-var_bd_lv1[0]) * P_trafo_mv[0]\
                + c[1] * (var_bd_mv[1]-var_bd_lv1[1]) * Q_trafo_mv[0]\
                + c[2] * (var_bd_mv[2]-var_bd_lv1[2]) * v2_mv[1]\
                + c[3] * (var_bd_mv[3]-var_bd_lv2[0]) * P_trafo_mv[1]\
                + c[4] * (var_bd_mv[4]-var_bd_lv2[1]) * Q_trafo_mv[1]\
                + c[5] * (var_bd_mv[5]-var_bd_lv2[2]) * v2_mv[5]\
                + c[6] * (var_bd_mv[6]-var_bd_lv3[0]) * P_trafo_mv[2]\
                + c[7] * (var_bd_mv[7]-var_bd_lv3[1]) * Q_trafo_mv[2]\
                + c[8] * (var_bd_mv[8]-var_bd_lv3[2]) * v2_mv[9]\
                + 0.5 * (
                     arr_b[0] * (P_trafo_mv[0] - var_bd_mv[0]) * (P_trafo_mv[0] - var_bd_mv[0]) 
                    +  arr_b[1] * (Q_trafo_mv[0] - var_bd_mv[1]) * (Q_trafo_mv[0] - var_bd_mv[1])
                    +  arr_b[2] * (v2_mv[1] - var_bd_mv[2]) * (v2_mv[1] - var_bd_mv[2])
                    +  arr_b[3] * (P_trafo_mv[1] - var_bd_mv[3]) * (P_trafo_mv[1] - var_bd_mv[3]) 
                    +  arr_b[4] * (Q_trafo_mv[1] - var_bd_mv[4]) * (Q_trafo_mv[1] - var_bd_mv[4])
                    +  arr_b[5] * (v2_mv[5] - var_bd_mv[5]) * (v2_mv[5] - var_bd_mv[5])
                    +  arr_b[6] * (P_trafo_mv[2] - var_bd_mv[6]) * (P_trafo_mv[2] - var_bd_mv[6]) 
                    +  arr_b[7] * (Q_trafo_mv[2] - var_bd_mv[7]) * (Q_trafo_mv[2] - var_bd_mv[7])
                    +  arr_b[8] * (v2_mv[9] - var_bd_mv[8]) * (v2_mv[9] - var_bd_mv[8])
                        )
                           
    # ----------------- objective function -----------------------------------
    np.seterr(divide='ignore')
    arr_sgen_mv_inv = 1/arr_mv_sgen
    arr_sgen_mv_inv[arr_sgen_mv_inv == inf] = 0    
    obj_prop = quicksum((arr_mv_sgen[i]-p_sgen_mv[i])*(arr_mv_sgen[i]-p_sgen_mv[i])*(arr_sgen_mv_inv[i]) for i in range(n_bus_mv)) + modification
    obj_quad = quicksum((arr_mv_sgen[i]-p_sgen_mv[i])*(arr_mv_sgen[i]-p_sgen_mv[i]) for i in range(n_bus_mv)) + modification
    obj_lin = quicksum((arr_mv_sgen[i]-p_sgen_mv[i]) for i in range(n_bus_mv)) + modification
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
    dict_output["P_trafo_mv"] = np.array([P_trafo_mv[i].X for i in range(n_trafo)])
    dict_output["Q_trafo_mv"] = np.array([Q_trafo_mv[i].X for i in range(n_trafo)])
    dict_output["v2_trafo_mv"] = np.array([v2_mv[1].X, v2_mv[5].X, v2_mv[9].X])
    dict_output["p_sgen_mv"] = np.array([p_sgen_mv[i].X for i in range(n_bus_mv)])
    dict_output["q_sgen_mv"] = np.array([q_sgen_mv[i].X for i in range(n_bus_mv)]) 
    dict_output["vm_pu_mv"] = np.array([np.sqrt(v2_mv[i].X)/v_mv for i in range(n_bus_mv+1)])
    dict_output["loading_line_mv"] = 100*np.array([np.sqrt((P_mv[l].X)**2+(Q_mv[l].X)**2)/(nw_mv_rural_cable.max_i_ka[l] * v_mv) for l in range(n_cable_mv)])
    return dict_output
    
def opf_lv1(dict_input_lv1, lambda1, var_bd_mv, var_bd_lv1, c,  arr_b, objective):
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
    Q_lv1 = m.addVars(n_bus_lv1_vec-1, lb = -GRB.INFINITY, name = "LV1_cable_Qflow") # length of 13
    # transformer power flow
    P_trafo_lv = m.addVar(lb = -GRB.INFINITY, name = "Trafo_Pflow_LV") # flow in LV
    Q_trafo_lv = m.addVar(lb = -GRB.INFINITY, name = "Trafo_Qflow_LV") # flow in LV
    v2_trafo = m.addVar(lb = -GRB.INFINITY, name = "Trafo_volt2_mvside") 
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
    # voltage relations
    m.addConstr(v2_trafo*turns_ratio**2-v2_lv1[3] == 2*(r_ohm_160kva*P_trafo_lv+x_ohm_160kva*Q_trafo_lv))  
    # --------------- dd obj modification ------------------------------------
    modification = -(lambda1[0]*P_trafo_lv + lambda1[1]*Q_trafo_lv + lambda1[2]*v2_trafo) - c[0] * (var_bd_mv[0]-var_bd_lv1[0]) * P_trafo_lv\
                - c[1] * (var_bd_mv[1]-var_bd_lv1[1]) * Q_trafo_lv\
                - c[2] * (var_bd_mv[2]-var_bd_lv1[2]) * v2_trafo + 0.5*(
         arr_b[0] * (P_trafo_lv-var_bd_lv1[0])*(P_trafo_lv-var_bd_lv1[0]) +  arr_b[1] * (Q_trafo_lv-var_bd_lv1[1])*(Q_trafo_lv-var_bd_lv1[1]) + 
         arr_b[2] * (v2_trafo-var_bd_lv1[2])*(v2_trafo-var_bd_lv1[2]))       
    # ----------------------- objective function -----------------------------  
    obj_quad = quicksum((arr_lv1_sgen[i]-p_sgen_lv1[i])*(arr_lv1_sgen[i]-p_sgen_lv1[i]) for i in range(n_bus_lv1_vec)) + modification
    obj_lin = quicksum((arr_lv1_sgen[i]-p_sgen_lv1[i]) for i in range(n_bus_lv1_vec)) + modification
    np.seterr(divide='ignore')
    arr_sgen_lv1_inv = 1/arr_lv1_sgen
    arr_sgen_lv1_inv[arr_sgen_lv1_inv == inf] = 0
    obj_prop = quicksum((arr_lv1_sgen[i]-p_sgen_lv1[i])*(arr_lv1_sgen[i]-p_sgen_lv1[i])*(arr_sgen_lv1_inv[i]) for i in range(n_bus_lv1_vec)) + modification
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
        dict_output["objVal"] = m.objVal
    return dict_output


def opf_lv2(dict_input_lv2, lambda2, var_bd_mv, var_bd_lv2, c,  arr_b, objective):
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
    m.Params.NumericFocus = 3
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
    v2_trafo = m.addVar(lb = -GRB.INFINITY, name = "Trafo_volt2_mvside") 
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
            
    # voltage relations
    m.addConstr(1e1*v2_trafo*turns_ratio**2-1e1*v2_lv2[62] ==1e1*2*(r_ohm_630kva*P_trafo_lv+x_ohm_630kva*Q_trafo_lv))  
    # --------------- dd obj modification ------------------------------------
    modification = -(lambda2[0]*P_trafo_lv + lambda2[1]*Q_trafo_lv + lambda2[2]*v2_trafo) - c[0] * (var_bd_mv[0]-var_bd_lv2[0]) * P_trafo_lv\
                - c[1] * (var_bd_mv[1]-var_bd_lv2[1]) * Q_trafo_lv\
                - c[2] * (var_bd_mv[2]-var_bd_lv2[2]) * v2_trafo + 0.5*(
         arr_b[0] * (P_trafo_lv-var_bd_lv2[0])*(P_trafo_lv-var_bd_lv2[0]) +  arr_b[1] * (Q_trafo_lv-var_bd_lv2[1])*(Q_trafo_lv-var_bd_lv2[1]) + 
         arr_b[2] * (v2_trafo-var_bd_lv2[2])*(v2_trafo-var_bd_lv2[2]))      
    # ----------------------- objective function -----------------------------  
    obj_quad = quicksum((arr_lv2_sgen[i]-p_sgen_lv2[i])*(arr_lv2_sgen[i]-p_sgen_lv2[i]) for i in range(n_bus_lv2_vec)) + modification
    obj_lin = quicksum((arr_lv2_sgen[i]-p_sgen_lv2[i]) for i in range(n_bus_lv2_vec)) + modification
    np.seterr(divide='ignore')
    arr_sgen_lv2_inv = 1/arr_lv2_sgen
    arr_sgen_lv2_inv[arr_sgen_lv2_inv == inf] = 0
    obj_prop = quicksum((arr_lv2_sgen[i]-p_sgen_lv2[i])*(arr_lv2_sgen[i]-p_sgen_lv2[i])*(arr_sgen_lv2_inv[i]) for i in range(n_bus_lv2_vec))+ modification      
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
        dict_output["objVal"] = m.objVal
    return dict_output


def opf_lv3(dict_input_lv3, lambda3, var_bd_mv, var_bd_lv3, c,  arr_b, objective):
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
    m.Params.NumericFocus = 3
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
     
    # voltage relations
    m.addConstr(v2_trafo*turns_ratio**2-v2_lv3[104]==2*(r_ohm_630kva*P_trafo_lv+x_ohm_630kva*Q_trafo_lv))    
    # --------------- dd obj modification ------------------------------------
    modification = -(lambda3[0]*P_trafo_lv + lambda3[1]*Q_trafo_lv + lambda3[2]*v2_trafo) - c[0] * (var_bd_mv[0]-var_bd_lv3[0]) * P_trafo_lv\
                - c[1] * (var_bd_mv[1]-var_bd_lv3[1]) * Q_trafo_lv\
                - c[2] * (var_bd_mv[2]-var_bd_lv3[2]) * v2_trafo + 0.5*(
         arr_b[0] * (P_trafo_lv-var_bd_lv3[0])*(P_trafo_lv-var_bd_lv3[0]) +  arr_b[1] * (Q_trafo_lv-var_bd_lv3[1])*(Q_trafo_lv-var_bd_lv3[1]) + 
         arr_b[2] * (v2_trafo-var_bd_lv3[2])*(v2_trafo-var_bd_lv3[2]))   
    # ----------------------- objective function -----------------------------  
    obj_quad = quicksum((arr_lv3_sgen[i]-p_sgen_lv3[i])*(arr_lv3_sgen[i]-p_sgen_lv3[i]) for i in range(n_bus_lv3_vec)) + modification
    obj_lin = quicksum((arr_lv3_sgen[i]-p_sgen_lv3[i]) for i in range(n_bus_lv3_vec))+ modification
    np.seterr(divide='ignore')
    arr_sgen_lv3_inv = 1/arr_lv3_sgen
    arr_sgen_lv3_inv[arr_sgen_lv3_inv == inf] = 0
    obj_prop = quicksum((arr_lv3_sgen[i]-p_sgen_lv3[i])*(arr_lv3_sgen[i]-p_sgen_lv3[i])*(arr_sgen_lv3_inv[i]) for i in range(n_bus_lv3_vec)) + modification
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
        dict_output["objVal"] = m.objVal
    return dict_output

def opf_mv_lv(dict_input, objective, epsilon = 1E-4):
    '''
    unit in kv, kA, ohm, mva,
    decision variables are sgen output (real & reactive power),
    dual decomposition
    '''  
    # grid topology
    n_bus_mv = 12 # 4~15
    n_bus_lv1_vec = 14
    n_bus_lv2_vec = 96
    n_bus_lv3_vec = 128
    v_mv = 20/np.sqrt(3)
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
    # ----------------- dual decomposition iterations ------------------------
    lambda1 = np.array([1,0,0])*1.0
    lambda2 = np.array([1,0,0])*1.0
    lambda3 = np.array([1,0,0])*1.0
    resid = 1E3
    resid_best = 1E3
    n_iterations = 0
    max_iterations=1000
    c = np.ones(9)
    arr_b = np.ones(9) * 2
    var_bd_lv1 = np.array([0,0,v_mv**2])
    var_bd_lv2 = np.array([0,0,v_mv**2])
    var_bd_lv3 = np.array([0,0,v_mv**2])
    var_bd_mv = np.array([0,0,v_mv**2]*3)
    while (resid >= epsilon and n_iterations < max_iterations):
        n_iterations += 1
        print("Iteration {}".format(n_iterations))
        dict_output_opf_mv = opf_mv(dict_input_mv, lambda1, lambda2, lambda3, var_bd_mv, var_bd_lv1, var_bd_lv2, var_bd_lv3, c,  arr_b, objective)
        P_trafo_mv = dict_output_opf_mv["P_trafo_mv"]
        Q_trafo_mv = dict_output_opf_mv["Q_trafo_mv"]
        v2_trafo_mv = dict_output_opf_mv["v2_trafo_mv"] 
        dict_output_opf_lv1 = opf_lv1(dict_input_lv1, lambda1, var_bd_mv[0:3], var_bd_lv1, c[0:3],  arr_b[0:3], objective)
        dict_output_opf_lv2 = opf_lv2(dict_input_lv2, lambda2, var_bd_mv[3:6], var_bd_lv2, c[3:6],  arr_b[3:6], objective)
        dict_output_opf_lv3 = opf_lv3(dict_input_lv3, lambda3, var_bd_mv[6:9], var_bd_lv3, c[6:9],  arr_b[6:9], objective)
        # primal residual
        P_trafo_lv1 = dict_output_opf_lv1["P_trafo_lv"]
        Q_trafo_lv1 = dict_output_opf_lv1["Q_trafo_lv"]
        v2_trafo_lv1 = dict_output_opf_lv1["v2_trafo"]     
        P_trafo_lv2 = dict_output_opf_lv2["P_trafo_lv"]
        Q_trafo_lv2 = dict_output_opf_lv2["Q_trafo_lv"]
        v2_trafo_lv2 = dict_output_opf_lv2["v2_trafo"] 
        P_trafo_lv3 = dict_output_opf_lv3["P_trafo_lv"]
        Q_trafo_lv3 = dict_output_opf_lv3["Q_trafo_lv"]
        v2_trafo_lv3 = dict_output_opf_lv3["v2_trafo"] 
        resid_primary = np.array([abs(P_trafo_mv[0]-P_trafo_lv1), abs(Q_trafo_mv[0]-Q_trafo_lv1), abs(v2_trafo_mv[0]-v2_trafo_lv1), 
                                  abs(P_trafo_mv[1]-P_trafo_lv2), abs(Q_trafo_mv[1]-Q_trafo_lv2), abs(v2_trafo_mv[1]-v2_trafo_lv2),
                                  abs(P_trafo_mv[2]-P_trafo_lv3), abs(Q_trafo_mv[2]-Q_trafo_lv3), abs(v2_trafo_mv[2]-v2_trafo_lv3) ])
        # dual residual
        resid_dual = c * abs(np.array([P_trafo_lv1, Q_trafo_lv1, v2_trafo_lv1,P_trafo_lv2, Q_trafo_lv2, v2_trafo_lv2,P_trafo_lv3, Q_trafo_lv3, v2_trafo_lv3])
                            -np.concatenate((var_bd_lv1, var_bd_lv2, var_bd_lv3)))
        resid = max(np.concatenate((resid_primary, resid_dual)))
        print("Primary residual is {}, dual residual is {}.".format(max(resid_primary), max(resid_dual)))
        if resid < resid_best:
            dict_output_opf_mv_best = dict_output_opf_mv
            dict_output_opf_lv1_best = dict_output_opf_lv1
            dict_output_opf_lv2_best = dict_output_opf_lv2
            dict_output_opf_lv3_best = dict_output_opf_lv3
            resid_best = resid
            
        var_bd_lv1 = np.array([P_trafo_lv1, Q_trafo_lv1, v2_trafo_lv1])
        var_bd_lv2 = np.array([P_trafo_lv2, Q_trafo_lv2, v2_trafo_lv2])
        var_bd_lv3 = np.array([P_trafo_lv3, Q_trafo_lv3, v2_trafo_lv3])
        var_bd_mv = np.array([P_trafo_mv[0],Q_trafo_mv[0],v2_trafo_mv[0],P_trafo_mv[1],Q_trafo_mv[1],v2_trafo_mv[1],P_trafo_mv[2],Q_trafo_mv[2],v2_trafo_mv[2]])
        # store results for dual residual calculation
        lambda1 += c[0:3] * np.array([P_trafo_mv[0]-P_trafo_lv1, Q_trafo_mv[0]-Q_trafo_lv1, v2_trafo_mv[0]-v2_trafo_lv1])
        lambda2 += c[3:6] * np.array([P_trafo_mv[1]-P_trafo_lv2, Q_trafo_mv[1]-Q_trafo_lv2, v2_trafo_mv[1]-v2_trafo_lv2])
        lambda3 += c[6:9] * np.array([P_trafo_mv[2]-P_trafo_lv3, Q_trafo_mv[2]-Q_trafo_lv3, v2_trafo_mv[2]-v2_trafo_lv3])
        # update c, b
        for i in range(9):
            if resid_primary[i] > 10 * resid_dual[i]:
                c[i] = c[i] * 2
                arr_b[i] = 2 * c[i]
            if resid_dual[i] > 10 * resid_primary[i]:
                c[i] = c[i] / 2
                arr_b[i] = 2 * c[i]
            
    dict_output_opf_mv = dict_output_opf_mv_best
    dict_output_opf_lv1 = dict_output_opf_lv1_best
    dict_output_opf_lv2 = dict_output_opf_lv2_best
    dict_output_opf_lv3 = dict_output_opf_lv3_best
    dict_output = {}  
    dict_output["resid"] = resid_best
    dict_output["n_iter"] = n_iterations
    dict_output["p_sgen_mv"] = dict_output_opf_mv["p_sgen_mv"]
    dict_output["p_sgen_lv1"] = dict_output_opf_lv1["p_sgen_lv1"]
    dict_output["p_sgen_lv2"] = dict_output_opf_lv2["p_sgen_lv2"]
    dict_output["p_sgen_lv3"] = dict_output_opf_lv3["p_sgen_lv3"]
    dict_output["q_sgen_mv"] = dict_output_opf_mv["q_sgen_mv"]
    dict_output["q_sgen_lv1"] = dict_output_opf_lv1["q_sgen_lv1"]
    dict_output["q_sgen_lv2"] = dict_output_opf_lv2["q_sgen_lv2"]
    dict_output["q_sgen_lv3"] = dict_output_opf_lv3["q_sgen_lv3"]
    dict_output["vm_pu_mv"] = dict_output_opf_mv["vm_pu_mv"]
    dict_output["vm_pu_lv1"] = dict_output_opf_lv1["vm_pu_lv1"]
    dict_output["vm_pu_lv2"] = dict_output_opf_lv2["vm_pu_lv2"]
    dict_output["vm_pu_lv3"] = dict_output_opf_lv3["vm_pu_lv3"]
    dict_output["P_trafo_lv"] = np.array([dict_output_opf_lv1["P_trafo_lv"],dict_output_opf_lv2["P_trafo_lv"],dict_output_opf_lv3["P_trafo_lv"]])
    dict_output["Q_trafo_lv"] = np.array([dict_output_opf_lv1["Q_trafo_lv"],dict_output_opf_lv2["Q_trafo_lv"],dict_output_opf_lv3["Q_trafo_lv"]])
    dict_output["loading_trafo"] = np.array([dict_output_opf_lv1["loading_trafo"],dict_output_opf_lv2["loading_trafo"],dict_output_opf_lv3["loading_trafo"]])
    dict_output["loading_line_mv"] = dict_output_opf_mv["loading_line_mv"]
    dict_output["loading_line_lv1"] = dict_output_opf_lv1["loading_line_lv1"]
    dict_output["loading_line_lv2"] = dict_output_opf_lv2["loading_line_lv2"]
    dict_output["loading_line_lv3"] = dict_output_opf_lv3["loading_line_lv3"]
    dict_output["p_curt_mv"] = np.array([arr_mv_sgen[i]-dict_output_opf_mv["p_sgen_mv"][i] for i in range(n_bus_mv)])
    dict_output["p_curt_lv1"] = np.array([arr_lv1_sgen[i]-dict_output_opf_lv1["p_sgen_lv1"][i] for i in range(n_bus_lv1_vec)])
    dict_output["p_curt_lv2"] = np.array([arr_lv2_sgen[i]-dict_output_opf_lv2["p_sgen_lv2"][i] for i in range(n_bus_lv2_vec)])
    dict_output["p_curt_lv3"] = np.array([arr_lv3_sgen[i]-dict_output_opf_lv3["p_sgen_lv3"][i] for i in range(n_bus_lv3_vec)])
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

