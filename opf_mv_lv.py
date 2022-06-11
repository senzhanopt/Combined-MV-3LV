import gurobipy as gp
from gurobipy import GRB, quicksum
import pandas as pd
import numpy as np
import math
from numpy import inf

nw_mv_rural_cable = pd.read_excel("mv_rural.xlsx", sheet_name="cable")
nw_lv_rural1_cable = pd.read_excel("lv_rural1.xlsx", sheet_name="cable")
nw_lv_rural2_cable = pd.read_excel("lv_rural2.xlsx", sheet_name="cable")
nw_lv_rural3_cable = pd.read_excel("lv_rural3.xlsx", sheet_name="cable")

non_linear = False

def jain(x): # Jain's fairness index
    x = np.array(x)
    return (np.mean(x))**2/np.mean(x*x) # if np.mean(x*x) > 0.01 else 0

def opf_mv_lv(dict_input, objective = "obj_prop", epsilon = 1E-4):
    '''
    unit in kv, kA, ohm, mva,
    decision variables are sgen output (real & reactive power)
    '''
    #------------------------  Load Profiles ----------------------------------
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
    
    #------------------------- Parameters -------------------------------------
    # grid topology
    n_bus_mv = 12 # 4~15
    n_cable_mv = 12 # 4~15
    n_bus_lv1_vec = 14
    n_bus_lv2_vec = 96
    n_bus_lv3_vec = 128
    n_trafo = 3
    # voltage
    v_mv = 20/np.sqrt(3)
    v_lv = 0.4/np.sqrt(3)
    v_lb_mv = 0.965
    v_ub_mv = 1.055
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
    r_ohm_630kva = 1.206/100*1/0.63*0.4**2*1/0.63
    x_ohm_630kva = np.sqrt(6**2-1.206**2)/100*1/0.63*0.4**2*1/0.63
    
    #------------------------- Variables- -------------------------------------
    m = gp.Model("opf_mv_lv")
    m.Params.LogToConsole = 0
    if non_linear == False:
        m.Params.OptimalityTol = 1E-8
    # voltage magnitude squared
    v2_mv = m.addVars(n_bus_mv+1, lb = (v_mv*v_lb_mv)**2, ub = (v_mv*v_ub_mv)**2, name = "MV_bus_volt2") # of 13
    v2_lv1 = m.addVars(n_bus_lv1_vec, lb = (v_lv*v_lb_lv)**2, ub = (v_lv*v_ub_lv)**2, name = "LV1_bus_volt2") # of 14
    v2_lv2 = m.addVars(n_bus_lv2_vec, lb = (v_lv*v_lb_lv)**2, ub = (v_lv*v_ub_lv)**2, name = "LV2_bus_volt2") # of 96
    v2_lv3 = m.addVars(n_bus_lv3_vec, lb = (v_lv*v_lb_lv)**2, ub = (v_lv*v_ub_lv)**2, name = "LV3_bus_volt2") # of 128
    # sgen real power
    p_sgen_mv = m.addVars(n_bus_mv, name = "MV_P_sgen") # of 12
    p_sgen_lv1 = m.addVars(n_bus_lv1_vec, name = "LV1_P_sgen") # of 14
    p_sgen_lv2 = m.addVars(n_bus_lv2_vec, name = "LV2_P_sgen") # of 96
    p_sgen_lv3 = m.addVars(n_bus_lv3_vec, name = "LV3_P_sgen") # of 128
    if objective == "obj_log":
        log_p_sgen_mv = m.addVars(n_bus_mv, lb = -GRB.INFINITY, name = "log_MV_P_sgen") # of 12
        log_p_sgen_lv1 = m.addVars(n_bus_lv1_vec, lb = -GRB.INFINITY, name = "log_LV1_P_sgen") # of 14
        log_p_sgen_lv2 = m.addVars(n_bus_lv2_vec, lb = -GRB.INFINITY, name = "log_LV2_P_sgen") # of 96
        log_p_sgen_lv3 = m.addVars(n_bus_lv3_vec, lb = -GRB.INFINITY, name = "log_LV3_P_sgen") # of 128
        
    # sgen reactive power
    q_sgen_mv = m.addVars(n_bus_mv, lb = -GRB.INFINITY, name = "MV_Q_sgen") # of 12
    q_sgen_lv1 = m.addVars(n_bus_lv1_vec, lb = -GRB.INFINITY, name = "LV1_Q_sgen") # of 14
    q_sgen_lv2 = m.addVars(n_bus_lv2_vec, lb = -GRB.INFINITY, name = "LV2_Q_sgen") # of 96
    q_sgen_lv3 = m.addVars(n_bus_lv3_vec, lb = -GRB.INFINITY, name = "LV3_Q_sgen") # of 128
    # real power injection
    p_mv = m.addVars(n_bus_mv, lb = -GRB.INFINITY, name = "MV_bus_P_injection") # of 12
    p_lv1 = m.addVars(n_bus_lv1_vec, lb = -GRB.INFINITY, name = "LV1_bus_P_injection") # of 14
    p_lv2 = m.addVars(n_bus_lv2_vec, lb = -GRB.INFINITY, name = "LV2_bus_P_injection") # of 96
    p_lv3 = m.addVars(n_bus_lv3_vec, lb = -GRB.INFINITY, name = "LV3_bus_P_injection") # of 128
    # reactive power injection
    q_mv = m.addVars(n_bus_mv, lb = -GRB.INFINITY, name = "MV_bus_Q_injection") # of 12
    q_lv1 = m.addVars(n_bus_lv1_vec, lb = -GRB.INFINITY, name = "LV1_bus_Q_injection") # of 14
    q_lv2 = m.addVars(n_bus_lv2_vec, lb = -GRB.INFINITY, name = "LV2_bus_Q_injection") # of 96
    q_lv3 = m.addVars(n_bus_lv3_vec, lb = -GRB.INFINITY, name = "LV3_bus_Q_injection") # of 128
    # real power flow
    P_mv = m.addVars(n_cable_mv, lb = -GRB.INFINITY, name = "MV_cable_Pflow") # length of 12
    P_lv1 = m.addVars(n_bus_lv1_vec-1, lb = -GRB.INFINITY, name = "LV1_cable_Pflow") # length of 13
    P_lv2 = m.addVars(n_bus_lv2_vec-1, lb = -GRB.INFINITY, name = "LV2_cable_Pflow") # length of 95
    P_lv3 = m.addVars(n_bus_lv3_vec-1, lb = -GRB.INFINITY, name = "LV3_cable_Pflow") # length of 127
    # reactive power flow
    Q_mv = m.addVars(n_cable_mv, lb = -GRB.INFINITY, name = "MV_cable_Qflow") # length of 12
    Q_lv1 = m.addVars(n_bus_lv1_vec-1, lb = -GRB.INFINITY, name = "LV1_cable_Qflow") # length of 13
    Q_lv2 = m.addVars(n_bus_lv2_vec-1, lb = -GRB.INFINITY, name = "LV2_cable_Qflow") # length of 95
    Q_lv3 = m.addVars(n_bus_lv3_vec-1, lb = -GRB.INFINITY, name = "LV3_cable_Qflow") # length of 127
    # transformer power flow
    P_trafo_mv = m.addVars(n_trafo, lb = -GRB.INFINITY, name = "Trafo_Pflow_MV") # flow out MV
    Q_trafo_mv = m.addVars(n_trafo, lb = -GRB.INFINITY, name = "Trafo_Qflow_MV") # flow out MV
    P_trafo_lv = m.addVars(n_trafo, lb = -GRB.INFINITY, name = "Trafo_Pflow_LV") # flow in LV
    Q_trafo_lv = m.addVars(n_trafo, lb = -GRB.INFINITY, name = "Trafo_Qflow_LV") # flow in LV
    
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
            m.addConstr(P_trafo_lv[0] + p_lv1[b] == P_out)
            m.addConstr(Q_trafo_lv[0] + q_lv1[b] == Q_out)
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
            m.addConstr(P_trafo_lv[1] + p_lv2[b] == P_out)
            m.addConstr(Q_trafo_lv[1] + q_lv2[b] == Q_out)
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
            m.addConstr(P_trafo_lv[2] + p_lv3[b] == P_out)
            m.addConstr(Q_trafo_lv[2] + q_lv3[b] == Q_out)
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
    m.addConstrs(P_trafo_mv[t] == P_trafo_lv[t] for t in range(n_trafo))
    m.addConstrs(Q_trafo_mv[t] == Q_trafo_lv[t] for t in range(n_trafo))
    # voltage relations: for easier Benders decomposition here
    m.addConstr(v2_mv[1]*turns_ratio**2-v2_lv1[3] == 2*(r_ohm_160kva*P_trafo_lv[0]+x_ohm_160kva*Q_trafo_lv[0]))
    m.addConstr(v2_mv[5]*turns_ratio**2-v2_lv2[62] ==2*(r_ohm_630kva*P_trafo_lv[1]+x_ohm_630kva*Q_trafo_lv[1]))
    m.addConstr(v2_mv[9]*turns_ratio**2-v2_lv3[104]==2*(r_ohm_630kva*P_trafo_lv[2]+x_ohm_630kva*Q_trafo_lv[2]))
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
    
    # log constraints
    if objective == "obj_log":
        for i in range(n_bus_mv):
            m.addGenConstrLog(p_sgen_mv[i], log_p_sgen_mv[i])
        for i in range(n_bus_lv1_vec):
            m.addGenConstrLog(p_sgen_lv1[i], log_p_sgen_lv1[i])
        for i in range(n_bus_lv2_vec):
            m.addGenConstrLog(p_sgen_lv2[i], log_p_sgen_lv2[i])
        for i in range(n_bus_lv3_vec):
            m.addGenConstrLog(p_sgen_lv3[i], log_p_sgen_lv3[i])
            
    
    # ----------------------- objective function -----------------------------    
    obj_quad = quicksum((arr_mv_sgen[i]-p_sgen_mv[i])*(arr_mv_sgen[i]-p_sgen_mv[i]) for i in range(n_bus_mv)) + \
          quicksum((arr_lv1_sgen[i]-p_sgen_lv1[i])*(arr_lv1_sgen[i]-p_sgen_lv1[i]) for i in range(n_bus_lv1_vec)) + \
          quicksum((arr_lv2_sgen[i]-p_sgen_lv2[i])*(arr_lv2_sgen[i]-p_sgen_lv2[i]) for i in range(n_bus_lv2_vec)) + \
          quicksum((arr_lv3_sgen[i]-p_sgen_lv3[i])*(arr_lv3_sgen[i]-p_sgen_lv3[i]) for i in range(n_bus_lv3_vec)) 
    obj_lin = quicksum((arr_mv_sgen[i]-p_sgen_mv[i]) for i in range(n_bus_mv)) + \
          quicksum((arr_lv1_sgen[i]-p_sgen_lv1[i]) for i in range(n_bus_lv1_vec)) + \
          quicksum((arr_lv2_sgen[i]-p_sgen_lv2[i]) for i in range(n_bus_lv2_vec)) + \
          quicksum((arr_lv3_sgen[i]-p_sgen_lv3[i]) for i in range(n_bus_lv3_vec)) 
    arr_sgen_mv_inv = 1/arr_mv_sgen
    arr_sgen_mv_inv[arr_sgen_mv_inv == inf] = 0
    arr_sgen_lv1_inv = 1/arr_lv1_sgen
    arr_sgen_lv1_inv[arr_sgen_lv1_inv == inf] = 0
    arr_sgen_lv2_inv = 1/arr_lv2_sgen
    arr_sgen_lv2_inv[arr_sgen_lv2_inv == inf] = 0
    arr_sgen_lv3_inv = 1/arr_lv3_sgen
    arr_sgen_lv3_inv[arr_sgen_lv3_inv == inf] = 0
    obj_prop = quicksum((arr_mv_sgen[i]-p_sgen_mv[i])*(arr_mv_sgen[i]-p_sgen_mv[i])*(arr_sgen_mv_inv[i]) for i in range(n_bus_mv)) + \
          quicksum((arr_lv1_sgen[i]-p_sgen_lv1[i])*(arr_lv1_sgen[i]-p_sgen_lv1[i])*(arr_sgen_lv1_inv[i]) for i in range(n_bus_lv1_vec)) + \
          quicksum((arr_lv2_sgen[i]-p_sgen_lv2[i])*(arr_lv2_sgen[i]-p_sgen_lv2[i])*(arr_sgen_lv2_inv[i]) for i in range(n_bus_lv2_vec)) + \
          quicksum((arr_lv3_sgen[i]-p_sgen_lv3[i])*(arr_lv3_sgen[i]-p_sgen_lv3[i])*(arr_sgen_lv3_inv[i]) for i in range(n_bus_lv3_vec))
    if objective == "obj_log":
        obj_log = quicksum(log_p_sgen_mv[i]*(arr_mv_sgen[i]) for i in range(n_bus_mv)) + \
              quicksum(log_p_sgen_lv1[i]*(arr_lv1_sgen[i]) for i in range(n_bus_lv1_vec)) + \
              quicksum(log_p_sgen_lv2[i]*(arr_lv2_sgen[i]) for i in range(n_bus_lv2_vec)) + \
              quicksum(log_p_sgen_lv3[i]*(arr_lv3_sgen[i]) for i in range(n_bus_lv3_vec)) 
    if objective == "obj_lin":
        m.setObjective(obj_lin, GRB.MINIMIZE)
    elif objective == "obj_quad":
        m.setObjective(obj_quad, GRB.MINIMIZE)
    elif objective == "obj_prop":
        m.setObjective(obj_prop, GRB.MINIMIZE)
    elif objective == "obj_log":
        m.setObjective(obj_log, GRB.MAXIMIZE)
    else:
        print("Objective function error.")
        return
        
    print("======================= Starting Gurobi ===========================")
    m.optimize()
    # m.printQuality() # check constraint violation
    # ------------------------ output ----------------------------------------
    dict_output = {}
    dict_output["p_sgen_mv"] = np.array([p_sgen_mv[i].X for i in range(n_bus_mv)])
    dict_output["p_sgen_lv1"] = np.array([p_sgen_lv1[i].X for i in range(n_bus_lv1_vec)])
    dict_output["p_sgen_lv2"] = np.array([p_sgen_lv2[i].X for i in range(n_bus_lv2_vec)])
    dict_output["p_sgen_lv3"] = np.array([p_sgen_lv3[i].X for i in range(n_bus_lv3_vec)])
    dict_output["q_sgen_mv"] = np.array([q_sgen_mv[i].X for i in range(n_bus_mv)])
    dict_output["q_sgen_lv1"] = np.array([q_sgen_lv1[i].X for i in range(n_bus_lv1_vec)])
    dict_output["q_sgen_lv2"] = np.array([q_sgen_lv2[i].X for i in range(n_bus_lv2_vec)])
    dict_output["q_sgen_lv3"] = np.array([q_sgen_lv3[i].X for i in range(n_bus_lv3_vec)])
    dict_output["vm_pu_mv"] = np.array([np.sqrt(v2_mv[i].X)/v_mv for i in range(n_bus_mv+1)])
    dict_output["vm_pu_lv1"] = np.array([np.sqrt(v2_lv1[i].X)/v_lv for i in range(n_bus_lv1_vec)])
    dict_output["vm_pu_lv2"] = np.array([np.sqrt(v2_lv2[i].X)/v_lv for i in range(n_bus_lv2_vec)])
    dict_output["vm_pu_lv3"] = np.array([np.sqrt(v2_lv3[i].X)/v_lv for i in range(n_bus_lv3_vec)])
    dict_output["P_trafo_lv"] = np.array([P_trafo_lv[i].X for i in range(n_trafo)])
    dict_output["Q_trafo_lv"] = np.array([Q_trafo_lv[i].X for i in range(n_trafo)])
    dict_output["loading_trafo"] = 100*np.array([np.sqrt((P_trafo_lv[i].X)**2 + (Q_trafo_lv[i].X)**2)/[0.16/3,0.21,0.21][i] for i in range(n_trafo)])
    dict_output["loading_line_mv"] = 100*np.array([np.sqrt((P_mv[l].X)**2+(Q_mv[l].X)**2)/(nw_mv_rural_cable.max_i_ka[l] * v_mv) for l in range(n_cable_mv)])
    dict_output["loading_line_lv1"] = 100*np.array([np.sqrt((P_lv1[l].X)**2+(Q_lv1[l].X)**2)/(nw_lv_rural1_cable.max_i_ka[l] * v_lv) for l in range(n_bus_lv1_vec-1)])
    dict_output["loading_line_lv2"] = 100*np.array([np.sqrt((P_lv2[l].X)**2+(Q_lv2[l].X)**2)/(nw_lv_rural2_cable.max_i_ka[l] * v_lv) for l in range(n_bus_lv2_vec-1)])
    dict_output["loading_line_lv3"] = 100*np.array([np.sqrt((P_lv3[l].X)**2+(Q_lv3[l].X)**2)/(nw_lv_rural3_cable.max_i_ka[l] * v_lv) for l in range(n_bus_lv3_vec-1)])
    dict_output["p_curt_mv"] = np.array([arr_mv_sgen[i]-p_sgen_mv[i].X for i in range(n_bus_mv)])
    dict_output["p_curt_lv1"] = np.array([arr_lv1_sgen[i]-p_sgen_lv1[i].X for i in range(n_bus_lv1_vec)])
    dict_output["p_curt_lv2"] = np.array([arr_lv2_sgen[i]-p_sgen_lv2[i].X for i in range(n_bus_lv2_vec)])
    dict_output["p_curt_lv3"] = np.array([arr_lv3_sgen[i]-p_sgen_lv3[i].X for i in range(n_bus_lv3_vec)])
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

