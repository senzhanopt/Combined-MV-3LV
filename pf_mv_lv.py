import pandapower as pp
import pandas as pd
import numpy as np
#from pandapower.plotting import simple_plot

nw_mv_rural_bus = pd.read_excel("mv_rural.xlsx", sheet_name="bus")
nw_mv_rural_cable = pd.read_excel("mv_rural.xlsx", sheet_name="cable")
nw_lv_rural1_bus = pd.read_excel("lv_rural1.xlsx", sheet_name="bus")
nw_lv_rural1_cable = pd.read_excel("lv_rural1.xlsx", sheet_name="cable")
nw_lv_rural1_trafo = pd.read_excel("lv_rural1.xlsx", sheet_name="trafo")
nw_lv_rural2_bus = pd.read_excel("lv_rural2.xlsx", sheet_name="bus")
nw_lv_rural2_cable = pd.read_excel("lv_rural2.xlsx", sheet_name="cable")
nw_lv_rural3_bus = pd.read_excel("lv_rural3.xlsx", sheet_name="bus")
nw_lv_rural3_cable = pd.read_excel("lv_rural3.xlsx", sheet_name="cable")

def pf_mv_lv(dict_input):
    '''
    returns power flow results for a combined MV-LV network,
    topology data from simBench,
    MV_rural feeder 1, LV_rural1, LV_rural2, LV_rural3
    '''
    #======================== Load profiles ==================================
    p_load_mv = dict_input["p_load_mv"] # array of length 12: bus 4~15
    q_load_mv = dict_input["q_load_mv"]
    p_load_lv1 = dict_input["p_load_lv1"] # array of length 14
    q_load_lv1 = dict_input["q_load_lv1"]
    p_load_lv2 = dict_input["p_load_lv2"] # array of length 96
    q_load_lv2 = dict_input["q_load_lv2"]
    p_load_lv3 = dict_input["p_load_lv3"] # array of length 128
    q_load_lv3 = dict_input["q_load_lv3"]
    #======================= Medium voltage ==================================
    net = pp.create_empty_network()
    n_bus_mv_vec = 16 # final bus of feeder 1 indexed 15
    bus_mv_vec = [0] * n_bus_mv_vec
    idx_ext = 2
    for i in range(1,14): # 13 bus, bus 0 is the HV bus, HV/MV trafo excluded
        bus_mv_vec[nw_mv_rural_bus.idx[i]] = pp.create_bus(net, 
                                           name = nw_mv_rural_bus.name[i],
                                           type = nw_mv_rural_bus.type[i],
                                         vn_kv = nw_mv_rural_bus.vn_kv[i],
                     geodata = (nw_mv_rural_bus.x[i],nw_mv_rural_bus.y[i])
                     )
    pp.create_ext_grid(net, bus_mv_vec[idx_ext])       
    # create cable connections
    n_cable_mv = 12
    for l in range(n_cable_mv): # radial structure
        if nw_mv_rural_cable.switch[l] != 'OPEN': # useful with full grid
            pp.create_line(net,
                        from_bus = bus_mv_vec[nw_mv_rural_cable.from_bus[l]], 
                        to_bus = bus_mv_vec[nw_mv_rural_cable.to_bus[l]], 
                        length_km = nw_mv_rural_cable.length_km[l],
                        std_type = nw_mv_rural_cable.std_type[l],
                        name = nw_mv_rural_cable.name[l])
            
    #======================== Low voltage 1===================================
    n_bus_lv1_vec = 14
    bus_lv1_vec = [0] * n_bus_lv1_vec
    for i in range(n_bus_lv1_vec):
        bus_lv1_vec[i] = pp.create_bus(net, vn_kv = nw_lv_rural1_bus.vn_kv[i],
            name = nw_lv_rural1_bus.name[i], type = nw_lv_rural1_bus.type[i],
            geodata = (nw_lv_rural1_bus.x[i]+3E-3,nw_lv_rural1_bus.y[i]-1E-3)
            )    # shift geodata for better visualization
    # create cable connections
    for l in nw_lv_rural1_cable.index:
        pp.create_line(net,
                    from_bus = bus_lv1_vec[nw_lv_rural1_cable.from_bus[l]], 
                    to_bus = bus_lv1_vec[nw_lv_rural1_cable.to_bus[l]], 
                    length_km = nw_lv_rural1_cable.length_km[l],
                    std_type = 'NAYY 4x150 SE',
                    name = nw_lv_rural1_cable.name[l])  
        
    #======================== Low voltage 2===================================
    n_bus_lv2_vec = 96
    bus_lv2_vec = [0] * n_bus_lv2_vec
    for i in range(n_bus_lv2_vec):
        bus_lv2_vec[i] = pp.create_bus(net, vn_kv = nw_lv_rural2_bus.vn_kv[i],
            name = nw_lv_rural2_bus.name[i], type = nw_lv_rural2_bus.type[i],
            geodata = (nw_lv_rural2_bus.x[i]+4E-3,nw_lv_rural2_bus.y[i]+0E-3)
            )   # shift geodata for better visualization 
    # create cable connections
    for l in nw_lv_rural2_cable.index:
        pp.create_line(net,
                    from_bus = bus_lv2_vec[nw_lv_rural2_cable.from_bus[l]], 
                    to_bus = bus_lv2_vec[nw_lv_rural2_cable.to_bus[l]], 
                    length_km = nw_lv_rural2_cable.length_km[l],
                    std_type = 'NAYY 4x150 SE',
                    name = nw_lv_rural2_cable.name[l])  
        
    #======================== Low voltage 3===================================
    n_bus_lv3_vec = 128
    bus_lv3_vec = [0] * n_bus_lv3_vec
    for i in range(n_bus_lv3_vec):
        bus_lv3_vec[i] = pp.create_bus(net, vn_kv = nw_lv_rural3_bus.vn_kv[i],
            name = nw_lv_rural3_bus.name[i], type = nw_lv_rural3_bus.type[i],
            geodata = (nw_lv_rural3_bus.x[i]-8E-3,nw_lv_rural3_bus.y[i]-1E-3)
            )    
    # create cable connections
    for l in nw_lv_rural3_cable.index:
        pp.create_line(net,
                    from_bus = bus_lv3_vec[nw_lv_rural3_cable.from_bus[l]], 
                    to_bus = bus_lv3_vec[nw_lv_rural3_cable.to_bus[l]], 
                    length_km = nw_lv_rural3_cable.length_km[l],
                    std_type = 'NAYY 4x150 SE',
                    name = nw_lv_rural3_cable.name[l]) 
    
    #=================== MV-LV connection points =============================
    idx_mv_lv1 = 4
    idx_mv_lv2 = 8
    idx_mv_lv3 = 12
    idx_lv1 = 3
    idx_lv2 = 62
    idx_lv3 = 104   
    pp.create_transformer_from_parameters(net, hv_bus = bus_mv_vec[idx_mv_lv1], 
                lv_bus = bus_lv1_vec[idx_lv1], 
                sn_mva = nw_lv_rural1_trafo.sn_mva,
                vn_hv_kv = nw_lv_rural1_trafo.vn_hv_kv,
                vn_lv_kv = nw_lv_rural1_trafo.vn_lv_kv,
                vk_percent = nw_lv_rural1_trafo.vk_percent,
                vkr_percent = nw_lv_rural1_trafo.vkr_percent,
                pfe_kw = nw_lv_rural1_trafo.pfe_kw,
                i0_percent = nw_lv_rural1_trafo.i0_percent,
                shift_degree = nw_lv_rural1_trafo.shift_degree,
                tap_side = nw_lv_rural1_trafo.tap_side,
                tap_neutral = nw_lv_rural1_trafo.tap_neutral,
                tap_max = nw_lv_rural1_trafo.tap_max,
                tap_min = nw_lv_rural1_trafo.tap_min,
                tap_step_percent = nw_lv_rural1_trafo.tap_step_percent,
                tap_step_degree = nw_lv_rural1_trafo.tap_step_degree,
                tap_pos = nw_lv_rural1_trafo.tap_pos
                          )
    pp.create_transformer(net, hv_bus = bus_mv_vec[idx_mv_lv2], 
                          lv_bus = bus_lv2_vec[idx_lv2],
                          std_type = '0.63 MVA 20/0.4 kV')    
    pp.create_transformer(net, hv_bus = bus_mv_vec[idx_mv_lv3], 
                          lv_bus = bus_lv3_vec[idx_lv3],
                          std_type = '0.63 MVA 20/0.4 kV') 
    # =================== Load ===============================================
    for b in range(4,n_bus_mv_vec):
        pp.create_load(net,bus=bus_mv_vec[b],p_mw=p_load_mv[b-4], #4~15
                       q_mvar=q_load_mv[b-4]) 
    for b in range(n_bus_lv1_vec):
        pp.create_load(net,bus=bus_lv1_vec[b],p_mw=p_load_lv1[b],
                       q_mvar=q_load_lv1[b]) 
    for b in range(n_bus_lv2_vec):
        pp.create_load(net,bus=bus_lv2_vec[b],p_mw=p_load_lv2[b],
                       q_mvar=q_load_lv2[b])
    for b in range(n_bus_lv3_vec):
        pp.create_load(net,bus=bus_lv3_vec[b],p_mw=p_load_lv3[b],
                       q_mvar=q_load_lv3[b])
    pp.runpp(net)  
    dict_output = {}
    dict_output["res_bus"] = net.res_bus
    dict_output["res_line"] = net.res_line
    dict_output["res_trafo"] = net.res_trafo
    dict_output["res_ext_grid"] = net.res_ext_grid
    return dict_output

if __name__ == '__main__':
    dict_input = {}
    dict_input["p_load_mv"] = np.ones(12) * 0.2
    dict_input["q_load_mv"] = np.ones(12) * 0.08
    dict_input["p_load_lv1"] = np.ones(14) * 0.002
    dict_input["q_load_lv1"] = np.ones(14) * 0.0008
    dict_input["p_load_lv2"] = np.ones(96) * 0.002
    dict_input["q_load_lv2"] = np.ones(96) * 0.0008
    dict_input["p_load_lv3"] = np.ones(128) * 0.002
    dict_input["q_load_lv3"] = np.ones(128) * 0.0008
    dict_output = pf_mv_lv(dict_input)