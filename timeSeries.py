objective_format = "obj_prop"
fashion = "int" # integrated, separated, benders, ADMM, APP
epsilon = 1E-4
visualization = False

if fashion == "int":
    from opf_mv_lv  import opf_mv_lv
elif fashion == "sep":
    from opf_separated import opf_mv_lv
elif fashion == "ben":
    from opf_benders import opf_mv_lv
elif fashion == "dd":
    from opf_dd import opf_mv_lv
elif fashion == "admm":
    from opf_admm import opf_mv_lv
elif fashion == "app":
    from opf_app import opf_mv_lv
else:
    print("Error -- import opf module.")

from pf_mv_lv import pf_mv_lv
import simbench as sb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as md
import time
plt.style.use(['ieee'])

n_bus_mv = 12
n_cable_mv = 12
n_bus_lv1_vec = 14
n_bus_lv2_vec = 96
n_bus_lv3_vec = 128
n_trafo = 3
pv_lvl = 0.8 # LV2/LV3
scale = 2
v_lb_mv = 0.965
v_ub_mv = 1.055
v_lb_lv = 0.9
v_ub_lv = 1.1

net_lv1 = sb.get_simbench_net('1-LV-rural1--2-no_sw')
net_lv2 = sb.get_simbench_net('1-LV-rural2--2-no_sw')
net_lv3 = sb.get_simbench_net('1-LV-rural3--2-no_sw')
net_mv = sb.get_simbench_net('1-MV-rural--2-no_sw')
sgen_lv2 = pd.read_excel("lv_rural2.xlsx", sheet_name = "sgen")[0:int(n_bus_lv2_vec*pv_lvl)]
sgen_lv3 = pd.read_excel("lv_rural3.xlsx", sheet_name = "sgen")[0:int(n_bus_lv3_vec*pv_lvl)]
pvdata = pd.read_excel("pvdata.xlsx")

#================== Test case ===============================================
'''
power in pandapower is 3 phase
voltage is line voltage
'''
d = 149
n_timesteps = 96
# MV 
df_mv_p = np.zeros((n_timesteps, n_bus_mv)) #4~15
df_mv_q = np.zeros((n_timesteps, n_bus_mv))
df_mv_sgen = np.zeros((n_timesteps, n_bus_mv))
cap_sgen_mv = np.zeros(n_bus_mv)
profile_abs_mv = sb.get_absolute_values(net_mv, True)
for b in range(4,16):
    # load
    arr_load_p = np.zeros(n_timesteps)
    arr_load_q = np.zeros(n_timesteps)
    for i in net_mv.load.index:
        if b not in [4, 8, 12] and net_mv.load.bus[i] == b:# LVs connected
            arr_load_p += profile_abs_mv['load', 'p_mw'][i][d*96:d*96+n_timesteps]
            arr_load_q += profile_abs_mv['load', 'q_mvar'][i][d*96:d*96+n_timesteps]
    df_mv_p[:,b-4] = arr_load_p
    df_mv_q[:,b-4] = arr_load_q
    # sgen
    arr_sgen = np.zeros(n_timesteps)
    for i in net_mv.sgen.index:
        if b in [5, 6, 7, 14, 15] and net_mv.sgen.bus[i] == b:# LV 1 and Wind not scaled up
            arr_sgen += profile_abs_mv['sgen', 'p_mw'][i][d*96:d*96+n_timesteps]
            cap_sgen_mv[b-4] = net_mv.sgen.p_mw[i] # capacity
        if b in [9, 10, 11, 13] and net_mv.sgen.bus[i] == b:# LV2/3 connected
            arr_sgen += profile_abs_mv['sgen', 'p_mw'][i][d*96:d*96+n_timesteps] * scale 
            cap_sgen_mv[b-4] = net_mv.sgen.p_mw[i] * scale # capacity
    df_mv_sgen[:,b-4] = arr_sgen

# LV 1
df_lv1_p = np.zeros((n_timesteps, n_bus_lv1_vec))
df_lv1_q = np.zeros((n_timesteps, n_bus_lv1_vec))
df_lv1_sgen = np.zeros((n_timesteps, n_bus_lv1_vec)) 
cap_sgen_lv1 = np.zeros(n_bus_lv1_vec)
profile_abs_lv1 = sb.get_absolute_values(net_lv1, True)
for b in range(n_bus_lv1_vec):
    # load
    arr_load_p = np.zeros(n_timesteps)
    arr_load_q = np.zeros(n_timesteps)
    for i in net_lv1.load.index:
        if net_lv1.load.bus[i] == b + 1: # scenario 2 vs 0 LV bus index + 1
            arr_load_p += profile_abs_lv1['load', 'p_mw'][i][d*96:d*96+n_timesteps]
            arr_load_q += profile_abs_lv1['load', 'q_mvar'][i][d*96:d*96+n_timesteps]
    df_lv1_p[:,b] = arr_load_p
    df_lv1_q[:,b] = arr_load_q
    # sgen
    arr_sgen = np.zeros(n_timesteps)
    for i in net_lv1.sgen.index:
        if net_lv1.sgen.bus[i] == b + 1:
            arr_sgen += profile_abs_lv1['sgen', 'p_mw'][i][d*96:d*96+n_timesteps]
            cap_sgen_lv1[b] = net_lv1.sgen.p_mw[i] # capacity
    df_lv1_sgen[:,b] = arr_sgen

# LV 2
df_lv2_p = np.zeros((n_timesteps, n_bus_lv2_vec))
df_lv2_q = np.zeros((n_timesteps, n_bus_lv2_vec))
df_lv2_sgen = np.zeros((n_timesteps, n_bus_lv2_vec)) 
cap_sgen_lv2 = np.zeros(n_bus_lv2_vec)
profile_abs_lv2 = sb.get_absolute_values(net_lv2, True)
for b in range(n_bus_lv2_vec):
    # load
    arr_load_p = np.zeros(n_timesteps)
    arr_load_q = np.zeros(n_timesteps)
    for i in net_lv2.load.index:
        if net_lv2.load.bus[i] == b + 1: # scenario 2 vs 0 LV bus index + 2
            arr_load_p += profile_abs_lv2['load', 'p_mw'][i][d*96:d*96+n_timesteps]
            arr_load_q += profile_abs_lv2['load', 'q_mvar'][i][d*96:d*96+n_timesteps]
    df_lv2_p[:,b] = arr_load_p
    df_lv2_q[:,b] = arr_load_q
    # sgen
    arr_sgen = np.zeros(n_timesteps)
    for i in sgen_lv2.idx:
        if sgen_lv2.bus[i] == b:
            arr_sgen += sgen_lv2.p_mw[i] * pvdata["{}".format(sgen_lv2.profile[i])][d*96:d*96+n_timesteps]
            cap_sgen_lv2[b] = sgen_lv2.p_mw[i] # capacity
    df_lv2_sgen[:,b] = arr_sgen
        
# LV 3
df_lv3_p = np.zeros((n_timesteps, n_bus_lv3_vec))
df_lv3_q = np.zeros((n_timesteps, n_bus_lv3_vec))
df_lv3_sgen = np.zeros((n_timesteps, n_bus_lv3_vec)) 
cap_sgen_lv3 = np.zeros(n_bus_lv3_vec)
profile_abs_lv3 = sb.get_absolute_values(net_lv3, True)
for b in range(n_bus_lv3_vec):
    # load
    arr_load_p = np.zeros(n_timesteps)
    arr_load_q = np.zeros(n_timesteps)
    for i in net_lv3.load.index:
        if net_lv3.load.bus[i] == b + 1: # scenario 2 vs 0 LV bus index + 3
            arr_load_p += profile_abs_lv3['load', 'p_mw'][i][d*96:d*96+n_timesteps]
            arr_load_q += profile_abs_lv3['load', 'q_mvar'][i][d*96:d*96+n_timesteps]
    df_lv3_p[:,b] = arr_load_p
    df_lv3_q[:,b] = arr_load_q
    # sgen
    arr_sgen = np.zeros(n_timesteps)
    for i in sgen_lv3.idx:
        if sgen_lv3.bus[i] == b:
            arr_sgen += sgen_lv3.p_mw[i] * pvdata["{}".format(sgen_lv3.profile[i])][d*96:d*96+n_timesteps]
            cap_sgen_lv3[b] = sgen_lv3.p_mw[i] # capacity
    df_lv3_sgen[:,b] = arr_sgen


# ====================== Simulation ==========================================
list_dict_output_pf_prior = []
list_dict_output_pf_after = []
list_dict_output_opf = []
start_time = time.time()
for t in range(38,60):
    print("===================================================================")
    print("Time step {}.".format(t))
    # ======================== power flow ====================================
    dict_input = {}
    dict_input["p_load_mv"] = df_mv_p[t,:] - df_mv_sgen[t,:] 
    dict_input["q_load_mv"] = df_mv_q[t,:]
    dict_input["p_load_lv1"] = df_lv1_p[t,:] - df_lv1_sgen[t,:]
    dict_input["q_load_lv1"] = df_lv1_q[t,:]
    dict_input["p_load_lv2"] = df_lv2_p[t,:] - df_lv2_sgen[t,:]
    dict_input["q_load_lv2"] = df_lv2_q[t,:]
    dict_input["p_load_lv3"] = df_lv3_p[t,:] - df_lv3_sgen[t,:]
    dict_input["q_load_lv3"] = df_lv3_q[t,:]
    dict_output_pf = pf_mv_lv(dict_input)
    dict_output_pf["t"] = t
    list_dict_output_pf_prior.append(dict_output_pf)
    # power flow results
    vm_mv_max = dict_output_pf["res_bus"].vm_pu[0:13].max()
    vm_lv_max = dict_output_pf["res_bus"].vm_pu[13:251].max()
    vm_mv_min = dict_output_pf["res_bus"].vm_pu[0:13].min()
    vm_lv_min = dict_output_pf["res_bus"].vm_pu[13:251].min()
    line_max = dict_output_pf["res_line"].loading_percent.max()
    trafo_max = dict_output_pf["res_trafo"].loading_percent.max()
    print("Max MV bus voltage is {:.3f} pu.".format(vm_mv_max)) # 0~12 MV bus
    print("Max LV bus voltage is {:.3f} pu.".format(vm_lv_max)) #13~250 LV bus
    print("Min MV bus voltage is {:.3f} pu.".format(vm_mv_min)) # 0~12 MV bus
    print("Min LV bus voltage is {:.3f} pu.".format(vm_lv_min)) #13~250 LV bus    
    print("Max line loading is {:.2f}%.".format(line_max))
    print("Max transformer loading is {:.2f}%.".format(trafo_max))
    # check grid states
    if vm_mv_max > v_ub_mv or vm_lv_max > v_ub_lv or line_max > 100 or trafo_max > 100:
        #======================= opf =========================================
        dict_input = {}
        # sgen capacity
        dict_input["cap_sgen_mv"] = cap_sgen_mv / 3
        dict_input["cap_sgen_lv1"] = cap_sgen_lv1 / 3
        dict_input["cap_sgen_lv2"] = cap_sgen_lv2 / 3
        dict_input["cap_sgen_lv3"] = cap_sgen_lv3 / 3
        # load: P, Q
        dict_input["arr_mv_p"] = df_mv_p[t,:] / 3
        dict_input["arr_mv_q"] = df_mv_q[t,:] / 3
        dict_input["arr_lv1_p"] = df_lv1_p[t,:] / 3
        dict_input["arr_lv1_q"] = df_lv1_q[t,:] / 3
        dict_input["arr_lv2_p"] = df_lv2_p[t,:] / 3
        dict_input["arr_lv2_q"] = df_lv2_q[t,:] / 3
        dict_input["arr_lv3_p"] = df_lv3_p[t,:] / 3
        dict_input["arr_lv3_q"] = df_lv3_q[t,:] / 3
        # sgen: P
        dict_input["arr_mv_sgen"] = df_mv_sgen[t,:] / 3
        dict_input["arr_lv1_sgen"] = df_lv1_sgen[t,:] / 3
        dict_input["arr_lv2_sgen"] = df_lv2_sgen[t,:] / 3
        dict_input["arr_lv3_sgen"] = df_lv3_sgen[t,:] / 3
        dict_output_opf = opf_mv_lv(dict_input, objective = objective_format, epsilon = epsilon)
        dict_output_opf["t"] = t
        list_dict_output_opf.append(dict_output_opf)
        # ============== extract opf results =================================
        p_sgen_mv = dict_output_opf["p_sgen_mv"] * 3
        p_sgen_lv1 = dict_output_opf["p_sgen_lv1"] * 3
        p_sgen_lv2 = dict_output_opf["p_sgen_lv2"] * 3
        p_sgen_lv3 = dict_output_opf["p_sgen_lv3"] * 3
        q_sgen_mv = dict_output_opf["q_sgen_mv"] * 3
        q_sgen_lv1 = dict_output_opf["q_sgen_lv1"] * 3
        q_sgen_lv2 = dict_output_opf["q_sgen_lv2"] * 3
        q_sgen_lv3 = dict_output_opf["q_sgen_lv3"] * 3
        # =================== power flow test ================================
        dict_input = {}
        dict_input["p_load_mv"] = df_mv_p[t,:] - p_sgen_mv 
        dict_input["q_load_mv"] = df_mv_q[t,:] - q_sgen_mv
        dict_input["p_load_lv1"] = df_lv1_p[t,:] - p_sgen_lv1
        dict_input["q_load_lv1"] = df_lv1_q[t,:] - q_sgen_lv1
        dict_input["p_load_lv2"] = df_lv2_p[t,:] - p_sgen_lv2
        dict_input["q_load_lv2"] = df_lv2_q[t,:] - q_sgen_lv2
        dict_input["p_load_lv3"] = df_lv3_p[t,:] - p_sgen_lv3
        dict_input["q_load_lv3"] = df_lv3_q[t,:] - q_sgen_lv3
        dict_output_pf = pf_mv_lv(dict_input)
        dict_output_pf["t"] = t
        list_dict_output_pf_after.append(dict_output_pf)
        # ================= optimized grid states ============================
        vm_mv_max = dict_output_pf["res_bus"].vm_pu[0:13].max()
        vm_lv_max = dict_output_pf["res_bus"].vm_pu[13:251].max()
        vm_mv_min = dict_output_pf["res_bus"].vm_pu[0:13].min()
        vm_lv_min = dict_output_pf["res_bus"].vm_pu[13:251].min()
        line_max = dict_output_pf["res_line"].loading_percent.max()
        trafo_max = dict_output_pf["res_trafo"].loading_percent.max()
        print("Max MV bus voltage is {:.3f} pu.".format(vm_mv_max)) # 0~12 MV bus
        print("Max LV bus voltage is {:.3f} pu.".format(vm_lv_max)) #13~250 LV bus
        print("Min MV bus voltage is {:.3f} pu.".format(vm_mv_min)) # 0~12 MV bus
        print("Min LV bus voltage is {:.3f} pu.".format(vm_lv_min)) #13~250 LV bus    
        print("Max line loading is {:.2f}%.".format(line_max))
        print("Max transformer loading is {:.2f}%.".format(trafo_max))
end_time = time.time()   
print("Simulation time is {:.1f} s.".format((end_time-start_time)/22))  
if visualization:       
    # ==================== result visualization ==================================
    time = ['01-{} {}:{}'.format(1+(15*i//60)//24, 
                                 (15*i//60)%24, 15*i%60) for i in range(97)]
    time = pd.to_datetime(time, format='%m-%d %H:%M')
        
    # ===================== voltage before =======================================
    vm_pu_before = np.zeros((n_timesteps, n_bus_mv+n_bus_lv1_vec+n_bus_lv2_vec+n_bus_lv3_vec+1))
    for t in range(n_timesteps):
        vm_pu_before[t,:] = list_dict_output_pf_prior[t]["res_bus"].vm_pu
        
    fig, ax = plt.subplots()
    ax.set_xlim(time.min(), time.max())
    for b in range(n_bus_mv+1):
        ax.plot(time[0:96], vm_pu_before[:,b])
    ax.plot(time[0:96], [1.055]*96,linestyle = 'dashed',color='red')
    ax.plot(time[0:96], [0.965]*96,linestyle = 'dashed',color='red')
    ax.xaxis.set_major_locator(md.HourLocator(interval = 4))
    ax.xaxis.set_major_formatter(md.DateFormatter('%H:%M'))
    ax.set_ylabel("MV bus voltage, pu")
    fig.autofmt_xdate()
    plt.show() 
    
    fig, ax = plt.subplots()
    ax.set_xlim(time.min(), time.max())
    ax.set_ylim(0.95,1.15)
    for b in range(n_bus_mv+1,n_bus_mv+n_bus_lv1_vec+n_bus_lv2_vec+n_bus_lv3_vec+1):
        ax.plot(time[0:96], vm_pu_before[:,b])
    ax.plot(time[0:96], [1.1]*96,linestyle = 'dashed',color='red')
    # ax.plot(time[0:96], [0.9]*96,linestyle = 'dashed',color='red')
    ax.xaxis.set_major_locator(md.HourLocator(interval = 4))
    ax.xaxis.set_major_formatter(md.DateFormatter('%H:%M'))
    ax.set_ylabel("Voltage (pu)",fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    fig.autofmt_xdate()
    plt.show() 
    
    # ================== line loading before ====================================
    line_loading_before = np.zeros((n_timesteps, n_cable_mv+n_bus_lv1_vec+n_bus_lv2_vec+n_bus_lv3_vec-3))
    for t in range(n_timesteps):
        line_loading_before[t,:] = list_dict_output_pf_prior[t]["res_line"].loading_percent
        
    fig, ax = plt.subplots()
    ax.set_xlim(time.min(), time.max())
    ax.set_ylim(0,160)
    for l in range(n_cable_mv+n_bus_lv1_vec+n_bus_lv2_vec+n_bus_lv3_vec-3):
        ax.plot(time[0:96], line_loading_before[:,l])
    ax.plot(time[0:96], [100]*96,linestyle = 'dashed',color='red')
    ax.xaxis.set_major_locator(md.HourLocator(interval = 4))
    ax.xaxis.set_major_formatter(md.DateFormatter('%H:%M'))
    ax.set_ylabel("Cable load (%)",fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    fig.autofmt_xdate()
    plt.show() 
    
    # ==================== trafo loading before ==================================
    trafo_loading_before = np.zeros((n_timesteps, n_trafo))
    for t in range(n_timesteps):
        trafo_loading_before[t,:] = list_dict_output_pf_prior[t]["res_trafo"].loading_percent
        
    fig, ax = plt.subplots()
    ax.set_xlim(time.min(), time.max())
    ax.set_ylim(0,160)
    for t in range(n_trafo):
        ax.plot(time[0:96], trafo_loading_before[:,t])
    ax.plot(time[0:96], [100]*96,linestyle = 'dashed',color='red')
    ax.xaxis.set_major_locator(md.HourLocator(interval = 4))
    ax.xaxis.set_major_formatter(md.DateFormatter('%H:%M'))
    ax.set_ylabel("Trans. load (%)",fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    fig.autofmt_xdate()
    plt.show() 
    
    # ===================== voltage after ========================================
    vm_pu_after = vm_pu_before
    for dict_output_pf in list_dict_output_pf_after:
        t = dict_output_pf["t"]
        vm_pu_after[t,:] = dict_output_pf["res_bus"].vm_pu
        
    fig, ax = plt.subplots()
    ax.set_xlim(time.min(), time.max())
    for b in range(n_bus_mv+1):
        ax.plot(time[0:96], vm_pu_after[:,b])
    ax.plot(time[0:96], [1.055]*96,linestyle = 'dashed',color='red')
    ax.plot(time[0:96], [0.965]*96,linestyle = 'dashed',color='red')
    ax.xaxis.set_major_locator(md.HourLocator(interval = 4))
    ax.xaxis.set_major_formatter(md.DateFormatter('%H:%M'))
    ax.set_ylabel("Optimized MV bus voltage, pu")
    fig.autofmt_xdate()
    plt.show() 
    
    fig, ax = plt.subplots()
    ax.set_xlim(time.min(), time.max())
    ax.set_ylim(0.95,1.15)
    for b in range(n_bus_mv+1,n_bus_mv+n_bus_lv1_vec+n_bus_lv2_vec+n_bus_lv3_vec+1):
        ax.plot(time[0:96], vm_pu_after[:,b])
    ax.plot(time[0:96], [1.1]*96,linestyle = 'dashed',color='red')
    # ax.plot(time[0:96], [0.9]*96,linestyle = 'dashed',color='red')
    ax.xaxis.set_major_locator(md.HourLocator(interval = 4))
    ax.xaxis.set_major_formatter(md.DateFormatter('%H:%M'))
    # ax.set_ylabel("Optimized LV bus voltage, pu")
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    fig.autofmt_xdate()
    plt.show()
    
    # ==================== line loading after ====================================
    line_loading_after = line_loading_before
    for dict_output_pf in list_dict_output_pf_after:
        t = dict_output_pf["t"]
        line_loading_after[t,:] = dict_output_pf["res_line"].loading_percent
    
    fig, ax = plt.subplots()
    ax.set_xlim(time.min(), time.max())
    ax.set_ylim(0,160)
    for l in range(n_cable_mv+n_bus_lv1_vec+n_bus_lv2_vec+n_bus_lv3_vec-3):
        ax.plot(time[0:96], line_loading_after[:,l])
    ax.plot(time[0:96], [100]*96,linestyle = 'dashed',color='red')
    ax.xaxis.set_major_locator(md.HourLocator(interval = 4))
    ax.xaxis.set_major_formatter(md.DateFormatter('%H:%M'))
    # ax.set_ylabel("Optimized cable loading percentage")
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    fig.autofmt_xdate()
    plt.show() 
    
    # =================== trafo loading after ====================================
    trafo_loading_after = trafo_loading_before
    for dict_output_pf in list_dict_output_pf_after:
        t = dict_output_pf["t"]
        trafo_loading_after[t,:] = dict_output_pf["res_trafo"].loading_percent
        
    fig, ax = plt.subplots()
    ax.set_xlim(time.min(), time.max())
    ax.set_ylim(0,160)
    for t in range(n_trafo):
        ax.plot(time[0:96], trafo_loading_after[:,t])
    ax.plot(time[0:96], [100]*96,linestyle = 'dashed',color='red')
    ax.xaxis.set_major_locator(md.HourLocator(interval = 4))
    ax.xaxis.set_major_formatter(md.DateFormatter('%H:%M'))
    # ax.set_ylabel("Optimized transformer loading percentage")
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    fig.autofmt_xdate()
    plt.show() 
    
# =================power curtailment & fairness===============================
total_curt = [0] * n_timesteps
std = [0] * n_timesteps
mean = [0] * n_timesteps
# zero elements are nodes where no sgen is connected, due to numerical issues
jain_idx = [0] * n_timesteps
for dict_output_opf in list_dict_output_opf:
    t = dict_output_opf["t"]
    total_curt[t] =  3 * (dict_output_opf["p_curt_lv1"].sum() + dict_output_opf["p_curt_lv2"].sum() + \
        dict_output_opf["p_curt_lv3"].sum() + dict_output_opf["p_curt_mv"].sum())
    jain_idx[t] = dict_output_opf["jain_idx"]
    std[t] = dict_output_opf["std"]
    mean[t] = dict_output_opf["mean"]
print("Total curtailment is {:.1f} kWh.".format(250*sum(total_curt)))

if fashion in ["ben", "admm", "app"]:
    iter_list = []
    resid_list = []
    for elem in list_dict_output_opf:
        iter_list.append(elem["n_iter"])
        resid_list.append(elem["resid"])
    print("Average no. of iterations is {}.".format(sum(iter_list)/22))
    print("Number of un-converged time steps is {}.".format(sum(np.array(resid_list)>epsilon)))

# voltage diff
vm_pu_mv_max = []
vm_pu_lv_max = []
vm_pu_diff = []
for i in range(len(list_dict_output_opf)):
    dict_output_opf = list_dict_output_opf[i]
    dict_output_pf = list_dict_output_pf_after[i]
    vm_pu_mv = dict_output_opf["vm_pu_mv"]
    vm_pu_lv1 = dict_output_opf["vm_pu_lv1"]
    vm_pu_lv2 = dict_output_opf["vm_pu_lv2"]
    vm_pu_lv3 = dict_output_opf["vm_pu_lv3"]
    vm_pu_opf = np.concatenate((vm_pu_mv,vm_pu_lv1,vm_pu_lv2,vm_pu_lv3))
    vm_pu_pf = dict_output_pf["res_bus"].vm_pu
    vm_pu_diff.append(abs(vm_pu_opf - vm_pu_pf).mean())
    vm_pu_mv_max.append(max(vm_pu_pf[0:13]))
    vm_pu_lv_max.append(max(vm_pu_pf[13:]))
print("Maximum MV voltage is {} pu".format(max(vm_pu_mv_max)))
print("Maximum LV voltage is {} pu".format(max(vm_pu_lv_max)))

# line loading diff
loading_line_diff = []
loading_line_max = []
for i in range(len(list_dict_output_opf)):
    dict_output_opf = list_dict_output_opf[i]
    dict_output_pf = list_dict_output_pf_after[i]
    loading_line_mv = dict_output_opf["loading_line_mv"]
    loading_line_lv1 = dict_output_opf["loading_line_lv1"]
    loading_line_lv2 = dict_output_opf["loading_line_lv2"]
    loading_line_lv3 = dict_output_opf["loading_line_lv3"]
    loading_line_opf = np.concatenate((loading_line_mv,loading_line_lv1,loading_line_lv2,loading_line_lv3))
    loading_line_pf = dict_output_pf["res_line"].loading_percent
    loading_line_diff.append(abs(loading_line_opf - loading_line_pf).mean())
    loading_line_max.append(max(loading_line_pf))
print("Maximum line loading is {} %".format(max(loading_line_max)))

# trafo loading diff
loading_trafo_diff = []
loading_trafo_max = []
for i in range(len(list_dict_output_opf)):
    dict_output_opf = list_dict_output_opf[i]
    dict_output_pf = list_dict_output_pf_after[i]
    loading_trafo_opf = dict_output_opf["loading_trafo"]
    loading_trafo_pf = dict_output_pf["res_trafo"].loading_percent
    loading_trafo_diff.append(abs(loading_trafo_opf - loading_trafo_pf).mean())
    loading_trafo_max.append(max(loading_trafo_pf))
print("Maximum trafo loading is {} %".format(max(loading_trafo_max)))
    

























