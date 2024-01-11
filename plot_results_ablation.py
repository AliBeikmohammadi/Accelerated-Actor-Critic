import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
import argparse
import pathlib
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')

def create_df(Env, ex_adress):
    Input_file_dir='./Results_Ablation_Studies/Results_'+Env
    j=0
    dataframes_list = []
    for root, dirs, files in os.walk(Input_file_dir):
        for file in files:
            if file.endswith(".csv"):
                path=os.path.join(root, file)
                if path.find(ex_adress) != -1:  
                    temp_df = pd.read_csv(path)
                    globals()['df%s' % j]= temp_df[['time/total_timesteps', 'rollout/ep_rew_mean', 'rollout/ep_len_mean', 'train/Beta']].copy().set_index('time/total_timesteps')
                    j+=1             
    union_idx =  df0.index
    for i in range(j):
        union_idx= union_idx.union(globals()['df%s' % i].index)
    dfList=[]
    for i in range(j):
        globals()['df%s' % i] = globals()['df%s' % i].reindex(union_idx)
        globals()['df%s' % i].iloc[0, globals()['df%s' % i].columns.get_loc("train/Beta")] = 1   
        globals()['df%s' % i] = globals()['df%s' % i].interpolate(method='linear', limit_direction='both', axis=0)
        dfList.append(globals()['df%s' % i])
    return dfList


def plot_ablation(Env, Backbone, plot_type, experiment_list, ax1, clrs, note, std_coaf):
    for i, ex in enumerate(experiment_list):
        dfList = create_df(Env, ex)
        DATA = pd.concat(dfList)
        MEAN= DATA.groupby(level=0).mean()
        MIN= DATA.groupby(level=0).min()
        MAX= DATA.groupby(level=0).max()
        STD= DATA.groupby(level=0).std()
        ci = STD['rollout/ep_rew_mean']/MEAN['rollout/ep_rew_mean']
        if plot_type=='min_max':
            with sns.axes_style("darkgrid"):
                MEAN.plot(y="rollout/ep_rew_mean", ax=ax1, label=note[-1]+'='+str(note[i]), linewidth=3, c=clrs[i])
                plt.fill_between(MEAN.index, MIN['rollout/ep_rew_mean'], MAX['rollout/ep_rew_mean'], facecolor=clrs[i] ,alpha=0.3)

        if plot_type=='std':
            with sns.axes_style("darkgrid"):
                MEAN.plot(y="rollout/ep_rew_mean", ax=ax1, label=note[-1]+'='+str(note[i]), linewidth=3, c=clrs[i])
                plt.fill_between(MEAN.index, MEAN['rollout/ep_rew_mean']-std_coaf*STD['rollout/ep_rew_mean'], 
                                 MEAN['rollout/ep_rew_mean']+std_coaf*STD['rollout/ep_rew_mean'], facecolor=clrs[i] ,alpha=0.3)


def Plot_ablation(Env, Backbone, experiment_list, note, plot_type, std_coaf=1):
    Save_fig_dir='./Plot_Ablation_Study_Results/'
    plt.rcParams["figure.figsize"] = [10, 5]
    ax1 = plt.subplot()
    clrs = sns.color_palette("husl", len(experiment_list))
    parameters = {'axes.labelsize': 28, 'axes.titlesize': 28, 'legend.fontsize': 25}
    plot_ablation(Env, Backbone, plot_type, experiment_list, ax1, clrs, note, std_coaf)
    ax1.set_ylabel('Average Return')
    ax1.set_xlabel('Time steps')
    ax1.legend()
    plt.rcParams.update(parameters)
    plt.tight_layout(pad=0.01)
    if plot_type=='min_max':
        N1=Save_fig_dir+Env+'_'+Backbone+'_'+plot_type+note[-1]+'.png'
        N2="ENV: "+Env+' | Backbone: '+Backbone+' | Shadow type: '+plot_type+' | Note: '+note[-1] 
    if plot_type=='std':
        N1=Save_fig_dir+Env+'_'+Backbone+'_'+plot_type+'_'+str(std_coaf)+note[-1]+'.png'
        N2="ENV: "+Env+' | Backbone: '+Backbone+' | Shadow type: '+plot_type+' | STD Coaf: '+str(std_coaf)+' | Note: '+note[-1] 
    plt.savefig(N1,dpi=144, format=None, metadata=None, bbox_inches=None, pad_inches=0.1,facecolor='auto', edgecolor='auto', backend=None)
    print(N2)
    print("The figures saved in:", N1)
    plt.show()


Backbone= 'SAC'
Env= 'CartPoleContinuous-v1'

ex11='CartPoleContinuous-v1_SAC_Weight1.0_Just_Beta1.0_s0_e1000_sig1.0_mu0.0'
ex12='CartPoleContinuous-v1_SAC_Weight0.1_Just_Beta1.0_s0_e1000_sig1.0_mu0.0'
ex13='CartPoleContinuous-v1_SAC_Weight0.5_Just_Beta1.0_s0_e1000_sig1.0_mu0.0'
ex14='CartPoleContinuous-v1_SAC_Weight5.0_Just_Beta1.0_s0_e1000_sig1.0_mu0.0'
ex15='CartPoleContinuous-v1_SAC_Weight10.0_Just_Beta1.0_s0_e1000_sig1.0_mu0.0'

ex21=ex11
ex22='CartPoleContinuous-v1_SAC_Weight1.0_Just_Beta1.0_s0_e100_sig1.0_mu0.0'
ex25='CartPoleContinuous-v1_SAC_Weight1.0_Just_Beta1.0_s0_e25000_sig1.0_mu0.0'

ex31=ex11
ex32='CartPoleContinuous-v1_SAC_Weight1.0_Just_Beta1.0_s0_e1000_sig0.0_mu0.0'
ex34='CartPoleContinuous-v1_SAC_Weight1.0_Just_Beta1.0_s0_e1000_sig2.0_mu0.0'

ex1=[ex11, ex12, ex13, ex14, ex15]
ex2=[ex21, ex22, ex25]
ex3=[ex31, ex32, ex34]

leg1=[1, 0.1, 0.5, 5, 10, r'$\beta_0$']
leg2=[1000, 100, 25000, r'$E$']
leg3=[1, 0, 2, r'$c_2$']


Plot_ablation(Env, Backbone, ex1, leg1 , plot_type='min_max')
Plot_ablation(Env, Backbone, ex1, leg1 , plot_type='std', std_coaf=1)

Plot_ablation(Env, Backbone, ex2, leg2 , plot_type='min_max')
Plot_ablation(Env, Backbone, ex2, leg2 , plot_type='std', std_coaf=1)

Plot_ablation(Env, Backbone, ex3, leg3 , plot_type='min_max')
Plot_ablation(Env, Backbone, ex3, leg3 , plot_type='std', std_coaf=1)


Backbone= 'TD3'
Env= 'CartPoleContinuous-v1'


ex11='CartPoleContinuous-v1_TD3_Weight10.0_Just_Beta1.0_s0_e500_sig0.5_mu0.0'
ex12='CartPoleContinuous-v1_TD3_Weight0.1_Just_Beta1.0_s0_e500_sig0.5_mu0.0'
ex13='CartPoleContinuous-v1_TD3_Weight1.0_Just_Beta1.0_s0_e500_sig0.5_mu0.0'
ex14='CartPoleContinuous-v1_TD3_Weight5.0_Just_Beta1.0_s0_e500_sig0.5_mu0.0'
ex15='CartPoleContinuous-v1_TD3_Weight100.0_Just_Beta1.0_s0_e500_sig0.5_mu0.0'

ex21=ex11
ex22='CartPoleContinuous-v1_TD3_Weight10.0_Just_Beta1.0_s0_e200_sig0.5_mu0.0'
ex23='CartPoleContinuous-v1_TD3_Weight10.0_Just_Beta1.0_s0_e800_sig0.5_mu0.0'

ex31=ex11
ex32='CartPoleContinuous-v1_TD3_Weight10.0_Just_Beta1.0_s0_e500_sig0.0_mu0.0'
ex34='CartPoleContinuous-v1_TD3_Weight10.0_Just_Beta1.0_s0_e500_sig2.0_mu0.0'

ex1=[ex11, ex12, ex13, ex14, ex15]
ex2=[ex21, ex22, ex23]
ex3=[ex31, ex32, ex34]

leg1=[10, 0.1, 1, 5, 100, r'$\beta_0$']
leg2=[500, 200, 800, r'$E$']
leg3=[0.5, 0, 2, r'$c_2$']


Plot_ablation(Env, Backbone, ex1, leg1 , plot_type='min_max')
Plot_ablation(Env, Backbone, ex1, leg1 , plot_type='std', std_coaf=1)

Plot_ablation(Env, Backbone, ex2, leg2 , plot_type='min_max')
Plot_ablation(Env, Backbone, ex2, leg2 , plot_type='std', std_coaf=1)

Plot_ablation(Env, Backbone, ex3, leg3 , plot_type='min_max')
Plot_ablation(Env, Backbone, ex3, leg3 , plot_type='std', std_coaf=1)


Backbone= 'SAC'
Env= 'MountainCarContinuous-v0'

ex11='MountainCarContinuous-v0_SAC_Weight5.0_Just_Beta1.0_s0_e300_Noise_Prob0.5_Noise_STD1.0'
ex12='MountainCarContinuous-v0_SAC_Weight0.1_Just_Beta1.0_s0_e300_Noise_Prob0.5_Noise_STD1.0'
ex13='MountainCarContinuous-v0_SAC_Weight0.5_Just_Beta1.0_s0_e300_Noise_Prob0.5_Noise_STD1.0'
ex14='MountainCarContinuous-v0_SAC_Weight1.0_Just_Beta1.0_s0_e300_Noise_Prob0.5_Noise_STD1.0'
ex15='MountainCarContinuous-v0_SAC_Weight10.0_Just_Beta1.0_s0_e300_Noise_Prob0.5_Noise_STD1.0'

ex21=ex11
ex22='MountainCarContinuous-v0_SAC_Weight5.0_Just_Beta1.0_s0_e50_Noise_Prob0.5_Noise_STD1.0'
ex23='MountainCarContinuous-v0_SAC_Weight5.0_Just_Beta1.0_s0_e600_Noise_Prob0.5_Noise_STD1.0'

ex31=ex11
ex32='MountainCarContinuous-v0_SAC_Weight5.0_Just_Beta1.0_s0_e300_Noise_Prob0.25_Noise_STD1.0'
ex33='MountainCarContinuous-v0_SAC_Weight5.0_Just_Beta1.0_s0_e300_Noise_Prob0.75_Noise_STD1.0'
ex34='MountainCarContinuous-v0_SAC_Weight5.0_Just_Beta1.0_s0_e300_Noise_Prob1.0_Noise_STD1.0'

ex41=ex11
ex42='MountainCarContinuous-v0_SAC_Weight5.0_Just_Beta1.0_s0_e300_Noise_Prob0.5_Noise_STD0.0'
ex43='MountainCarContinuous-v0_SAC_Weight5.0_Just_Beta1.0_s0_e300_Noise_Prob0.5_Noise_STD2.0'


ex1=[ex11, ex12, ex13, ex14, ex15]
ex2=[ex21, ex22, ex23]
ex3=[ex31, ex32, ex33, ex34]
ex4=[ex41, ex42, ex43]

leg1=[5, 0.1, 0.5, 1, 10, r'$\beta_0$']
leg2=[300, 50, 600, r'$E$']
leg3=[0.5, 0.25, 0.75, 1, r'$\epsilon$']
leg4=[1, 0, 2, r'$c_1$']


Plot_ablation(Env, Backbone, ex1, leg1 , plot_type='min_max')
Plot_ablation(Env, Backbone, ex1, leg1 , plot_type='std', std_coaf=1)

Plot_ablation(Env, Backbone, ex2, leg2 , plot_type='min_max')
Plot_ablation(Env, Backbone, ex2, leg2 , plot_type='std', std_coaf=1)

Plot_ablation(Env, Backbone, ex3, leg3 , plot_type='min_max')
Plot_ablation(Env, Backbone, ex3, leg3 , plot_type='std', std_coaf=1)

Plot_ablation(Env, Backbone, ex4, leg4 , plot_type='min_max')
Plot_ablation(Env, Backbone, ex4, leg4 , plot_type='std', std_coaf=1)


Backbone= 'TD3'
Env= 'MountainCarContinuous-v0'

ex11='MountainCarContinuous-v0_TD3_Weight1.0_Just_Beta1.0_s0_e50_Noise_Prob0.5_Noise_STD1.0'
ex12='MountainCarContinuous-v0_TD3_Weight0.1_Just_Beta1.0_s0_e50_Noise_Prob0.5_Noise_STD1.0'
ex13='MountainCarContinuous-v0_TD3_Weight0.5_Just_Beta1.0_s0_e50_Noise_Prob0.5_Noise_STD1.0'
ex14='MountainCarContinuous-v0_TD3_Weight5.0_Just_Beta1.0_s0_e50_Noise_Prob0.5_Noise_STD1.0'
ex15='MountainCarContinuous-v0_TD3_Weight10.0_Just_Beta1.0_s0_e50_Noise_Prob0.5_Noise_STD1.0'

ex21=ex11
ex22='MountainCarContinuous-v0_TD3_Weight1.0_Just_Beta1.0_s0_e10_Noise_Prob0.5_Noise_STD1.0'
ex24='MountainCarContinuous-v0_TD3_Weight1.0_Just_Beta1.0_s0_e250_Noise_Prob0.5_Noise_STD1.0'

ex31=ex11
ex32='MountainCarContinuous-v0_TD3_Weight1.0_Just_Beta1.0_s0_e50_Noise_Prob0.25_Noise_STD1.0'
ex33='MountainCarContinuous-v0_TD3_Weight1.0_Just_Beta1.0_s0_e50_Noise_Prob0.75_Noise_STD1.0'
ex34='MountainCarContinuous-v0_TD3_Weight1.0_Just_Beta1.0_s0_e50_Noise_Prob1.0_Noise_STD1.0'

ex41=ex11
ex42='MountainCarContinuous-v0_TD3_Weight1.0_Just_Beta1.0_s0_e50_Noise_Prob0.5_Noise_STD0.0'
ex43='MountainCarContinuous-v0_TD3_Weight1.0_Just_Beta1.0_s0_e50_Noise_Prob0.5_Noise_STD2.0'


ex1=[ex11, ex12, ex13, ex14, ex15]
ex2=[ex21, ex22, ex24]
ex3=[ex31, ex32, ex33, ex34]
ex4=[ex41, ex42, ex43]

leg1=[1, 0.1, 0.5, 5, 10, r'$\beta_0$']
leg2=[50, 10, 250, r'$E$']
leg3=[0.5, 0.25, 0.75, 1, r'$\epsilon$']
leg4=[1, 0, 2, r'$c_1$']


Plot_ablation(Env, Backbone, ex1, leg1 , plot_type='min_max')
Plot_ablation(Env, Backbone, ex1, leg1 , plot_type='std', std_coaf=1)

Plot_ablation(Env, Backbone, ex2, leg2 , plot_type='min_max')
Plot_ablation(Env, Backbone, ex2, leg2 , plot_type='std', std_coaf=1)

Plot_ablation(Env, Backbone, ex3, leg3 , plot_type='min_max')
Plot_ablation(Env, Backbone, ex3, leg3 , plot_type='std', std_coaf=1)

Plot_ablation(Env, Backbone, ex4, leg4 , plot_type='min_max')
Plot_ablation(Env, Backbone, ex4, leg4 , plot_type='std', std_coaf=1)




