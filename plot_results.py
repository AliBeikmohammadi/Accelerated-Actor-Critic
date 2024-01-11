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

def plot_result(Env, Backbone, plot_type, std_coaf= 1.):
    Input_file_dir='./Results_'+Env
    Save_fig_dir='./Plot_Results/'
    dataframes_list = []
    dataframes_list2 = []
    for root, dirs, files in os.walk(Input_file_dir):
        for file in files:
            if file.endswith(".csv"):
                path=os.path.join(root, file)
                if path.find(Backbone+'_only_RL') != -1:
                    temp_df = pd.read_csv(path)
                    dataframes_list.append(temp_df)
                if path.find(Backbone+'_Weight') != -1:

                    temp_df2 = pd.read_csv(path)
                    dataframes_list2.append(temp_df2)
                    
                    
                    
    for i in range(len(dataframes_list)):
        globals()['df%s' % i] = dataframes_list[i][['time/total_timesteps', 'rollout/ep_rew_mean', 'rollout/ep_len_mean']].copy().set_index('time/total_timesteps') 
        globals()['df2%s' % i] = dataframes_list2[i][['time/total_timesteps', 'rollout/ep_rew_mean', 'rollout/ep_len_mean', 'train/Beta']].copy().set_index('time/total_timesteps') 

    union_idx =  df0.index
    union_idx2 =  df20.index
    for i in range(len(dataframes_list)):
        union_idx= union_idx.union(globals()['df%s' % i].index)
        union_idx2= union_idx2.union(globals()['df2%s' % i].index)
    dfList=[]
    dfList2=[]
    for i in range(len(dataframes_list)):
        globals()['df%s' % i] = globals()['df%s' % i].reindex(union_idx)
        globals()['df2%s' % i] = globals()['df2%s' % i].reindex(union_idx2)  
        globals()['df2%s' % i].iloc[0, globals()['df2%s' % i].columns.get_loc("train/Beta")] = 1   
        globals()['df%s' % i] = globals()['df%s' % i].interpolate(method='linear', limit_direction='both', axis=0) 
        globals()['df2%s' % i] = globals()['df2%s' % i].interpolate(method='linear', limit_direction='both', axis=0) 
        dfList.append(globals()['df%s' % i])
        dfList2.append(globals()['df2%s' % i])
    DATA = pd.concat(dfList)
    MEAN= DATA.groupby(level=0).mean()
    MIN= DATA.groupby(level=0).min()
    MAX= DATA.groupby(level=0).max()
    STD= DATA.groupby(level=0).std()
    DATA2 = pd.concat(dfList2)
    MEAN2= DATA2.groupby(level=0).mean()
    MIN2= DATA2.groupby(level=0).min()
    MAX2= DATA2.groupby(level=0).max()
    STD2= DATA2.groupby(level=0).std()
    ci = STD['rollout/ep_rew_mean']/MEAN['rollout/ep_rew_mean']
    ci2 = STD2['rollout/ep_rew_mean']/MEAN2['rollout/ep_rew_mean']
    
    if plot_type=='min_max':
        plt.rcParams["figure.figsize"] = [10, 5]
        ax1 = plt.subplot()
        clrs = sns.color_palette("husl", 2)
        parameters = {'axes.labelsize': 28, 'axes.titlesize': 28, 'legend.fontsize': 25}
        with sns.axes_style("darkgrid"):
            MEAN.plot(y="rollout/ep_rew_mean", ax=ax1, label=Backbone, linewidth=3, c=clrs[0])
            MEAN2.plot(y="rollout/ep_rew_mean", ax=ax1, label='Accelerated '+Backbone, linewidth=3, c=clrs[1])
            plt.fill_between(MEAN.index, MIN['rollout/ep_rew_mean'], MAX['rollout/ep_rew_mean'], facecolor=clrs[0] ,alpha=0.3)
            plt.fill_between(MEAN2.index, MIN2['rollout/ep_rew_mean'], MAX2['rollout/ep_rew_mean'], facecolor=clrs[1] ,alpha=0.3)
            ax2 = ax1.twinx()
            MEAN2.plot(y="train/Beta", ax=ax2, legend=False, linewidth=3, c='tab:gray', linestyle='--')
            ax1.set_ylabel('Average Return')
            ax2.set_ylabel(r'Normalized $\beta_e$')
            ax1.set_xlabel('Time steps')
            ax2.set_ylim((0, 1))
            ax1.legend()
            plt.rcParams.update(parameters)
            plt.tight_layout(pad=0.01)
            plt.savefig(Save_fig_dir+'/'+Env+'_'+Backbone+'_'+plot_type+'.png', dpi=144, format=None, metadata=None, bbox_inches=None, pad_inches=0.1,
                     facecolor='auto', edgecolor='auto', backend=None)
            print("ENV: "+Env+' | Backbone: '+Backbone+' | Shadow type: '+plot_type)
            plt.show()
            print("The figures saved in:"+Save_fig_dir+'/'+Env+'_'+Backbone+'_'+plot_type+'.png')
       
    if plot_type=='std':
        plt.rcParams["figure.figsize"] = [10, 5]
        ax1 = plt.subplot()
        clrs = sns.color_palette("husl", 2)
        parameters = {'axes.labelsize': 28, 'axes.titlesize': 28, 'legend.fontsize': 25}
        with sns.axes_style("darkgrid"):
            MEAN.plot(y="rollout/ep_rew_mean", ax=ax1, label=Backbone, linewidth=3, c=clrs[0])
            MEAN2.plot(y="rollout/ep_rew_mean", ax=ax1, label='Accelerated '+Backbone, linewidth=3, c=clrs[1])
            plt.fill_between(MEAN.index, MEAN['rollout/ep_rew_mean']-std_coaf*STD['rollout/ep_rew_mean'], 
                             MEAN['rollout/ep_rew_mean']+std_coaf*STD['rollout/ep_rew_mean'], facecolor=clrs[0] ,alpha=0.3)
            plt.fill_between(MEAN2.index, MEAN2['rollout/ep_rew_mean']-std_coaf*STD2['rollout/ep_rew_mean'], 
                             MEAN2['rollout/ep_rew_mean']+std_coaf*STD2['rollout/ep_rew_mean'], facecolor=clrs[1] ,alpha=0.3)
            ax2 = ax1.twinx()
            MEAN2.plot(y="train/Beta", ax=ax2, legend=False, linewidth=3, c='tab:gray', linestyle='--')
            ax1.set_ylabel('Average Return')
            ax2.set_ylabel(r'Normalized $\beta_e$')
            ax1.set_xlabel('Time steps')
            ax2.set_ylim((0, 1))
            ax1.legend()
            plt.rcParams.update(parameters)
            plt.tight_layout(pad=0.01)
            plt.savefig(Save_fig_dir+'/'+Env+'_'+Backbone+'_'+plot_type+'_'+str(std_coaf)+'.png', dpi=144, format=None, metadata=None, bbox_inches=None, pad_inches=0.1,
                     facecolor='auto', edgecolor='auto', backend=None)
            print("ENV: "+Env+' | Backbone: '+Backbone+' | Shadow type: '+plot_type)
            plt.show()
            print("The figures saved in:"+Save_fig_dir+'/'+Env+'_'+Backbone+'_'+plot_type+'_'+str(std_coaf)+'.png')


Backbone= ['SAC', 'TD3']
Env= ['MountainCarContinuous-v0', 'BipedalWalker-v3', 'CartPoleContinuous-v1', 'LunarLanderContinuous-v2', 'Pendulum-v1']
plot_type=['min_max', 'std'] #'std' 
std_coaf= 0.5
for i in Env:
    for j in Backbone:
        for k in plot_type:
            plot_result(Env=i, Backbone= j, plot_type=k)
            if k=='std':
                plot_result(Env=i, Backbone= j, plot_type=k, std_coaf= std_coaf)