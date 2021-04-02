# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 03:33:41 2019

@author: bdb3m, cw8xk
"""
from datetime import datetime
import os
import random as rn
import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, Concatenate, BatchNormalization
from keras.optimizers import Adam
# from keras.callbacks import ModelCheckpoint
import tensorflow as tf
from keras import backend as K
from rl.agents import DDPGAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess
from rl.callbacks import WandbLogger
from rl_wq.hague_ddpg_Model import BasicEnv
# from processing.read_rpt import get_ele_df, get_file_contents, get_summary_df, get_total_flooding
from processing.swmm_utils import get_result_df

wgt_dir = "rl_wq/agent_weights"
result_dir = "rl_wq/results_082019"
train_steps = 365925
rwd_num = 36

swmm_file = "swmm_models/hague_v22_simpleRL.inp"
fcst = "timeseries_data/hague_fcst_2010_2019_dated.csv"
env = BasicEnv(inp_file=swmm_file, fcst_file=fcst,
               start_date=datetime(2019, 8, 1, 0, 0, 0), end_date=datetime(2019, 9, 1, 0, 0, 0))  # env with forecasts
assert len(env.action_space.shape) == 1
nb_actions = env.action_space.shape[0]

os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)
rn.seed(12345)
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
tf.set_random_seed(1234)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

actor = Sequential()
# actor.add(BatchNormalization(input_shape=(1,) + env.observation_space.shape))  # batch norm
actor.add(Flatten(input_shape=(1,) + env.observation_space.shape))  # is this needed with batch norm?
actor.add(Dense(300))  # was 16
actor.add(Activation('relu'))  # try leaky relu?
actor.add(Dense(150))
actor.add(Activation('relu'))
# actor.add(Dense(8))
# actor.add(Activation('relu'))
actor.add(Dense(nb_actions))
actor.add(Activation('sigmoid'))
#print(actor.summary())

action_input = Input(shape=(nb_actions,), name='action_input')
observation_input = Input(shape=(1,) + env.observation_space.shape, name='observation_input')
flattened_observation = Flatten()(observation_input)
x = Concatenate()([action_input, flattened_observation])
x = Dense(300)(x)  # was 32
x = Activation('relu')(x)
x = Dense(150)(x)
x = Activation('relu')(x)
# x = Dense(32)(x)
# x = Activation('relu')(x)
x = Dense(1)(x)
x = Activation('linear')(x)
critic = Model(inputs=[action_input, observation_input], outputs=x)
# print(critic.summary())

memory = SequentialMemory(limit=370000, window_length=1)
random_process = OrnsteinUhlenbeckProcess(size=nb_actions, theta=.15, mu=0., sigma=.25)  # maybe increase sigma, more exploration, was 0.1
agent = DDPGAgent(nb_actions=nb_actions, actor=actor, critic=critic, critic_action_input=action_input,
                  memory=memory, nb_steps_warmup_critic=50, nb_steps_warmup_actor=50,
                  random_process=random_process, gamma=.99, target_model_update=1e-3)
agent.compile(Adam(lr=.0001, clipnorm=1.), metrics=['mae'])  # maybe decrease lr, was 0.001
# agent.load_weights(os.path.join(wgt_dir,'ddpg_weights_300000_rwd31.h5f'))

# cb_list = [WandbLogger(name='Rwd{}, {}stp'.format(rwd_num, train_steps), project='hague_rl'),
#            ModelCheckpoint('rl_wq/agent_weights/ddpg_weights_best_rwd{}.h5f'.format(rwd_num),
#                            monitor='mae', save_best_only=True, mode='min', verbose=1, save_weights_only=True)]
cb_list = [WandbLogger(name='Rwd{}, {}stp'.format(rwd_num, train_steps), project='hague_rl')]
train_start = datetime.now()
agent.fit(env, nb_steps=train_steps, verbose=2, callbacks=cb_list)
agent.save_weights(os.path.join(wgt_dir, 'ddpg_weights_{}_rwd{}.h5f'.format(train_steps, rwd_num)), overwrite=True)
train_end = datetime.now()

# # get agent weights and names
# actor_weights = agent.actor.get_weights()
# critic_weights = agent.critic.get_weights()
# actor_names = [weight.name for layer in agent.actor.layers for weight in layer.weights]
# critic_names = [weight.name for layer in agent.critic.layers for weight in layer.weights]
#
# # plot agent weights
# import seaborn as sns
# data = critic_weights
#
# plt.subplot(141)
# sns.heatmap(data[0], vmin=-3, vmax=3, linewidth=0.5, cbar=False)
# plt.title("Hidden 1")
# plt.subplot(142)
# sns.heatmap(data[2], vmin=-3, vmax=3, linewidth=0.5, cbar=False)
# plt.title("Hidden 2")
# plt.subplot(143)
# sns.heatmap(data[4], vmin=-3, vmax=3, linewidth=0.5, cbar=False)
# plt.title("Hidden 3")
# plt.subplot(144)
# sns.heatmap(data[6], vmin=-3, vmax=3, linewidth=0.5)
# plt.title("Output")
# plt.tight_layout()
# plt.show()

# test learned policy on training data, no exploration
history = agent.test(env, nb_episodes=1, visualize=False, nb_max_start_steps=0)
env.close()
rpt_df = get_result_df(swmm_file.split('.')[0] + ".rpt")
rpt_df.to_csv(os.path.join(result_dir, "ddpg_{}steps_rwd{}_bignet.csv".format(train_steps, rwd_num)), index=True)

# # plot average rewards per episode
# avg_reward = []
# num_episodes = int(memory.nb_entries/env.T)
#
# for i in range(num_episodes):
#     temp_rwd = memory.rewards.data[env.T * i: env.T * i + env.T]
#     avg_reward.append(np.mean(temp_rwd))
#
# # plot reward per step
# plt.plot(memory.rewards.data)
# plt.tight_layout()
# plt.show()
#
# # plot state variables
# st1_depth, st3_depth = [], []
# r1_act, r3_act = [], []
# r1_flow, r3_flow = [], []
# r1_tss, r3_tss = [], []
# rain_fcst, tide_fcst = [], []
#
# for i in memory.observations.data[:memory.nb_entries]:
#     st1_depth.append(i[0])
#     st3_depth.append(i[1])
#     r1_act.append(i[2])
#     r3_act.append(i[3])
#     r1_flow.append(i[4])
#     r3_flow.append(i[5])
#     r1_tss.append(i[6])
#     r3_tss.append(i[7])
#     rain_fcst.append(i[8])
#     tide_fcst.append(i[9])
#
# fig = plt.figure(figsize=(6.5, 8.5))
# ax1 = fig.add_subplot(6, 1, 1)  # set up axes with sharing
# ax2 = fig.add_subplot(6, 1, 2, sharex=ax1)
# ax3 = fig.add_subplot(6, 1, 3, sharex=ax1)
# ax4 = fig.add_subplot(6, 1, 4, sharex=ax1)
# ax5 = fig.add_subplot(6, 1, 5, sharex=ax1)
# ax6 = fig.add_subplot(6, 1, 6, sharex=ax1)
# ax1.plot(st1_depth)
# ax1.plot(st3_depth)
# ax2.plot(r1_act)
# ax2.plot(r3_act)
# ax3.plot(r1_flow)
# ax3.plot(r3_flow)
# ax4.plot(r1_tss)
# ax4.plot(r3_tss)
# ax5.plot(rain_fcst)
# ax6.plot(tide_fcst)
# ax1.set_ylabel("Depth (ft)")
# ax2.set_ylabel("Valve Position")
# ax3.set_ylabel("Outflow (cfs)")
# ax4.set_ylabel("TSS (mg/L)")
# ax5.set_ylabel("Rain Fcst (in)")
# ax6.set_ylabel("Tide Fcst (ft)")
# fig.tight_layout()
# fig.show()
#
# # plot results from test with learned policy
# fig = plt.figure(figsize=(6.5, 8.5))
#
# ax1 = fig.add_subplot(6, 1, 1)  # set up axes with sharing
# ax2 = fig.add_subplot(6, 1, 2, sharex=ax1)
# ax3 = fig.add_subplot(6, 1, 3, sharex=ax1)
# ax4 = fig.add_subplot(6, 1, 4, sharex=ax1)
# ax5 = fig.add_subplot(6, 1, 5, sharex=ax1)
# ax6 = fig.add_subplot(6, 1, 6)
#
# ax1.plot(rpt_df["St1_Depth"], label='St1')
# ax1.plot(rpt_df["St3_Depth"], label='St3')
# ax1.plot(rpt_df["St1_Max"], linestyle='--', color='k', label='St1 Max')
# ax1.plot(rpt_df["St3_Max"], linestyle='--', color='grey', label='St3 Max')
# # ax1.plot(obs_pond_df["Pond"], ':', label="Obs. Level")
# ax1.set_ylabel("Pond Level (ft)")
# # ax1.set_ylim(-5)
# ax1.set_xticklabels([])
# # ax1.legend(loc='lower center', bbox_to_anchor=(0.5, -0.25), frameon=False, ncol=4)
# ax1.legend(ncol=4)
#
# ax2.plot(rpt_df["R1_act"], '-', label='R1')
# ax2.plot(rpt_df["R3_act"], ':', label='R3')
# ax2.set_ylim(0, 1.05)
# ax2.set_ylabel("Valve Position")
# ax2.set_xticklabels([])
# ax2.legend()
#
# ax3.plot(rpt_df["St1_TSS"], '-', label='St1')
# ax3.plot(rpt_df["St3_TSS"], ':', label='St3')
# # ax3.set_ylim(0, 1.05)
# ax3.set_ylabel("Pond TSS (mg/L)")
# ax3.set_xticklabels([])
# ax3.legend()
#
# ax4.plot(rpt_df["R1_TSS"], '-', label='R1')
# ax4.plot(rpt_df["R3_TSS"], ':', label='R3')
# # ax6.set_ylim(0, 1.05)
# ax4.set_ylabel("Valve TSS (mg/L)")
# ax4.legend()
#
# ax5.plot(rpt_df["St1_Flooding"], label='St1')
# ax5.plot(rpt_df["St3_Flooding"], label='St3')
# ax5.set_ylabel("Flooding (cfs)")
# ax5.legend()
#
# ax6.plot(avg_reward, color='k')
# # ax5.title('Average reward per episode')
# ax6.set_ylabel("Avg. Reward")
# ax6.set_xlabel("Episode")
#
# fig.tight_layout()
# # fig.subplots_adjust(top=0.991, bottom=0.027, left=0.097, right=0.919, hspace=0.235, wspace=0.2)
# # fig.show()
# fig.savefig("C:/PycharmProjects/swmm_rl_hague/results_082019/ddpg_" + str(train_steps) + "steps_rwd3_valve.png", dpi=300)
# plt.close()

# ############################### test agent on all data ###############################################################
test_dates = {'052015': [datetime(2015, 5, 1, 0, 0, 0), datetime(2015, 6, 1, 0, 0, 0)],  # TODO add 2010-2019 test
              '092014': [datetime(2014, 9, 1, 0, 0, 0), datetime(2014, 10, 1, 0, 0, 0)],
              '092016': [datetime(2016, 9, 1, 0, 0, 0), datetime(2016, 10, 1, 0, 0, 0)]}

for key, value in test_dates.items():
    env = BasicEnv(inp_file=swmm_file, fcst_file=fcst, start_date=value[0], end_date=value[1])
    agent.test(env, nb_episodes=1, visualize=False, nb_max_start_steps=0)
    env.close()
    rpt_df = get_result_df(swmm_file.split('.')[0] + ".rpt")
    rpt_df.to_csv(os.path.join(result_dir, "ddpg_{}steps_rwd{}_test{}.csv".format(train_steps, rwd_num, key)),
                  index=True)
