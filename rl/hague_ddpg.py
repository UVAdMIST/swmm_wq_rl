# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 03:33:41 2019

@author: cw8xk
"""
from datetime import datetime
import os
import random as rn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, Concatenate
from keras.optimizers import Adam
import tensorflow as tf
from keras import backend as K
from rl.agents import DDPGAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess
from swmm_rl_hague.hague_ddpg_Model import BasicEnv


def get_end_line(file_lines, start_line):
    for i in range(len(file_lines[start_line:])):
        line_no = start_line + i
        if file_lines[line_no].strip() == "" and file_lines[line_no + 1].strip() == "":
            return line_no
    # raise error if end line of section not found
    raise KeyError('Did not find end of section starting on line {}'.format(start_line))


print("start time: ", datetime.now())

swmm_file = "C:/PycharmProjects/swmm_rl_hague/hague_inp_files_pyswmm/hague_v17_simpleRL.inp"
fcst = "C:/Users/Ben Bowes/Documents/LongTerm_SWMM/hague_fcst_2010_2019_dated.csv"
env = BasicEnv(inp_file=swmm_file, fcst_file=fcst)  # env with forecasts
# env = BasicEnv(depth=Depth)  # basic env
assert len(env.action_space.shape) == 1
nb_actions = env.action_space.shape[0]

# np.random.seed(123)
# set_random_seed(1234)

os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)
rn.seed(12345)
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
tf.set_random_seed(1234)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

actor = Sequential()
actor.add(Flatten(input_shape=(1,) + env.observation_space.shape))
actor.add(Dense(16))
actor.add(Activation('relu'))
actor.add(Dense(16))
actor.add(Activation('relu'))
actor.add(Dense(8))
actor.add(Activation('relu'))
actor.add(Dense(nb_actions))
actor.add(Activation('sigmoid'))
#print(actor.summary())

action_input = Input(shape=(nb_actions,), name='action_input')
observation_input = Input(shape=(1,) + env.observation_space.shape, name='observation_input')
flattened_observation = Flatten()(observation_input)
x = Concatenate()([action_input, flattened_observation])
x = Dense(32)(x)
x = Activation('relu')(x)
x = Dense(32)(x)
x = Activation('relu')(x)
x = Dense(32)(x)
x = Activation('relu')(x)
x = Dense(1)(x)
x = Activation('linear')(x)
critic = Model(inputs=[action_input, observation_input], outputs=x)
# print(critic.summary())

memory = SequentialMemory(limit=1000000, window_length=1)
random_process = OrnsteinUhlenbeckProcess(size=nb_actions, theta=.15, mu=0., sigma=.1)
agent = DDPGAgent(nb_actions=nb_actions, actor=actor, critic=critic, critic_action_input=action_input,
                  memory=memory, nb_steps_warmup_critic=50, nb_steps_warmup_actor=50,
                  random_process=random_process, gamma=.99, target_model_update=1e-3)
agent.compile(Adam(lr=.001, clipnorm=1.), metrics=['mae'])
train_steps = 297000  # 197000
# agent.load_weights('swmm_rl_multi_inp_forecast/agent_weights/ddpg_swmm_weights_100000_depth_4.61.h5f'.format(Depth))  # these wgts are from training XXX, but are used in the forecast models
# agent.load_weights('C:/PycharmProjects/swmm_rl_multi_inp_forecast/agent_weights/ddpg_swmm_weights2_197000_depth_4.61.h5f')  # final weights from study 2

agent.fit(env, nb_steps=train_steps, verbose=1)
agent.save_weights('swmm_rl_hague/agent_weights/ddpg_weights_{}.h5f'.format(train_steps), overwrite=True)

print("training finished at: ", datetime.now())

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

history = agent.test(env, nb_episodes=1, visualize=False, nb_max_start_steps=0)
env.close()
all_actions = np.array(history.history['action'])
all_states = np.array(history.history['states'])
all_depths = all_states[:, :, :2]
all_flooding = all_states[:, :, 2:4]
st1_max = [10] * len(all_depths[0])
st3_max = [7.16] * len(all_depths[0])

# get total flood volume (10^6 gal) from .rpt file
total_flood = 0
# node_flooded = []
    if all_flooding.max() > 0:
    with open(swmm_file.split('.')[0] + ".rpt", 'r') as rpt_file:
        lines = rpt_file.readlines()
    for i, l in enumerate(lines):
        if l.startswith("  Node Flooding Summary"):  # find flooding section
            start = i + 10
            end = get_end_line(file_lines=lines, start_line=start)
            for line in lines[start:end]:
                # node_flooded.append(line.strip().split()[0])
                total_flood += float(line.strip().split()[-2])

# plot average rewards per episode
avg_reward = []
num_episodes = int(memory.nb_entries/env.T)

for i in range(num_episodes):
    temp_rwd = memory.rewards.data[env.T * i: env.T * i + env.T]
    avg_reward.append(np.mean(temp_rwd))

# plot results from test with learned policy
fig = plt.figure(1, figsize=(6, 8))

plt.subplot(4, 1, 1)
depth_plot = plt.plot(all_depths[0])
max_1 = plt.plot(st1_max, linestyle=':', color='k', label='St1 max')
max_2 = plt.plot(st3_max, linestyle=':', color='grey', label='St3 max')
# plt.ylim(0, 6)
plt.title('Depths')
plt.ylabel("ft")
# plt.xlabel("time step")
first_legend = plt.legend(depth_plot, ('St1', 'St3'), bbox_to_anchor=(0.03, -.5, 1., .102), loc=3,
                          ncol=2, borderaxespad=0.1, frameon=False, columnspacing=1)
ax = plt.gca().add_artist(first_legend)
plt.legend(loc=4, bbox_to_anchor=(-0.025, -.5, 1., .102), ncol=2, borderaxespad=0.1, frameon=False, columnspacing=1)

plt.subplot(4, 1, 2)
act_plot = plt.plot(all_actions[0][:, 0], '-', all_actions[0][:, 1], ':')
plt.ylim(0, 1.05)
plt.title('Policy')
plt.ylabel("Valve Position")
plt.xlabel("time step")
plt.legend(act_plot, ('R1', 'R3'))

plt.subplot(4, 1, 3)
plt.plot(all_flooding[0],  label=['St1', 'St3'])
plt.ylim(0)
plt.title('Flooding')
plt.ylabel("CFS")
# plt.xlabel("time step")
flood_str = "Total Vol. = " + str(round(total_flood, 3)) + "MG"
# flood_max = max(max(all_flooding[0][0:, 0]), max(all_flooding[0][0:, 1]), max(all_flooding[0][0:, 2])) * 0.9
_, top = plt.gca().get_ylim()
flood_max = top * 0.85
plt.text(0, flood_max, flood_str)

plt.subplot(4, 1, 4)
plt.plot(avg_reward, color='k')
plt.title('Average reward per episode')
plt.ylabel("reward")
plt.xlabel("episode")

plt.tight_layout()
# plt.show()
plt.savefig("C:/PycharmProjects/swmm_rl_hague/results_082019/ddpg_" + str(train_steps) + "steps_rwd1.png", dpi=300)
plt.close()

print("testing and plotting 1 episode finished at: ", datetime.now())

# ############################### test agent on all data ###############################################################
test_dir = "C:/PycharmProjects/LongTerm_SWMM/observed_data/obs_data_1month_all_controlled"
results_dir = "C:/PycharmProjects/LongTerm_SWMM/observed_data/obs_test_data/1month_results"
fcst_dir = "C:/PycharmProjects/LongTerm_SWMM/observed_data/obs_data_daily_fcsts"

for file in os.scandir(test_dir):
    if file.name.endswith('.inp'):
        print('testing ', file.name)
        env = BasicEnv(inp_file=os.path.join(test_dir, file), fcst_file=os.path.join(fcst_dir, file.name.split('.')[0] + ".csv"), depth=Depth)
        history = agent.test(env, nb_episodes=1, visualize=False, nb_max_start_steps=0)
        env.close()

        # get rain/tide data from inp file
        rain1_str = []
        rain2_str = []
        tide_str = []
        with open(file.path, 'r') as tmp_file:
            lines = tmp_file.readlines()
            for i, l in enumerate(lines):
                if l.startswith("[TIMESERIES]"):  # find time series section
                    start = i + 3
        for i, l in enumerate(lines[start:]):
            if l.startswith('Rain1'):
                rain1_str.append(l)
            if l.startswith('Rain2'):
                rain2_str.append(l)
            if l.startswith('Tide1'):
                tide_str.append(l)

        rain1_data = []
        rain1_time = []
        rain2_data = []
        rain2_time = []
        tide_data = []
        tide_time = []
        for i in rain1_str:
            rain1_data.append(i.split(' ')[3].rstrip())
            rain1_time.append(i.split(' ')[1] + " " + i.split(' ')[2])

        for i in rain2_str:
            rain2_data.append(i.split(' ')[3].rstrip())
            rain2_time.append(i.split(' ')[1] + " " + i.split(' ')[2])

        for i in tide_str:
            tide_data.append(i.split(' ')[3].rstrip())
            tide_time.append(i.split(' ')[1] + " " + i.split(' ')[2])

        rain1_df = pd.DataFrame([rain1_time, rain1_data]).transpose()
        rain1_df.columns = ['datetime1', 'rain1']
        rain1_df['datetime1'] = pd.to_datetime(rain1_df['datetime1'], infer_datetime_format=True)
        rain1_df.set_index(pd.DatetimeIndex(rain1_df['datetime1']), inplace=True)
        rain1_df['rain1'] = rain1_df['rain1'].astype('float')
        rain1_df = rain1_df.resample('H').sum()

        rain2_df = pd.DataFrame([rain2_time, rain2_data]).transpose()
        rain2_df.columns = ['datetime2', 'rain2']
        rain2_df['datetime2'] = pd.to_datetime(rain2_df['datetime2'], infer_datetime_format=True)
        rain2_df.set_index(pd.DatetimeIndex(rain2_df['datetime2']), inplace=True)
        rain2_df['rain2'] = rain2_df['rain2'].astype('float')
        rain2_df = rain2_df.resample('H').sum()

        tide_df = pd.DataFrame([tide_time, tide_data], dtype='float64').transpose()
        tide_df.columns = ['datetime', 'tide']
        tide_df['datetime'] = pd.to_datetime(tide_df['datetime'], infer_datetime_format=True)
        tide_df.set_index(pd.DatetimeIndex(tide_df['datetime']), inplace=True)
        tide_df['tide'] = tide_df['tide'].astype('float')

        df = pd.concat([rain1_df['rain1'], rain2_df['rain2'], tide_df['tide']], axis=1)
        df[['rain1', 'rain2']].fillna(0, inplace=True)
        df.reset_index(inplace=True)

        all_actions = np.array(history.history['action'])
        all_states = np.array(history.history['states'])
        all_depths = all_states[:, :, :3]
        all_flooding = all_states[:, :, 3:6]
        st_max = [Depth] * len(all_depths[0])
        st3_max = [2] * len(all_depths[0])

        # get total flood volume (10^6 gal) from .rpt file
        total_flood = 0
        # node_flooded = []
        if all_flooding.max() > 0:
            rpt_name = file.name.split('.')[0] + ".rpt"
            # with open(os.path.join("/home/bdb3m/swmm_rl/observed_data/obs_test_data", rpt_name), 'r') as rpt_file:
            with open(os.path.join(test_dir, rpt_name), 'r') as rpt_file:
                lines = rpt_file.readlines()
            for i, l in enumerate(lines):
                if l.startswith("  Node Flooding Summary"):  # find flooding section
                    start = i + 10
                    end = get_end_line(file_lines=lines, start_line=start)
                    for line in lines[start:end]:
                        # node_flooded.append(line.strip().split()[0])
                        total_flood += float(line.strip().split()[-2])
        total_flood = [total_flood] * len(all_depths[0])

        # save results data
        result_list = [all_depths[0][0:, 0].tolist(), all_depths[0][0:, 1].tolist(), all_depths[0][0:, 2].tolist(),
                       all_flooding[0][0:, 0].tolist(), all_flooding[0][0:, 1].tolist(),
                       all_flooding[0][0:, 2].tolist(),
                       total_flood, st_max, st3_max, all_actions[0][0:, 0].tolist(), all_actions[0][0:, 1].tolist()]
        result_cols = ["St1_depth", "St2_depth", "J1_depth",
                       "St1_flooding", "St2_flooding", "J1_flooding",
                       "total_flood", "St_full", "J1_full", "R1_act", "R2_act"]
        results_df = pd.DataFrame(result_list).transpose()
        results_df.columns = result_cols
        results_df = pd.concat([results_df, df], axis=1)
        results_df.to_csv(os.path.join(results_dir, file.name.split('.')[0] + ".csv"), index=False)

        # # plot average rewards per episode
        # avg_reward = []
        # num_episodes = int(memory.nb_entries/env.T)
        #
        # for i in range(num_episodes):
        #     temp_rwd = memory.rewards.data[env.T * i: env.T * i + env.T]
        #     avg_reward.append(np.mean(temp_rwd))

        # plot results from test with learned policy
        fig, axs = plt.subplots(4, sharey='none', sharex='none', figsize=(6, 8))
        # first plot is tide and rainfall
        ax = axs[0]
        df["tide"].plot(ax=ax, color='c', legend=None)
        ax2 = ax.twinx()
        ax2.invert_yaxis()
        df["rain1"].plot.bar(ax=ax2, color="b", legend=None)
        ax2.set_xticks([])
        ax.set_xticks([], minor=True)
        ax.set_ylabel("Sea Level (ft.)")
        ax2.set_ylabel("Rainfall (in.)")
        ax.set_title('Inputs')
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, ("Sea Level", "Cumulative Rainfall"), bbox_to_anchor=(0.85, -0.05), ncol=2,
                  frameon=False)

        # plot depths
        depth_plot = axs[1].plot(all_depths[0])
        max_1 = axs[1].plot(st_max, linestyle=':', color='k', label='Storage max')
        max_2 = axs[1].plot(st3_max, linestyle=':', color='grey', label='Pipe max')
        axs[1].set_ylim(0, 6)
        axs[1].set_title('Depths')
        axs[1].set_ylabel("ft")
        # plt.xlabel("time step")
        # lines, labels = axs[1].get_legend_handles_labels()
        # axs[1].legend(lines, labels, bbox_to_anchor=(0.8, -0.05), ncol=5)
        first_legend = axs[1].legend(depth_plot, ('St1', 'St2', 'J1'), bbox_to_anchor=(0.0, -.5, 1., .102),
                                     loc=3, ncol=3, borderaxespad=0.1, frameon=False, columnspacing=.75)
        legend_ax = axs[1].add_artist(first_legend)
        axs[1].legend(loc=4, bbox_to_anchor=(0.025, -.5, 1., .11), ncol=2,
                      borderaxespad=0.1, frameon=False, columnspacing=.75)

        # plot actions
        act_plot = axs[2].plot(all_actions[0][:, 0], '-', all_actions[0][:, 1], ':')
        axs[2].set_ylim(0, 1.05)
        axs[2].set_title('Policy')
        axs[2].set_ylabel("Valve Position")
        axs[2].legend(act_plot, ['R1', 'R2'], bbox_to_anchor=(0.0, -.5, 1., .102),
                      loc=10, ncol=2, borderaxespad=0.1, frameon=False, columnspacing=.75)

        # plot flooding
        axs[3].plot(all_flooding[0])
        axs[3].set_ylim(0)
        axs[3].set_title('Flooding')
        axs[3].set_ylabel("CFS")
        axs[3].set_xlabel("time step")
        flood_str = "Total Vol. = " + str(round(total_flood[-1], 3)) + "MG"
        # flood_max = max(max(all_flooding[0][0:, 0]), max(all_flooding[0][0:, 1]), max(all_flooding[0][0:, 2])) * 0.9
        _, top = axs[3].get_ylim()
        flood_max = top * 0.85
        axs[3].text(0, flood_max, flood_str)

        # plot average reward
        # plt.subplot(4, 1, 4)
        # plt.plot(avg_reward, color='k')
        # plt.ylim(plt.ylim()[0], 0)
        # plt.title('Average reward per episode')
        # plt.ylabel("reward")
        # plt.xlabel("episode")

        plt.tight_layout()
        # plt.show()
        # plt.savefig("/home/bdb3m/swmm_rl/observed_data/obs_test_data/" + file.name.split('.')[0] + ".png", dpi=300)
        plt.savefig(os.path.join(results_dir, file.name.split('.')[0] + ".png"), dpi=300)
        plt.close()
