# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 03:33:41 2019

@author: bdb3m, cw8xk
"""
from datetime import datetime
import os
import random as rn
import numpy as np
import pickle
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
from rl.random import OrnsteinUhlenbeckProcess, GaussianWhiteNoiseProcess
from rl.callbacks import WandbLogger
from rl_wq.hague_ddpg_Model import BasicEnv
# from processing.read_rpt import get_ele_df, get_file_contents, get_summary_df, get_total_flooding
from processing.swmm_utils import get_result_df

wgt_dir = "rl_wq/agent_weights"
mem_dir = "rl_wq/agent_memory"
test_dir = "rl_wq/hague_inp_files_test"
result_dir = "rl_wq/results_082019"
train_steps = 50000
rwd_num = 31
config_num = 38

swmm_file = "swmm_models/hague_v22_simpleRL.inp"
fcst = "timeseries_data/hague_fcst_2010_2019_dated.csv"
env = BasicEnv(inp_file=swmm_file, fcst_file=fcst)  # env with forecasts
assert len(env.action_space.shape) == 1
nb_actions = env.action_space.shape[0]

os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(22)
rn.seed(54321)
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
tf.set_random_seed(4321)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

actor = Sequential()
# actor.add(BatchNormalization(input_shape=(1,) + env.observation_space.shape))  # batch norm
actor.add(Flatten(input_shape=(1,) + env.observation_space.shape))  # is this needed with batch norm?
actor.add(Dense(50))
actor.add(Activation('relu'))  # try leaky relu?
actor.add(Dense(25))
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
x = Dense(50)(x)  # was 32
x = Activation('relu')(x)
x = Dense(25)(x)
x = Activation('relu')(x)
# x = Dense(32)(x)
# x = Activation('relu')(x)
x = Dense(1)(x)
x = Activation('linear')(x)
critic = Model(inputs=[action_input, observation_input], outputs=x)
# print(critic.summary())

# unpickle previous memory if retraining from saved weights
# memory = pickle.load(open(os.path.join(mem_dir, 'ddpg_mem_100000_rwd39_config26_gamma50'), 'rb'))
memory = SequentialMemory(limit=train_steps, window_length=1)
# random_process = OrnsteinUhlenbeckProcess(size=nb_actions, theta=.15, mu=0., sigma=.25)  # maybe increase sigma, more exploration, was 0.1
random_process = GaussianWhiteNoiseProcess(mu=0., sigma=.3, sigma_min=0.01, n_steps_annealing=train_steps, size=nb_actions)
agent = DDPGAgent(nb_actions=nb_actions, actor=actor, critic=critic, critic_action_input=action_input,
                  memory=memory, nb_steps_warmup_critic=0, nb_steps_warmup_actor=0,
                  random_process=random_process, gamma=.99, target_model_update=1e-3)
agent.compile(Adam(lr=.001, clipnorm=1.), metrics=['mae'])  # maybe decrease lr, was 0.001
agent.load_weights(os.path.join(wgt_dir, 'ddpg_weights_50000_config38.h5f'))

# cb_list = [WandbLogger(name='Rwd{}, {}stp'.format(rwd_num, train_steps), project='hague_rl'),
#            ModelCheckpoint('rl_wq/agent_weights/ddpg_weights_best_rwd{}.h5f'.format(rwd_num),
#                            monitor='mae', save_best_only=True, mode='min', verbose=1, save_weights_only=True)]
wb_config = {'rwd': rwd_num, 'steps': train_steps, "nets": "(50, 25)", "lr": 0.001, "noise": "gaussian",
             'seed': "(22, 54321, 4321)", 'activation': 'relu', 'annealing': train_steps, 'sigma': 0.3, 'gamma': .99,
             'sigma_min': 0.01, 'fcst threshold': 0.5, 'rain threshold': 0.1, 'dry rwd': 'with system flooding/35000'}
cb_list = [WandbLogger(config=wb_config, name='Config{}_{}'.format(config_num, train_steps), project='tuning')]
# cb_list = [WandbLogger(config={"lr": 0.001, "annealing": 100000}, name='Config{}'.format(config_num), project='tuning')]
train_start = datetime.now()
agent.fit(env, nb_steps=train_steps, verbose=2, callbacks=cb_list)
# agent.save_weights(os.path.join(wgt_dir, 'ddpg_weights_{}_rwd{}_config{}.h5f'.format(train_steps, rwd_num, config_num)), overwrite=True)
agent.save_weights(os.path.join(wgt_dir, 'ddpg_weights_{}_rwd{}_config{}.h5f'.format(train_steps, rwd_num, config_num)), overwrite=True)
pickle.dump(memory, open(os.path.join(mem_dir, 'ddpg_mem_{}_rwd{}_config{}'.format(train_steps, rwd_num, config_num)), 'wb'))
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
rpt_df.to_csv(os.path.join(result_dir, "ddpg_{}steps_rwd{}_config{}.csv".format(train_steps, rwd_num, config_num)), index=True)

# test agent on all data
test_files = os.listdir(test_dir)
test_files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
for file in test_files:
    if file.endswith('.inp'):
        print(file)
        env = BasicEnv(inp_file=os.path.join(test_dir, file), fcst_file=fcst)
        agent.test(env, nb_episodes=1, visualize=False, nb_max_start_steps=0)
        env.close()

        rpt_df = get_result_df(os.path.join(test_dir, file.split('.')[0] + ".rpt"))
        rpt_df.to_csv(os.path.join(result_dir,
                                   "ddpg_{}steps_rwd{}_test{}_config38_mass.csv".format(train_steps, rwd_num,
                                                                                   file.split('_')[3].split('.')[0])),
                      index=True)
