# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 03:33:41 2019

@author: bdb3m, cw8xk
"""
from datetime import datetime
import os
import glob
import random as rn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, Concatenate, BatchNormalization
from keras.optimizers import Adam
import tensorflow as tf
from keras import backend as K
from rl.agents import DDPGAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess, GaussianWhiteNoiseProcess
from rl.callbacks import WandbLogger, ModelIntervalCheckpoint, TrainEpisodeLogger
from swmm_rl_hague.hague_ddpg_Model_tune import BasicEnv
# from swmm_rl_hague.read_rpt import get_ele_df, get_file_contents, get_summary_df, get_total_flooding
from swmm_rl_hague.swmm_utils import get_result_df
total_steps = 100000
train_steps = 25000
rwd_num = 31

chkpnt_path = "C:/PycharmProjects/swmm_rl_hague/agent_weights/checkpoint_weights"
inp_dir = "C:/PycharmProjects/swmm_rl_hague/hague_inp_train/"
out_dir = "C:/PycharmProjects/swmm_rl_hague/multi_inp/results"
wgt_dir = "C:/PycharmProjects/swmm_rl_hague/multi_inp/agent_weights"
inp_files = glob.glob(inp_dir + '*.inp')
fcst = "C:/Users/Ben Bowes/Documents/LongTerm_SWMM/hague_fcst_2010_2019_dated.csv"

file_num = 1
for file in inp_files:
    if file_num == 1:
        env = BasicEnv(inp_file=file, fcst_file=fcst)  # env with forecasts
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
        actor.add(Dense(50))  # was 16
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

        memory = SequentialMemory(limit=total_steps, window_length=1)
        # random_process = OrnsteinUhlenbeckProcess(size=nb_actions, theta=.15, mu=0., sigma=.25)  # maybe increase sigma, more exploration, was 0.1
        random_process = GaussianWhiteNoiseProcess(mu=0., sigma=1., sigma_min=0.01, n_steps_annealing=total_steps, size=nb_actions)  # sigma decreasing from 1 to 0 in 1000steps?
        agent = DDPGAgent(nb_actions=nb_actions, actor=actor, critic=critic, critic_action_input=action_input,  # get sigma/noise from here?
                          memory=memory, nb_steps_warmup_critic=50, nb_steps_warmup_actor=50,
                          random_process=random_process, gamma=.99, target_model_update=1e-3)
        agent.compile(Adam(lr=.001, clipnorm=1.), metrics=['mae'])  # maybe decrease lr, was 0.001
        # agent.load_weights('swmm_rl_hague/tuning/agent_weights/ddpg_weights_config19_150k.h5f')

        wb_config = {'rwd': rwd_num, 'steps': train_steps, "nets": "(50, 25)", "lr": 0.001, "noise": "gaussian",
                     'seed': "(22, 54321, 4321)", 'activation': 'relu', 'annealing': total_steps, 'sigma_min': 0.01,
                     'rwd threshold': 0.5, 'dry rwd': 'with system flooding/15000'}
        # cb_list = [WandbLogger(config=wb_config, name='Config{}'.format(config_num), project='tuning'),
        #            ModelCheckpoint('C:/PycharmProjects/swmm_rl_hague/tuning/agent_weights/ddpg_weights_best_config{}.h5f'.format(config_num),
        #                            monitor='loss', save_best_only=True, mode='min', verbose=1, save_weights_only=True)]
        cb_list = [WandbLogger(config=wb_config, name='Config{}_200k'.format(config_num), project='multi_train')]
                  # , ModelIntervalCheckpoint(os.path.join(chkpnt_path, 'chkpnt.h5f'), 10000, verbose=1)

        agent.fit(env, nb_steps=train_steps, verbose=2, callbacks=cb_list)  # TODO remove wandb.finish from callback
        env.close()

    else:
        env = BasicEnv(inp_file=file, fcst_file=fcst)  # env with forecasts
        agent.fit(env, nb_steps=train_steps, verbose=2, callbacks=cb_list)
        env.close()
    file_num += 1

agent.save_weights(os.path.join(wgt_dir, 'ddpg_{}steps_rwd{}.h5f'.format(train_steps, rwd_num)), overwrite=True)

# test on all training files
for file in inp_files:
    env = BasicEnv(inp_file=file, fcst_file=fcst)
    agent.test(env, nb_episodes=1, visualize=False, nb_max_start_steps=0)
    env.close()

    rpt_df = get_result_df(file.split('.')[0] + '.rpt')
    rpt_df.to_csv(os.path.join(out_dir, "ddpg_{}steps_rwd{}.csv".format(train_steps, rwd_num)), index=True)

# # test on train data with actions repeated
# agent.test(env, nb_episodes=1, action_repetition=4, visualize=False, nb_max_start_steps=0)
# env.close()
