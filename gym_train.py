import time
import json
import gym
import numpy as np
import pickle
import copy
from gym import wrappers
import argparse
import os
from td import SarsaAgent, ExpectedSarsaAgent, QAgent, SarsaLambdaAgent 
from td import HLSAgent

def suffix_dir(params):
    r = ''
    for p, v in params.iteritems():
        r += str(p) + '-' + str(v) + '_'
    return r

parser = argparse.ArgumentParser(description='Train model')
parser.add_argument('-m', dest='model_dir', type=str, required=True,
                    help='Directory of the gym model')
parser.add_argument('-l', dest='learning_alg', type=str,required=True,
                    help='Learning algorithm: "sarsa" or "expsarsa" or "q"')
parser.add_argument('-n', dest='n_episodes', type=int, required=True,
                    help='# episodes')
parser.add_argument('--alpha', dest='alpha', type=float,default=0.2,
                    help='learning rate')
parser.add_argument('--gamma', dest='gamma', type=float,default=1.0,
                    help='discounting factor')
parser.add_argument('--lambda', dest='lambda_', type=float,default=0.5)
parser.add_argument('--eps', dest='eps', type=float,default=0.1,
                    help='epsilon-greediness')
parser.add_argument('--n-trials', dest='n_trials', type=int, default=10,
                    help='number of trials')

args = parser.parse_args()

alpha = args.alpha
gamma = args.gamma
lambda_ = args.lambda_
eps = args.eps
n_episodes = args.n_episodes
params = copy.deepcopy(vars(args))
del params['model_dir']
del params['n_episodes']
model_dir = args.model_dir + '_' + suffix_dir(params)

agent = None

if not os.path.exists(model_dir):
    os.makedirs(model_dir)

#if os.path.isdir(model_dir) and os.path.exists(model_dir):
#    print "Directory " + model_dir + " already exists. Resume training."
#    agent = pickle.load(open(model_dir + "/model.p", "rb"))


env_name = "Taxi-v1"
env = gym.make(env_name)

A = env.action_space.n
if isinstance(env.observation_space, gym.spaces.discrete.Discrete):
    S = env.observation_space.n
elif isinstance(env.observation_space, gym.spaces.box.Box):
    S = env.observation_space.shape[0]
else:
    print env.observation_space

features_size = S
feature_func = lambda x: x

if not agent:
    if args.learning_alg == 'sarsa':
        create_agent = lambda: SarsaAgent(env_name, A, S, alpha, gamma, eps)
    elif args.learning_alg == 'q':
        create_agent = lambda: QAgent(env_name, A, S, alpha, gamma, eps)
    elif args.learning_alg == 'expsarsa':
        create_agent = lambda: ExpectedSarsaAgent(env_name, A, S, alpha, gamma, eps)
    elif args.learning_alg == 'sarsalambda':
        create_agent = lambda: SarsaLambdaAgent(env_name, A, S, alpha, gamma, eps, lambda_)
    elif args.learning_alg == 'hls':
        create_agent = lambda: HLSAgent(env_name, A, S, gamma, eps, lambda_)
    else:
        raise Exception("Not Implemented")

#np.seterr(all='raise')

rewards_count = np.zeros((args.n_trials, n_episodes))
episodes_len = np.zeros((args.n_trials, n_episodes))
for trial in range(args.n_trials):
    agent = create_agent()
    for i_episode in range(n_episodes):
        if i_episode % 100 == 0:
            print "# episode:", i_episode
        observation = env.reset()
        agent.reset(observation)
        done = False
        t = 0
        r = 0
        while not done and t < 100:
            action = agent.sample_action(observation)
            observation, reward, done, info = env.step(action)
            r += reward
            agent.update(action, observation, reward, False)
            t += 1
            if done:
                break
        rewards_count[trial, i_episode] = r
        episodes_len[trial, i_episode] = t
        action = agent.sample_action(observation)
        # only action matter, as observation and reward are not used
        agent.update(action, observation, reward, True)

pickle.dump(rewards_count, open(model_dir + "/rewards_count.p", "wb"))
pickle.dump(episodes_len, open(model_dir + "/episode_lengths.p", "wb"))
#pickle.dump(agent, open(model_dir + "/model.p", "wb"))
