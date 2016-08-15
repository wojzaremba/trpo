from utils import *
import numpy as np
import random
import tensorflow as tf
import time
import os
import logging
import gym
from gym import envs, scoreboard
from gym.spaces import Discrete, Box
import prettytensor as pt
from space_conversion import SpaceConversionEnv
import tempfile
import sys

class TRPOAgent(object):

    config = dict2(**{
        "timesteps_per_batch": 1000,
        "max_pathlength": 10000,
        "max_kl": 0.01,
        "cg_damping": 0.1,
        "gamma": 0.95})

    def __init__(self, env):
        self.env = env
        if not isinstance(env.observation_space, Box) or \
           not isinstance(env.action_space, Discrete):
            print("Incompatible spaces.")
            exit(-1)
        print("Observation Space", env.observation_space)
        print("Action Space", env.action_space)
        self.session = tf.Session()
        self.end_count = 0
        self.train = True
        self.obs = obs = tf.placeholder(
            dtype, shape=[
                None, 2 * env.observation_space.shape[0] + env.action_space.n], name="obs")
        self.prev_obs = np.zeros((1, env.observation_space.shape[0]))
        self.prev_action = np.zeros((1, env.action_space.n))
        self.action = action = tf.placeholder(tf.int64, shape=[None], name="action")
        self.advant = advant = tf.placeholder(dtype, shape=[None], name="advant")
        self.oldaction_dist = oldaction_dist = tf.placeholder(dtype, shape=[None, env.action_space.n], name="oldaction_dist")

        # Create neural network.
        action_dist_n, _ = (pt.wrap(self.obs).
                            fully_connected(64, activation_fn=tf.nn.tanh).
                            softmax_classifier(env.action_space.n))
        eps = 1e-6
        self.action_dist_n = action_dist_n
        N = tf.shape(obs)[0]
        p_n = slice_2d(action_dist_n, tf.range(0, N), action)
        oldp_n = slice_2d(oldaction_dist, tf.range(0, N), action)
        ratio_n = p_n / oldp_n
        Nf = tf.cast(N, dtype)
        surr = -tf.reduce_mean(ratio_n * advant)  # Surrogate loss
        var_list = tf.trainable_variables()
        kl = tf.reduce_sum(oldaction_dist * tf.log((oldaction_dist + eps) / (action_dist_n + eps))) / Nf
        ent = tf.reduce_sum(-action_dist_n * tf.log(action_dist_n + eps)) / Nf

        self.losses = [surr, kl, ent]
        self.pg = flatgrad(surr, var_list)
        # KL divergence where first arg is fixed
        # replace old->tf.stop_gradient from previous kl
        kl_firstfixed = tf.reduce_sum(tf.stop_gradient(
            action_dist_n) * tf.log(tf.stop_gradient(action_dist_n + eps) / (action_dist_n + eps))) / Nf
        grads = tf.gradients(kl_firstfixed, var_list)
        self.flat_tangent = tf.placeholder(dtype, shape=[None])
        shapes = map(var_shape, var_list)
        start = 0
        tangents = []
        for shape in shapes:
            size = np.prod(shape)
            param = tf.reshape(self.flat_tangent[start:(start + size)], shape)
            tangents.append(param)
            start += size
        gvp = [tf.reduce_sum(g * t) for (g, t) in zip(grads, tangents)]
        self.fvp = flatgrad(gvp, var_list)
        self.gf = GetFlat(self.session, var_list)
        self.sff = SetFromFlat(self.session, var_list)
        self.vf = VF(self.session)
        self.session.run(tf.initialize_all_variables())

    def act(self, obs, *args):
        obs = np.expand_dims(obs, 0)
        self.prev_obs = obs
        obs_new = np.concatenate([obs, self.prev_obs, self.prev_action], 1)

        action_dist_n = self.session.run(self.action_dist_n, {self.obs: obs_new})

        if self.train:
            action = int(cat_sample(action_dist_n)[0])
        else:
            action = int(np.argmax(action_dist_n))
        self.prev_action *= 0.0
        self.prev_action[0, action] = 1.0
        return action, action_dist_n, np.squeeze(obs_new)

    def learn(self):
        config = self.config
        start_time = time.time()
        numeptotal = 0
        i = 0
        while True:
            # Generating paths.
            print("Rollout")
            paths = rollout(
                self.env,
                self,
                config.max_pathlength,
                config.timesteps_per_batch)

            # Computing returns and estimating advantage function.
            for path in paths:
                path["baseline"] = self.vf.predict(path)
                path["returns"] = discount(path["rewards"], config.gamma)
                path["advant"] = path["returns"] - path["baseline"]

            # Updating policy.
            action_dist_n = np.concatenate([path["action_dists"] for path in paths])
            obs_n = np.concatenate([path["obs"] for path in paths])
            action_n = np.concatenate([path["actions"] for path in paths])
            baseline_n = np.concatenate([path["baseline"] for path in paths])
            returns_n = np.concatenate([path["returns"] for path in paths])

            # Standardize the advantage function to have mean=0 and std=1.
            advant_n = np.concatenate([path["advant"] for path in paths])
            advant_n -= advant_n.mean()

            # Computing baseline function for next iter.

            advant_n /= (advant_n.std() + 1e-8)

            feed = {self.obs: obs_n,
                    self.action: action_n,
                self.advant: advant_n,
                    self.oldaction_dist: action_dist_n}


            episoderewards = np.array(
                [path["rewards"].sum() for path in paths])

            print "\n********** Iteration %i ************" % i
            if episoderewards.mean() > 1.1 * self.env._env.spec.reward_threshold:
                self.train = False
            if not self.train:
                print("Episode mean: %f" % episoderewards.mean())
                self.end_count += 1
                if self.end_count > 100:
                    break
            if self.train:
                self.vf.fit(paths)
                thprev = self.gf()

                def fisher_vector_product(p):
                    feed[self.flat_tangent] = p
                    return self.session.run(self.fvp, feed) + config.cg_damping * p

                g = self.session.run(self.pg, feed_dict=feed)
                stepdir = conjugate_gradient(fisher_vector_product, -g)
                shs = .5 * stepdir.dot(fisher_vector_product(stepdir))
                lm = np.sqrt(shs / config.max_kl)
                fullstep = stepdir / lm
                neggdotstepdir = -g.dot(stepdir)

                def loss(th):
                    self.sff(th)
                    return self.session.run(self.losses[0], feed_dict=feed)
                theta = linesearch(loss, thprev, fullstep, neggdotstepdir / lm)
                self.sff(theta)

                surrafter, kloldnew, entropy = self.session.run(
                    self.losses, feed_dict=feed)
                if kloldnew > 2.0 * config.max_kl:
                    self.sff(thprev)

                stats = {}

                numeptotal += len(episoderewards)
                stats["Total number of episodes"] = numeptotal
                stats["Average sum of rewards per episode"] = episoderewards.mean()
                stats["Entropy"] = entropy
                exp = explained_variance(np.array(baseline_n), np.array(returns_n))
                stats["Baseline explained"] = exp
                stats["Time elapsed"] = "%.2f mins" % ((time.time() - start_time) / 60.0)
                stats["KL between old and new distribution"] = kloldnew
                stats["Surrogate loss"] = surrafter
                for k, v in stats.iteritems():
                    print(k + ": " + " " * (40 - len(k)) + str(v))
                if entropy != entropy:
                    exit(-1)
                if exp > 0.8:
                    self.train = False
            i += 1

training_dir = tempfile.mkdtemp()
logging.getLogger().setLevel(logging.DEBUG)

if len(sys.argv) > 1:
    task = sys.argv[1]
else:
    task = "RepeatCopy-v0"

env = envs.make(task)
env.monitor.start(training_dir)

env = SpaceConversionEnv(env, Box, Discrete)

agent = TRPOAgent(env)
agent.learn()
env.monitor.close()
gym.upload(training_dir,
           algorithm_id='trpo_ff')


