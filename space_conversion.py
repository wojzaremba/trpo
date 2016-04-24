"""
`SpaceConversionEnv` acts as a wrapper on
any environment. It allows to convert some action spaces, and observation spaces to others.
"""

import numpy as np
from gym.spaces import Discrete, Box, Tuple
from gym import Env


def box2box4obj(x, old_space_obj, new_space_obj):
    assert(old_space_obj.contains(x))
    action = np.reshape(x, new_space_obj.shape)
    assert(new_space_obj.contains(action))
    return action

def box2box4class(box_space):
    shape = np.prod(box_space.shape)
    low = box_space.low
    high = box_space.high
    if isinstance(low, np.ndarray):
        low = np.reshape(low, (shape, ))
    if isinstance(high, np.ndarray):
        high = np.reshape(high, (shape, ))
    return Box(low, high)

def discrete2tuple4obj(x, discrete_space, tuple_space):
    assert(discrete_space.contains(x))
    action = []
    for space in tuple_space.spaces:
        assert(isinstance(space, Discrete))
        action.append(x % space.n)
        x = int(x / space.n)
    action = tuple(action)
    assert(tuple_space.contains(action))
    return action

def tuple2discrete4obj(x, old_space_obj, new_space_obj):
    assert(False)

def tuple2discrete4class(tuple_space):
    n = 1
    for space in tuple_space.spaces:
        assert(isinstance(space, Discrete))
        n *= space.n
    return Discrete(n)

def box2discrete4obj(x, box_space, discrete_space):
    assert(False)

def discrete2box4obj(x, discrete_space, box_space):
    ret = np.zeros(discrete_space.n)
    ret[x] = 1.0
    return ret

def discrete2box4class(discrete_space):
    return Box(0.0, 1.0, discrete_space.n)

def ident4obj(x, old_space_obj, new_space_obj):
    return x

class SpaceConversionEnv(Env):
    convertable = {(Tuple, Discrete): (tuple2discrete4obj, discrete2tuple4obj, tuple2discrete4class), \
                   (Discrete, Box): (discrete2box4obj, box2discrete4obj, discrete2box4class), \
                   (Box, Box): (box2box4obj, box2box4obj, box2box4class)}
    
    def __init__(self, env, target_observation_space=None, target_action_space=None, verbose=False):
        self._verbose = verbose
        self._env = env
        self.action_convert = None
        self.observation_convert = None
        for pairs, convert in self.convertable.iteritems():
            if env.action_space.__class__ == pairs[0] and \
               target_action_space == pairs[1] and \
               self.action_convert is None:
                self.action_convert = convert[1]
                self._action_space_ = convert[2](env.action_space)
            if env.observation_space.__class__ == pairs[0] and \
               target_observation_space == pairs[1] and \
               self.observation_convert is None:
                self.observation_convert = convert[0]
                self._observation_space_ = convert[2](env.observation_space)

        if self.action_convert is None and \
           (self.action_space.__class__ == target_action_space or 
             target_action_space is None):
            self.action_convert = ident4obj
            self._action_space = env.action_space
        if self.observation_convert is None and \
           (self.observation_space.__class__ == target_observation_space or \
           target_observation_space is None):
            self.observation_convert = ident4obj
            self._observation_space = env.observation_space

        assert(self.action_convert is not None)
        assert(self.observation_convert is not None)

    def step(self, action, **kwargs):
        conv_action = self.action_convert(action, self.action_space, self._env.action_space)
        if self._verbose and self.action_convert != ident4obj:
            print("Input action: %s, converted action: %s" % (action, conv_action))
        step = self._env.step(conv_action, **kwargs)
        observation, reward, done, info = step

        conv_observation = self.observation_convert(observation, self._env.observation_space, self.observation_space)  

        if self._verbose and self.observation_convert != ident4obj:
            print("Input observation: %s, converted observation: %s" % (observation, conv_observation))
        return conv_observation, reward, done, {}

    def reset(self, **kwargs):
        observation = self._env.reset(**kwargs)
        conv_observation = self.observation_convert(observation, self._env.observation_space, self.observation_space)

        if self._verbose and self.observation_convert != ident4obj:
            print("Input observation: %s, converted observation: %s" % (observation, conv_observation))
        return conv_observation
  
    @property
    def action_space(self):
        return self._action_space_

    @property
    def observation_space(self):
        return self._observation_space_

    def __getattr__(self, field):
        """
        proxy everything to underlying env
        """
        if hasattr(self._env, field):
            return getattr(self._env, field)
        raise AttributeError(field)
  
    def __repr__(self):
        if "object at" not in str(self._env):
            env_name = str(env._env)
        else:
            env_name = self._env.__class__.__name__
        return env_name
