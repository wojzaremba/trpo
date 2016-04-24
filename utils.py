import numpy as np
import tensorflow as tf
import random
import scipy.signal

seed = 1
random.seed(seed)
np.random.seed(seed)
tf.set_random_seed(seed)

dtype = tf.float32

def discount(x, gamma):
    assert x.ndim >= 1
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

def rollout(env, agent, max_pathlength, n_timesteps):
    paths = []
    timesteps_sofar = 0
    while timesteps_sofar < n_timesteps:
        obs, actions, rewards, action_dists = [], [], [], []
        ob = env.reset()
        for _ in xrange(max_pathlength):
            action, action_dist, ob = agent.act(ob)
            obs.append(ob)
            actions.append(action)
            action_dists.append(action_dist)
            res = env.step(action)
            ob = res[0]
            rewards.append(res[1])
            if res[2]:
                path = {"obs": np.concatenate(np.expand_dims(obs, 0)),
                        "action_dists": np.concatenate(action_dists),
                        "rewards": np.array(rewards),
                        "actions": np.array(actions)}
                paths.append(path)
                agent.prev_action *= 0.0
                agent.prev_obs *= 0.0
                break
        timesteps_sofar += len(path["rewards"])
    return paths


class LinearVF(object):
    coeffs = None

    def _features(self, path):
        o = path["obs"].astype('float32')
        o = o.reshape(o.shape[0], -1)
        l = len(path["rewards"])
        al = np.arange(l).reshape(-1, 1) / 100.0
        return np.concatenate([o, o**2, al, al**2, np.ones((l, 1))], axis=1)

    def fit(self, paths):
        featmat = np.concatenate([self._features(path) for path in paths])
        returns = np.concatenate([path["returns"] for path in paths])
        n_col = featmat.shape[1]
        lamb = 2.0
        self.coeffs = np.linalg.lstsq(featmat.T.dot(featmat) + lamb * np.identity(n_col), featmat.T.dot(returns))[0]

    def predict(self, path):
        return np.zeros(len(path["rewards"])) if self.coeffs is None else self._features(
            path).dot(self.coeffs)


def cat_sample(prob_nk):
    assert prob_nk.ndim == 2
    N = prob_nk.shape[0]
    csprob_nk = np.cumsum(prob_nk, axis=1)
    out = np.zeros(N, dtype='i')
    for (n, csprob_k, r) in zip(xrange(N), csprob_nk, np.random.rand(N)):
        for (k, csprob) in enumerate(csprob_k):
            if csprob > r:
                out[n] = k
                break
    return out


def var_shape(x):
    out = [k.value for k in x.get_shape()]
    assert all(isinstance(a, int) for a in out), \
        "shape function assumes that shape is fully known"
    return out


def numel(x):
    return np.prod(var_shape(x))


def flatgrad(loss, var_list):
    grads = tf.gradients(loss, var_list)
    return tf.concat(0, [tf.reshape(grad, [numel(v)])
                         for (v, grad) in zip(var_list, grads)])


class SetFromFlat(object):

    def __init__(self, session, var_list):
        self.session = session
        assigns = []
        shapes = map(var_shape, var_list)
        total_size = sum(np.prod(shape) for shape in shapes)
        self.theta = theta = tf.placeholder(dtype, [total_size])
        start = 0
        assigns = []
        for (shape, v) in zip(shapes, var_list):
            size = np.prod(shape)
            assigns.append(
                tf.assign(
                    v,
                    tf.reshape(
                        theta[
                            start:start +
                            size],
                        shape)))
            start += size
        self.op = tf.group(*assigns)

    def __call__(self, theta):
        self.session.run(self.op, feed_dict={self.theta: theta})


class GetFlat(object):

    def __init__(self, session, var_list):
        self.session = session
        self.op = tf.concat(0, [tf.reshape(v, [numel(v)]) for v in var_list])

    def __call__(self):
        return self.op.eval(session=self.session)


def slice_2d(x, inds0, inds1):
    inds0 = tf.cast(inds0, tf.int64)
    inds1 = tf.cast(inds1, tf.int64)
    shape = tf.cast(tf.shape(x), tf.int64)
    ncols = shape[1]
    x_flat = tf.reshape(x, [-1])
    return tf.gather(x_flat, inds0 * ncols + inds1)


def linesearch(f, x, fullstep, expected_improve_rate):
    accept_ratio = .1
    max_backtracks = 10
    fval = f(x)
    for (_n_backtracks, stepfrac) in enumerate(.5**np.arange(max_backtracks)):
        xnew = x + stepfrac * fullstep
        newfval = f(xnew)
        actual_improve = fval - newfval
        expected_improve = expected_improve_rate * stepfrac
        ratio = actual_improve / expected_improve
        if ratio > accept_ratio and actual_improve > 0:
            return xnew
    return x


def conjugate_gradient(f_Ax, b, cg_iters=10, residual_tol=1e-10):
    p = b.copy()
    r = b.copy()
    x = np.zeros_like(b)
    rdotr = r.dot(r)
    for i in xrange(cg_iters):
        z = f_Ax(p)
        v = rdotr / p.dot(z)
        x += v * p
        r -= v * z
        newrdotr = r.dot(r)
        mu = newrdotr / rdotr
        p = r + mu * p
        rdotr = newrdotr
        if rdotr < residual_tol:
            break
    return x

class dict2(dict):
    def __init__(self, **kwargs):
        dict.__init__(self, kwargs)
        self.__dict__ = self


