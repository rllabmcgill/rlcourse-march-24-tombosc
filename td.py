from __future__ import division

import numpy as np
from collections import deque

from utils import oh_encode, epsilon_greedify

"""
TD methods in the tabular case
"""

class SarsaAgent(object):
    """
    a Sarsa agent in a discrete action space and discrete state space
    """
    def __init__(self, env_name, A, S, alpha, gamma, eps):
        # TODO: assert action space is discrete and obs space is discrete
        self.env_name = env_name
        self.A = A
        self.S = S
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        self._allocate()

    def _allocate(self):
        # Optimistic initialisation
        self.q = np.ones((self.S, self.A))*10
        self._s, self._a, self._r = deque(), deque(), deque()

    def _update_dequeue(self, l, x):
        #if len(l) > 1:
        #    l.popleft()
        l.append(x)
    
    def reset(self, s):
        self._s.clear()
        self._r.clear()
        self._a.clear()
        self._s.append(s)

    def update(self, a, s_p, r, done):
        # TODO: use done?
        self._update_dequeue(self._s, s_p)
        self._update_dequeue(self._a, a)
        self._update_dequeue(self._r, r)

        if len(self._a) < 2 and len(self._r) < 2:
            return

        # SARS'A' update
        s, s_p = self._s[-3], self._s[-2]
        a, a_p = self._a[-2], self._a[-1]
        r = self._r[-2]
        delta = (r + self.gamma * self.q[s_p, a_p] - self.q[s,a])
        self.q[s,a] += self.alpha * delta

    def _eps_greedy_step(self, p):
        uniform = np.random.choice(2, p=[self.eps, 1-self.eps])
        if uniform == 0:
            return np.random.choice(self.A)
        else:
            return np.argmax(p)
 
    def sample_action(self, s):
        return self._eps_greedy_step(self.q[s,:])

    def __getstate__(self):
        return {
                "A": self.A,
                "S": self.S,
                "q": self.q,
                "alpha": self.alpha,
                "gamma": self.gamma,
                "eps": self.eps,
                "env_name": self.env_name
               }

    def __setstate__(self, state):
        for k in state.keys():
            self.__dict__[k] = state[k]
        self._allocate()


class ExpectedSarsaAgent(SarsaAgent):
    """
    Expected Sarsa agent in a discrete action space and discrete state space
    """
    def update(self, a, s_p, r, done):
        self._update_dequeue(self._s, s_p)
        self._update_dequeue(self._a, a)
        self._update_dequeue(self._r, r)

        if len(self._a) < 2 and len(self._r) < 2:
            return

        s, s_p = self._s[-3], self._s[-2]
        a, a_p = self._a[-2], self._a[-1]
        r = self._r[-2]
        pi = self.q[s_p,:]
        dot_prod = np.dot(pi, epsilon_greedify(pi, self.eps))
        self.q[s,a] += self.alpha * (r + self.gamma * dot_prod - self.q[s,a])

class QAgent(SarsaAgent):
    """
    Q-learning agent in a discrete action space and discrete state space
    """
    def update(self, a, s_p, r, done):
        self._update_dequeue(self._s, s_p)
        self._update_dequeue(self._a, a)
        self._update_dequeue(self._r, r)

        if len(self._a) < 2 and len(self._r) < 2:
            return

        s, s_p = self._s[-3], self._s[-2]
        a, a_p = self._a[-2], self._a[-1]
        r = self._r[-2]
        q_max = max(self.q[s_p,:])
        self.q[s,a] += self.alpha * (r + self.gamma * q_max - self.q[s,a])


class SarsaLambdaAgent(SarsaAgent):
    """
    a Sarsa agent in a discrete action space and discrete state space
    """
    def __init__(self, env_name, A, S, alpha, gamma, eps, lambda_):
        super(SarsaLambdaAgent, self).__init__(env_name, A, S, alpha, gamma, eps)
        self.lambda_ = lambda_
        self._allocate()

    def _allocate(self):
        SarsaAgent._allocate(self)
        self.e = np.zeros((self.S, self.A))

    def reset(self, s):
        SarsaAgent.reset(self, s)
        self.e = np.zeros((self.S, self.A))

    def update(self, a, s_p, r, done):
        # TODO: use done?
        self._update_dequeue(self._s, s_p)
        self._update_dequeue(self._a, a)
        self._update_dequeue(self._r, r)

        if len(self._a) < 2 and len(self._r) < 2:
            return

        # SARS'A' update
        s, s_p = self._s[-3], self._s[-2]
        a, a_p = self._a[-2], self._a[-1]
        r = self._r[-2]
        g = self.gamma

        # compute eligibility
        mask = np.zeros((self.S, self.A))
        mask[s, a] = 1
        self.e = g * self.lambda_ * self.e + mask
        delta = (r + self.gamma * self.q[s_p, a_p] - self.q[s,a])
        self.q += self.alpha * delta * self.e

    def __getstate__(self):
        state = SarsaAgent.__getstate__(self)
        state["lambda_"] = self.lambda_
        return state

class HLSAgent(SarsaLambdaAgent):
    """
    a HLS(lambda) agent in a discrete action space and discrete state space
    """
    def __init__(self, env_name, A, S, gamma, eps, lambda_):
        super(HLSAgent, self).__init__(env_name, A, S, 0.1, gamma, eps, lambda_)
        self.lambda_ = lambda_
        self._allocate()

    def _allocate(self):
        SarsaLambdaAgent._allocate(self)
        self.N = np.ones((self.S, self.A))

    def reset(self, s):
        SarsaLambdaAgent.reset(self, s)
        self.e = np.zeros((self.S, self.A))
        self.N = np.ones((self.S, self.A))

    def update(self, a, s_p, r, done):
        self._update_dequeue(self._s, s_p)
        self._update_dequeue(self._a, a)
        self._update_dequeue(self._r, r)

        if len(self._a) < 2 and len(self._r) < 2:
            return

        # SARS'A' update
        s, s_p = self._s[-3], self._s[-2]
        a, a_p = self._a[-2], self._a[-1]
        r = self._r[-2]

        g = self.gamma
        l = self.lambda_

        self.e[s,a] += 1
        self.N[s,a] += 1

        delta = r + g * self.q[s_p,a_p] - self.q[s,a]
        beta = np.zeros((self.S, self.A))

        #except:
        #print "Err in delta up:"
        #print "r:", r
        #print "self.q[s_p, a_p]:", self.q[s_p, a_p]
        #print "self.q[s,a]:", self.q[s,a]
        K = (self.N[s_p,a_p]/(self.N[s_p,a_p] - g*self.e[s_p,a_p]))
        #ratio = self.N[s_p, a_p] / self.N
        #print "ratio (min, max mean)", np.min(ratio), np.max(ratio), np.mean(ratio)
        beta = K / self.N
        self.q = self.q + beta * self.e * delta
        self.e = self.e * l * g
        self.N = self.N * l
        #print "delta:", delta
        #print "s, a, r, s', a'", s, a, r, s_p, a_p
        #print "e", self.e
        #print "q:", self.q
        #print "N (min, max mean)", np.min(self.N), np.max(self.N), np.mean(self.N)
        #print "beta (min max mean):", np.min(beta), np.max(beta), np.mean(beta)
        #beta = np.ones((self.S, self.A)) * 0.5

    def __getstate__(self):
        state = SarsaAgent.__getstate__(self)
        del state["alpha"]
        return state
