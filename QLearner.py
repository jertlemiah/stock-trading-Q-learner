import time

"""
Template for implementing QLearner  (c) 2015 Tucker Balch

Copyright 2018, Georgia Institute of Technology (Georgia Tech)
Atlanta, Georgia 30332
All Rights Reserved

Template code for CS 4646/7646

Georgia Tech asserts copyright ownership of this template and all derivative
works, including solutions to the projects assigned in this course. Students
and other users of this template code are advised not to share it with others
or to make it available on publicly viewable websites including repositories
such as github and gitlab.  This copyright statement should not be removed
or edited.

We do grant permission to share solutions privately with non-students such
as potential employers. However, sharing with other current or future
students of CS 7646 is prohibited and subject to being investigated as a
GT honor code violation.

-----do not edit anything above this line---

Student Name: Jeremiah Plauche
GT User ID: jplauch3
GT ID: 903051398
"""

import random as rand

import numpy as np


class QLearner(object):
    """
    This is a Q learner object.
    :param num_states: The number of states to consider.
    :type num_states: int
    :param num_actions: The number of actions available.
    :type num_actions: int
    :param alpha: The learning rate used in the update rule. Should range between 0.0 and 1.0 with 0.2 as a typical value.
    :type alpha: float
    :param gamma: The discount rate used in the update rule. Should range between 0.0 and 1.0 with 0.9 as a typical value.
    :type gamma: float
    :param rar: Random action rate: the probability of selecting a random action at each step. Should range between 0.0 (no random actions) to 1.0 (always random action) with 0.5 as a typical value.
    :type rar: float
    :param radr: Random action decay rate, after each update, rar = rar * radr. Ranges between 0.0 (immediate decay to 0) and 1.0 (no decay). Typically 0.99.
    :type radr: float
    :param dyna: The number of dyna updates for each regular update. When Dyna is used, 200 is a typical value.
    :type dyna: int
    :param verbose: If “verbose” is True, your code can print out information for debugging.
    :type verbose: bool
    """

    def __init__(
            self,
            num_states=100,
            num_actions=4,
            alpha=0.2,
            gamma=0.9,
            rar=0.5,
            radr=0.99,
            dyna=0,
            verbose=False,
    ):
        """
        Constructor method
        """
        # np.random.seed() # unfortunately I can't query the rand.seed being used

        self.num_states = num_states
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.rand_act_rate = rar
        self.rand_act_decay_rate = radr
        self.dyna = dyna
        self.verbose = verbose

        self.s = 0
        self.a = 0

        self.Q = np.zeros([num_states, num_actions])
        self.Q = np.zeros((num_states, num_actions))

        self.T_count = np.full(shape=(num_states, num_actions, num_states), fill_value=0.00001)
        self.T = np.zeros(shape=(num_states, num_actions, num_states))
        self.R = np.zeros((num_states, num_actions))
        self.observed_states = np.zeros(shape=(num_states, num_actions))

    def author(self):
        """
        :return: The GT username of the student
        :rtype: str
        """
        return "jplauch3"

    def querysetstate(self, s):
        """
        Update the state without updating the Q-table

        :param s: The new state
        :type s: int
        :return: The selected action
        :rtype: int
        """
        self.s = s
        self.a = self.get_possibly_rand_action(s)
        if self.verbose:
            print(f"s = {s}, a = {self.a}")
        return self.a

    def update_q_table(self, s, a, s_prime, r):
        new_estimate = r + self.gamma * self.Q[s_prime, np.argmax(self.Q[s_prime])]

        self.Q[s, a] = (1 - self.alpha) * self.Q[s, a] + self.alpha * new_estimate

    def apply_dyna(self):
        # Precompute as much as possible to reduce the cost of subsequent calls
        # For some reason this doesn't give good enough answers
        # rand_vals = np.full(shape=(self.dyna, 2), fill_value=0)
        # rand_vals[:, 0] = np.random.randint(0, self.num_states, size=self.dyna)
        # rand_vals[:, 1] = np.random.randint(0, self.num_actions, size=self.dyna)
        # func =
        # rand_vals[:, 2] =
        # np.apply_along_axis(lambda row: np.random.choice(1, self.T[row[0], row[1], :]), 1, rand_vals[:, 2])
        # for (state, action) in rand_vals:
            # state_prime = np.random.choice(1, self.T[state, action, :])

        for _ in range(self.dyna):
            # state = rand.randint(0, self.num_states - 1)
            # action = rand.randint(0, self.num_actions - 1)
            # if rand.uniform(0.0, 1.0) < self.rand_act_rate:
            #     state = rand.randint(0, self.num_states - 1)
            #     action = rand.randint(0, self.num_actions - 1)
            # else:
            #     nonzero = np.nonzero(self.observed_states)
            #     choice = rand.randint(0, len(nonzero[0]) - 1)
            #     state = nonzero[0][choice]
            #     action = nonzero[1][choice]

            nonzero = np.nonzero(self.observed_states)
            choice = rand.randint(0, len(nonzero[0]) - 1)
            state = nonzero[0][choice]
            action = nonzero[1][choice]

            state_prime = np.argmax(self.T[state, action, :])

            # this is the best middle ground I could manage
            # The true solution (else) is 10x slower than arg max
            # if rand.uniform(0.0, 1.0) > self.rand_act_rate / 2:
            #     state_prime = np.argmax(self.T[state, action, :])
            # else:
            #     state_prime = np.random.choice(a=np.arange(0, self.T[state, action, :].shape[0], dtype=int),
            #                                    p=self.T[state, action, :] / sum(self.T[state, action, :]))


            # CURRENT CORRECT SOLUTION
            # state_prime = np.random.choice(a=np.arange(0, self.T[state, action, :].shape[0], dtype=int),
            #                                p=self.T[state, action, :]/sum(self.T[state, action, :]))


            # toc = time.perf_counter()
            # print(f"non-Slice took {(toc - tic) * 10000:0.4f} micro seconds, choice {state_prime}")
            # toc = time.perf_counter()
            # print(f"true took {(toc - tic)*10000:0.4f} micro seconds, choice {state_prime}")
            # state_prime = rand.choices(population=range(self.num_states), weights=self.T[state, action, :])
            reward = self.R[state, action]

            self.update_q_table(s=state, a=action, s_prime=state_prime, r=reward)

        return

    def query(self, s_prime, r):
        """
        Update the Q table and return an action

        :param s_prime: The new state
        :type s_prime: int
        :param r: The immediate reward
        :type r: float
        :return: The selected action
        :rtype: int
        """
        action = self.get_possibly_rand_action(s_prime)

        self.update_q_table(self.s, self.a, s_prime, r)

        if self.dyna > 0:
            self.observed_states[self.s, self.a] += 1
            self.T_count[self.s, self.a, s_prime] += 1
            self.R[self.s, self.a] += self.alpha * (r - self.R[self.s, self.a])
            self.T = np.round(self.T_count) / np.sum(self.T_count, axis=0)
            self.apply_dyna()

        self.rand_act_rate *= self.rand_act_decay_rate
        self.s = s_prime
        self.a = action

        if self.verbose:
            print(f"s = {s_prime}, a = {action}, r={r}")
        return action

    def get_possibly_rand_action(self, state):
        if rand.uniform(0.0, 1.0) <= self.rand_act_rate:
            return rand.randint(0, self.num_actions - 1)
        else:
            return np.argmax(self.Q[state, :])

if __name__ == "__main__":
    print("Remember Q from Star Trek? Well, this isn't him")
