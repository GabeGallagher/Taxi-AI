import gym
import numpy as np
import random
import math
import matplotlib.pyplot as plt

# Environment constants
GAME = gym.make("Taxi-v2")
STATE_SPACE = GAME.observation_space.n
ACTION_SPACE = GAME.action_space.n
MAX_STEPS = GAME.spec.timestep_limit

# Hyperparams
NUM_EPISODES = 50000
GAMMA = 0.95
ALPHA_START = 0.1
ALPHA_TAPER = 0.01
EPSILON_START = 1.0
EPSILON_TAPER = 0.01


# Get action based on epsilon greedy
def get_action(state, e=EPSILON_START):
    # if we decide to explore or are new to this state, take a random action
    if random.random() < e or math.isclose(max(q_table[state]), 0, rel_tol=1e-9):
        return np.random.choice(q_table[state])

    return np.argmax(q_table[state])


def get_alpha():
    return ALPHA_START


def update_q(state, action, a, reward, new_state):
    q_table[state][action] += a * (reward + GAMMA * max(q_table[new_state]) - q_table[state][action])


def update_memory(state):
    if state not in count_table:
        count_table[state] = 1
    else:
        count_table[state] += 1


def play():
    state = GAME.reset()
    is_done = False
    total_reward = 0
    moves = 0

    while not is_done and moves < MAX_STEPS:
        action = get_action(state)
        new_state, is_done, reward, prob = GAME.step(action)
        update_q(state, action, get_alpha(), reward, new_state)
        update_memory(state)
        total_reward += reward
        moves += 1


if __name__ == "__main__":
    q_table = np.zeros((STATE_SPACE, ACTION_SPACE))
    delta_array = []
    count_table = {}

    # Play game
    play()

    # Record results

    # update state values

    # Repeat until convergence
