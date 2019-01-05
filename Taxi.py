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
EPSILON_START = 0.1
EPSILON_TAPER = 0.01


# Get action based on epsilon greedy
def get_action(state, e=EPSILON_START):
    # if we decide to explore or are new to this state, take a random action
    #if random.random() < e or math.isclose(max(q_table[state]), 0, rel_tol=1e-9):
    if random.random() < e:
        return random.randint(0, len(q_table[state]) - 1)

    return np.argmax(q_table[state])


def get_alpha():
    return ALPHA_START


def update_q(state, action, a, reward, new_state):
    q_table[state][action] += a * (reward + GAMMA * max(q_table[new_state]) - q_table[state][action])


# def update_memory(state):
#     if state not in count_table:
#         count_table[state] = 1
#     else:
#         count_table[state] += 1


def learn():
    state = GAME.reset()
    is_done = False
    moves = 0
    did_win = False

    while not is_done and moves < MAX_STEPS:
        action = get_action(state)
        new_state, reward, is_done, prob = GAME.step(action)
        if reward == 20:
            did_win = True
        update_q(state, action, get_alpha(), reward, new_state)
        state = new_state
        moves += 1

    if did_win:
        return 1

    return 0


def play():
    success = False
    is_done = False
    moves = 0
    state = GAME.reset()

    while not is_done and moves < MAX_STEPS:
        new_state, reward, is_done, prob = GAME.step(get_action(state))
        if reward == 20:
            success = True
        state = new_state
        moves += 1

        if plays == 0:
            GAME.render()

    return success


if __name__ == "__main__":
    q_table = np.zeros((STATE_SPACE, ACTION_SPACE))
    wins = 0
    thousands = 1

    for episode in range(NUM_EPISODES):
        wins += learn()
        if (episode + 1) % (1000 * thousands) == 0:
            print("success rate: ", wins / (episode + 1))
            print("Total Wins: ", wins)
            print("Episodes: ", episode + 1)
            thousands += 1
            plays = 0
            passed = True
            # play the game ten times or to failure
            while plays < 10 and passed:
                passed = play()
                plays += 1

            if passed:
                print("Episodes to convergence: ", episode + 1)
                episode = NUM_EPISODES
