import gymnasium as gym
import numpy as np
from collections import deque


def epsilon_greedy_action(Q, state, n_actions, eps, rng):
    # return random action value if random number < esp
    # otherwise return the action value with largest reward value
    if rng.random() < eps:
        return int(rng.integers(n_actions))
    return int(np.argmax(Q[state])) 	# find the largest action value within the state row of Q table

def run_one_episode(Q, env, alpha, gamma, eps, rng, max_steps=200):

    state, info = env.reset() 	# reset the state of the env to the starting state
    total_reward = 0.0          # total reward for the episode

    for step in range(max_steps):
        # choose action using epsilon greedy
        action = epsilon_greedy_action(Q, state, env.action_space.n, eps, rng)
        # take the action, return next states and signals
        next_state, reward, terminated, truncated, _ = env.step(action)

        # get the best action value for the next state
        best_next = np.max(Q[next_state])

        # set best_next to 0 if terminated or truncated
        if terminated or truncated:
            best_next = 0.0

        # implement Q learning formula to update Q table
        Q[state, action] = Q[state, action] + alpha * (reward + gamma * best_next - Q[state, action])

        state = next_state         # move to the next state
        total_reward += reward     # update the total reward

        if terminated or truncated:
            break

    return total_reward            # if the reward is 1.0, you reached the goal, else 0.0


if __name__ == "__main__":
    env = gym.make('FrozenLake-v1', map_name='8x8', is_slippery=True, render_mode='human')
    env.metadata['render_fps'] = 3000

    rng = np.random.default_rng(0)	    # random number generator

    Q = np.zeros((env.observation_space.n, env.action_space.n), dtype=np.float32)	# generate Q table

    alpha = .9		        # learning rate (how big a nudge toward a direction each step)
    gamma = .95		        # discount (how much future rewards matter compared to immediate ones)
    eps = .9		        #  exploration vs exploitation (how likely to try something new vs something known to work well)
    eps_end = .05           # final value of eps
    eps_decay = 0.999       # decay rate per episode

    episodes = 8000         # total number of episodes to train on
    rewards = deque(maxlen=100)   # last 100 rewards

    for ep in range(episodes):
        reward = run_one_episode(Q, env, alpha, gamma, eps, rng)
        rewards.append(reward)     # save the most recent reward

        # decay eps after each episode until it reaches eps_end
        eps = max(eps_end, eps * eps_decay)

        # print a progress report every 500 episodes
        if (ep + 1) % 500 == 0:
            if ep % 500 == 0 or ep == 1: print(f"Ep {ep:5d} | success@100={np.mean(rewards):.2%} | eps={eps:.3f}")

    reward = run_one_episode(Q, env, alpha, gamma, eps, rng)
    print("reward:", reward)