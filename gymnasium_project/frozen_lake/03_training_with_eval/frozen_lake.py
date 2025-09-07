import gymnasium as gym
import numpy as np
from collections import deque
import os


def epsilon_greedy_action(Q, state, n_actions, eps, rng):
    # return random action value if random number < esp
    # otherwise return the action value with largest reward value
    if rng.random() < eps:
        return int(rng.integers(n_actions))
    row = Q[state] 
    best = np.flatnonzero(row == row.max()) 
    return int(rng.choice(best)) 	# find the largest action value within the state row of Q table

def run_one_episode(Q, env, alpha, gamma, eps, rng, max_steps=200, reset_seed=None):

    state, info = env.reset(seed=reset_seed) 	# reset the state of the env to the starting state
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

def evaluate_greedy(Q, episodes=10, max_steps=200, map_name='8x8', is_slippery=True, seed=999):
    # Evaluate the Q table by running episodes with greedy actoins only
    env = gym.make('FrozenLake-v1', map_name=map_name, is_slippery=is_slippery, render_mode='human')
    env.metadata['render_fps'] = 15
    env.action_space.seed(seed)      # set the seed for action space for reproducibility

    wins = 0
    for ep in range(episodes):
        state, info = env.reset(seed=seed + ep) 	# reset the state of the env to the starting state
        total_reward = 0.0          # total reward for the episode

        for step in range(max_steps):
            action = int(np.argmax(Q[state])) 	# choose the best action (greedy)
            next_state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward     # update the total reward
            if terminated or truncated:
                break

        wins += int(total_reward > 0)   # count the win if total_reward > 0
        print(f"[EVAL] Episode {ep}: reward={total_reward}")

    env.close()         # close window
    print(f"Greedy success rate over {episodes} episodes: {wins / episodes:.2%}")



if __name__ == "__main__":
    train_env = gym.make('FrozenLake-v1', map_name='8x8', is_slippery=True, render_mode=None)
    train_env.action_space.seed(0)      # set the seed for action space for reproducibility
    

    rng = np.random.default_rng(0)	    # random number generator

    Q = np.full((train_env.observation_space.n, train_env.action_space.n), .5, dtype=np.float32)	# generate Q table

    # hyperparameters
    alpha = .9		        # learning rate (how big a nudge toward a direction each step)
    gamma = .95		        # discount (how much future rewards matter compared to immediate ones)
    eps = .9		        #  exploration vs exploitation (how likely to try something new vs something known to work well)
    eps_end = .05  			# final value of eps
    eps_decay = 0.999    	# decay rate per episode

    episodes = 20000         # total number of episodes to train on
    rewards = deque(maxlen=100)   # last 100 rewards

    for ep in range(episodes):
        # vary reset seed to get different starting positions
        reward = run_one_episode(Q, train_env, alpha, gamma, eps, rng, reset_seed=10_000 + ep)

        rewards.append(reward)     # save the most recent reward

        # decay eps after each episode until it reaches eps_end
        eps = max(eps_end, eps * eps_decay)

        # print a progress report every 500 episodes
        if (ep + 1) % 1000 == 0:
            print(f"Ep {ep + 1:5d} | success@100={np.mean(rewards):.2%} | eps={eps:.3f}")


    print(f"Traning complete. Last-100 success:", f"{np.mean(rewards):.2%}")

    current_path = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(current_path, "frozenlake_q.npy")

    # Save Q for reuse (optional)
    np.save(output_path, Q)
    # Close training env
    train_env.close()

    # --- Evaluate with rendering (greedy policy) --
    evaluate_greedy(Q)