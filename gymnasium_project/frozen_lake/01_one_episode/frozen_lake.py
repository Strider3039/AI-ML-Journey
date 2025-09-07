import gymnasium as gym
import numpy as np


def epsilon_greedy_action(Q, state, n_actions, eps, rng):
    # return random action value if random number < esp
    # otherwise return the action value with largest reward value
    if rng.random() < eps:
        return int(rng.integers(n_actions))
    return int(np.argmax(Q[state])) 	# find the largest action value within the state row of Q table

def main():

    # set q learning function values
    alpha = .9		# learning rate (how big a nudge toward a direction each step)
    gamma = .95		# discount (how much future rewards matter compared to immediate ones)
    eps = .9		# exploration vs exploitation (how likely to try something new vs something known to work well)

    # create the environment (8x8 Grid; Slippery Ice; Show Window)
    env = gym.make('FrozenLake-v1', map_name='8x8', is_slippery=True, render_mode='human')
    # env.metadata['render_fps'] = 15
    
    rng = np.random.default_rng(123)	# random number generator
    
    nS = env.observation_space.n
    nA = env.action_space.n
    Q = np.zeros((nS, nA), dtype=np.float32)	# generate Q table
    

    # reset state to the starting state
    # in frozen lake state is a grid number (0-63)
    # info for debugging
    state, info = env.reset(seed=123) 	# reset env and set seed for reproducibility
    
	# choose action using epsilon greedy, then print the Q value for that state and action
    action = epsilon_greedy_action(Q, state, nA, eps, rng)
    print(f"Before: Q[{state}, {action}] = {Q[state, action]:.3f}")

    # take the action, return next states and signals
    next_state, reward, terminated, truncated, _ = env.step(action)

    terminated = False          # true if you reached hole or end
    truncated = False           # true if you exceeded max actions (200 default)

    # get the best action value for the next state
    best_next = np.max(Q[next_state])

    # set best_next to 0 if terminated or truncated
    if terminated or truncated:
        best_next = 0.0

    # implement Q learning formula to update Q table
    Q[state, action] = Q[state, action] + alpha * (reward + gamma * best_next - Q[state, action])

    print(f"After:  Q[{state}, {action}] = {Q[state, action]:.3f}")
    print("reward:", reward, "| terminated:", terminated, "| truncated:", truncated)

if __name__ == "__main__":
    main()