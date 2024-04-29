import gymnasium as gym

# The CartPole (version 1) is a very simple environment composed of a cart 
# that can move left or right, and pole placed vertically on top of it. 
# The agent must move the cart left or right to keep the pole upright.
env = gym.make("CartPole-v1", render_mode='human')

# Let's initialize the environment by calling is reset() method. 
# This returns an observation, as well as a dictionary that may contain extra information. 
# Both are environment-specific. In the CartPole environment, this dictionary is empty.
obs, info = env.reset(seed=42)

for _ in range(100):
    # The agent will need to select an action from an "action space" (the set of possible actions).
    # Two possible actions: accelerate towards the left (0) or towards the right (1).
    action = env.action_space.sample()

    # The step() method executes the desired action and returns five values:
    # obs: This is the new observation.
    # reward: tells the agent how much reward it got during the last step
    # done: When the game is over, the environment returns done=True
    # trunacted: This value will be True when an episode is interrupted early
    # info: This environment-specific dictionary may provide extra information, just like the one returned by the reset() method.
    obs, reward, done, truncated, info = env.step(action)

    # The sequence of steps between the moment the environment is reset until 
    # it is done or truncated is called an "episode". 
    # At the end of an episode (i.e., when step() returns done=True or truncated=True), 
    # you should reset the environment before you continue to use it.
    if done or truncated:
        obs, info = env.reset()
        print(obs)

# Once you have finished using an environment, you should call close() method to free resources.
env.close()
