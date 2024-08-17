import numpy as np

# Function for updating rolling average rewards
def update_rolling_average(episode, episode_rewards, average_rewards, solved_episode=None):
    if episode >= 100:
        avg_reward = sum(episode_rewards[episode-100:episode]) / 100
        average_rewards.append(avg_reward)
        if avg_reward >= 475 and solved_episode is None:
            solved_episode = episode
    else:
        average_rewards.append(sum(episode_rewards[:episode]) / 100)
    return solved_episode

# Print training metrics
def print_metrics(episode_rewards, solved_episode, training_time):
    print("---------------------------------------------------------------------------------------------------")
    print("Metrics")
    print("---------------------------------------------------------------------------------------------------")
    print(f"avg_reward: {np.mean(episode_rewards)} - avg_reward_last_100: {np.mean(episode_rewards[-100:])} - solved_at_ep: {solved_episode} - training_time: {training_time:.2f} sec")
    print("---------------------------------------------------------------------------------------------------")

# Print robustness metrics
def print_robustness_metrics(variants, rewards_data):
    print("---------------------------------------------------------------------------------------------------")
    print("Metrics")
    print("---------------------------------------------------------------------------------------------------")
    for i in range(len(variants)):
        print(f"{variants[i][0]}: mean: {np.mean(rewards_data[i])} - std: {np.std(rewards_data[i])}")
    print("---------------------------------------------------------------------------------------------------")
