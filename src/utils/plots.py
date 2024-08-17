import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def plot_training_result(algorithm, episodes, episode_rewards, average_reward):
    df = pd.DataFrame({
        'Episode': range(episodes),
        'Reward': episode_rewards,
        'Average Reward': average_reward
    })
    sns.set(style="darkgrid")
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df, x='Episode', y='Reward', label='Reward')
    sns.lineplot(data=df, x='Episode', y='Average Reward', label='Average Reward (over 100 ep.)')
    plt.axhline(y=475, color='r', linestyle='--', label='475 Steps')
    plt.title(f'{algorithm} Training Result', fontsize=16)
    plt.xlabel('Episode', fontsize=14)
    plt.ylabel('Reward', fontsize=14)
    plt.legend(loc='upper left')
    plt.savefig(f'{algorithm}.png')

def plot_robustness_performance(algorithm, variants, rewards_data) -> None:
    data = []
    for variant, variant_rewards in zip(variants, rewards_data):
        for seed_index, seed_rewards in enumerate(variant_rewards):
            episodes = np.arange(1, len(seed_rewards) + 1)
            df = pd.DataFrame({
                'Episode': episodes,
                'Average Reward': seed_rewards,
                'Variant': variant,
                'Seed': seed_index + 1
            })
            data.append(df)
    df_combined = pd.concat(data, ignore_index=True)
    plt.figure(figsize=(12, 6))
    sns.set_theme(style="darkgrid", context="talk", palette="rainbow")
    ax = sns.lineplot(x='Episode', y='Average Reward', hue='Variant', data=df_combined)
    ax.set_title(f"{algorithm} Robustness Performance", fontsize=16)
    ax.set_xlabel("Episode", fontsize=14)
    ax.set_ylabel("Average Reward", fontsize=14)
    plt.ylim(bottom=0)
    plt.legend(loc='upper right', fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=12)
    plt.savefig(f'{algorithm}-robustness.png')
