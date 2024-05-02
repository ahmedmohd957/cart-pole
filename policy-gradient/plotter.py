import matplotlib.pyplot as plt

def plot_graph_1(title, xlabel, ylabel, rewards, mean_rewards, episodes, file_name):
    plt.figure(figsize=(10, 6))
    plt.plot(rewards, label='Reward')
    plt.plot(mean_rewards, label='Average Reward', color='orange', linewidth=2)
    plt.axhline(y=475, color='r', linestyle='--', label='475 Steps')
    plt.axvline(x=episodes, color='g', linestyle='--', label=f'{episodes} Episode')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc='upper left')
    plt.savefig(f'{file_name}.png')
    plt.show()

def plot_graph_2(title, xlabel, ylabel, rewards, file_name):
    plt.figure(figsize=(10, 6))
    plt.plot(rewards, linewidth=1)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(f'{file_name}.png')
    plt.show()

def plot_graph_3(title, xlabel, ylabel, rewards, file_name):
    plt.figure(figsize=(8, 4))
    plt.plot(rewards)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.savefig(f'{file_name}.png')
    plt.show()