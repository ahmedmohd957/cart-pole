from reinforce import REINFORCE

learning_rates = [0.01, 0.005, 0.001]
discount_factors = [0.95, 0.99, 0.999]

best_performance = float('-inf')
best_config = None

# Loop over all combinations of hyperparameters
for lr in learning_rates:
    for df in discount_factors:
        config = {
            'learning_rate': lr,
            'discount_factor': df,
            'seed': 543,
            'n_episodes': 1000,
            'n_max_steps': 500,
            'log_interval': 10,
        }

        # Initialize the RL agent with the current configuration
        agent = REINFORCE(config)
        performance = agent.train()

        # Update best performing configuration
        if performance > best_performance:
            best_performance = performance
            best_config = config

# Output the best configuration and its performance
print("Best Configuration:", best_config)
print("Best Performance:", best_performance)