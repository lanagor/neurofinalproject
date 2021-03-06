# Neuro 1401
# Spring 2019
# Group 6: Project 2

"""Note that in the original paper they average the results from five reversals. For simplicity, this script only executes 
one reversal, but (with some added variance) the results are the same as those from the paper. 

The plot produced at the end of the script replicates Figure 1 from the Wilson paper. 
"""

import numpy as np
import matplotlib.pyplot as plt 

# Define q-learning parameters 

q_learning_params = {
                    'alpha': 0.03, 
                    'beta': 3, 
                    'adhd_temp': 0.8,
                    'neurotypical_temp': 0.2,
                    'num_trial': 400
                    }

def reward_dict(p, r , v):
    return {'p': p, 'v': v, 'r': r}

# Set-up for each experimental condition
state_1 = {'A': reward_dict(0.7, 1, 0), 'B': reward_dict(0.25, 1, 0), 'C': reward_dict(0.05, 7, 0)}
state_2 = {'A': reward_dict(0.7, 4, 2), 'B': reward_dict(0.25, 4, 2)}

def plot_vals(adhd_vals, neurtypical_vals): 
    plt.scatter(adhd_vals)
     
    plt.xlabel('Trial Number')
    plt.ylabel('Errors to Criterion')
    plt.title('Reversal Learning Performance')
    plt.xticks(index + 0.5 * bar_width, ('Before Reversal', 'After Reversal'))
    plt.legend()
     
    plt.tight_layout()
    plt.show()

def get_reward(action, states): 
    p, r, v = states[action]['p'], states[action]['r'], states[action]['v']
    is_reward = np.random.binomial(1, p)
    reward_magnitude = np.random.normal(r, v)
    return max(0, is_reward * reward_magnitude)

def softmax(q_vals, tau):
    probs = [0] * len(q_vals)
    total_prob = sum(np.exp(q_vals / tau))
    for i in range(len(q_vals)):
        probs[i] = (np.exp(q_vals[i] / tau)) /  total_prob
    return probs

def select_action(q_learning_params, q_values, tau, states):
    '''Select an action given the current state and using the Luce rule.'''
    # Compute the probability of choosing 'right'
    weighted_q_values = np.zeros(len(states))
    actions = states.keys()
    for action_index in range(len(q_values)):
        weighted_q_values[action_index] = q_learning_params['beta'] * q_values[actions[action_index]]
    probs = softmax(weighted_q_values, tau)
    # Choose an action based on the probability of choosing 'right' / 'left'
    return np.random.choice(actions, p=probs)

def q_update(state, reward, q_values, q_learning_params, action):
    '''Given the most recently taken action (and the resulting reward), update the stored q values.'''
    q_values[action] += q_learning_params['alpha'] * (reward - q_values[action])
    return q_values 

def run_trial(states, q_learning_params, tau, num_trial):
    q_values = {key: 0 for key in states.keys()}
    choices = [0] * num_trial
    rewards = [0] * num_trial
    for i in range(num_trial): 
        action = select_action(q_learning_params, q_values, tau, states)
        reward = get_reward(action, states)
        choices[i] = action
        rewards[i] = reward
        q_values = q_update(states, reward, q_values, q_learning_params, action)
    return choices, rewards

def single_agent_trial(states, q_learning_params):
    adhd_choices, adhd_rewards = run_trial(states, q_learning_params, q_learning_params['adhd_temp'], q_learning_params['num_trial'])
    neurotypical_chioces, neurotypical_rewards = run_trial(states, q_learning_params, q_learning_params['neurotypical_temp'], q_learning_params['num_trial'])
    print(sum(adhd_rewards), sum(neurotypical_rewards))

if __name__ == "__main__":
    single_agent_trial(state_1, q_learning_params)






