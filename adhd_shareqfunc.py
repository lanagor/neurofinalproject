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
                    'adhd_temp': 20,
                    'neurotypical_temp': .01,
                    'num_trial': 400
                    }

# Set-up for each experimental condition
state_1 = {'A': (0.7, 10), 'B': (0.5, 20), 'C': (0.1, 300)}
state_2 = {'C': (0.7, 10), 'B': (0.5, 20), 'A': (0.1, 300)}
num_trials = 10

def plot_vals(adhd_vals, neurtypical_vals): 
    plt.scatter(adhd_vals)
     
    plt.xlabel('Trial Number')
    plt.ylabel('Errors to Criterion')
    plt.title('Reversal Learning Performance')
    plt.xticks(index + 0.5 * bar_width, ('Before Reversal', 'After Reversal'))
    plt.legend()
     
    plt.tight_layout()
    plt.show()


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
    actions = list(states.keys())
    for action_index in range(len(q_values)):
        weighted_q_values[action_index] = q_learning_params['beta'] * q_values[actions[action_index]]
    probabilities = softmax(weighted_q_values, tau)
    # Choose an action based on the probability of choosing 'right' / 'left'
    return np.random.choice(actions, p=probabilities)

def q_update(state, reward, q_values, q_learning_params, action):
    '''Given the most recently taken action (and the resulting reward), update the stored q values.'''
    q_values[action] += q_learning_params['alpha'] * (reward - q_values[action])
    return q_values, reward

def run_trial(state_1, state_2, q_learning_params, num_neurtypical, num_adhd, num_trial):
    q_values = {'A': 0, 'B': 0, 'C': 0}

    num_trials = q_learning_params['num_trial']
    total_rewards = [0]*num_trials

    choices = [0] * num_trial
    total_rewards = [0] * num_trial
    states = state_1
    for i in range(num_trial):
        # if i  % 100 == 0:
        #     if states == state_1:
        #         states = state_2
        #     else:
        #         states = state_1
        for j in range(num_adhd):
            action = select_action(q_learning_params, q_values, q_learning_params['adhd_temp'], states)
            p, r = states[action][0], states[action][1]
            reward = np.random.choice([0, r], p=[1 - p, p])        
            q_values, received_reward = q_update(states, reward, q_values, q_learning_params, action)
            total_rewards[i] += received_reward

        for j in range(num_neurtypical):
            action = select_action(q_learning_params, q_values, q_learning_params['neurotypical_temp'], states)
            p, r = states[action][0], states[action][1]
            reward = np.random.choice([0, r], p=[1 - p, p])        
            q_values, received_reward = q_update(states, reward, q_values, q_learning_params, action)
            total_rewards[i] += received_reward

        total_rewards[i] = total_rewards[i]/(num_adhd+num_neurtypical)
    return total_rewards

total_rewards = run_trial(state_1, state_2, q_learning_params, 100, 10, q_learning_params['num_trial'])
for i in range(q_learning_params['num_trial']):
    plt.scatter(i, total_rewards[i], c="blue", s=5)

plt.ylim(0, 20)
plt.xlim(0, q_learning_params['num_trial'])
plt.xlabel("trial")
plt.ylabel("reward from action")
plt.title("mixed population")
print(sum(total_rewards) / float(len(total_rewards)))
plt.show()


#6.993636363636364 neurotypical
#7.4759090909090835 mixed


