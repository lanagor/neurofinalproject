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
                    'adhd_temp': 8,
                    'neurotypical_temp': 1,
                    'num_trial': 100
                    }

# Set-up for each experimental condition
state_1 = {'A': (0.7, 1), 'B': (0.25, 1), 'C': (0.05, 7)}

def plot_vals(adhd_vals, neurtypical_vals): 
    plt.scatter(adhd_vals)
     
    plt.xlabel('Trial Number')
    plt.ylabel('Errors to Criterion')
    plt.title('Reversal Learning Performance')
    plt.xticks(index + 0.5 * bar_width, ('Before Reversal', 'After Reversal'))
    plt.legend()
     
    plt.tight_layout()
    plt.show()


def softmax(q_vals, tau)
    probs = [0] * len(q_vals)
    total_prob = sum(np.exp(q_vals / tau))
    for i in len(q_vals):
        probs[i] = (np.exp(q_vals[i] / tau)) /  total_prob
    return probs

def select_action(q_learning_params, q_values, tau, states):
    '''Select an action given the current state and using the Luce rule.'''
    # Compute the probability of choosing 'right'
    weighted_q_values = np.zeros(len(states))
    actions = states.keys()
    for action_index in range(len(q_vals)):
        weighted_q_values[action_index] = q_learning_params['beta'] * q_values[actions[action_index]]
    probabilities = softmax(weighted_q_values, tau)
    # Choose an action based on the probability of choosing 'right' / 'left'
    return np.random.choice(actions, p=probs)

def q_update(state, q_values, q_learning_params, action):
    '''Given the most recently taken action (and the resulting reward), update the stored q values.'''
    q_values[action] += q_learning_params['alpha'] * (state[action][1] - q_values[action])
    return q_values 

def run_trial(states, q_learning_params, tau, num_trial):
    q_values = {'A': 0, 'B': 0, 'C': 0}
    choices = [0] * num_trial
    for i in range(num_trial): 
        action = select_action(q_learning_params, q_values, tau, states)
        p, r = states[action][0], states[action][1]
        reward = np.random.choice([0, r], p=prob)
        choices[i] = reward
        q_vals = q_update(state, q_values, q_learning_params, action)
    return choices

def single_agent_trial(states, q_learning_params):
    adhd_trials = run_trial(states, q_learning_params, q_learning_params['adhd_temp', q_learning_params['num_trial']])
    neurtypical_trials = run_trial(states, q_learning_params, q_learning_params['neurotypical_temp'], q_learning_params['num_trial'])



if __name__ == "__main__":








