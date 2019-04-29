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
                    }

# Set-up for each experimental condition
state = {}

def plot_reversal_errors(sham_initial_errors, sham_reversal_errors, OFC_initial_errors, OFC_reversal_errors):
    conditions = 2
    initial_errors = (int(sham_initial_errors), int(OFC_initial_errors))
    reversal_errors = (int(sham_reversal_errors), int(OFC_reversal_errors))
    
    fig, ax = plt.subplots()
    index = np.arange(conditions)
    bar_width = 0.2
    opacity = 0.8
     
    plt.bar(index, initial_errors, bar_width,
    color='b',
    label='Sham Lesions')
     
    plt.bar(index + bar_width, reversal_errors, bar_width,
    color='r',
    label='OFC Lesions')
     
    plt.xlabel('Condition')
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

def select_action(q_learning_params, q_values, state, tau):
    '''Select an action given the current state and using the Luce rule.'''
    # Compute the probability of choosing 'right'
    weighted_q_values = np.zeros(2)
    for action_index in range(len(q_vals)):
        weighted_q_values[action_index] = q_learning_params['beta'] * q_values[actions[action_index]]
    probabilities = softmax(weighted_q_values, tau)
    # Choose an action based on the probability of choosing 'right' / 'left'
    return np.random.choice(['left', 'right'], p=probs)

def q_update(state_action_reward_dict, q_values, q_learning_params, state, action):
    '''Given the most recently taken action (and the resulting reward), update the stored q values.'''
    q_values[str(state)][action] += q_learning_params['alpha'] * (state_action_reward_dict[str(state)][action] - q_values[str(state)][action])
    return q_values 

def reversal_learning(state_action_reward_dict, q_learning_params):
    '''Compute how many errors our simulated agent makes until they reach 90% accuracy in a reversal learning task.'''
    # Initialize q_values 
    if len(state_action_reward_dict) == 2:
        q_values = {'1': {'left': 0, 'right': 0}, 
                    '2': {'left': 0, 'right': 0}}
    else:
        q_values = {'1': {'left': 0, 'right': 0}}
    # Start in state 1
    state = 1
    # Take actions until 90% correct 
    initial_correct, initial_trials = run_trial(state_action_reward_dict, q_learning_params, q_values, state)  
    # Compute performance metric 
    initial_errors_to_criterion = initial_trials - initial_correct
    # Change state if sham lesioned
    if len(state_action_reward_dict) == 2:
        state = 2
    # Otherwise, change action receiving reward (since OFC lesioned)
    else:
        state_action_reward_dict['1']['left'] = q_learning_params['reward']
        state_action_reward_dict['1']['right'] = 0
    # Take actions until 90% correct
    reversal_correct, reversal_trials = run_trial(state_action_reward_dict, q_learning_params, q_values, state)
    # Compute performance metric 
    reversal_errors_to_criterion = reversal_trials - reversal_correct
    return initial_errors_to_criterion, reversal_errors_to_criterion 

# Run experiment with sham lesioned agent
sham_initial_errors, sham_reversal_errors = reversal_learning(sham_lesioned, q_learning_params)

# Run experiment with OFC lesioned agent 
OFC_initial_errors, OFC_reversal_errors = reversal_learning(OFC_lesioned, q_learning_params)

# Plot results for comparison with paper plots 
plot_reversal_errors(sham_initial_errors, sham_reversal_errors, OFC_initial_errors, OFC_reversal_errors)