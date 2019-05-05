import q_learning_reversal_final 

q_learning_params['num_generations'] = 300
q_learning_params['num_trials'] = 100
 
def run_group_trial(states, agents, q_learning_params):
	q_values = {key: 0 for key in states.keys()}
	rewards = 0
    # results = {'actions': [0] * num_trial, 'rewards': [0] * num_trial}
    for i in range(q_learning_params['num_trial']):
		for agent in agents:
			action = select_action(q_learning_params, q_values, agents[i], states)
		    reward = get_reward(action, states)
		    q_values = q_update(states, reward, q_values, q_learning_params, action)
		    # results[i]['actions'], restuls[i]['rewards'] = action, reward
		    rewards += reward
	return rewards

def reproduce(agents, prob_survival, q_learning_params):
	num_successful = np.random.binomial(len(agents), prob_survival)
	np.random.shuffle(agents)
	surviving_agents = agents[:num_successful + 1]
	num_kids = np.random.normal(2, 1, size=surviving_agents)



def expected_value(states):
	return sum(map(lambda x: x['r'] * x['p'])) / float(len(states))


def generation_trial(states, groups, q_learning_params):
	expected_value = expected_value(states)
	for group in groups:
		rewards = run_group_trial(states, group, q_learning_params)
		prob_survival = softmax(rewards, expected_value)[0]
		next_gen = reproduce(agents, prob_survival, q_learning_params)


