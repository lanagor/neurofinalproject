import q_learning_reversal_final as q 
import numpy as np

params = q.q_learning_params.copy()
params['num_generations'] = 300
params['num_trials'] = 50
params['temp_variance'] = 0.1
params['adhd_heredity'] = 0.75
params['mutation_prob'] = 0.05
params['generation_size'] = 100
 
def run_group_trial(states, agents, q_learning_params):
	q_values = {key: 0 for key in states.keys()}
	rewards = 0
    # results = {'actions': [0] * num_trial, 'rewards': [0] * num_trial}
	for i in range(q_learning_params['num_trial']):
		for agent in agents:
			action = q.select_action(q_learning_params, q_values, agent, states)
			reward = q.get_reward(action, states)
			q_values = q.q_update(states, reward, q_values, q_learning_params, action)
			# results[i]['actions'], restuls[i]['rewards'] = action, reward
			rewards += reward
	return rewards

def reproduce(agents, prob_survival, q_learning_params):
	print(prob_survival)
	num_successful = np.random.binomial(len(agents), prob_survival)
	print(num_successful)
	np.random.shuffle(agents)
	surviving_agents = agents[:num_successful + 1]
	# num_kids = np.random.normal(2, 1, size=surviving_agents)
	new_gen = np.zeros(sum(num_kids))
	for i in range(len(surviving_agents)):
		# for j in range(max(0, round(num_kids[i]))):
		new_gen[j] = recombination(surviving_agents[i], q_learning_params)
	return new_gen

def recombination(tau, q_learning_params):
	sd, adhd_tau, nt_tau = q_learning_params['temp_variance'], q_learning_params['adhd_temp'], q_learning_params['neurotypical_temp']
	# neurotypical parent: inherits tau or mutates to adhd temp
	if tau - adhd_tauh > sd:
		return np.random.choice([draw_normal(tau), draw_normal(adhd_tau)], p=[1 - q_learning_params['mutation_prob'], q_learning_params['mutation_prob']])
	# adhd parent: inherits tau or passes neurotypical with probability of adhd hereditability
	else:
		return np.random.choice([draw_normal(tau), draw_normal(nt_tau)], p=[q_learning_params['adhd_heredity'], 1 - q_learning_params['adhd_heredity']])

def draw_normal(tau):
	return np.random.normal(tau, q_learning_params['temp_variance'])

def calc_expected_value(states):
	return sum([states[s]['p'] * states[s]['r'] for s in states]) / float(len(states))

def generation_trial(states, groups, q_learning_params):
	expected_value = calc_expected_value(states)
	group_rewards = np.zeros((q_learning_params['num_generations'], len(groups)))
	for i in range(q_learning_params['num_generations']):
		for j in range(len(groups)):
			rewards = run_group_trial(states, groups[j], q_learning_params)
			print("rewards ", rewards, "expected value ", expected_value)
			prob_survival = q.softmax(np.array([rewards, expected_value], dtype=np.float128), 1)
			groups[j] = reproduce(groups[j], prob_survival, q_learning_params)
			group_rewards[i][j]
		if i % 10 == 0: 
			print("Generation: ", i, "group rewards: ", group_rewards[i])
	return group_rewards, groups

def generate_groups(adhd_proportions, size, adhd_temp, nt_temp):
	groups = [np.random.choice([adhd_temp, nt_temp], p=[p, 1 - p], size=size) for p in adhd_proportions]
	return groups


if __name__ == "__main__":
	groups = generate_groups([0, 0.2, 0.4, 0.8, 1],params['generation_size'], params['adhd_temp'], params['neurotypical_temp'])
	print(generation_trial(q.state_2, groups, params))


