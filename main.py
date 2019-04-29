import math
import random

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy
from random import choices
from numpy.random import beta

num_actions = 3
#70, 25, 5
actions = list(range(num_actions))
a_values = [1]*num_actions
b_values = [1]*num_actions
q_values = [0]*num_actions
betas = [0]*num_actions

num_trials = 200

sigma = 0.5
temperature = 0.1

#11% of people have ADHD

degrees = [0]*num_trials
entropies = [0]*num_trials

true_source = [.8, .7, .6, .5]

def softmax(x, temp):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x/temp) / np.sum((np.exp(x))/temp, axis=0)

for i in range(num_trials):

	#get action decision values
	for j in range(num_actions):
		a = a_values[j]
		b = b_values[j]
		v = (a*b)/(((a+b)**2)*(a+b+1))
		q_values[j]=a/(a+b)+sigma*v

	#draw from beta distribution:
	for j in range(num_actions):
		betas[j] = beta(a_values[j], b_values[j])

	for_softmax = [a*b for a,b in zip(q_values,betas)]

	normalized_values = softmax(for_softmax, temperature)

	max_value = np.argmax(true_source)

	decision = choices(actions, normalized_values)[0]

	#update chosen values
	for j in range(num_actions):
		if j == decision:
			a_values[j] = a_values[j]+1
		else:
			b_values[j] = b_values[j]+1

	#exploit
	if max_value == decision:
		color = "red"
		exploit_sum += entr
		exploit_num += 1

	#blue is explore
	else:
		color="blue"
		explore_sum += entr
		explore_num += 1

	degree = true_source[max_value]-true_source[decision]
	degrees[i]=degree
	entropies[i]=entr

	plt.scatter(i, entr, color=color, s=4)
	plt.title("entropy by degree of exploration")
	plt.ylabel("entropy")
	plt.xlabel("degree of exploration")
	#plt.xlim(-.01, 0.02)
	#plt.ylim(1.383, 1.387)
	#plt.ylim(1.384, 1.387)

	for j in range(len(true_source)):
		true_source[j] += .01*random.randint(-3, 3)

print("exploit trials: ", exploit_sum/exploit_num)
print("explore trials: (larger?) ", explore_sum/explore_num) #should be higher for explore

idx_sorted_degrees = np.argsort(degrees)

first_third = 0
second_third = 0
third_third = 0
print(num_trials//3)
for j in range(num_trials//3):
	idx = idx_sorted_degrees[j]
	first_third += entropies[idx]/(num_trials//3)

for j in range(num_trials//3, 2*num_trials//3):
	idx = idx_sorted_degrees[j]
	second_third += entropies[idx]/(num_trials//3)

for j in range(2*num_trials//3, num_trials):
	idx = idx_sorted_degrees[j]
	third_third += entropies[idx]/(num_trials//3)

print(first_third) #should be smallest
print(second_third)
print(third_third)

max_degree = 1.75
num_divisions = 7

def chunkIt(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0

    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg

    return out

mini_degrees = chunkIt(degrees, num_divisions)
mini_entropies = chunkIt(entropies, num_divisions)

print(exploit_num/num_trials)

# for d in range(num_divisions):
# 	plt.scatter(sum(mini_degrees[d])/len(mini_degrees[d]), sum(mini_entropies[d])/len(mini_entropies[d]))

plt.show()