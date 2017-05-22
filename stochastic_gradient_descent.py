import numpy as np 

def generate_data(num_datapoints=10000, num_features=10):
	data = np.random.uniform(-1.0, 1.0, (num_datapoints, num_features+1))
	weights = np.zeros((num_features, 1))
	return data, weights

def train(data, weights, step_size=1.0, batch_fraction=1.0, num_iterations=100):
	print(compute_cost(data, weights))
	computed_weights = weights
	batch_size = int(batch_fraction*len(data))
	error_over_time = np.zeros(num_iterations)
	for i in range(0, num_iterations):
		np.random.shuffle(data)
		x = data[:, 0:len(weights)]
		y = data[:, len(weights):len(weights)+1]
		for j in range(0, len(y), batch_size):
			if(j+batch_size - 1 >= len(y)):
				break
			computed_weights -= step_size/batch_size*(np.dot((x[[j,j+batch_size-1], :]).transpose(), np.dot(x[[j,j+batch_size-1], :], computed_weights)-y[[j,j+batch_size-1], :]))
		error_over_time[i] = compute_cost(data, computed_weights)

	return error_over_time, computed_weights

def normal_equation(data):
	x = data[:, 0:len(data[0])-1]
	y = data[:, len(data[0])-1:len(data[0])]
	return np.dot(np.dot(np.linalg.pinv(np.dot(x.transpose(), x)), x.transpose()), y)

def compute_cost(data, weights):
	x = data[:, 0:len(weights)]
	y = data[:, len(weights):len(weights)+1]
	return np.sum(np.square(np.dot(x, weights)-y))/(2.0*len(y))

def main():
	data, weights = generate_data()
	errors, sgd_model_weights = train(data, weights)
	optimal_weights = normal_equation(data)
	print(optimal_weights)
	print("\n")
	print(sgd_model_weights)
	print("\n")
	print("Cost using optimal weights: " + str(compute_cost(data, optimal_weights)))
	print("Cost using stochastic model weights: " + str(compute_cost(data, sgd_model_weights)))
	
main()