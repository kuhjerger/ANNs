import numpy as np
import theano
import theano.tensor as T
import matplotlib.pyplot as plt


def one_hot_encode(y):
	N = len(y)
	K = len(set(y))

	Y = np.zeros((N,K))

	for i in range(N):
		Y[i,y[i]] = 1

	return Y


def shuffle(X, Y):
	N = len(X)
	idx = np.random.permutation(N)
	return X[idx], Y[idx]


def accuracy(y, y_hat):
	return np.mean(y == y_hat)


class HiddenLayer():
	def __init__(self, layer_num, M1, M2):
		W = np.random.randn(M1,M2)/np.sqrt(M1)
		b = np.random.randn(M2)

		self.layer_num = layer_num
		self.W = theano.shared(W.astype(np.float32), "W{}".format(layer_num))
		self.b = theano.shared(b.astype(np.float32), "b{}".format(layer_num))
		self.params = [self.W, self.b]

	def forward(self, X, activation = T.nnet.relu):
		return activation(X.dot(self.W) + self.b)


class ANN():
	def __init__(self, hidden_layer_sizes, activations = None):
		self.hidden_layer_sizes = hidden_layer_sizes
		self.activations = activations

	def forward(self, X):
		Z = X
		if self.activations is None:
			for h in self.hidden_layers:
				Z = h.forward(Z)
		else:
			for h, a in zip(self.hidden_layers, self.activations):
				Z = h.forward(Z, a)
		return T.nnet.softmax(Z.dot(self.W) + self.b)

	def fit(self, X, y, train_splits = [0.6, 0.8], eta = 1e-4, lambda2 = 0, lambda1 = 0, mu = 0.9, gamma = 0.999, epsilon = 1e-10, batch_sz = 100, epochs = 500, show_fig = False):
		N, D = X.shape
		K = len(set(y))

		Y = one_hot_encode(y)

		idx = np.random.permutation(N)
		X = X[idx,:].astype(np.float32)
		Y = Y[idx,:].astype(np.int32)
		y = y[idx].astype(np.int32)

		X_train = X[:int(N*train_splits[0]),:]
		Y_train = Y[:int(N*train_splits[0]),:]
		y_train = y[:int(N*train_splits[0])]

		X_cv = X[int(N*train_splits[0]):int(N*train_splits[1]),:]
		Y_cv = Y[int(N*train_splits[0]):int(N*train_splits[1]),:]
		y_cv = y[int(N*train_splits[0]):int(N*train_splits[1])]

		X_test = X[int(N*train_splits[1]):,:]
		Y_test = Y[int(N*train_splits[1]):,:]
		y_test = y[int(N*train_splits[1]):]

		N_train = len(X_train)
		N_cv = len(X_cv)
		N_test = len(X_test)

		X_tensor = T.matrix(name = "X", dtype = config.float32)
		Y_tensor = T.matrix(name = "Y", dtype = config.int32)
		y_tensor = T.vector(name = "y", dtype = config.int32)
		P = self.forward(X_tensor)


		self.hidden_layers = []
		layer_num = 0
		M1 = D

		for M2 in self.hidden_layer_sizes:
			self.hidden_layers.append(HiddenLayer(layer_num, M1, M2))
			layer_num += 1
			M1 = M2

		W = np.random.randn(M1,K)/np.sqrt(M1)
		b = np.random.randn(K)
		self.W = theano.shared(W.astype(np.float32), "W{}".format(layer_num))
		self.b = theano.shared(b.astype(np.float32), "b{}".format(layer_num))
		self.params = [self.W, self.b]

		for h in self.hidden_layers:
			self.params += h.params


		vparams = [theano.shared(np.zeros(p.get_value().shape).astype(np.float32)) for p in self.params]
		Gparams = [theano.shared(np.ones(p.get_value().shape).astype(np.float32)) for p in self.params]


		L2_penalty = (lambda2/2)*T.sum([(p*p).sum for p in self.params])
		L1_penalty = lambda1*T.sum([T.abs(p).sum for p in self.params])

		objective = -T.sum(Y_tensor*T.log(P)) + L2_penalty + L1_penalty

		pred = self.predict(X_tensor)

		objective_op = theano.function(
			inputs = [X_tensor, Y_tensor],
			outputs = [objective]
			)

		predict_op = theano.function(
			inputs = [X_tensor],
			outputs = [pred]
			)


		updates = [
		(G, gamma*G + (1 - gamma)*T.grad(objective, p)*T.grad(objective, p)) for p, G in zip(self.params, Gparams)
		] + [
		(v, mu*v - (eta/T.sqrt(G + epsilon))*T.grad(objective, p)) for p, v, G in zip(self.params, vparams, Gparams)
		] + [
		(p, p + mu*v - (eta/T.sqrt(G + epsion)*T.grad(objective, p)) for p, v, G in zip(self.params, vparams, Gparams))
		]

		train_op = theano.function(
			inputs = [X_tensor, Y_tensor],
			updates = updates
			)

		n_batches = N_train//batch_sz

		J_train = []
		J_cv = []
		J_test = []

		for epoch in range(epochs):
			idx = np.random.permutation(N_train)
			X_train = X_train[idx,:]
			Y_train = Y_train[idx,:]
			y_train = y_train[idx]

			for i in range(N_train):
				X_batch = X_train[(i*batch_sz):((i + 1)*batch_sz),:]
				Y_batch = Y_train[(i*batch_sz):((i + 1)*batch_sz),:]

				train_op(X_batch, Y_batch)

				if i % 10 == 0:
					j_train = objective_op(X_train, Y_train)
					j_cv = objective_op(X_cv, Y_cv)
					j_test = objective_op(X_test, Y_test)
					acc = accuracy(y_train, self.predict(X_train))

					J_train.append(j_train/N_train)
					J_cv.append(j_cv/N_cv)
					J_test.append(j_test/N_test)

					print("epoch: {} of {} -- batch: {} of {} -- J train: {} -- training accuracy: {}".format(epoch + 1, epochs, i + 1, n_batches, j_train, acc))


		if show_fig:
			plt.plot(J_train, label = "Training Error")
			plt.plot(J_cv, label = "Validation Error")
			plt.plot(J-test, label = "Test Error")
			plt.legend()
			plt.xlabel("Training Epochs")
			plt.ylabel("Error")
			plt.title("Training Curve")
			plt.show()

	def predict(self, X):
		P = self.forward(X)
		return T.argmax(P, axis = 1)

