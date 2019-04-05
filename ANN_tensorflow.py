import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def one_hot_encode(y):
	N = len(y)
	K = len(set(y))

	Y = np.zeros((N,K))

	for i in range(N):
		Y[i,y[i]] = 1

	return Y


def accuracy(y, y_hat):
	return np.mean(y == y_hat)


class HiddenLayer():
	def __init__(self, layer_num, M1, M2):
		W = np.random.randn(M1, M2)/np.sqrt(M1)
		b = np.random.randn(M2)

		self.layer_num = layer_num
		self.W = tf.Variable(W.astype(np.float32), name = "W{}".format(layer_num))
		self.b = tf.Variable(b.astype(np.float32), name = "b{}".format(layer_num))
		self.params = [self.W, self.b]

	def forward(self, X, activation = tf.nn.relu):
		return activation(tf.matmul(X,self.W) + self.b)


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
		return tf.matmul(Z,self.W) + self.b

	def fit(self, X, y, train_splits = [0.6, 0.8], eta = 1e-4, lambda2 = 0, lambda1 = 0, mu = 0.9, gamma = 0.999, epsilon = 1e-10, epochs = 500, batch_sz = 100, show_fig = False):
		N, D = X.shape
		K = len(set(y))

		Y = one_hot_encode(y)

		idx = np.random.permutation(N)
		X = X[idx,:].astype(np.float32)
		Y = Y[idx,:].astype(np.int32)
		y = y[idx].astype(np.float32)

		X_train = X[:int(N*train_splits[0]),:]
		Y_train = Y[:int(N*train_splits[0]),:]
		y-train = y[:int(N*train_splits[0])]

		X_cv = X[int(N*train_splits[0]):int(N*train_splits[1]),:]
		Y_cv = Y[int(N*train_splits[0]):int(N*train_splits[1]),:]
		y_cv = y[int(N*train_splits[0]):int(N*train_splits[1])]

		X_test = X[int(N*train_splits[1]):,:]
		Y_test = Y[int(N*train_splits[1]):,:]
		y_test = y[int(N*train_splits[1]):]

		X_tensor = tf.placeholder(dtype = tf.float32, shape = (None,D), name = "X")
		Y_tensor = tf.placeholder(dtype = tf.int32, shape = (None,K), name = "Y")
		y_tensor = tf.placeholder(dtype = tf.int32, shape = (None,), name = "y")
		H = self.forward(X_tensor)

		N_train = len(X_train)
		N_cv = len(X_cv)
		N_test = len(X_test)


		self.hidden_layers = []
		layer_num = 0
		M1 = D

		for M2 in self.hidden_layer_sizes:
			self.hidden_layers.append(HiddenLayer(layer_num, M1, M2))
			layer_num += 1
			M1 = M2

		W = np.random.randn(M1,K)/np.sqrt(M1)
		b = np.random.randn(K)

		self.W = tf.Variable(W.astype(np.float32), "W{}".format(layer_num))
		self.b = tf.Variable(b.astype(np.float32), "b{}".format(layer_num))
		self.params = [self.W, self.b]

		for h in self.hidden_layers:
			self.params += h.params

		L1_norm = tf.contrib.layers.l1_regularizer(lambda1)

		L2_penalty = (lambda2/2)*sum(tf.nn.l2_loss(p) for p in self.params)
		L1_penalty = sum(L1_norm(p) for p in self.params)

		objective = tf.reduce_sum(tf.softmax_cross_entropy_with_logits_v2(labels = Y_tensor, logits = H)) + L2_penalty + L1_penalty

		pred = self.predict(X_tensor)

		train_op = tf.train.RMSPropOptimizer(eta, decay = gamma, momentum = mu).minimize(objective)

		n_batches = N_train//batch_sz
		J_train = []
		J_cv = []
		J_test = []

		init = tf.global_variables_initializer()

		with tf.Session() as sess:
			sess.run(init)

			for epoch in range(epochs):
				idx = np.random.permutation(N_train)
				X_train = X_train[idx,:]
				Y_train = Y_train[idx,:]
				y_train = y_train[idx,:]

				for i in range(n_batches):
					X_batch = X_train[(i*batch_sz):((i + 1)*batch_sz),:]
					Y_batch = Y_train[(i*batch-sz):((i + 1)*batch_sz),:]

					sess.run(train_op, feed_dict = {X_tensor: X_batch, Y_tensor: Y_batch})

					if i % 10 == 0:
						j_train = sess.run(objective, feed_dict = {X_tensor: X_train, Y_tensor: Y_train})
						j_cv = sess.run(objective, feed_dict = {X_tensor: X_cv, Y_tensor: Y_cv})
						j_test = sess.run(objective, feed_dict = {X_tensor: X_test, Y_tensor: Y_test})
						acc = accuracy(y_train, sess.run(pred, feed_dict = {X_tensor: X_train}))

						J_train.append(j_train)
						J_cv.append(j_cv)
						J_test.append(j_test)

						print("epoch: {} of {} -- batch: {} of {} -- J train: {} -- accuracy: {}".format(epoch, epochs, i, n_batches, j_train, acc))


		if show_fig:
			plt.plot(J_train, label = "Training Error")
			plt.plot(J_cv, label = "Validation Error")
			plt.plot(J_test, label = "Test Error")
			plt.legend()
			plt.xlabel("epochs")
			plt.ylabel("Error")
			plt.show()

	def predict(self, X):
		H = self.forward(X)
		return tf.argmax(H, axis = 1)
		