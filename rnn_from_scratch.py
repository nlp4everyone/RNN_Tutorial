import numpy as np

# Feed-forward
class RNN():
    def __init__(self,embedding_space =100,hidden_space = 384):
        # Define parameter
        self.embedding_space = embedding_space
        self.hidden_space = hidden_space

        # Define matrix input
        self.W_xh = np.random.uniform(low=(-1/np.sqrt(embedding_space)),high=(1/np.sqrt(embedding_space)),size=(embedding_space,hidden_space))
        self.W_hh = np.random.uniform(low=(-1/np.sqrt(hidden_space)),high=(1/np.sqrt(hidden_space)),size=(hidden_space,hidden_space))
        self.hidden_state = np.zeros(shape=(1,hidden_space))
        self.bias1 = np.random.uniform(low=(-1/np.sqrt(hidden_space)),high=(1/np.sqrt(hidden_space)),size=(1,hidden_space))
        self.bias2 = np.random.uniform(low=(-1 / np.sqrt(hidden_space)), high=(1 / np.sqrt(hidden_space)),
                                       size=(1, embedding_space))
        self.W_out = np.random.uniform(low=(-1/np.sqrt(hidden_space)),high=(1/np.sqrt(hidden_space)),size=(hidden_space,embedding_space))
    def calculate_soft_max(self,X):
        return (np.exp(X) / np.exp(X).sum())

    def forward(self,X):
        self.hidden_state = np.dot(X,self.W_xh) + np.dot(self.hidden_state,self.W_hh) + self.bias1
        # Apply tanh function
        self.hidden_state = np.tanh(self.hidden_state)
        # prob
        prob = np.dot(self.hidden_state,self.W_out) + self.bias2
        print(prob.shape)
        prob = self.calculate_soft_max(prob)
        print(np.max(prob))
        return self.hidden_state

input_vector = np.random.rand(1,100)
rnn = RNN()
hidden_state = rnn.forward(input_vector)

# (np.exp(X) / np.exp(X).sum())



