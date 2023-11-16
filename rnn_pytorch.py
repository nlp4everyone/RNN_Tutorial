from gensim import corpora
from gensim.parsing.preprocessing import split_on_space


##### Data #####
train_X = ['good', 'bad', 'happy', 'sad', 'not good', 'not bad', 'not happy', 'not sad', 'very good', 'very bad', 'very happy', 'very sad', 'i am happy', 'this is good', 'i am bad', 'this is bad', 'i am sad', 'this is sad', 'i am not happy', 'this is not good', 'i am not bad', 'this is not sad', 'i am very happy', 'this is very good', 'i am very bad', 'this is very sad', 'this is very happy', 'i am good not bad', 'this is good not bad', 'i am bad not good', 'i am good and happy', 'this is not good and not happy', 'i am not at all good', 'i am not at all bad', 'i am not at all happy', 'this is not at all sad', 'this is not at all happy', 'i am good right now', 'i am bad right now', 'this is bad right now', 'i am sad right now', 'i was good earlier', 'i was happy earlier', 'i was bad earlier', 'i was sad earlier', 'i am very bad right now', 'this is very good right now', 'this is very sad right now', 'this was bad earlier', 'this was very good earlier', 'this was very bad earlier', 'this was very happy earlier', 'this was very sad earlier', 'i was good and not bad earlier', 'i was not good and not happy earlier', 'i am not at all bad or sad right now', 'i am not at all good or happy right now', 'this was not happy and not good earlier']
train_y = [1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0]

test_X = ['this is happy', 'i am good', 'this is not happy', 'i am not good', 'this is not bad', 'i am not sad', 'i am very good', 'this is very bad', 'i am very sad', 'this is bad not good', 'this is good and happy', 'i am not good and not happy', 'i am not at all sad', 'this is not at all good', 'this is not at all bad', 'this is good right now', 'this is sad right now', 'this is very bad right now', 'this was good earlier', 'i was not happy and not good earlier']
test_y = [1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0]

# BUild a corpus
import numpy as np
array = np.random.rand(3,5)
print(array.T.shape)