import pickle

# load data from pickle
with open('x_train.pickle', 'rb') as file:
    x_train = pickle.load(file)
    # print(x_train[0])
with open('y_train.pickle', 'rb') as file:
    y_train = pickle.load(file)
with open('x_test.pickle', 'rb') as file:
    x_test = pickle.load(file)
with open('y_test.pickle', 'rb') as file:
    y_test = pickle.load(file)

from collections import Counter

def getVocabularyList(content_list, vocabulary_size):
    allContent_str = ''.join(content_list)
    counter = Counter(allContent_str) # sort words according to their frequency
    vocabulary_list = [k[0] for k in counter.most_common(vocabulary_size)] # get the most frequent 2000 words
    return ['PAD'] + vocabulary_list 

def makeVocabularyFile(content_list, vocabulary_size):
    vocabulary_list = getVocabularyList(content_list, vocabulary_size)
    with open('vocabulary2.txt', 'w', encoding='utf8') as file:
        for vocabulary in vocabulary_list:
            file.write(vocabulary + '\n')

makeVocabularyFile(x_train, vocabulary_size = 2000) # build a dict, containing most frequent 2000 words

# read vocabulary
with open('vocabulary.txt', encoding='utf8') as file:
    vocabulary_list = [k.strip() for k in file.readlines()]

# build word-id dict
word2id_dict = dict([(b, a) for a, b in enumerate(vocabulary_list)])
# print(word2id_dict)

# transform contents into id sequences
content2idList = lambda content : [word2id_dict[word] for word in content if word in word2id_dict]
x_train = [content2idList(content) for content in x_train] 
# print(x_train[0])
# print(type(x_train[0])) # output <class 'list'>
x_test = [content2idList(content) for content in x_test]

from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from keras.preprocessing import sequence

# transform label to one-hot vector
labelEncoder = LabelEncoder()
y_train = labelEncoder.fit_transform(y_train)
y_train = np_utils.to_categorical(y_train, num_classes = 4)
y_test = labelEncoder.fit_transform(y_test)
y_test = np_utils.to_categorical(y_test, num_classes = 4)
with open('labelEncoder.pickle', 'wb') as file:
    pickle.dump(labelEncoder, file, protocol=pickle.HIGHEST_PROTOCOL)

vocab_size = 5000 if len(vocabulary_list) > 5000 else len(vocabulary_list)
print('vocabulary size:->>', vocab_size)

# contentLength_list = [len(k) for k in x_train]
# seq_length = max(contentLength_list) # make the length of the longest sentence as the unifed length of vectors
# print('sequence size:->>', seq_length)

seq_length = 200
# unify x_train list sequences into numpy.narrays of length seq_length
x_train = sequence.pad_sequences(x_train, maxlen = seq_length, padding = 'post')
x_test = sequence.pad_sequences(x_test, maxlen = seq_length, padding = 'post')
# print(x_train[0])
# print(type(x_train[0])) # output <class 'numpy.ndarray'>

from keras.models import Sequential
from keras.layers.core import  Activation, Flatten, Dense, Dropout
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam

nb_class = 4
nb_epoch = 7
batchsize = 128
embedding_dim = 64


model = Sequential()

model.add(Embedding(
    output_dim = embedding_dim,
    input_dim = vocab_size,
    input_length = seq_length
))

model.add(Dropout(0.1))

# cnn layer
model.add(Conv1D(
    filters = 256,
    kernel_size = 5,

))

# max pooling layer
model.add(MaxPooling1D(
    pool_size = 5,
    strides = 5,
    padding = 'valid'

))

# 1st fully connected layer
model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))


# 2nd fully connected layer
model.add(Dense(4))
model.add(Activation('softmax'))


# compile model
model.compile(
    optimizer = 'Adam',
    loss = 'categorical_crossentropy',
    metrics= ['accuracy']
)

# train model
model.fit(
    x = x_train,
    y = y_train,
    epochs = nb_epoch,
    batch_size = batchsize,
    validation_data = (x_test, y_test)
)

model.save('model')