from keras.models import load_model
from keras.preprocessing import sequence
import pickle
from sklearn.preprocessing import LabelEncoder

model = load_model('model')
seq_length = 200
labelEncoder = LabelEncoder()
def predict(text):
    # read vocabulary
    with open('vocabulary2.txt', encoding='utf8') as file:
        vocabulary_list = [k.strip() for k in file.readlines()]
    with open('labelEncoder.pickle', 'rb') as file:
        labelEncoder =  pickle.load(file)
    # build word-id dict
    word2id_dict = dict([(b, a) for a, b in enumerate(vocabulary_list)])
    # print(word2id_dict)

    text = [text]
    # transform contents into id sequences
    content2idList = lambda content : [word2id_dict[word] for word in content if word in word2id_dict]
    text_list = [content2idList(content) for content in text]
    # print(text_list)

    text_array = sequence.pad_sequences(text_list, maxlen = seq_length, padding = 'post')
    # print(text_array)

    prediction = model.predict(text_array)
    final_prediction = [result.argmax() for result in prediction][0]
    final_prediction = [final_prediction]
    final_prediction = labelEncoder.inverse_transform(final_prediction)
    return final_prediction[0]



text = "报警人被老公打，无人伤，请民警到场处理。"
result = predict(text)
print('predict class:->>', result)


