import pickle
import os
folder_path = './data/'
os.listdir(folder_path)
fileName_list = os.listdir(folder_path)[:-1]
labelType_list = [k.strip('.csv') for k in fileName_list]
content_list = []
label_list = []
fileName = os.path.join(folder_path, fileName_list[0])

with open(fileName, encoding='utf8') as file:
    line_list = [k.strip() for k in file.readlines()][:-1]
    for line in line_list:
        content_list.append(line.split(',', maxsplit=1)[1])
        label_list.append(fileName.strip('.csv').strip('/data'))

content_list = []
label_list = []
for fileName in fileName_list:
    filePath = os.path.join(folder_path, fileName)
    with open(filePath, encoding='utf8') as file:
        line_list = [k.strip() for k in file.readlines()][:-1]
        for line in line_list:
            content_list.append(line.split(',', maxsplit=1)[1])
            label_list.append(fileName.strip('.csv').strip('/data'))

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(content_list, label_list)

with open('x_train.pickle', 'wb') as file:
    pickle.dump(x_train, file)
    
with open('y_train.pickle', 'wb') as file:
    pickle.dump(y_train, file)

with open('x_test.pickle', 'wb') as file:
    pickle.dump(x_test, file)
    
with open('y_test.pickle', 'wb') as file:
    pickle.dump(y_test, file)

