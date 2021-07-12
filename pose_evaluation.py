from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
from os import listdir
import argparse
import numpy as np
import os
import glog as log





parser = argparse.ArgumentParser(description='pose_evaluation')
parser.add_argument('--pathNameGood', required = True)
parser.add_argument('--pathNameBad', required = True)
parser.add_argument('--pathTest', required = True)

args = parser.parse_args()



pathGood = args.pathNameGood
pathBad = args.pathNameBad
pathTest = args.pathTest

csvsGood = os.listdir(pathGood)
csvsBad = os.listdir(pathBad)
csvsTest = os.listdir(pathTest)

dataset = np.empty([0, 39])
test = np.empty([0, 39])

for csv in csvsGood:
    fileName = pathGood + '/' + csv
    video_csv = loadtxt(fileName, delimiter=',')
      
    dataset = np.concatenate((dataset, video_csv), axis=0)

for csv in csvsBad:
    fileName = pathBad + '/' + csv
    video_csv = loadtxt(fileName, delimiter=',')
      
    dataset = np.concatenate((dataset, video_csv), axis=0)

for csv in csvsTest:
    fileName = pathTest + '/' + csv
    video_csv = loadtxt(fileName, delimiter=',')
      
    test = np.concatenate((test, video_csv), axis=0)

 
   


np.random.shuffle(dataset)
np.random.shuffle(test)

trainX = dataset[:, 0:38]
trainY = dataset[:, 38]

testX = test[:, 0:38]
testY = test[:, 38]

#trainX = dataset[:-100,0:38]
#trainY = dataset[:-100,38]
#testX = dataset[-100:, 0:38]
#testY = dataset[-100:, 38]


model = Sequential()
model.add(Dense(4, input_dim=38, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


model.fit(trainX, trainY, epochs=150, batch_size=10)

print('***************************')

_, accuracy = model.evaluate(testX, testY)
print('Accuracy: %.2f' % (accuracy*100))