from sklearn import svm
from random import shuffle

x_file = open('game-by-game-feature-vectors.csv', 'r')
lines = x_file.readlines()

x = []

for line in lines:
	line = line.replace('\n', '')[:-1]
	vector = line.split(',')
	vector = list(map(float, vector))
	x.append(vector)

y_file = open('game-by-game-results.csv')
lines = y_file.readlines()

y = []

for line in lines:
	line = line.replace('\n', '')
	y.append(int(line))

shuffled_x = []
shuffled_y = []
indices = list(range(len(x)))
shuffle(indices)
for i in indices:
	shuffled_x.append(x[i])
	shuffled_y.append(y[i])

num_samples = len(y)
num_train_samples = num_samples - int(num_samples * 0.8)
num_test_samples = num_samples - num_train_samples

train_x = shuffled_x[:num_train_samples]
train_y = shuffled_y[:num_train_samples]

test_x = shuffled_x[num_train_samples:]
test_y = shuffled_y[num_train_samples:]

classifier = svm.SVC(gamma=0.00001, C=100)
classifier.fit(train_x, train_y)

predictions = classifier.predict(test_x)
correct = 0
for i, p in enumerate(predictions):
	if p == test_y[i]:
		correct += 1
accuracy = (correct / num_test_samples) * 100

print('Accuracy: {0}%'.format(round(accuracy, 2)))
