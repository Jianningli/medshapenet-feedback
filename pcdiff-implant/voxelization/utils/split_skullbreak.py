import csv
import glob
import random

data = []
defects = ['bilateral', 'frontoorbital', 'parietotemporal', 'random_1', 'random_2']

with open('datasets/SkullBreak/train.csv', 'r', newline='') as file:
    csvreader = csv.reader(file)
    for row in csvreader:
        for i in range(5):
            data.append(row[0].split('complete')[0] + 'voxelization/' + row[0].split('skull/')[1].split('.nrrd')[0] +
                        '_' + defects[i])

train = random.sample(data, 400)
test = [elem for elem in data if elem not in train]

# Training set
with open('datasets/SkullBreak/voxelization/train.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    for i in range(len(train)):
        datapoint = train[i]
        writer.writerow([datapoint])

# Test set
with open('datasets/SkullBreak/voxelization/eval.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    for i in range(len(test)):
        datapoint = test[i]
        writer.writerow([datapoint])

print("Successfully created training and evaluation split for SkullBreak")
