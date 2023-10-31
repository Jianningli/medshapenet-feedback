import csv
import glob
import random

completes = sorted(glob.glob('datasets/SkullFix/complete_skull/*.nrrd'))

with open('datasets/SkullFix/skullfix.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    for i in range(len(completes)):
        complete = completes[i]
        writer.writerow([complete])

train = random.sample(completes, 75)
test = [elem for elem in completes if elem not in train]

# Training set
with open('datasets/SkullFix/train.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    for i in range(len(train)):
        complete = train[i]
        writer.writerow([complete])

# Test set
with open('datasets/SkullFix/test.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    for i in range(len(test)):
        complete = test[i]
        writer.writerow([complete])

print("Successfully created training and test split for SkullFix")
