import csv
import glob
import random

completes = sorted(glob.glob('datasets/SkullBreak/complete_skull/*.nrrd'))

with open('datasets/SkullBreak/skullbreak.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    for i in range(len(completes)):
        complete = completes[i]
        writer.writerow([complete])

train = random.sample(completes, 86)
test = [elem for elem in completes if elem not in train]

# Training set
with open('datasets/SkullBreak/train.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    for i in range(len(train)):
        complete = train[i]
        writer.writerow([complete])

# Test set
with open('datasets/SkullBreak/test.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    for i in range(len(test)):
        complete = test[i]
        writer.writerow([complete])

print("Successfully created training and test split for SkullBreak")
