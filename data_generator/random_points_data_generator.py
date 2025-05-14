'''
this file generates synthetic data for training EBM:
5 random points in a 2D (10x10) space:
    each point has 3 features: x, y, and colour: {0:red, 1:green, 2:blue}
1 observer at random position in the 2D space

a set of labels for <concepts object>:
    0: observer is close (<3 units)
    1: observer is far (>5 units)
    2: observer is to the right
    3: observer is to the left
    4: observer is above
    5: observer is below

for each <concept object>, stores a set of <concept subjects>:
    0: red
    1: green
    2: blue

for example:
if a red point and a green point are close to the observer, a set of labels are attached: {[0,0], [0,1]}
'''

import os
import numpy as np
import random
import pickle
import tqdm
import matplotlib.pyplot as plt

COLOURS = ['red', 'green', 'blue', 'yellow']


class RandomPointsDataGenerator:
    def __init__(self, data_directory, data_file, data_size=1000, num_points=5):
        self.data_directory = data_directory
        self.data_file = data_file
        self.data_size = data_size
        self.num_points = num_points
        self.data_and_labels = []

    def decode_concept(self, concept):
        # Decode the concept into a human-readable format
        concept_str = ""
        concept_object = int(concept[0])
        concept_subject = int(concept[1])
        colour = COLOURS[concept_subject]
        if concept_object == 0:
            concept_str = f"Observer is close to {colour} point"
        elif concept_object == 1:
            concept_str = f"Observer is far from {colour} point"
        elif concept_object == 2:
            concept_str = f"Observer is to the right of {colour} point"
        elif concept_object == 3:
            concept_str = f"Observer is to the left of {colour} point"
        elif concept_object == 4:
            concept_str = f"Observer is above {colour} point"
        elif concept_object == 5:
            concept_str = f"Observer is below {colour} point"
        else:
            concept_str = "Unknown concept"
        
        return concept_str

    def generate_data(self, save=True):
        while len(self.data_and_labels) < self.data_size:
            # Generate random points
            points = []
            for _ in range(self.num_points):
                x = random.uniform(-1, 1)
                y = random.uniform(-1, 1)
                colour = random.choice(np.arange(len(COLOURS)))
                points.append([x, y, colour])
            points = np.array(points)
            # Generate random observer position
            observer_x = random.uniform(-1, 1)
            observer_y = random.uniform(-1, 1)
            observer = np.array([observer_x, observer_y])
            # Generate labels
            labels = []
            for i in range(self.num_points):
                point = points[i]
                distance = np.linalg.norm(point[:2] - observer)
                if distance < 0.3:
                    labels.append([0, point[2]])
                elif distance > 0.7:
                    labels.append([1, point[2]])
                if point[0] < observer[0]:
                    labels.append([2, point[2]])
                elif point[0] > observer[0]:
                    labels.append([3, point[2]])
                if point[1] < observer[1]:
                    labels.append([4, point[2]])
                elif point[1] > observer[1]:
                    labels.append([5, point[2]])
            # Store data and labels
            # add small noise to the labels
            labels = np.array(labels)
            noise = np.random.normal(0, 0.1, labels.shape)
            # pick one unique labels
            labels = np.unique(labels, axis=0)
            # labels = labels + noise
            # self.data_and_labels.append((points, observer, labels))
            # remove opposite labels for example: being left and right at the same time for some colour
            indices_to_remove = np.zeros(labels.shape[0], dtype=bool)
            for i in range(len(labels)):
                if labels[i][0] == 2:
                    if [3., labels[i][1]] in labels.tolist():
                        indices_to_remove[i] = True

                elif labels[i][0] == 3:
                    if [2., labels[i][1]] in labels.tolist():
                        indices_to_remove[i] = True

                elif labels[i][0] == 4:
                    if [5., labels[i][1]] in labels.tolist():
                        indices_to_remove[i] = True
                elif labels[i][0] == 5:
                    if [4., labels[i][1]] in labels.tolist():
                        indices_to_remove[i] = True
            labels = labels[~indices_to_remove]

            if len(labels) == 0:
                continue

            for label in labels:
                self.data_and_labels.append((points, observer, label))

            # show scatter plot of points and observer
            # plt.scatter(points[:, 0], points[:, 1], c=[COLOURS[int(p[2])] for p in points])
            # plt.scatter(observer[0], observer[1], c='black', marker='x')
            # plt.show()
            
        # Save data and labels to file
        if save:
            os.makedirs(self.data_directory, exist_ok=True)
            with open(os.path.join(self.data_directory, self.data_file), 'wb') as f:
                pickle.dump(self.data_and_labels, f)
            print(f"Data saved to {os.path.join(self.data_directory, self.data_file)}")

        else:
            return self.data_and_labels
    
if __name__ == "__main__":
    data_directory = "../data"
    data_file = "random_points_data.pkl"
    data_size = 1000000
    num_points = 5

    generator = RandomPointsDataGenerator(data_directory, data_file, data_size, num_points)
    generator.generate_data()
