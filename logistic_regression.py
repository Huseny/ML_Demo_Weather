from math import exp
import numpy as np


class BinaryLogisticRegression:
    def __init__(self, learning_rate=0.01, num_iterations=100):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None

    def train(self, features, labels):
        num_features = len(features[0])
        num_examples = len(features)
        self.weights = np.zeros(num_features)
        self.bias = 0

        for _ in range(self.num_iterations):
            for i in range(num_examples):
                y = labels[i]
                z = self.bias + np.dot(self.weights, features[i])
                z = np.clip(z, -500, 500)
                sigmoid = 1 / (1 + np.exp(-z))
                error = sigmoid - y

                self.bias -= self.learning_rate * error
                self.weights -= self.learning_rate * error * np.array(features[i])

    def predict(self, features):
        predictions = []
        for i in range(len(features)):
            z = self.bias + np.dot(self.weights, features[i])
            sigmoid = 1 / (1 + np.exp(-z))
            predictions.append(1 if sigmoid >= 0.5 else 0)
        return predictions


class MulticlassLogisticRegression:
    def __init__(self):
        pass

    def train(self, features, labels, learning_rate=0.01, num_iterations=100):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.models = {}
        unique_labels = list(set(labels))
        for label in unique_labels:
            binary_labels = [1 if l == label else 0 for l in labels]
            model = BinaryLogisticRegression(self.learning_rate, self.num_iterations)
            model.train(features, binary_labels)
            self.models[label] = model

    def predict(self, features):
        predictions = []
        for feature in features:
            label_predictions = {}
            for label, model in self.models.items():
                label_predictions[label] = model.predict([feature])[0]
            predicted_label = max(label_predictions, key=label_predictions.get)
            predictions.append(predicted_label)
        return predictions
