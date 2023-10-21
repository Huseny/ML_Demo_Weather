class Experiment:
    def __init__(self):
        pass

    def load_csv(self, file_path):
        data = []
        with open(file_path, "r") as file:
            lines = file.readlines()
            for line in lines:
                row = line.strip().split(",")
                data.append(row)
        return data

    def preprocess_data(self, data):
        features = []
        labels = []
        for row in data:
            features.append(row[:-1])
            labels.append(row[-1])
        return features, labels

    def calculate_class_priors(self, labels):
        class_counts = {}
        total_instances = len(labels)

        for value in labels:
            if value in class_counts:
                class_counts[value] += 1
            else:
                class_counts[value] = 1

        priors = {}
        for class_value, count in class_counts.items():
            priors[class_value] = count / total_instances

        return priors

    def calculate_likelihoods(self, features, labels, laplace_smoothing):
        likelihoods = {}
        feature_counts = {}
        class_values = set(labels)
        total_classes = len(class_values)

        for i in range(len(features)):
            for j in range(len(features[i])):
                feature_value = features[i][j]
                class_value = labels[i]

                if class_value not in likelihoods:
                    likelihoods[class_value] = {}

                if j not in likelihoods[class_value]:
                    likelihoods[class_value][j] = {}

                if j not in feature_counts:
                    feature_counts[j] = {}

                if feature_value not in feature_counts[j]:
                    feature_counts[j][feature_value] = 0

                if feature_value not in likelihoods[class_value][j]:
                    likelihoods[class_value][j][feature_value] = 0

                feature_counts[j][feature_value] += 1
                likelihoods[class_value][j][feature_value] += 1

        for class_value in likelihoods:
            for j in likelihoods[class_value]:
                for feature_value in likelihoods[class_value][j]:
                    count = likelihoods[class_value][j][feature_value]
                    likelihoods[class_value][j][feature_value] = (
                        count + laplace_smoothing
                    ) / (
                        feature_counts[j][feature_value]
                        + laplace_smoothing * total_classes
                    )

        return likelihoods

    def extract_features_labels(self, data):
        features = []
        labels = []
        for row in data:
            features.append(row[:-1])
            labels.append(row[-1])
        return features, labels

    def one_hot_encode(self, features):
        all_categories = set()
        for row in features:
            for val in row:
                all_categories.add(val)

        all_categories = sorted(list(all_categories))
        encoded_features = []
        for row in features:
            encoded_row = [0] * len(all_categories)
            for i, category in enumerate(all_categories):
                if category in row:
                    encoded_row[i] = 1
            encoded_features.append(encoded_row)

        return encoded_features

    def encode_labels(self, labels):
        encoded_labels = []
        for label in labels:
            encoded_labels.append(label)
        return encoded_labels

    def calculate_accuracy(self, actual, predicted):
        correct = 0
        for i in range(len(actual)):
            if actual[i] == predicted[i]:
                correct += 1
        return (correct / len(actual)) * 100

    def calculate_accuracy_logistic(self, test_labels_encoded, test_predictions):
        accuracy = sum(
            1
            for i in range(len(test_labels_encoded))
            if test_labels_encoded[i] == test_predictions[i]
        ) / len(test_labels_encoded)
        return accuracy * 100
