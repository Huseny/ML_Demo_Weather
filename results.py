from experiment import Experiment
from naive_bayes import NaiveBayes
from logistic_regression import MulticlassLogisticRegression
from matplotlib import pyplot as plt


def try_naive_bayes(e: Experiment):
    train_data = e.load_csv("train.csv")
    test_data = e.load_csv("test.csv")

    train_features, train_target = e.preprocess_data(train_data)
    test_features, test_target = e.preprocess_data(test_data)

    priors = e.calculate_class_priors(train_target)

    smoothing = [0.001, 0.01, 0.1, 0.2, 0.3, 0.5, 0.6, 0.8, 1, 2, 4]
    accuracies = []
    naive_bayes = NaiveBayes()
    for laplace_smoothing in smoothing:
        likelihoods = e.calculate_likelihoods(
            train_features, train_target, laplace_smoothing
        )

        predictions = naive_bayes.classify_instances(
            priors, likelihoods, test_features, laplace_smoothing
        )

        accuracy = e.calculate_accuracy(test_target, predictions)
        accuracies.append(accuracy)

    plt.plot(smoothing, accuracies, marker="o")
    plt.xlabel("Smoothing")
    plt.ylabel("Accuracy (%)")
    plt.title("Smoothing vs Accuracy in Naive Bayes for Demo Weather Dataset")
    plt.grid(True)
    plt.show()


def try_logistic(e: Experiment):
    train_data = e.load_csv("./train.csv")[1:]
    test_data = e.load_csv("./test.csv")[1:]

    train_features, train_labels = e.extract_features_labels(train_data)
    test_features, test_labels = e.extract_features_labels(test_data)

    train_features_encoded = e.one_hot_encode(train_features)
    test_features_encoded = e.one_hot_encode(test_features)

    train_labels_encoded = e.encode_labels(train_labels)
    test_labels_encoded = e.encode_labels(test_labels)

    logistic_regression = MulticlassLogisticRegression()

    learning_rates = [0.001, 0.01, 0.1, 0.5, 1.0, 1.5, 2, 5, 10]
    accuracies = []

    for learning_rate in learning_rates:
        logistic_regression.train(
            train_features_encoded,
            train_labels_encoded,
            learning_rate=learning_rate,
            num_iterations=100,
        )

        predictions = logistic_regression.predict(test_features_encoded)
        accuracy = e.calculate_accuracy_logistic(test_labels_encoded, predictions)
        accuracies.append(accuracy)

    plt.plot(learning_rates, accuracies, marker="o")
    plt.xlabel("Learning Rates")
    plt.ylabel("Accuracy (%)")
    plt.title(
        "Learning Rate vs Accuracy in Logistic Regression for Demo Weather Dataset"
    )
    plt.grid(True)
    plt.show()


e = Experiment()
try_logistic(e)
