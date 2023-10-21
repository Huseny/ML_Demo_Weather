from decimal import Decimal


class NaiveBayes:
    def calculate_posterior(self, priors, likelihoods, instance, laplace_smoothing):
        probabilities = {}
        for class_value, prior in priors.items():
            probability = Decimal(prior)
            for j in range(len(instance)):
                feature_value = instance[j]
                if feature_value in likelihoods[class_value][j]:
                    probability = Decimal(probability) * Decimal(
                        likelihoods[class_value][j][feature_value]
                    )
                else:
                    probability = Decimal(probability) * (
                        Decimal(laplace_smoothing)
                        / Decimal(laplace_smoothing * len(likelihoods[class_value][j]))
                        + Decimal(laplace_smoothing)
                    )

            probabilities[class_value] = probability
        return probabilities

    def classify_instances(self, priors, likelihoods, test_features, laplace_smoothing):
        predictions = []
        for instance in test_features:
            probabilities = self.calculate_posterior(
                priors, likelihoods, instance, laplace_smoothing
            )
            max_probability = Decimal("-inf")
            predicted_class = None
            for class_value, probability in probabilities.items():
                if probability > max_probability:
                    max_probability = probability
                    predicted_class = class_value
            predictions.append(predicted_class)
        return predictions
