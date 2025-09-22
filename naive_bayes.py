import numpy as np

class NaiveBayesTextClassifier:
    """
    Naive Bayes for text classification based on numeric input (e.g., TF-IDF, manual embedding).
    Uses the Multinomial approach with Laplace smoothing.

    Assumptions:
        - X: array (n_samples, n_features), non-negative values
        - y: binary labels (0 or 1)
    """

    def __init__(self):
        self.class_log_prior = None     # log(P(class))
        self.feature_log_prob = None    # log(P(feature|class))
        self.num_classes = 2

    def fit(self, X, y):
        """
        Trains the model by computing the prior and likelihood for each class.

        Args:
            X: np.array, shape (n_samples, n_features)
            y: np.array, shape (n_samples, 1)
        """
        y = y.flatten()
        n_samples, n_features = X.shape
        self.class_log_prior = np.zeros(self.num_classes)
        self.feature_log_prob = np.zeros((self.num_classes, n_features))

        for c in range(self.num_classes):
            X_c = X[y == c]
            class_count = X_c.shape[0]
            self.class_log_prior[c] = np.log(class_count / n_samples)

            # Compute likelihoods with Laplace smoothing
            #  Step 1: Compute total word frequency per feature for class `c`: sum over samples in class
            #  Step 2: Add 1 to every count for Laplace smoothing to avoid log(0)
            #  Step 3: Normalize by the total smoothed word count
            #  Step 4: Take log to prepare for log-probability summation

            alpha = 1.0
            feature_counts = X_c.sum(axis=0) ## normalised feature count sum of row
            denom = feature_counts.sum() + alpha * n_features


            for j in range(n_features):
                j_prob = ((feature_counts[j]) + alpha) / denom
                self.feature_log_prob[c, j] = np.log(j_prob)



    def predict(self, X):
        """
        Predicts the class based on log probabilities.

        Args:
            X: np.array, shape (n_samples, n_features)
        Returns:
            pred: np.array of predicted labels (0 or 1)
        """
        log_probs = []
        for c in range(self.num_classes):
            # Implement Naive Bayes prediction formula:
            #  Step 1: Use class_log_prior[c] → log P(class c)
            #  Step 2: Add dot product between input X and log P(feature|class=c)
            #  Step 3: This gives log posterior score for class c
            class_prob = self.class_log_prior[c] + X @ self.feature_log_prob[c]
            log_probs.append(class_prob)
            
        #  Choose class with the highest log-probability
        #  Stack both class scores and take argmax to get predicted class
        log_probs = np.column_stack(log_probs)
        return np.argmax(log_probs, axis = 1)
