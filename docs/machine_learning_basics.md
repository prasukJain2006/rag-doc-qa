# Machine Learning Basics

## Supervised and Unsupervised Learning
Supervised learning uses labelled data, where each input has a known output. Classification predicts a category, such as spam or not spam. Regression predicts a number, such as a house price. Unsupervised learning uses unlabelled data. Clustering groups similar examples together, and dimensionality reduction compresses features while keeping the main structure.

## Train and Test Split
A common split is 80 percent for training and 20 percent for testing. The test set must not be used to choose the model, otherwise the reported score is optimistic. This mistake is called data leakage. Always split before scaling or feature selection that learns from the data.

## Overfitting and Underfitting
Overfitting happens when a model memorises the training data and fails on new data. Underfitting happens when a model is too simple to capture the pattern. Regularisation, more training data, and simpler models can reduce overfitting. Cross-validation helps detect it.

## Cross-Validation
In 5-fold cross-validation, the data is divided into 5 equal parts. The model is trained on 4 parts and tested on the remaining part, and this is repeated 5 times so each part is tested once. The average score across folds is more reliable than a single split.

## Evaluation Metrics
Accuracy is the share of correct predictions, but it is misleading for imbalanced data. Precision measures how many predicted positives were truly positive. Recall measures how many true positives were found. F1-score is the harmonic mean of precision and recall. For regression, common metrics are mean squared error and R-squared.

## Feature Scaling
Many models work better when features are on a similar scale. Standardisation subtracts the mean and divides by the standard deviation. Normalisation scales values into a fixed range, often 0 to 1. Tree-based models are less sensitive to scaling than distance-based models.
