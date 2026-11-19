# Python for Data Work

## NumPy Arrays
NumPy arrays store data of one type in a fixed shape and support fast vectorised operations. Broadcasting allows operations between arrays of different shapes when the shapes are compatible. Reshaping changes the shape without changing the data. Vectorised code is usually much faster than Python loops over individual elements.

## Pandas DataFrames
A Pandas DataFrame is a table with labelled rows and columns. Common operations are filtering rows, selecting columns, grouping by a column, and merging two tables on a shared key. Missing values can be found with isna and handled by dropping rows or filling them with a statistic such as the median.

## Scikit-learn Workflow
The standard Scikit-learn workflow is: split the data, fit a model on the training set, predict on the test set, and score the predictions. The function train_test_split from sklearn.model_selection performs the split. Pipelines chain preprocessing and modelling steps so the same steps are applied to training and test data without leakage.

## Working with CSV Files
Pandas read_csv loads a CSV file into a DataFrame. The first row is treated as the header by default. Arguments such as usecols, dtype, and parse_dates control which columns are loaded and how dates are parsed. Writing back to disk uses to_csv, with index set to False to avoid saving the row index as an extra column.

## Visualisation Basics
Matplotlib is the base plotting library in Python. A line plot shows a trend over time, a bar chart compares categories, and a histogram shows the distribution of a numeric variable. Seaborn builds on Matplotlib and adds statistical plots such as box plots and heatmaps with less code.

## Reproducibility
Set a random_state in train_test_split and in models that use randomness, so results can be reproduced. Record library versions in requirements.txt. Keep raw data separate from processed data, and keep notebooks and scripts under version control with Git.
