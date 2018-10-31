# Yelp-review-ratings-analysis:

Its a model for analysis and prediction of ratings and reviews using different machine learning algorithms. Its also an attempt to give more precise ratings based on customer reviews rather than customer given ratings due to findings that suggest that reviews and ratings given has very less covariance based on the words used in reviews.

For this, two models are considered, one which has text-based features and other has numerical features which are extracted and derived from the reviews and corresponding ratings with the use of libraries like NLTK, numpy and AFFIN scores. It analyze their performance to see which predicts better.

For the project, the database of Yelp reviews is taken and analyzed. I used the upvotes(cool, funny and useful) to collect only the reviews that are meaningful and actually reflect the rating. Yelp mentioned that good reviews receive more number of upvotes, so used that as a metric in pre-processing.

Some other preprocessing steps like, removing stop words, stemming etc. are performed for pre-processing. For gathering numerical features out of text, I used AFINN score which is a list containing 2500 words with values from -5 to 5 to convey the wight of the word. Along with this I developed a similar list of words and values, based on the available dataset, which gives all the words in all the reviews a value from 0 to 5.

For the text-based analysis, I used NBClassifier model and for numerical features we used Logistic regression model.

## Files:

- filtered_yelp1.csv: filtered dataset (original dataset obtained from: https://www.kaggle.com/yelp-dataset/yelp-dataset).
- Dataset Filtering.ipynb: Filtering of the dataset.
- NB_Classifier.ipynb: Implementaion and analysis of NBClassifier model on the dataset.
- Logistic_regression.ipynb: Implementaion and analysis of Logistic regression model on the dataset.
- avgVal.txt, weightSum.txt: files dervied and generated for the Logistic_regression.ipynb.
