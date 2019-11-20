# Project: Web Scraping Reddit 
---

### Table of Contents

- [Problem Statement](#Problem-Statement)
- [Executive Summary](#Executive-Summary)
- [Data set](#Data-set)
- [Evaluation of Model](#Evaluation-of-Model)
- [Conclusion](#Conclusion)
- [Executive Summary](#Executive-Summary)


## Problem Statement

When writing and publishing an article in the news, it is important to have great headlines to draw the reader's attention. As a small publisher, we want to emphasize on positivity of news all over the world. Our stories and headlines should be different and unique compared to other headlines. Our readers should feel the positivity we want to convey when they read our headlines. 

The problem we are trying to answer here is: **Are there any distinct keywords in titles, that would make someone perceive news as positive?**

Reddit is a network of communities based on people's interests. We are going to use the posts of `r/worldnews` and `r/positive_news` to analyze this problem. Titles of each Subreddit are being extracted and analyzed using classification models, Multinomial Naive Bayes and Logistic Regression. We will use key words in titles to predict whether the title are listed in `r/positive_news` or `r/worldnews`. Our evaluation metrics will be accuracy score. False positive/false negative in this case are equally bad. Therefore, accuracy is likely the best metric to use to measure success for our models.

## Executive Summary 

In the [notebook](./code/reddit_project.ipynb), the first part of the project was to use web scraping to collect data from two Subreddits, r/worldnews and r/positive_news. By using a test case, we determined that the information we needed was located in the title of the posts. After pulling the data, the information was then saved in a new data frame. Punctuation and duplicates were then removed during data cleaning. Exploratory data analysis was then conducted by counting the word frequency to determine if additional cleaning or preprocessing was required. Our model was predicting, whether a title was from r/positive_news or not by using key words in the training data set.

The following classification models were used to analyze our data:

- Multinomial Naive Bayes with CountVectorizer and TfidfVectorizer

- Logistic Regression with CountVectorizer and TfidfVectorizer

Additionally, GridSearch and Pipeline were used to find out the best parameters for these models. Accuracy score was then used to evaluate these models. Since in this case any misclassification is equally bad, accuracy score is the likely the best metric to evaluate these model. We also further investigated our worst and best model to understand our incorrect predictions better by looking at coefficients as well as misclassification. 

Conclusions and results were successfully obtained to answer our problem statement and a presentation was created and presented [here](./presentation.pdf). 

## Data set

The data composes of `r/worldnews` and r/`positive_news` with titles and a total of 1992 posts. 
- csv file can be found [here](./data/subreddit.csv)

## Evaluation of Model 

_All scores here are accuracy scores._

Model|Vectorizer|Train Score|Test Score|
---|---|---|---|
Baseline||0.568|0.570
**MB Naive Bayes**| **Tfidf**|**0.968**|**0.866**
MB Naive Bayes| Count|0.979|0.863
Logistic Regression| Tfidf|0.926|0.835
Logistic Regression| Count|0.996|0.866

Comparing all these models we can see that our best model is Multinomial Naive Bayes Model with `TfidfVectorizer` with an accuracy score of 97% in training and accuracy score of 87% in testing. The training accuracy scores are very high for each of the models, but all of the models did not do well in unfamiliar environments. However, all our models performed much better than the baseline model.

## Conclusion

In conclusion, our best predictive model was MB Naive Bayes model, where we predicted an accuracy score of 95% in training and accuracy score 87% in testing. Surprisingly, our best model used n-gram range of 1, which means that it predicted our outcome only by using single words. 

- One disadvantage of the Naive Bayes model is that it assumes that all the words are independent of each other. Therefore, grammar, order and structure in the text are lost with this model. However, sometimes in titles, context of the whole corpus plays a big role in determining how the titles are perceived by humans. This is a big downside of the model. 
- On the other hand, the model is very efficient and great with many features. In this case we used 3000 features in our model. It outperformed our logistic regression models, because it is not sensitive to outliers due to variable independence . 

During the entire analysis of predicting our `r/worldnews` and `r/positive_news`, we saw that world news titles have the trend of talking about politics and government in many different countries, whereas positive news had a lot of information about the environment. Therefore to answer our problem, we can conclude that there are possible keywords in headlines that would make someone perceive news as positive. If we want to have our readers perceive news as positive, we should write more about the environment instead of politics and government is our recommendation. 

The next step would be improving our models by including more diverse data to see if there are more features, not just environment, that would give us better recommendations. Including more data about others topics might diversify our results even further. During modeling n-gram range 2 showed in EDA that there was significant differences between the titles, however, our model was not able to predict great results based on that. With this in mind, by adding more data, we might be able to tune our parameters better such as lowering maximum features and increasing n-gram range to add more context, structure and order to our prediction model. Additionally, another way to improve our model is using more a advanced classification model, where we won't lose the context of the corpus when analyzing the data.
