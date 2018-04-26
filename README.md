# customer_retention
Machine learning and logistic regression predictions of retained customers

This code runs a customer retention analysis of customers over the course of 6 months. The data include 50,000 customer records in 3 cities, and 12 factors that can be used to predict retention.

The goals of the analysis are:
  • Perform an exploratory analysi and visualizations to understand the data.
  • Build a model to predict whether or not a customer will be active in their 6th month after joining.

The analysis consists of:
  • Exploratory summary stats, pairs plot, select boxplots
  • Partition data into training & test sets
  • Logistic Regression model
     	• identify key factors
      • assess model prediction accuracy
        • ROC & test set confusion matrix
  • Machine Learning: Random Forest model
      • assess model prediction accuracy
        • Test set confusion matrix
      • Variable importance (compare with logistic regression)

The analysis was originally done late 2016. I made minor updates to post it here.
