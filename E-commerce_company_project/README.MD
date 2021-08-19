# Some parts of projects in e-commerce company

## SelectWords.py
###### Background of the project: 
  This is an automated promotion project. 
  We need to determine the keywords for the next purchase based on the effects we have previously spent on different keywords on the online shopping platform.
  So we need to sort out the return of historical keywords (including exposure, click-through rate, purchase conversion rate, etc.), and use multiple simple machine learning models to predict future performance.
  Finally, the prediction results of the model with the lowest MSE and the highest R2 are selected as the most likely performance of the keyword in the future.
##### Code usage：
  Use simple multiple models to make basic predictions on the future performance of keywords and provide effective keyword pools for the promotion department.



## app_func.py
##### Background of the project:
  The company hopes to create a new website application to help brands understand the reasons for their returns.
  Behind this new website application, the functions that need to be provided include: word cloud, return reason text topic extraction, word frequency analysis, etc.
##### Code usage:
  Provide packaged functions to front-end engineers. 
  Provide background word cloud image generation, word frequency statistics, synonym dictionary configuration and LSI model generation and result output.


## Return_Order_FollowUp.sql
##### Backbround of the project:
  E-commerce merchants want to know the follow-ups of the return order.
  It is necessary to query through tables of the database to understand the product that the same user has placed the most recent order within 30 days after the return.
  The resulting data can be used to analyze the efficiency of re-ordering after the return and the reason for the replacement.
##### Code usage:
  MySQL script, used to query the four tables (return order table, all order table, product table, order product table) by using join.
