# CSE158_A1

task1: Reading prediction
description: 
I followed the same method from hw3 but built a new popular set called newReturn1 using the threshold from hw3 solution: "count > 1.5 * totalRead/2". I also incorporated the code from hw3 solution, which is "len(ratingsPerItem[b]) > 30", in addition to my original maximum similarity comparison and whether the item is in newReturn1.

task2: Category prediction
description:
I used the same method as hw3 to generate a dictionary that is ordered by the frequency of words. I also eliminate the stop words to get better accuracy. Compared to the dictionary size of 100, I increase it to 80000. Then I used sklearn.feature_extraction.text.TfidfVectorizer to encode the review texts and generate the model by logistic regression to predict the test data.
