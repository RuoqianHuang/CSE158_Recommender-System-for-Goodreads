#!/usr/bin/env python
# coding: utf-8

# In[97]:


import gzip
from collections import defaultdict
import math
import scipy.optimize
from sklearn import svm
import numpy
import string
import random
import string
from sklearn import linear_model


# In[98]:


import warnings
warnings.filterwarnings("ignore")


# In[99]:


def assertFloat(x):
    assert type(float(x)) == float

def assertFloatList(items, N):
    assert len(items) == N
    assert [type(float(x)) for x in items] == [float]*N


# In[100]:


def readGz(path):
    for l in gzip.open(path, 'rt'):
        yield eval(l)


# In[101]:


def readCSV(path):
    f = gzip.open(path, 'rt')
    f.readline()
    for l in f:
        u,b,r = l.strip().split(',')
        r = int(r)
        yield u,b,r


# In[102]:


answers = {}


# # Some data structures that will be useful

# In[7]:


allRatings = []
for l in readCSV("train_Interactions.csv.gz"):
    allRatings.append(l)


# In[8]:


len(allRatings)


# In[9]:


ratingsTrain = allRatings[:190000]
ratingsValid = allRatings[190000:]
ratingsPerUser = defaultdict(list)
ratingsPerItem = defaultdict(list)
for u,b,r in ratingsTrain:
    ratingsPerUser[u].append((b,r))
    ratingsPerItem[b].append((u,r))


# ##################################################
# # Read prediction                                #
# ##################################################

# In[10]:


# Copied from baseline code
bookCount = defaultdict(int)
totalRead = 0

for user,book,_ in readCSV("train_Interactions.csv.gz"):
    bookCount[book] += 1
    totalRead += 1

mostPopular = [(bookCount[x], x) for x in bookCount]
mostPopular.sort()
mostPopular.reverse()

return1 = set()
count = 0
for ic, i in mostPopular:
    count += ic
    return1.add(i)
    if count > totalRead/2: break


# ### Question 1

# In[67]:


allBooks = set()
for u,b,r in allRatings:
    allBooks.add(b)
booksPerUser = defaultdict(set)
for u,b,r in ratingsTrain:
    booksPerUser[u].add(b)


# In[68]:


negSamples = []
for u,b,r in ratingsValid:
    bookRange = allBooks - booksPerUser[u]
    negativeBook = random.sample(bookRange,k=1)[0]
    negSamples.append((u,negativeBook,0))


# In[69]:


ratingsValidLabel = []
for u,b,r in ratingsValid:
    ratingsValidLabel.append((u,b,1))
newValid = ratingsValidLabel + negSamples


# In[70]:


len(newValid)


# In[71]:


# check whether the book in newValid is in return1
predictions = []
y = []
for u,b,r in newValid:
    y.append(r)
    if (b in return1):
        predictions.append(1)
    else:
        predictions.append(0)


# In[72]:


# compare predictions with newValid
pred = numpy.array(predictions)
correct = pred == y # Binary vector indicating which predictions were correct
acc1 = sum(correct) / len(correct)


# In[73]:


acc1


# In[74]:


answers['Q1'] = acc1


# In[75]:


assertFloat(answers['Q1'])


# ### Question 2

# In[93]:


# Copied from baseline code
bookCount = defaultdict(int)
totalRead = 0

for user,book,_ in readCSV("train_Interactions.csv.gz"):
    bookCount[book] += 1
    totalRead += 1

mostPopular = [(bookCount[x], x) for x in bookCount]
mostPopular.sort()
mostPopular.reverse()

newReturn1 = set()
count = 0
for ic, i in mostPopular:
    count += ic
    newReturn1.add(i)
    if count > 1.5 * totalRead/2: break


# In[94]:


# check whether the book in newValid is in newReturn1
predictions = []
y = []
for u,b,r in newValid:
    y.append(r)
    if (b in newReturn1):
        predictions.append(1)
    else:
        predictions.append(0)


# In[95]:


# compare predictions with newValid
pred = numpy.array(predictions)
correct = pred == y # Binary vector indicating which predictions were correct
acc2 = sum(correct) / len(correct)


# In[96]:


threshold = 1.5/2


# In[97]:


answers['Q2'] = [threshold, acc2]


# In[98]:


assertFloat(answers['Q2'][0])
assertFloat(answers['Q2'][1])


# In[99]:


answers['Q2']


# ### Question 3/4

# In[100]:


usersPerBook = defaultdict(set)
for u,b,r in ratingsTrain:
    usersPerBook[b].add(u)
allUsers = set()
for u,b,r in allRatings:
    allUsers.add(u)


# In[101]:


def Jaccard(s1, s2):
    numer = len(s1.intersection(s2))
    denom = len(s1.union(s2))
    if denom == 0:
        return 0
    return numer / denom


# In[102]:


predictions3 = []
for u,b,r in newValid:
    maxSim = 0
    for uRead in booksPerUser[u]:
        s1 = usersPerBook[b]
        s2 = usersPerBook[uRead]
        sim = Jaccard(s1, s2)
        if (sim > maxSim):
            maxSim = sim
    if (maxSim > 0.003 and len(ratingsPerItem[b]) > 30):
        predictions3.append(1)
    else:
        predictions3.append(0)


# In[103]:


len(predictions3)


# In[104]:


# compare predictions with newValid
pred3 = numpy.array(predictions3)
correct = pred3 == y # Binary vector indicating which predictions were correct
acc3 = sum(correct) / len(correct)


# In[105]:


acc3


# In[110]:


predictions4 = []
for u,b,r in newValid:
    maxSim = 0
    for uRead in booksPerUser[u]:
        s1 = usersPerBook[b]
        s2 = usersPerBook[uRead]
        sim = Jaccard(s1, s2)
        if (sim > maxSim):
            maxSim = sim
    if ((maxSim > 0.003 and b in newReturn1) or len(ratingsPerItem[b]) > 30):
        predictions4.append(1)
    else:
        predictions4.append(0)


# In[111]:


len(predictions4)


# In[112]:


# compare predictions with newValid
pred4 = numpy.array(predictions4)
correct = pred4 == y # Binary vector indicating which predictions were correct
acc4 = sum(correct) / len(correct)


# In[113]:


acc4


# In[114]:


answers['Q3'] = acc3
answers['Q4'] = acc4


# In[115]:


assertFloat(answers['Q3'])
assertFloat(answers['Q4'])


# In[116]:


predictions = open("predictions_Read.csv", 'w')
for l in open("pairs_Read.csv"):
    if l.startswith("userID"):
        predictions.write(l)
        continue
    u,b = l.strip().split(',')
    # (etc.)
    maxSim = 0
    for uRead in booksPerUser[u]:
        s1 = usersPerBook[b]
        s2 = usersPerBook[uRead]
        sim = Jaccard(s1, s2)
        if (sim > maxSim):
            maxSim = sim
    if ((maxSim > 0.003 and b in newReturn1) or len(ratingsPerItem[b]) > 30):
        predictions.write(u + ',' + b + ",1\n")
    else:
        predictions.write(u + ',' + b + ",0\n")
predictions.close()


# In[117]:


answers['Q5'] = "I confirm that I have uploaded an assignment submission to gradescope"


# In[118]:


assert type(answers['Q5']) == str


# ##################################################
# # Category prediction (CSE158 only)              #
# ##################################################

# ### Question 6

# In[26]:


stop_words = {'a',
 'about',
 'above',
 'after',
 'again',
 'against',
 'ain',
 'all',
 'am',
 'an',
 'and',
 'any',
 'are',
 'aren',
 "aren't",
 'as',
 'at',
 'be',
 'because',
 'been',
 'before',
 'being',
 'below',
 'between',
 'both',
 'but',
 'by',
 'can',
 'couldn',
 "couldn't",
 'd',
 'did',
 'didn',
 "didn't",
 'do',
 'does',
 'doesn',
 "doesn't",
 'doing',
 'don',
 "don't",
 'down',
 'during',
 'each',
 'few',
 'for',
 'from',
 'further',
 'had',
 'hadn',
 "hadn't",
 'has',
 'hasn',
 "hasn't",
 'have',
 'haven',
 "haven't",
 'having',
 'he',
 'her',
 'here',
 'hers',
 'herself',
 'him',
 'himself',
 'his',
 'how',
 'i',
 'if',
 'in',
 'into',
 'is',
 'isn',
 "isn't",
 'it',
 "it's",
 'its',
 'itself',
 'just',
 'll',
 'm',
 'ma',
 'me',
 'mightn',
 "mightn't",
 'more',
 'most',
 'mustn',
 "mustn't",
 'my',
 'myself',
 'needn',
 "needn't",
 'no',
 'nor',
 'not',
 'now',
 'o',
 'of',
 'off',
 'on',
 'once',
 'only',
 'or',
 'other',
 'our',
 'ours',
 'ourselves',
 'out',
 'over',
 'own',
 're',
 's',
 'same',
 'shan',
 "shan't",
 'she',
 "she's",
 'should',
 "should've",
 'shouldn',
 "shouldn't",
 'so',
 'some',
 'such',
 't',
 'than',
 'that',
 "that'll",
 'the',
 'their',
 'theirs',
 'them',
 'themselves',
 'then',
 'there',
 'these',
 'they',
 'this',
 'those',
 'through',
 'to',
 'too',
 'under',
 'until',
 'up',
 've',
 'very',
 'was',
 'wasn',
 "wasn't",
 'we',
 'were',
 'weren',
 "weren't",
 'what',
 'when',
 'where',
 'which',
 'while',
 'who',
 'whom',
 'why',
 'will',
 'with',
 'won',
 "won't",
 'wouldn',
 "wouldn't",
 'y',
 'you',
 "you'd",
 "you'll",
 "you're",
 "you've",
 'your',
 'yours',
 'yourself',
 'yourselves', 'book', 'read', 'story', 'one', 'like', 'really', 'characters', 'character', 'series', 'love', 'first', 'books', 'much', 'many', 'reading', 'im', 'good', 'great', 'well', 'also', 'will', 'would',}


# In[114]:


data = []

for d in readGz("train_Category.json.gz"):
    data.append(d)


# In[115]:


data[0]


# In[116]:


train = data[:90000]
valid = data[90000:]


# In[154]:


# Just build our feature vector by taking the most popular words (lowercase, punctuation removed, but no stemming)
wordCount = defaultdict(int)
punctuation = set(string.punctuation)
for d in train:
    r = ''.join([c for c in d['review_text'].lower() if not c in punctuation])
    for w in r.split():
        if (w in stop_words): continue
        wordCount[w] += 1

counts = [(wordCount[w], w) for w in wordCount]
counts.sort()
counts.reverse()


# In[155]:


words = [x[1] for x in counts[:80000]]


# In[156]:


from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(max_features=80000, vocabulary=words)


# In[157]:


reviews = []
for d in train:
    reviews.append(d['review_text'])


# In[158]:


X = vectorizer.fit_transform(reviews)
y = [d['genreID'] for d in train]


# In[159]:


model = linear_model.LogisticRegression(C=1)
model.fit(X, y)


# # Run on test set

# In[160]:


dataTest = []
for d in readGz("test_Category.json.gz"):
    dataTest.append(d)


# In[161]:


dataTest[0]


# In[162]:


Xtest_reviews = [d['review_text'] for d in dataTest]
Xtest = vectorizer.fit_transform(Xtest_reviews)
pred_test = model.predict(Xtest)


# In[163]:


predictions = open("predictions_Category.csv", 'w')
pos = 0

for l in open("pairs_Category.csv"):
    if l.startswith("userID"):
        predictions.write(l)
        continue
    u,b = l.strip().split(',')
    _ = predictions.write(u + ',' + b + ',' + str(pred_test[pos]) + '\n')
    pos += 1

predictions.close()


# In[ ]:
