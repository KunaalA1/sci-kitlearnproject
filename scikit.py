  import json
import random
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
#there are many models of data classifiers this model is called support vector machine(svm) classifier
#naive bayes
#logistic regression
#decision tree
from sklearn.svm import SVC
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV



class Sentiment:
	NEGATIVE = "NEGATIVE"
	NEUTRAL = "NEUTRAL"
	POSITIVE = "POSITIVE"



class Review:
	def __init__(self, text, score):
		self.text = text
		self.score = score
		self.sentiment = self.get_sentiment()

	def get_sentiment(self):
		if self.score <= 2:
			return Sentiment.NEGATIVE
		elif self.score == 3:
			return Sentiment.NEUTRAL
		else:
			return Sentiment.POSITIVE




class ReviewContainer:
	

	def __init__(self, reviews):
		self.reviews = reviews
	

	def get_text(self):
		return [x.text for x in self.reviews]

	def get_sentiment(self):

		return [x.sentiment for x in self.reviews]


	def evenly_distribute(self):
		negative = list(filter(lambda x: x.sentiment == Sentiment.NEGATIVE, self.reviews))
		positive = list(filter(lambda x: x.sentiment == Sentiment.POSITIVE, self.reviews))

		positive_shrunk = positive[:len(negative)]
		self.reviews = negative + positive_shrunk
		random.shuffle(self.reviews)

		print(negative[0].text)
		print(len(negative))
		print(len(positive))



file_name = "Books_small_10000.json"

reviews = []

with open(file_name) as f:
	
	for line in f:
		
		review = json.loads(line)
		
		reviews.append(Review(review["reviewText"], review['overall']))




training, test = train_test_split(reviews, test_size = 0.33, random_state = 42)
train_container = ReviewContainer(training)
test_container = ReviewContainer(test)





train_x = [x.text for x in training]
train_y = [x.sentiment for x in training]

test_x = [x.text for x in  test]
test_y = [x.sentiment for x in test]



train_y.count(Sentiment.POSITIVE)
train_y.count(Sentiment.NEGATIVE)


train_container.evenly_distribute()
test_container.evenly_distribute()
len(train_container.reviews)

train_x = train_container.get_text()
train_y = train_container.get_sentiment()


test_x = test_container.get_text()
test_y = test_container.get_sentiment()

print(train_y.count(Sentiment.POSITIVE))
print(train_y.count(Sentiment.NEGATIVE))



vectorizer = CountVectorizer()
train_x_vectors = vectorizer.fit_transform(train_x)
test_x_vectors = vectorizer.transform(test_x)


#checking which data classifier is best
#SVM Classifier test 
clf_svm = svm.SVC(kernel = "linear")
clf_svm.fit(train_x_vectors,train_y)
test_x[0]
print(clf_svm.predict(test_x_vectors[0]))


#Decision Tree Classifier Test
clf_dec = DecisionTreeClassifier()
clf_dec.fit(train_x_vectors, train_y)
print(clf_dec.predict(test_x_vectors[0]))

clf_gnb = GaussianNB()
clf_gnb.fit(train_x_vectors.todense(), train_y)
clf_gnb.predict(test_x_vectors[0].todense())

clf_log = LogisticRegression()
clf_log.fit(train_x_vectors.todense(), train_y)
clf_log.predict(test_x_vectors[0].todense())

print(train_x[0])
print(train_x_vectors[0].toarray())

print(clf_svm.score(test_x_vectors, test_y))
#f1 score measures are more important
print(f1_score(test_y,clf_svm.predict(test_x_vectors), average = None, labels = [Sentiment.POSITIVE,Sentiment.NEUTRAL, Sentiment.NEGATIVE]))
print(train_x[0])
print(train_y.count(Sentiment.NEGATIVE))

#gridsearch can help us choose which hyperparameter is best.

parameters = {'kernel' : ('linear', 'rbf'), 'C' : (1,4,8,16,32)}
svc = svm.SVC()
clf = GridSearchCV(svc,parameters, cv = 5)
print(clf.fit(train_x_vectors, train_y))


test_set = ["very awesome book", "bad book do not buy", "horrible waste of time"]
new_test = vectorizer.transform(test_set)

print(clf_svm.predict(new_test))



