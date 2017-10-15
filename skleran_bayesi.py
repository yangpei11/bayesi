# *-* coding:utf-8 *-*
from numpy import *
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import GaussianNB,MultinomialNB
from sklearn.svm import SVC

def loadDataSet(filename):
	j = 0
	dataMat = []; labelMat = []
	f = open(filename)
	for line in f.readlines():
		lineArr = line.strip().split(',')
		labelMat.append( ord(lineArr[0])-ord('A') )
		vector = []
		for i in range(1, len(lineArr)):
			vector.append( int(lineArr[i]) )
		dataMat.append(vector)
		#j += 1
		#if(j == 10):
			#break
	return dataMat, labelMat

count_vec = TfidfVectorizer(binary = False, decode_error = 'ignore',\
	stop_words = 'english')

dataMat, labelMat = loadDataSet('letter-recognition.data')
x_train, x_test, y_train, y_test = train_test_split(dataMat, labelMat, test_size=0.2)

#x_train = count_vec.fit_transform(x_train)
#x_test = count_vec.transform(x_test)

clf  = GaussianNB().fit(x_train, y_train)
doc_class_predicted = clf.predict(x_test)

#clf = SVC()
#clf.fit(x_train, y_train)
#doc_class_predicted = clf.predict(x_test)

print( mean(doc_class_predicted == y_test) )