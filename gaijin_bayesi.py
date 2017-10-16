# *-* coding:utf-8 *-*
from numpy import *
from math import log

'''
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
	return dataMat, labelMat


dataMat, labelMat = loadDataSet('letter-recognition.data')
dataMat = array(dataMat)
labelMat = array(labelMat)

dataMat, testDataMat, labelMat, testLabelMat = train_test_split(dataMat, labelMat, test_size = 0.2)
'''
#返回训练集，测试集数据
def loadDataSet():
	f = open('letter-recognition.data')
	dic = {} #存放每个类别的特征矩阵
	X = [None]*26
	U = [None]*26
	labelNum = zeros(26)
	j = 0
	for line in f.readlines():
		lineArr = line.strip().split(',')
		vector = []
		for i in range(1, len(lineArr)):
			vector.append( int(lineArr[i]) )
		if(dic.get( lineArr[0] )  == None):
			dic[ lineArr[0] ] = []
			dic[ lineArr[0] ].append(vector)
		else:
			dic[ lineArr[0] ].append(vector)
			
	testDataMat = []; testLabelMat = []
	for i in dic:
		Xi = []
		for j in range( len(dic[i]) ):
			if(j < len(dic[i])*0.8 ):
				Xi.append( dic[i][j] )
			else:
 				testDataMat.append( dic[i][j] )
 				testLabelMat.append( ord(i) - ord('A') )
		Xi = array(Xi)
		labelNum[ ord(i) - ord('A') ] = len(Xi)  #每一类别的个数
		X[ ord(i) - ord('A') ] = mean(Xi, axis = 0) #每一类别的平均值向量
		U[ ord(i) - ord('A') ] = cov(Xi.T)  #每一类的协方差矩阵

	return testDataMat, testLabelMat, X, U, labelNum


testDataMat, testLabelMat, X, U, labelNum= loadDataSet()

print(labelNum)
sum1 = sum(labelNum)
KlableP = labelNum/sum1
print(sum1)

m, n = shape(testDataMat)
print(m)

count = 0
for i in range(m):
	predictClass = 0
	predictP = -99999

	for j in range(26):
		x = -0.5 * mat ( (testDataMat[i]-X[j]) )*  mat(U[j]).I * mat( (testDataMat[i]-X[j]) ).T + log( KlableP[j] ) - 0.5 * log( linalg.det( mat(U[j]) ) )	
		if(x[0][0] > predictP):
			predictP = x
			predictClass = j
	if(predictClass == testLabelMat[i]):
		count += 1		


print(count*1.0/m)

#(1, 16) *(16 ,16) * (16, 1)


