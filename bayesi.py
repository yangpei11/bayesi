# *-* coding:utf-8 *-*
from numpy import *
from sklearn.cross_validation import train_test_split

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
	dic = {}
	for line in f.readlines():
		lineArr = line.strip().split(',')
		vector = []
		for i in range(1, len(lineArr)):
			vector.append( int(lineArr[i]) )
		if(dic.get( lineArr[0] )  == None):
			dic[ lineArr[0] ] = []
		else:
			dic[ lineArr[0] ].append(vector)
	dataMat = []; labelMat = []; testDataMat = []; testLabelMat = []
	for i in dic:
		for j in range( len(dic[i]) ):
			if(j < len(dic[i])*0.8 ):
				dataMat.append( dic[i][j] )
				labelMat.append( ord(i) - ord('A') )
			else:
 				testDataMat.append( dic[i][j] )
				testLabelMat.append( ord(i) - ord('A') )

	return dataMat, testDataMat, labelMat, testLabelMat


dataMat, testDataMat, labelMat, testLabelMat = loadDataSet()
dataMat = array(dataMat)
testDataMat = array(testDataMat)
labelMat = array(labelMat)
testLabelMat = array(testLabelMat)

'''
print(shape(dataMat))
print(shape(testDataMat))
print(shape(labelMat))
print(shape(testLabelMat))
'''


labelNum = zeros(26)
for i in range( len(labelMat) ):
	labelNum[ labelMat[i] ] = labelNum[ labelMat[i] ] + 1

m, n = shape(dataMat)

KtrainP = ones((26, 16, n))  #第k类的第n个属性出现数字（0-15）的个数

for i in range(m):
	for j in range(n):
		KtrainP[labelMat[i], dataMat[i, j], j]  = KtrainP[labelMat[i], dataMat[i, j], j] + 1

for i in range(26):
	for j in range(16):
		for k in range(n):
			KtrainP[i, j, k] = KtrainP[i, j, k]/labelNum[i]


KlableP = labelNum/m

m, n = shape(testDataMat)

count = 0
for i in range(m):
	predictClass = 0
	predictP = -1

	for j in range(26):
		x = KlableP[j]
		for k in range(n):
			x *= KtrainP[j, testDataMat[i,k], k]
		if(x > predictP):
			predictP = x
			predictClass = j
	if(predictClass == testLabelMat[i]):
		count += 1		


print(count*1.0/m)

