# -*- coding:utf-8 -*-

from math import *
import numpy as np
import numpy.matlib
from numpy.random import *
import matplotlib.pyplot as plt
import os
import random
from PIL import Image

DIRPATH = u"C:\\Users\\user\\Documents\\なかむら\\つくばチャレンジ2015\\測定データ\\20151023130844\\"
DIRPATH = u"C:\\Users\\user\\Documents\\なかむら\\つくばチャレンジ2015\\測定データ\\20160524154532\\"
#DIRPATH = u"C:\\Users\\user\\Documents\\なかむら\\つくばチャレンジ2015\\測定データ\\20151017135820\\"

def ICPsample():

    #Simulation Parameters
    nPoint = 25  #レーザ点の数
    fieldLength = 50   #点をばら撒く最大距離
    motion = np.array([3, 5, 20])   #真の移動量[並進x[m],並進y[m],回転[deg]]
    transitionSigma = 1    #並進方向の移動誤差標準偏差[m]
    thetaSigma = 1    #回転方向の誤差標準偏差[deg]

    # 点をランダムでばら撒く(t - 1の時の点群)
    # data1:2行nPoint列
    data1 = fieldLength * rand(2,nPoint) - fieldLength / 2
    plt.scatter(data1[0,:],data1[1,:],marker = "o",color = "r",s = 60, label = "data1(before matching)")

    # data2 =  data1を移動させる & ノイズ付加
    # 回転方向 ＆ ノイズ付加
    theta = toRadian(motion[2]) + toRadian(thetaSigma) * rand()

    # 並進ベクトル ＆ ノイズ付加
    t = np.matlib.repmat(transposition( motion[0:2] ),1,nPoint) + transitionSigma * randn(2,nPoint)

    # 回転行列の作成
    A = np.array([[cos(theta), sin(theta)],
                 [-sin(theta), cos(theta)]])

    # data1を移動させてdata2を作る
    data2 = t + A.dot(data1)
    plt.scatter(data2[0,:],data2[1,:],marker = "x",color = "b",s = 60, label = "data2")

    # ICPアルゴリズム data2とdata1のMatching
    # R:回転行列　t:併進ベクトル
    # R,T = icp(data1,data2)
    R,T,matchData = ICPMatching(data2,data1)
    plt.scatter(matchData[0,:],matchData[1,:],marker = "o",color = "g",s = 60, label = "data1(after matching)")

    #結果の表示
    print('True Motion [m m deg]:')
    print( motion )

    print('Estimated Motion [m m deg]:')
    theta  =  acos(R[0,0]) / pi * 180
    Est = np.hstack([transposition(T), transposition( np.array([theta]))])
    print("{:.4f}, {:.4f}, {:.4f}".format(Est[0][0],Est[0][1],Est[0][2]))

    print('Error [m m deg]:')
    Error = Est - motion
    print( "{:.4f}, {:.4f}, {:.4f}".format(Error[0][0],Error[0][1],Error[0][2]) )

    plt.grid(True)
    plt.legend(loc = "upper right")
    plt.show()

def ICPMatching(data1, data2 , boundaryList = []):
    # ICPアルゴリズムによる、並進ベクトルと回転行列の計算を実施する関数
    # data1  =  [x(t)1 x(t)2 x(t)3 ...]
    # data2  =  [x(t + 1)1 x(t + 1)2 x(t + 1)3 ...]
    # x = [x y z]'

    #ICP パラメータ
    preError = 0    #一つ前のイタレーションのerror値
    dError = 1000   #エラー値の差分
    EPS = 0.0001    #収束判定値
    maxIter = 100   #最大イタレーション数
    count = 0       #ループカウンタ

    R = np.identity(2)  #回転行列
    t = np.zeros([2,1])  #並進ベクトル

    while not(dError < EPS):
        count = count + 1
    
        ii, error = FindNearestPoint(data1, data2, boundaryList)  #最近傍点探索
        R1, t1 = SVDMotionEstimation(data1, data2, ii)    #特異値分解による移動量推定

        #計算したRとtで点群とRとtの値を更新
        data2 = R1.dot( data2 )
        data2 = np.array([data2[0] + t1[0],
                          data2[1] + t1[1]])
        R = R1.dot( R )
        t = R1.dot( t ) + t1 
    
        dError = abs(preError - error)  #エラーの改善量
        preError = error    #一つ前のエラーの総和値を保存
    
        if count > maxIter:  #収束しなかった
            print('Max Iteration')
            return

    print('Convergence:' + str( count ))
    
    return R , t , data2

def FindNearestPoint(data1, data2 , boundaryList = []):
    #data2に対するdata1の最近傍点のインデックスを計算する関数
    m1 = data1.shape[1]
    m2 = data2.shape[1]
    index = [[],[]]
    error = 0

    min = 100
    max = 0
    sum = 0.0
    distList = []

    for i in range(m1):
        dx = data2 - np.matlib.repmat( transposition( data1[:,i] ), 1, m2 )
        dist = np.sqrt(dx[0,:] ** 2 + dx[1,:] ** 2)

        ii = np.argmin(dist)
        dist = np.min(dist)
        
        if min > dist:min = dist
        if max < dist:max = dist
        sum += dist
        distList.append(dist)
        if (len(boundaryList) != 0 and ii in boundaryList) or dist > 0.1:
            continue
        index[0].append(i)
        index[1].append(ii)
        error = error + dist

    #plt.hist(distList,bins = 50)
    #plt.show()
    #print("min:{},max:{},ave:{}".format(min,max,sum/m1))
    return index , error

def SVDMotionEstimation(data1, data2, index):
    #特異値分解法による並進ベクトルと、回転行列の計算

    #print("data size:{}=>{}".format(len(data1[0]),len(index[0])))
    #各点群の重心の計算
    M = data1[:,index[0]]
    mm = np.c_[M.mean(1)]
    S = data2[:,index[1]]
    ms = np.c_[S.mean(1)]

    #各点群を重心中心の座標系に変換
    Sshifted = np.array([S[0,:] - ms[0],
                         S[1,:] - ms[1]])
    Mshifted = np.array([M[0,:] - mm[0],
                         M[1,:] - mm[1]])

    W = Sshifted.dot(transposition( Mshifted ))
    U,A,V = np.linalg.svd( W )    #特異値分解

    R = transposition( U.dot( transposition( V ) ))   #回転行列の計算
    t = mm - R.dot(ms) #並進ベクトルの計算

    return R , t

def toRadian(degree):
    # degree to radian
    return degree / 180 * pi

# arrayを転置する
# numpy標準では1行(ベクトル)だと転置されないので...
def transposition(array):
    if len(array.shape) == 1 or array.shape[0] == 1:
        return np.c_[array]
    else:
        return array.T

def getPcdList(dirPath):
    pcdList = []
    count = 0

    for path in os.listdir(dirPath):
        if ".pcd" in path:
            pcdList.append(path)
            count += 1

    sortedPcdList = [0] * count
    for path in pcdList:
        sortedPcdList[ int(path[11:-4]) ] = path

    return sortedPcdList

def getPointCloudData(filePath):
    pcd = open( DIRPATH + filePath , "r" )

    retData = [[],[]]
    boundaryData = []
    boundaryFlag = False #ゴミ値がfalse

    for index,line in enumerate( pcd ):
        if index < 11:
            continue
        
        data = line.split(",")

        if float(data[0]) == 0.0 and float( data[1]) == 0.0:
            if boundaryFlag:
                boundaryFlag = not boundaryFlag
                boundaryData.append(len(retData[0])-1)
            continue
        
        #retData[0].append(+(float(data[0])-float(data[2])))
        #retData[1].append(+(float(data[1])-float(data[0])))
        retData[0].append(float(data[0]))
        retData[1].append(float(data[1]))

        if not boundaryFlag:
                boundaryFlag = not boundaryFlag
                boundaryData.append(len(retData[0])-1)

    return retData , boundaryData

def adjustDataSize( data1 , data2 , max = 0):
    data1_length = len(data1[0]) 
    data2_length = len(data2[0])

    print("Default data size:{},{}".format(data1_length,data2_length))

    if data1_length > data2_length:
        if max != 0 and data2_length > max:
            return [random.sample(data1[0],max),
                    random.sample(data1[1],max)] ,\
                    [random.sample(data2[0],max),
                    random.sample(data2[1],max)]
        else:
            return [random.sample(data1[0],data2_length),random.sample(data1[1],data2_length)] , data2
    else:
        if max != 0 and data1_length > max:
            return [random.sample(data1[0],max),
                    random.sample(data1[1],max)] ,\
                    [random.sample(data2[0],max),
                    random.sample(data2[1],max)]
        else:
            return data1 , [random.sample(data2[0],data1_length),random.sample(data2[1],data1_length)]

def plotPoint2Image(data,img):
    imgMap = img.load()
    coefficient = 100.0 / 5.0
    origin_x = img.size[0] / 2.0
    origin_y = img.size[1] / 2.0
    
    for index in range(0,len(data[0])):
        x = int(data[0][index] * coefficient + origin_x)
        y = int(data[1][index] * coefficient + origin_y)
        
        if imgMap[x,y] != 250:
           imgMap[x,y] += 50
    return img

def pcdICPsample(minIndex = 0,maxIndex = 0):
    print (DIRPATH)
    pathList = getPcdList(DIRPATH)

    if maxIndex == 0:
        maxIndex = len(pathList)

    print("Index : {} ~ {} \n\n".format(minIndex,maxIndex))

    preData, boundaryList = getPointCloudData( pathList[minIndex] )
    preR = np.identity(2)
    preT = np.zeros([2,1])

    #plt.grid(True)
    #fig, ax = plt.subplots(1, 1)
    #fig = plt.figure()
    #ax = fig.add_subplot(1,1,1)

    #lines, = ax.plot(preData[0,:],preData[1,:],marker = "o",color = "b",s = 10)

    img = Image.new("L",(600,600))

    #for index, path in enumerate(pathList):
    for index in range(minIndex + 1,maxIndex,1):
        print ("\nindex:",index)

        #data = getPointCloudData( path )
        data, boundaryList = getPointCloudData( pathList[index] )
        #data = preR.dot( data )
        #data = np.array([data[0] + preT[0],
        #                  data[1] + preT[1]])

        try:
            #boundaryList = []
            R,T,matchData = ICPMatching(np.array(preData),np.array(data),boundaryList)

        except TypeError:
            print("error")
            continue

        else:
            #data1 , data2 = adjustDataSize(preData, data)
            #print("Adjusted data size:{},{}".format(len(data1[0]),len(data1[0])))
            #R,T,matchData = ICPMatching(np.array(data2),np.array(data1))

            #ax.scatter(matchData[0,:],matchData[1,:],marker = "o",color = "b",s = 10)
            
            #plt.scatter(matchData[0,:],matchData[1,:],marker = "o",color = "b",s = 10)
            #lines.set_data(matchData[0,:],matchData[1,:],marker = "o",color = "b",s = 10)
            #ax.set_xlim((x.min(), x.max()))
            #ax.set_ylim((y.min(), y.max()))
            #plt.pause(.01)

            matchData = preR.dot( matchData )
            matchData = np.array([matchData[0] + preT[0],
                                  matchData[1] + preT[1]])
            preR = R.dot( preR )
            preT = R.dot( preT ) + T 
            plt.scatter(matchData[0,:],matchData[1,:],marker = "o",color = "b",s = 10)

        img = plotPoint2Image(matchData,img)
        ##img.show()
        ##画像をarrayに変換
        #im_list = np.asarray(img)
        ##貼り付け
        #plt.imshow(im_list)
        #plt.gray()
        ##表示
        #plt.pause(.01)

        #img2 = Image.new("L",(500,500))
        #img2 = plotPoint2Image(data,img2)
        #img2.show()

        preData = data

    img.show()

    #画像をarrayに変換
    im_list = np.asarray(img)
    #貼り付け
    plt.imshow(im_list)
    plt.gray()
    #表示
    plt.show()

if __name__ == "__main__":
    #ICPsample()
    pcdICPsample( 35,40 )