# -*- coding:utf-8 -*-

from math import *
import numpy as np
import numpy.matlib
from numpy.random import *
import matplotlib.pyplot as plt

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
    plt.scatter(data1[0,:],data1[1,:],marker = "o",color = "r",s = 60, label = "data1")

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
    plt.scatter(data2[0,:],data2[1,:],marker = "x",color = "b",s = 60, label = "data2(before matching)")

    # ICPアルゴリズム data2とdata1のMatching
    # R:回転行列　t:併進ベクトル
    # R,T = icp(data1,data2)
    R,T = ICPMatching(data1,data2)

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

def ICPMatching(data1, data2):
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
    
        ii, error = FindNearestPoint(data1, data2)  #最近傍点探索
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
    plt.scatter(data2[0,:],data2[1,:],marker = "x",color = "g",s = 60, label = "data2(after matching)")
    return R , t

def FindNearestPoint(data1, data2):
    #data2に対するdata1の最近傍点のインデックスを計算する関数
    m1 = data1.shape[1]
    m2 = data2.shape[1]
    index = np.array([],dtype=np.integer)
    error = 0

    for i in range(m1):
        dx = data2 - np.matlib.repmat( transposition( data1[:,i] ), 1, m2 )
        dist = np.sqrt(dx[0,:] ** 2 + dx[1,:] ** 2)

        ii = np.argmin(dist)
        dist = np.min(dist)
        
        index = np.r_[index,[ii]]
        error = error + dist

    return index , error

def SVDMotionEstimation(data1, data2, index):
    #特異値分解法による並進ベクトルと、回転行列の計算

    #各点群の重心の計算
    M  =  data1 
    mm  =  np.c_[M.mean(1)]
    S  =  data2[:,index.tolist()]
    ms  = np.c_[S.mean(1)]

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


if __name__ == "__main__":
    ICPsample()