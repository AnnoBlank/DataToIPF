#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 11:14:00 2017

##Version Comment to 1.22 Height leveling Change at the sputter experiment --> Fromm 100 to mean Value


@author: karsten
"""

import ebsd_merge_evaluate.mapplot as mapplot 
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from ebsd_merge_evaluate.wedge import Wedge
import ebsd_merge_evaluate.ipfzXY as ipfzXY
import time
# imports specific to the plots in this example
import numpy as np
import matplotlib as mpl
import matplotlib.ticker

from multiprocessing import Pool



class EBSDdataEvaluation():
    def __init__(self):
        self.dataPathFile = 0
        self.mergeDataSet = 0
        self.refereceHeight = 0
        self.progressUpdate2 = 0
        self.resolution  = 2
        self.heightIPF = [0]
    
    # def loadData(self):
    #     self.mergeDataSet = np.loadtxt(self.dataPathFile)
    
    def loadBCAData(self):
            fileName = self.dataPathFile
               
            SoutterYIeldSim = np.loadtxt(fileName)
            grid_size = 92
            grid_sizej = 92j
            grid_x, grid_y = np.mgrid[0:45:grid_sizej, 0:45:grid_sizej]
            x = np.arange(0, 46, 1)
            pts = np.array([[i,j] for i in x for j in x] )
            vals = np.reshape(SoutterYIeldSim[1:, 1:], (46*46))
            grid_z = griddata(pts, vals, (grid_x, grid_y), method='linear')
    
            Data2 = -1*np.ones((grid_size*grid_size,11))
            k=0
            xmax = 2 * np.tan(np.radians(45)/2)
            for i in range(grid_size):
                x = xmax * i / (grid_size-1.)
                for j in range(grid_size):
                    y = xmax * j / (grid_size-1.)
                    Data2[k, 6] = 4*x / (4 + x**2 + y**2)
                    Data2[k, 7] = 4*y / (4 + x**2 + y**2)
                    Data2[k, 8] =  (4 - x**2 - y**2) / (4 + x**2 + y**2) 
                    Data2[k, 2] = grid_z[i][j]
                    k = k+1
                    
            XY= np.zeros((Data2.shape[0], 2))
            for k in range(0,Data2.shape[0]):
                    XY[k,:] = ipfzXY.poleA2(Data2[k, 6],Data2[k, 7],Data2[k, 8])
            Data2[:, 9] = XY[:, 0]
            Data2[:, 10] = XY[:, 1]
            self.mergeDataSet = Data2

    def loadMDData(self):
        try:
            fileName = self.dataPathFile
            with open(fileName, "r", errors="replace") as f:
                CRSTdata = f.readlines()  
            num = len(CRSTdata)
            Data1 = -1 * np.ones((num,11))
            euler1Index = self.euler1Index
            euler2Index = self.euler2Index
            dataIndex = self.dataIndex
            print('Load data')
            for i , lines in enumerate(CRSTdata):
                cells = lines.split()
                Data1[i, 2] = cells[dataIndex-1]
                Data1[i, 4:5] = cells[euler1Index-1:euler1Index]
                Data1[i, 5:6] = cells[euler2Index-1:euler2Index]
                
    
            grid_y, grid_x = np.mgrid[43:45:31j, 51:56:51j]
            grid_z = griddata(Data1[:, 4:6], Data1[:, 2], (grid_x, grid_y), method = 'linear')
            
            num2 = grid_z.shape[0] * grid_z.shape[1]
            Data = -1 * np.ones((num + num2, 11))      
            Data[:num,4: 5] = Data1[:num, 4:5]
            Data[:num,5: 6] = Data1[:num, 5:6]
            Data[:num,2: 3] = Data1[:num, 2:3]
            Data[num:,4: 5] = grid_x.reshape((num2, 1))
            Data[num:,5: 6] = grid_y.reshape((num2, 1))
            Data[num:,2: 3] = grid_z.reshape((num2, 1))
            mask = np.logical_not(Data[:, 5] == -1)
            Data = Data[mask]
    
            Data[:,6] = np.sin((Data[:, 4]) * np.pi / 180) * np.sin((Data[:, 5]) * np.pi / 180)
            Data[:,7] = np.sin((Data[:, 4]) * np.pi / 180) * np.cos((Data[:, 5]) * np.pi / 180)
            Data[:,8] = np.cos((Data[:, 4]) * np.pi / 180)
    
            mask = np.zeros(Data.shape[0])
            for i in range(0, Data.shape[0]):
                if (Data[i,6] <= Data[i, 8] and Data[i, 7] <= Data[i, 8] and Data[i, 7] >= Data[i, 6] ):
                    mask[i]= True
                else:
                    mask[i]= False
            Data = Data[mask == 1]
            XY= np.zeros((Data.shape[0], 2))
            
            for k in range(0, Data.shape[0]):
                    XY[k,:] = ipfzXY.poleA2(Data[k, 7], Data[k, 6], Data[k, 8])
            Data[:, 9] = XY[:, 0]
            Data[:, 10] = XY[:, 1]
    
            self.mergeDataSet = Data
        except:
            print('Data load failed! -------------- ??Column wrong??')
        
            
    def browse_merged_leveld_data(self):
        self.dataPathFiles
        self.mergeDataSet = np.loadtxt(self.dataPathFiles[0])
        print('Load data set 1')
        for i in range(1, len(self.dataPathFiles)):
            tmp = np.loadtxt(self.dataPathFiles[i])
            tmp[:, 0] = tmp[:, 0]+ np.max(self.mergeDataSet[:, 0])
            self.mergeDataSet = np.concatenate((self.mergeDataSet,tmp))
            print(f'Load{i+1}')
            
    def combine_data(self,data2merge):
        print('Load data set 1')
        data2merge[:, 0] = data2merge[:, 0] + np.max(self.mergeDataSet[:, 0])
        self.mergeDataSet = np.concatenate((self.mergeDataSet,data2merge))
        print(f'Load{1}')


    
    def relativeHeighttoAbsHeight(self):
    
        print(f'Mean relative height: {np.round(np.mean(self.mergeDataSet[:,2]),5)}    ------    Amount of  values: {len(self.mergeDataSet[:,2])}' )
        meanRelative100 = np.mean(self.mergeDataSet[:, 2])
        self.mergeDataSet[:, 2] = (self.mergeDataSet[:, 2] - meanRelative100) + self.refereceHeight 
        return True
    
    
    def loadMergeFile(Data_result):
    
        p = Pool()
        r = []
        for i in range(len(Data_result)):
            r.append(p.apply_async(np.loadtxt, [Data_result[i]]))
            print('Pool Load ',i ,'starts')
            
        for i in range(len(Data_result)):
            Data_result[i] = r[i].get()
            print('DataGet ', i)
        return Data_result
        
    
    def SpikesContrast(Data_result, lowerLimit = 20, UpperLimit = 235):
        Data_result_Sp = []
        Data_result_Sp = Data_result
        for i in range(len(Data_result_Sp)):
            Data_result_Sp[i] = Data_result_Sp[i][Data_result_Sp[i][:, 2] < UpperLimit]
            Data_result_Sp[i] = Data_result_Sp[i][Data_result_Sp[i][:, 2] > lowerLimit]
            
        return Data_result_Sp
    def normContrast2(merge_EBSD_CLSM_DATA, ContrastNromFactor = 256):
        i = 0
        print('Start Radius 10')
        mtmp1 = -1 * np.ones((len(merge_EBSD_CLSM_DATA),) + merge_EBSD_CLSM_DATA[0].shape)
        i1 = 0
        mtmp2 = -1 * np.ones((len(merge_EBSD_CLSM_DATA),) + merge_EBSD_CLSM_DATA[0].shape)
        i2 = 0
        for i in range(len(merge_EBSD_CLSM_DATA)):
            
    
            if1tmp = merge_EBSD_CLSM_DATA[i, 9]
            if2tmp = merge_EBSD_CLSM_DATA[i, 10]
            if (np.sqrt(if1tmp**2 + if2tmp**2)) < 10:
                tmp =  merge_EBSD_CLSM_DATA[i:i+1, :]
                mtmp1[i:i+len(tmp),] = tmp
                i1 = i1+len(tmp)
            if if1tmp < 100 and if1tmp > 88  and if2tmp < 36 and if2tmp > 24:
                tmp =  merge_EBSD_CLSM_DATA[i:i+1, :]
                mtmp2[i:i + len(tmp),] = tmp
                i2 = i2 + len(tmp)
        print('Finish')
        mtmp1 = mtmp1[mtmp1[:, 0] != -1]
        mtmp2 = mtmp2[mtmp2[:, 0] != -1]
    
        print('MeannormContrast100', np.median(mtmp1[:, 2]), 'Anzahl Werte', len(mtmp1))
        print('MeannormContrastXXX', np.median(mtmp2[:, 2]), 'Anzahl Werte', len(mtmp2))
        meanContrast100= np.median(mtmp1[:, 2])
        meanMaxContrast= np.median(mtmp2[:, 2]) - meanContrast100
        merge_EBSD_CLSM_DATA[:, 2] = (merge_EBSD_CLSM_DATA[:, 2] - meanContrast100) * (ContrastNromFactor/meanMaxContrast)
        return merge_EBSD_CLSM_DATA
    
    def normContrast(merge_EBSD_CLSM_DATA, ContrastNromFactor = 256):
        mtmp1 = 0
        mtmp2 = 0
        print('Start Radius 10')
        
        
        for i in range(len(merge_EBSD_CLSM_DATA)):     
            if1tmp = merge_EBSD_CLSM_DATA[i, 9]
            if2tmp = merge_EBSD_CLSM_DATA[i, 10]
            if (np.sqrt(if1tmp**2 + if2tmp**2)) < 10:
                tmp =  merge_EBSD_CLSM_DATA[i:i+1, :]
                if type(mtmp1) == int:
                    mtmp1 = tmp
                else:
                    mtmp1 = np.concatenate((mtmp1, tmp))
            if if1tmp < 98 and if1tmp > 90  and if2tmp < 36 and if2tmp > 28:
                tmp =  merge_EBSD_CLSM_DATA[i:i+1, :]
                if type(mtmp2) == int:
                    mtmp2 = tmp
                else:
                    mtmp2 = np.concatenate((mtmp2, tmp))
        print('Finish')
        print('MeannormContrast100', np.median(mtmp1[:, 2]), 'Anzahl Werte', len(mtmp1))
        print('MeannormContrastXXX', np.median(mtmp2[:, 2]), 'Anzahl Werte', len(mtmp2))
        meanContrast100 = np.median(mtmp1[:, 2]) - 30
        meanMaxContrast = np.median(mtmp2[:, 2]) - meanContrast100
        merge_EBSD_CLSM_DATA[:, 2] = (merge_EBSD_CLSM_DATA[:, 2] - meanContrast100) * (200/ meanMaxContrast)/ContrastNromFactor
        return merge_EBSD_CLSM_DATA

    #%%

    
    
    def relativeHeighttoAbsOxidTungstenHeight100(self):
        mtmp1 = -1 * np.ones((len(self.mergeDataSet),) + self.mergeDataSet[0].shape)
        i1 = 0
        for i in range(len(self.mergeDataSet)):     
            if1tmp = self.mergeDataSet[i,9]
            if2tmp = self.mergeDataSet[i,10]
            if (np.sqrt(if1tmp**2+if2tmp**2))<10:
                tmp =  self.mergeDataSet[i:i+1,:]
                mtmp1[i:i+len(tmp),] = tmp
                i1 = i1+len(tmp)
        print('Finish')
        mtmp1 = mtmp1[mtmp1[:,0] !=-1]
        print('MeanRelativeHeight', np.mean(mtmp1[:,2]),'Anzahl Werte', len(mtmp1))
        meanRelative100= np.mean(mtmp1[:,2])
        self.mergeDataSet[:,2] = (self.mergeDataSet[:,2] - meanRelative100)*(1+self.DepthgrowFactor) + self.ABSHeightOxide 
    

    
    def relativeHeighttoAbsOxidTungstenHeightmean(self):
    
        print(f'Mean relative height: {np.round(np.mean(self.mergeDataSet[:,2]),5)}    ------    Amount of  values: {len(self.mergeDataSet[:,2])}' )
        meanRelative= np.mean(self.mergeDataSet[:,2])
        self.mergeDataSet[:,2] = (self.mergeDataSet[:,2] - meanRelative) *(1+self.DepthgrowFactor)+ self.ABSHeightOxide
        return True    
    
    
    #%%
    
    
    def heighttoSputterYield(self):
        z1= np.multiply(self.heightIPF,-1)
        self.sputterYield=z1 *self.atomicDensity/self.fluence/1000000
        
        return True
 
    def heighttoSputterYieldCopy(self):
        z1= np.multiply(self.mergeDataSetcopy[:,2],-1)
        self.mergeDataSetcopy[:,2]=z1 *self.atomicDensity/self.fluence/1000000      
        return True

    
    #%%    
    
    def neighboursMedian(matrix, k, o, tmp1):
        tmp1 = np.median(tmp1)
        tmp2 = matrix[k+1][o]
        if type(tmp2) != int:
            tmp1=np.append(tmp1, np.median(tmp2[:,2]))
            
        tmp2 = matrix[k+1][o+1]
        if type(tmp2) != int:
            tmp1=np.append(tmp1, np.median(tmp2[:,2]))
            
        tmp2 = matrix[k+1][o-1]
        if type(tmp2) != int:
            tmp1=np.append(tmp1, np.median(tmp2[:,2]))
            
        tmp2 = matrix[k-1][o]
        if type(tmp2) != int:
            tmp1=np.append(tmp1, np.median(tmp2[:,2]))
            
        tmp2 = matrix[k-1][o+1]
        if type(tmp2) != int:
            tmp1=np.append(tmp1, np.median(tmp2[:,2]))
            
        tmp2 = matrix[k-1][o-1]
        if type(tmp2) != int:
            tmp1=np.append(tmp1, np.median(tmp2[:,2]))
    
        tmp2 = matrix[k][o+1]
        if type(tmp2) != int:
            tmp1=np.append(tmp1, np.median(tmp2[:,2]))
        tmp2 = matrix[k][o-1]
        if type(tmp2) != int:
            tmp1=np.append(tmp1, np.median(tmp2[:,2]))
        return tmp1
    
    ## Hier wurde noch eine Abfrage der Länge des Vectors tmp2 eingebaut
    def neighboursMedian1(self, matrix, k, o, tmp1):
        tmp1 = np.median(tmp1)
        tmp2 = matrix[k+1][o]
        if type(tmp2) != int:
            if len(tmp2) > 10:
                tmp1=np.append(tmp1, np.median(tmp2[:,2]))
            
        tmp2 = matrix[k+1][o+1]
        if type(tmp2) != int:
            if len(tmp2) > 10:
                tmp1=np.append(tmp1, np.median(tmp2[:,2]))
            
        tmp2 = matrix[k+1][o-1]
        if type(tmp2) != int:
            if len(tmp2) > 10:
                tmp1=np.append(tmp1, np.median(tmp2[:,2]))
            
        tmp2 = matrix[k-1][o]
        if type(tmp2) != int:
            if len(tmp2) > 10:
                tmp1=np.append(tmp1, np.median(tmp2[:,2]))
            
        tmp2 = matrix[k-1][o+1]
        if type(tmp2) != int:
            if len(tmp2) > 10:
                tmp1=np.append(tmp1, np.median(tmp2[:,2]))
            
        tmp2 = matrix[k-1][o-1]
        if type(tmp2) != int:
            if len(tmp2) > 10:
                tmp1=np.append(tmp1, np.median(tmp2[:,2]))
    
        tmp2 = matrix[k][o+1]
        if type(tmp2) != int:
            if len(tmp2) > 10:
                tmp1=np.append(tmp1, np.median(tmp2[:,2]))
        tmp2 = matrix[k][o-1]
        if type(tmp2) != int:
            if len(tmp2) > 10:
                tmp1=np.append(tmp1, np.median(tmp2[:,2]))
                
        return tmp1
    
        
    def medianDataXYZ(self):
            mat = self.mergeDataSet
            resulution = self.resolution 
            neighbours = self.smoothing
            DataLess = self.cutOffFilter
        
            X = []
            Y = []
            Z = []
            Zstd = []
            Zcounts = []
            h = []
            k1 = []
            l1 = []
        
            resulutiontmp = int(resulution *8)
            matrix = [[0]*38*resulutiontmp for i in range(40*resulutiontmp)]
            print('Sort Data')
            for i in range(len(mat[:,9])):
                tmp =  mat[i:i+1,:]
                if type(matrix[int(mat[i,9]*resulution)][int(mat[i,10]*resulution)]) == int:
                    matrix[int(mat[i,9]*resulution)][int(mat[i,10]*resulution)] = tmp*resulution
                else:
                    matrix[int(mat[i,9]*resulution)][int(mat[i,10]*resulution)] = np.concatenate((matrix[int(mat[i,9]*resulution)][int(mat[i,10]*resulution)],tmp*resulution))
              
            LessData=0 
            Data=0
            for k in range(38*resulutiontmp):
                for o in range(35*resulutiontmp):
                        tmp = matrix[k][o]
                        if type(tmp) == int:
                            pass   
                        elif len(tmp)<DataLess:
                           LessData=LessData+1               
                        
                        elif type(tmp) != int:
                            tmp1 = tmp[:,2]
                            if neighbours != 0:
                                tmp1 = self.neighboursMedian1(matrix, k, o, tmp1)
                            Data=Data+1
                            X.extend([int(np.mean(tmp[:,9]))])
                            Y.extend([int(np.mean(tmp[:,10]))])
                            Z.extend([np.median(tmp1)/resulution])
                            Zstd.extend([np.std(tmp[:,2]/resulution)])
                            Zcounts.extend([len(tmp[:,2])])
                            h.extend([np.mean(tmp[:,6])/resulution])
                            k1.extend([np.mean(tmp[:,7])/resulution]) 
                            l1.extend([np.mean(tmp[:,8])/resulution])
            
            print(f'Cut off filter used: {LessData}')
            print(f'IPF data point: {Data}')
            self.xIPFcoordinate = X
            self.yIPFcoordinate = Y
            self.heightIPF = Z
            self.heightSD = Zstd
            self.heightCounts = Zcounts
            num = len(X)
            self.IPF_Data_all  = np.zeros((num,9))           
            self.IPF_Data_all[:,0] = np.array(Z)
            self.IPF_Data_all[:,1] = np.array(Zstd)
            self.IPF_Data_all[:,2] = np.array(Zcounts)
            self.IPF_Data_all[:,3] = np.array(h)
            self.IPF_Data_all[:,4] = np.array(k1)
            self.IPF_Data_all[:,5] = np.array(l1)
            self.IPF_Data_all[:,6] = np.array(np.divide(X,resulution))
            self.IPF_Data_all[:,7] = np.array(np.divide(Y,resulution))
            
            return True
    #%%
    def meanDataXYZ(mat,resulution=2, DataLess=10):
    
        X = []
        Y = []
        Z = []
        Zstd = []
        Zcounts = []
        h = []
        k1 = []
        l = []
    
        resulutiontmp = int(resulution *4)
        matrix = [[0]*38*resulutiontmp for i in range(40*resulutiontmp)]
        print('Sort Data')
    
        for i in range(len(mat[:,9])):
            tmp =  mat[i:i+1,:]
            if type(matrix[int(mat[i,9]*resulution)][int(mat[i,10]*resulution)]) == int:
                matrix[int(mat[i,9]*resulution)][int(mat[i,10]*resulution)] = tmp*resulution
            else:
                matrix[int(mat[i,9]*resulution)][int(mat[i,10]*resulution)] = np.concatenate((matrix[int(mat[i,9]*resulution)][int(mat[i,10]*resulution)],tmp*resulution))
        
        print('Sort Data finished')    
        LessData=0 
        Data=0
        for k in range(38*resulutiontmp):
            for o in range(35*resulutiontmp):
                    tmp = matrix[k][o]
                    if type(tmp) == int:
                        pass   
                    elif len(tmp)<DataLess:
                       LessData=LessData+1               
                    
                    elif type(tmp) != int:
                        Data=Data+1
                        X.extend([int(np.mean(tmp[:,9]))])
                        Y.extend([int(np.mean(tmp[:,10]))])
                        Z.extend([np.mean(tmp[:,2])/resulution])
                        Zstd.extend([np.std(tmp[:,2]/resulution)])
                        Zcounts.extend([len(tmp[:,2])])
                        h.extend([np.mean(tmp[:,6])])
                        k1.extend([np.mean(tmp[:,7])]) 
                        l.extend([np.mean(tmp[:,8])])
    
        print('Cut off filter used: ',LessData)
        print('IPF data point: ',Data)
        return X,Y,Z,Zstd,Zcounts,h,k1,l
    
    
        
    def X_Y_X_area_Y_area(h,k,l,Z):
        X = np.array(h) #/300/resolution
        Y = np.array(k)
        l = np.array(l)
        r = np.sqrt(l**2+X**2+Y**2)
        X/=r
        Y/=r
        l/=r
        Z = np.array(Z)
        Z2 = []
        delta = []
        phi = []
        X2= []
        Y2 = []
        Xneu = []
        Yneu = []
        Xcluster = []
        Ycluster = []
        Zcluster = []
        for i in range(len(X)):
            Xtmp = X[i]
            Ytmp = Y[i]
            ltmp = l[i]
            Xtmp, Ytmp, ltmp = ipfzXY.pole22(Xtmp, Ytmp, ltmp)  
            if Xtmp > 0:
                phitmp = np.arctan(Ytmp/Xtmp)
                deltatmp = np.arcsin(Xtmp/np.cos(phitmp))    
                delta.append(deltatmp)
                phi.append(phitmp)
                X2.append(Xtmp)
                Y2.append(Ytmp)
                Z2.append(Z[i])
        for i in range(len(delta)):
            Xtmp = X2[i]
            Ytmp = Y2[i]
            z = np.cos(delta[i])
            zneu = 1-z
            delta[i] = np.pi/2-delta[i]/2
            r_neu = np.sqrt(zneu**2+Xtmp**2+Ytmp**2) 
            Xneu.append(r_neu * np.sin(delta[i]) * np.cos(phi[i])*100)
            Yneu.append(r_neu * np.sin(delta[i]) * np.sin(phi[i])*100)
            
            
        
        matrix = [[0]*int(max(Yneu)+10) for i in range(int(max(Xneu)+10))]
        print('Sort Data')
    
        for i in range(len(Xneu)):
            if type(matrix[int(Xneu[i])][int(Yneu[i])]) == int:
                matrix[int(Xneu[i])][int(Yneu[i])] = [Z2[i]]
            else:
                
                matrix[int(Xneu[i])][int(Yneu[i])].append(Z2[i]) # = np.concatenate(matrix[int(Xneu[i])][int(Yneu[i])],[Z[i]])
        
        for k in range(int(max(Xneu)+1)):
            for o in range(1,int(max(Yneu)+1)):
                tmp = matrix[k][o]
                if type(tmp) != int:
                    Xcluster.append(k)
                    Ycluster.append(o)
                    Zcluster.extend([np.mean(tmp)])
                    
                
        return Xcluster,Ycluster,Zcluster

    #%%
    def grainBounderies(self,Winkelfehler=30):
        self.progressUpdate2 = 0
        mat = self.mergeDataSetcopy
        grainboundary = self.grainboundaryOrNois 
        PixelCluster = self.pixelCluster
        PixelCluster = int(PixelCluster+1)
        
        ClusterVersetzung= int(PixelCluster/2)
        print('Start sort data')
    
        mi = int(np.max([np.max(mat[:,0]),  np.max(mat[:,1])])/PixelCluster+3)
        mi_all = int(np.max([np.max(mat[:,0])+3,  np.max(mat[:,1])])+3)
        matrix_all = [[0]*mi_all for i in range(mi_all)]
        matrix = [[0] for m in range(9)]
        for i in range(9):
            matrix[i] = [[0]*mi for i in range(mi)]
        
        self.progressUpdate2  = 3
        for i in range(len(mat)):
            m = -2
            tmp =  mat[i:i+1,:]
            for k in np.arange(-1,2):
                m+=3
                for o in np.arange(-1,2):
                    if type(matrix[m+o][int((mat[i,0]+k*ClusterVersetzung)/PixelCluster)][int((mat[i,1]+o*ClusterVersetzung)/PixelCluster)]) == int:
                        matrix[m+o][int((mat[i,0]+k*ClusterVersetzung)/PixelCluster)][int((mat[i,1]+o*ClusterVersetzung)/PixelCluster)] = tmp
                    else:
                        matrix[m+o][int((mat[i,0]+k*ClusterVersetzung)/PixelCluster)][int((mat[i,1]+o*ClusterVersetzung)/PixelCluster)] = np.concatenate((matrix[m+o][int((mat[i,0]+k*ClusterVersetzung)/PixelCluster)][int((mat[i,1]+o*ClusterVersetzung)/PixelCluster)],tmp))
    
       
        mtmp = -1*np.ones((len(mat)*13,)+mat[0].shape)
        i=0
        print('Apply filter')
        state = False
        
        for k in range(mi):
            time.sleep(0.001)
            self.progressUpdate2 = int(k / mi*86) +10
            for o in range(mi):
                    tmp = []
                    for l in range(9):
                       if type(matrix[l][k][o]) != int:
                           tmp.append(matrix[l][k][o])
                    if grainboundary == 1:
                        for m in range(len(tmp)):
                            if np.abs(np.var(tmp[m][:,9])) + np.abs(np.var(tmp[m][:,10])) <Winkelfehler and np.abs(np.var(tmp[m][:,11])) + np.abs(np.var(tmp[m][:,12])) <Winkelfehler and np.abs(np.var(tmp[m][:,13])) + np.abs(np.var(tmp[m][:,14])) <Winkelfehler:# and np.var(np.abs(np.cos((tmp[m][:,3])/180*np.pi))  )< rotationsfehler:
                                state = True
                            else:
                                state = False
                                break
                                
                        if state == True and len(tmp) != 0:
                            mtmp[i:i+len(tmp[0]),] = tmp[0]
                            i = i+len(tmp[0])
   
                    elif grainboundary == 0: 
                        for m in range(len(tmp)):
                            if np.abs(np.var(tmp[m][:,9]))  + np.abs(np.var(tmp[m][:,10]))<Winkelfehler and np.abs(np.var(tmp[m][:,11]))  + np.abs(np.var(tmp[m][:,12]))<Winkelfehler and np.abs(np.var(tmp[m][:,13])) + np.abs(np.var(tmp[m][:,14])) <Winkelfehler: # and np.var(np.abs(np.cos((tmp[m][:,3])/180*np.pi))  )< rotationsfehler:
                                mtmp[i:i+len(tmp[m]),] = tmp[m]
                                i = i+len(tmp[m])
    
        print('Remove double enteries')
        mtmp = mtmp[mtmp[:,0] !=-1]
        for i in range(len(mtmp)):
            tmp =  mtmp[i:i+1,:]
            if type(matrix_all[int(mtmp[i,0])][int(mtmp[i,1])]) == int:
                matrix_all[int(mtmp[i,0])][int(mtmp[i,1])] = tmp
            else:
                matrix_all[int(mtmp[i,0])][int(mtmp[i,1])] =  np.concatenate((matrix_all[int(mtmp[i,0])][int(mtmp[i,1])],tmp))
        self.progressUpdate2 = 100
        mtmp = -1*np.ones((len(mat)*2,)+mat[0].shape)
        i=0
        for k in range(mi_all):
            for o in range(mi_all):
                tmp = matrix_all[k][o]
                if type(tmp) == int:
                    pass
                else:
                    mtmp[i:i+1,] = tmp[0:1,]
                    i+=1
    
        mtmp = mtmp[mtmp[:,0] !=-1]
        self.mergeDataSetcopy = mtmp.copy()
        return True
    
    
    
    def grainBounderies_Reduce_Data(self):
        PixelCluster = int(self.pixelClusterBinning)*2
        self.mergeDataSetBinning = self.mergeDataSet.copy()
        mat = self.mergeDataSetBinning
        PixelCluster = int(PixelCluster)
        
        Winkelfehler=30
        mi = int(np.max([np.max(mat[:,0]),  np.max(mat[:,1])])/PixelCluster+3)
        
        matrix = [[0]*mi for i in range(mi)]
    
    
        print('Sort Cluster GrainBoundaries')
        
        for i in range(len(mat)):
            tmp =  mat[i:i+1,:]
            if type(matrix[int(mat[i,0]/PixelCluster)][int(mat[i,1]/PixelCluster)]) == int:
                matrix[int(mat[i,0]/PixelCluster)][int(mat[i,1]/PixelCluster)] = tmp
            else:
                matrix[int(mat[i,0]/PixelCluster)][int(mat[i,1]/PixelCluster)] = np.concatenate((matrix[int(mat[i,0]/PixelCluster)][int(mat[i,1]/PixelCluster)],tmp))
                
    
        
        mtmp = -1*np.ones((len(mat),)+mat[0].shape)
        i=0
       # pbarNoise = ProgressBar()    
        for k in range(mi):
            for o in range(mi):
                    tmp = matrix[k][o]
                    if type(tmp) == int:
                        pass
                    elif np.abs(np.var(np.sort(tmp[:,9])[:])) + np.abs(np.var(np.sort(tmp[:,10])[:]))<Winkelfehler:
                          mtmp[i,] = tmp[0,:]
                          i += 1
     
        mtmp = mtmp[mtmp[:,0] !=-1]
        self.mergeDataSetBinning = mtmp
        return True
    def optimisationSelctPointsIPF(self):
        opt_points1_xy = self.opt_points1_xy
        if  int(self.optReduceData) !=0:
            Data_resulttmp = self.mergeDataSetBinning
            print('Use reduced data set')
        else:
            Data_resulttmp = self.mergeDataSet
        mean_height = np.mean(Data_resulttmp[:,2])
        std_height = np.std(Data_resulttmp[:,2])
        IPF_min_scale = mean_height - 2.5 * std_height 
        IPF_max_scale = mean_height + 2.5 * std_height 
        X,Y,Z,Zstd,Zcounts,h,k1,l = self.meanDataXYZ(Data_resulttmp,
                                                               float(self.resolutionOptimisation), DataLess=3)
    
        fig, ax = self.plot_inter_IPFz(X, Y, Z,  float(self.resolutionOptimisation), 
                               cbarrange= [IPF_min_scale,IPF_max_scale], 
                               Title='IPFz', cbarlabel='data', ticks = (-256,-128,-64,-32,-16,-8,-4,-2,-1,-0.5,0,0.5,1,2,4,8,16,32,64,128,256),
                               contourplot=0,pltshow=2)
        if len(opt_points1_xy) != 4:
            ax.plot([opt_points1_xy[0][0], opt_points1_xy[1][0], opt_points1_xy[1][0],opt_points1_xy[0][0],opt_points1_xy[0][0]], 
                 [opt_points1_xy[0][1],opt_points1_xy[0][1], opt_points1_xy[1][1],opt_points1_xy[1][1],opt_points1_xy[0][1] ], linewidth=4, color='blue')
            ax.text(opt_points1_xy[0][0],opt_points1_xy[1][1], 'Old selection', color='black', fontsize=20, fontweight='bold')
            
        opt_points1_xy = plt.ginput(2, timeout=-1)
        opt_points1_xy = np.asarray(opt_points1_xy)
        if opt_points1_xy[0][0] > opt_points1_xy[1][0]:
            tmp = opt_points1_xy[0][0]
            opt_points1_xy[0][0] = opt_points1_xy[1][0]
            opt_points1_xy[1][0] = tmp
        if opt_points1_xy[0][1] > opt_points1_xy[1][1]:
            tmp = opt_points1_xy[0][1]
            opt_points1_xy[0][1] = opt_points1_xy[1][1]
            opt_points1_xy[1][1] = tmp
        print(opt_points1_xy)   
        self.opt_points1_xy = opt_points1_xy
        ax.plot([opt_points1_xy[0][0], opt_points1_xy[1][0], opt_points1_xy[1][0],opt_points1_xy[0][0],opt_points1_xy[0][0]], 
                 [opt_points1_xy[0][1],opt_points1_xy[0][1], opt_points1_xy[1][1],opt_points1_xy[1][1],opt_points1_xy[0][1] ], linewidth=4, color='black')
        plt.pause(1)
        fig.show()    
    
    #%%
    
    def GetCrystaorientationToPhiBeam(Data, hkl):
        XandSputterYield = 2*np.ones([int(len(Data)),2])
        Bereich = 3
        X, Y = ipfzXY.poleA2(hkl[0],hkl[1],hkl[2])
        print(X)
        print(Y)
        mat = [[] for i in range(390)]
        mat2 = [[] for i in range(390)]
        for i in range(0,Data.shape[0]):
            X1 = Data[i,9]
            Y1 = Data[i,10]
            if X1  <X+Bereich and X1 >X-Bereich and Y1 <Y+Bereich and Y1 >Y-Bereich:
                mat[int(Data[i,3]/4)*4].extend([Data[i,2]])
                mat2[int(Data[i,3]/4)*4].extend([Data[i,3]])
            
        for i in range(0,389):
                XandSputterYield[i,0] = np.mean(mat2[i])
                XandSputterYield[i,1]=np.mean(mat[i])
                
            
        XandSputterYield = XandSputterYield[XandSputterYield[:,0] !=2]
        XandSputterYield= XandSputterYield[~np.isnan(XandSputterYield).any(axis=1)]
        
        return XandSputterYield
    
    def get_crystal_orientation_data(self, hkl, area_pixel = 5):
        Data = self.mergeDataSet
        XandSputterYield = -1000*np.ones([int(len(Data)),19])
        Bereich = area_pixel
        X, Y = ipfzXY.poleA2(hkl[0],hkl[1],hkl[2])
        print(X)
        print(Y)
        for i in range(0,Data.shape[0]):
            X1 = Data[i,9]
            Y1 = Data[i,10]
            if (X1-X)**2+(Y1-Y)**2 < Bereich:    
                XandSputterYield[i,:] = Data[i,:] 
                
            
        XandSputterYield = XandSputterYield[XandSputterYield[:,0] !=-1000]
        XandSputterYield= XandSputterYield[~np.isnan(XandSputterYield).any(axis=1)]
        
        self.mergeDataSet = XandSputterYield
        return True
    
    
    
    
    def Linie100zu101(self, NameLabel):
        Data = self.IPF_Data_all
        XandSputterYield = -1*np.ones([int(np.max(Data[:,6])+4),2])

        mat = [[] for i in range(130)]
        mat2 = [[] for i in range(130)]
        for i in range(0,Data.shape[0]):
            if Data[i,7] < 3:
                mat[int(Data[i,6])].extend([Data[i,0]])
                mat2[int(Data[i,6])].extend([Data[i,6]])
    
         
        for i in range(0,int(max(Data[:,6]))):
                XandSputterYield[i,0] = np.mean(mat2[i])
                XandSputterYield[i,1]=np.mean(mat[i])
                
            
        XandSputterYield = XandSputterYield[XandSputterYield[:,0] !=-1]
        XandSputterYield= XandSputterYield[~np.isnan(XandSputterYield).any(axis=1)]

        self.data_Linie100zu101 = XandSputterYield
        self.plotLinescan(XandSputterYield, NameLabel, [0,36,70,91,124], MarkerName=['$%s$' %r'\langle 100\rangle', '$%s$' %r'\langle 401 \rangle','$%s$' %r'\langle 201 \rangle', '$%s$' %r'\langle 302 \rangle' , '$%s$' %r'\langle 101 \rangle']) 
        return True
    
    def Linie101zu111(self, NameLabel):
        Data = self.IPF_Data_all
        XandSputterYield = -1*np.ones([int(np.max(Data[:,7])+4),2])
        mat = [[] for i in range(130)]
        mat2 = [[] for i in range(130)]
        for i in range(0,Data.shape[0]):
            pole1, pole2, pole3 = ipfzXY.pole22(Data[i,3],Data[i,4],Data[i,5])
            norm = np.sqrt(pole1*pole1+pole2*pole2+pole3*pole3)
            pole1 = pole1/norm
            pole2 = pole2/norm
            pole3 = pole3/norm
            
            if pole1/pole3  > 0.95:
    
                mat[int(Data[i,7])].extend([Data[i,0]])
                mat2[int(Data[i,7])].extend([Data[i,7]])
            
        for i in range(0,int(max(Data[:,7]))):
                XandSputterYield[i,0] = np.mean(mat2[i])
                XandSputterYield[i,1]=np.mean(mat[i])
            
        XandSputterYield = XandSputterYield[XandSputterYield[:,0] !=-1]
        XandSputterYield= XandSputterYield[~np.isnan(XandSputterYield).any(axis=1)]
        self.data_Linie101zu111 = XandSputterYield
        self.plotLinescan(XandSputterYield, NameLabel, [0,30,60,78,110], MarkerName=['$%s$' %r'\langle 101 \rangle', '$%s$' %r'\langle 414 \rangle' , '$%s$' %r'\langle 212 \rangle' ,'$%s$' %r'\langle 323 \rangle' , '$%s$' %r'\langle 111 \rangle' ]) 
        return True 
    
    def Linie100zu111(self, NameLabel):
        Data = self.IPF_Data_all
        XandSputterYield = -1*np.ones([int(np.max(Data[:,6])+4),2])
        mat = [[] for i in range(130)]
        mat2 = [[] for i in range(130)]
        for i in range(0,Data.shape[0]):
            X1 = Data[i,6]
            Y1 = Data[i,7]
            if np.abs(X1-Y1)  <3:
                mat[int(Data[i,6])].extend([Data[i,0]])
                mat2[int(Data[i,6])].extend([Data[i,6]])

            
        for i in range(0,int(max(Data[:,6]))):
                XandSputterYield[i,0] = np.mean(mat2[i])
                XandSputterYield[i,1]=np.mean(mat[i])
            
        XandSputterYield = XandSputterYield[XandSputterYield[:,0] !=-1]
        XandSputterYield= XandSputterYield[~np.isnan(XandSputterYield).any(axis=1)]
        self.data_Linie100zu111 = XandSputterYield
        self.plotLinescan(XandSputterYield, NameLabel, [0,36,67,84,110], MarkerName=['$%s$' %r'\langle 100\rangle' , '$%s$' %r'\langle 411\rangle' , '$%s$' %r'\langle 211\rangle' , '$%s$' %r'\langle 322\rangle' , '$%s$' %r'\langle 111\rangle' ]) 
        
        return True
    
    
    
    def Linie100zu212(self, NameLabel):
        Data = self.IPF_Data_all
        XandSputterYield = -1*np.ones([int(np.max(Data[:,6])+4),2])
        mat = [[] for i in range(130)]
        mat2 = [[] for i in range(130)]
        for i in range(0,Data.shape[0]):
            X1 = Data[i,6]
            Y1 = Data[i,7]
            if np.abs(X1-Y1)  <Y1+3 and np.abs(X1-Y1)  >Y1-3 :
                mat[int(Data[i,6])].extend([Data[i,0]])
                mat2[int(Data[i,6])].extend([Data[i,6]])
               # Data[i,2] = 100
            
        for i in range(0,int(max(Data[:,6]))):
                XandSputterYield[i,0] = np.mean(mat2[i])
                XandSputterYield[i,1]=np.mean(mat[i])
                
            
        XandSputterYield = XandSputterYield[XandSputterYield[:,0] !=-1]
        XandSputterYield= XandSputterYield[~np.isnan(XandSputterYield).any(axis=1)]
        self.data_Linie100zu212 = XandSputterYield
        self.plotLinescan(XandSputterYield, NameLabel, [0,36,69,96,120], MarkerName=['$%s$' %r'\langle 100\rangle','$%s$' %r'\langle 812\rangle','$%s$' %r'\langle 412\rangle','$%s$' %r'\langle 312\rangle','$%s$' %r'\langle 212\rangle']) 
        
        return True
    
    def plotLinescan(self, data, NameLabel, MarkerPosition, MarkerName):
        plt.figure(figsize=(10,6))
        plt.plot(data[:,0],data[:,1], 'ko', label=NameLabel)
        plt.title('Linescan from IPFz sputter yield  ')
        plt.xticks(MarkerPosition,MarkerName)
        plt.ylabel('sputter yield Ga/W')
        plt.xlabel('crystal direction')
        plt.legend(loc='upper left')
        plt.ylim(0,(int(max(data[:,1]))+2))
        
        
    def plotEBSD_Data(self):
        ebsdFigure = plt.figure(figsize=(6,6))
        axebsd = ebsdFigure.add_subplot(1, 1, 1)
   
        Orientation = self.mergeDataSet[:,3:6]
        X = self.mergeDataSet[:,0:1]
        Y = self.mergeDataSet[:,1:2]
        Norma = Orientation.max(axis=0)
        color = np.abs(Orientation) / Norma     
        axebsd.scatter(X, Y, c=color, marker='o', s=2,  cmap=plt.get_cmap('jet'))
        ebsdFigure.gca().invert_yaxis()
        ebsdFigure.show()  

    def plotEBSD_Datacopy(self):
        ebsdFigurecopy = plt.figure(figsize=(6,6))
        axebsd = ebsdFigurecopy.add_subplot(1, 1, 1)
   
        Orientation = self.mergeDataSetcopy[:,3:6]
        X = self.mergeDataSetcopy[:,0:1]
        Y = self.mergeDataSetcopy[:,1:2]
        Norma = Orientation.max(axis=0)
        color = np.abs(Orientation) / Norma     
        axebsd.scatter(X, Y, c=color, marker='o', s=2,  cmap=plt.get_cmap('jet'))
        ebsdFigurecopy.gca().invert_yaxis()
        ebsdFigurecopy.show()  
        
        
    def plotEBSDDataBinning(self):
        ebsdFigure = plt.figure(figsize=(6,6))
        axebsd = ebsdFigure.add_subplot(1, 1, 1)
   
        Orientation = self.mergeDataSetBinning[:,3:6]
        X = self.mergeDataSetBinning[:,0:1]
        Y = self.mergeDataSetBinning[:,1:2]
        Norma = Orientation.max(axis=0)
        color = np.abs(Orientation) / Norma     
        axebsd.scatter(X, Y, c=color, marker='o', s=2, edgecolors='face', cmap='jet')
        ebsdFigure.gca().invert_yaxis()
        ebsdFigure.show()             
    
    def plotDataHeight(self):
        clsmigure = plt.figure(figsize=(6,6))
        axCLSM = clsmigure.add_subplot(1, 1, 1)
        x = self.mergeDataSet[:,0]
        y = self.mergeDataSet[:,1]
        z = self.mergeDataSet[:,2]
        im = axCLSM.scatter(x, y, c=z, s=1, vmin=np.median(z)-np.var(z)*6, vmax=np.median(z)+np.var(z)*6, cmap=plt.cm.gray)
        clsmigure.colorbar(im)
        clsmigure.gca().invert_yaxis()
        clsmigure.show()  
    
    def plotDataContrast(merge_EBSD_CLSM_DATA):
        plt.figure(figsize=(6, 6))
        x = merge_EBSD_CLSM_DATA[:,0]
        y = merge_EBSD_CLSM_DATA[:,1]
        z = merge_EBSD_CLSM_DATA[:,2]
        plt.scatter(x, y, c=z, s=0.5, cmap=plt.cm.gray)
    
      
    
    #%%
    def plot_IPFz(X, Y, Z, resolution, cbarrange= [0,4], Digits=5,Title='IPFz', cbarlabel='label',Markersize=20, cmap1=mpl.cm.jet):
        plt.title(Title, y=1.08, size=10*resolution)
        plt.axis('off')
        plt.xlim([-5*resolution, 125*resolution])
        plt.text(-10*resolution, -10*resolution, r'$\langle$100$\rangle$', color='black', size=20)
        plt.text(110*resolution, -10*resolution, r'$\langle$101$\rangle$', color='black', size=20)
        plt.text(105*resolution, 111*resolution, r'$\langle$111$\rangle$', color='black', size=20)
        plt.ylim([-5*resolution, 125*resolution])
    
    
        plt.scatter(X, Y, c=Z, s=Markersize/resolution, marker="s",  edgecolor='', cmap=cmap1)
        
        plt.clim(cbarrange)
        
        bounds = np.linspace(cbarrange[0],cbarrange[1],Digits)
        cbar = plt.colorbar(ticks=bounds)
        cbar.set_label(cbarlabel, size=24, labelpad=20*resolution)
        for l in cbar.ax.yaxis.get_ticklabels():
            l.set_size(24)
    
    def plot_IPFz_Log10(X, Y, Z, resolution, cbarrange= [0,4],Digits=4, Title='IPFz', cbarlabel='label'):
        plt.figure(figsize=(14,10))
        plt.title(Title, y=1.08, size=10*resolution)
        plt.text(10*resolution,130*resolution,'DataSum=', color='black', size=10*resolution)
        plt.text(40*resolution,130*resolution,np.sum(Z), color='black', size=10*resolution)
        plt.axis('off')
        plt.xlim([0, 250])
        plt.text(-10*resolution, -5*resolution, '$\langle$100>', color='black', size=10*resolution)
        plt.text(115*resolution, -5*resolution, '$\langle$101>', color='black', size=10*resolution)
        plt.text(105*resolution, 111*resolution, '$\langle$111>', color='black', size=10*resolution)
        plt.ylim([-5*resolution, 125*resolution])
    
    
        plt.scatter(X, Y, c=np.log10(Z), s=20/resolution, marker="s",  edgecolor='', cmap=mpl.cm.jet)
        
        plt.clim(cbarrange)
        
        bounds = np.linspace(cbarrange[0],cbarrange[1],Digits)
        cbar = plt.colorbar(ticks=bounds)
        cbar.set_label(cbarlabel, size=22, labelpad=20*resolution)
        for l in cbar.ax.yaxis.get_ticklabels():
            l.set_size(24)
    
    def plot_IPF_plane(self,IPF=[2,1,2],testsize=15,text='',yTextoffset=0):
        XY = ipfzXY.poleA2(IPF[0],IPF[1],IPF[2])
        fig = plt.gcf()
        ax0 = fig.get_axes()[0]
        XY/= 150
        XYtextposition = XY+0.005
        ax0.plot(XY[0], XY[1], 'k', marker='+',markersize=20)
        if (text == ''):
            klammer = r'\langle' +  f'{int(IPF[0])}   {int(IPF[1])} {int(IPF[2])}' + r'\rangle'
            ax0.text(XYtextposition[0],XYtextposition[1]-yTextoffset, r'$%s$' %klammer, color='black', size=testsize)
        else:
            
            ax0.text(XYtextposition[0],XYtextposition[1]-yTextoffset, r'{}'.format(text), color='black', size=testsize)
        fig.canvas.draw_idle()
            
    def diffIPFforOptimisation(self, X, Y, Z, X2,Y2, Z2, resolution2):

        resolution = 2
        resolution /=300
        figIPF, ax = plt.subplots(figsize=(10,10))
        
        Z = np.subtract(Z,np.min(Z))
        Z = Z/np.max(Z)
        Z2 = np.subtract(Z2,np.min(Z2))
        Z2 = Z2/np.max(Z2)
    

        xi, yi = np.mgrid[0:1:200j,0:1:200j]
        
        zi = griddata((np.divide(X,150*resolution2), np.divide(Y,150*resolution2)), Z, (xi, yi))
        zi2 = griddata((np.divide(X2,150*resolution2), np.divide(Y2,150*resolution2)), Z2, (xi, yi))
        ##Nomieren
        

        ax.scatter(xi, yi, c=zi-zi2, s=20,  vmin=-0.1, vmax= 0.2, cmap=plt.get_cmap('jet'))
        zdiff = zi-zi2
        mask = np.logical_not(np.isnan(zdiff))
        zdiff = zdiff[mask]
        print(np.var(zdiff))




    
    def plot_inter_IPFz(self, X, Y, Z, resolution2, cbarrange= [0,4], Title='IPFz', cbarlabel='label', ticks = (0,0.5,1,1.5,2),contourplot=0, cmap1=plt.get_cmap('jet'), log=0, savepath = 1, pltshow=1):

        resolution = 2
        resolution /=300
        if pltshow ==0:
                plt.ioff()
        figIPF, ax = plt.subplots(figsize=(10,10))
    
        frame = mapplot.Wedge()
        frame_polygon = frame.get_polygon()
        frame_patch = mpl.patches.Polygon(frame_polygon, closed=True, fill=False,
                               color='k', linewidth =4.5)
    
        ax.add_patch(frame_patch)
    
        mapplot.plot_grid(frame)
        plt.axis('off')
        klammer = r'\langle 100 \rangle'
        plt.text(-10*resolution, -5*resolution, '$%s$' %klammer, color='black', size=20)
        klammer = r'\langle 101 \rangle'
        plt.text(120*resolution, -5*resolution, '$%s$' %klammer, color='black', size=20)
        klammer = r'\langle 111 \rangle'
        plt.text(105*resolution, 111*resolution, '$%s$' %klammer, color='black', size=20)
        plt.ylim([-5*resolution, 125*resolution])
        plt.xlim([-5*resolution, 125*resolution])
        
    
        xi, yi = np.mgrid[0:1:500j,0:1:500j]
        
        zi = griddata((np.divide(X,150*resolution2), np.divide(Y,150*resolution2)), Z, (xi, yi))
        if contourplot==1:
            ax.scatter(xi, yi, c=zi, s=10,  vmin=cbarrange[0], vmax=cbarrange[1], cmap=cmap1, marker="s")
        else:
            ax.scatter(np.divide(X,150*resolution2), np.divide(Y,150*resolution2), c=Z, s=20/resolution2, marker="s",   vmin=cbarrange[0], vmax=cbarrange[1], cmap=cmap1)
        #
    
        cax = inset_axes(ax,
                         width="60%",  # width = 60% of parent_bbox width
                         height="5%",  # height = 5% of parent_bbox width
                         loc='upper left',
                         bbox_to_anchor=(0,-0.2,1,1),
                         bbox_transform=ax.transAxes,
                         borderpad=0,
                             )
        c = [cbarrange[0],cbarrange[1]]
        a = [0,0]   
        cs2 = ax.scatter(a, a, c=c, s=0.000001,  cmap=cmap1)
        cbar = figIPF.colorbar(cs2, cax=cax, orientation='horizontal', ticks=ticks)
    
        if (log ==1):
            cbar.ax.set_xticklabels(['1', '10', '100', '1000', '10000', '100000', '1000000'] )
            
        frame =  Wedge()
        frame_polygon = frame.get_polygon()
        frame_patch = mpl.patches.Polygon(frame_polygon, closed=True, fill=False,
                                      color='k')
    
        ax.add_patch(frame_patch)
    
    
        cbar.set_label( r'{}'.format(cbarlabel), size=20, labelpad=10)
        for l in cbar.ax.xaxis.get_ticklabels():
            #l.set_family("Times New Roman")
            l.set_size(18)
        ax.add_patch(frame_patch)
        if savepath != 1:
            figIPF.savefig(savepath)
        if pltshow ==1:
            figIPF.show()
        elif pltshow ==0:
            plt.close('all')
            plt.ion()
        elif pltshow ==2:
            print('select points')
            return figIPF, ax
        else:
            plt.close('all')
        
        
    def AngleSkalar(x,y):
        dot = np.dot(x,y)
        x_modulus = np.sqrt((x*x).sum())
        y_modulus = np.sqrt((y*y).sum())
        cos_angle = dot / x_modulus / y_modulus 
        angle = np.arccos(cos_angle) # Winkel in Bogenmaß
        return angle
    
    #%%
    def IPFYandIPFX(self):
        print('Berechne IPFx')
        b = np.zeros((self.mergeDataSet.shape[0],self.mergeDataSet.shape[1]+4))
        IPFX = self.rotationEBSD(0,90,0)
        print('Berechne IPFy')
        IPFY = self.rotationEBSD(90,90,0)
        b[:,:-4] = self.mergeDataSet
        b[:,-4:-2] = IPFX[:,-2:]
        b[:,-2:] = IPFY[:,-2:]
        self.mergeDataSet = b
    
    def rotateIPFZ(self, phi1,phi2,phi3):
        
        self.mergeDataSet = self.rotationEBSD(phi1,phi2,phi3)
    
    def rotationEBSD(self, phi1,phi2,phi3):
        mat2 = self.mergeDataSet.copy()
        posiotiondrehen = np.array([[0],[0],[1]])
        R1 = self.rotation(phi1, phi2, phi3)
        
        for i in range(0,len(mat2[:,6])):
            R2 = self.rotation(mat2[i,3], mat2[i,4], mat2[i,5])
            R = np.dot(np.linalg.inv(R1),R2)
            xx , yy , zz = np.dot(np.transpose(R),posiotiondrehen)
            mat2[i,6], mat2[i,7],  mat2[i,8] = xx[0], yy[0], zz[0]
            r = np.sqrt(mat2[i,6] ** 2 + mat2[i,7] ** 2 + mat2[i,8] ** 2)
            mat2[i,6] = mat2[i,6] / r
            mat2[i,7] = mat2[i,7] / r
            mat2[i,8] = mat2[i,8] / r
            
            
        for k in range(0,len(mat2[:,6])):
                mat2[k,9:11] =ipfzXY.poleA2(mat2[k,6],mat2[k,7],mat2[k,8])
        return mat2
        
    def rotation(self, phi1,phi,phi2):
       phi1=phi1*np.pi/180;
       phi=phi*np.pi/180;
       phi2=phi2*np.pi/180;
       R=np.array([[np.cos(phi1)*np.cos(phi2)-np.cos(phi)*np.sin(phi1)*np.sin(phi2),
                -np.cos(phi)*np.cos(phi2)*np.sin(phi1)-np.cos(phi1)*
                np.sin(phi2),np.sin(phi)*np.sin(phi1)],[np.cos(phi2)*np.sin(phi1)
                +np.cos(phi)*np.cos(phi1)*np.sin(phi2),np.cos(phi)*np.cos(phi1)
                *np.cos(phi2)-np.sin(phi1)*np.sin(phi2), -np.cos(phi1)*np.sin(phi)],
                [np.sin(phi)*np.sin(phi2), np.cos(phi2)*np.sin(phi), np.cos(phi)]],float)
       return R
