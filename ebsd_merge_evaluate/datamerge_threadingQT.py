# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 20:19:12 2017

@author: karsten
"""


import numpy as np
import cv2 as cv
from ebsd_merge_evaluate import ipfzXY
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
from matplotlib.patches import Rectangle
from scipy import interpolate
import scipy.linalg
from PIL import Image
from skimage.transform import PiecewiseAffineTransform, warp
from skimage import io
from os.path import join
from PyQt5.QtCore import QObject, pyqtSignal #, QThread
#from mpl_interactions import panhandler, zoom_factory


class parent_merge(QObject):
    def __init__(self):
        super().__init__()
        self.P1  = np.array([0,0])
        self.P2  = np.array([0,0])
        self.P  = np.array([0,0])
        self.Pc = np.array([0,0])
        self.Ptiff = np.array([0,0])
        
        self.confocal_data = np.zeros((2, 2))
        self.confocal_image = np.zeros((2, 2))
        self.confocal_image_data_1 = np.zeros((2, 2))
        self.confocal_image_data_2 = np.zeros((2, 2))
        
        
        self.X = np.zeros((2, 2))
        self.Y = np.zeros((2, 2))
        self.XY = np.zeros((2,2))
        
        self.HKL = np.zeros((2, 2))
        self.Heights = np.zeros((2,2))
        self.Categories = np.zeros((2,2))
        self.Orientation = np.zeros((2,2))
        self.color = np.zeros((2,2))
        self.Zerowerte_all = 0
        self.dataclsm = 0
        self.threading_true = False
        
    def render_confocal_image(self, data, lower_scale = -1, upper_scale = 1):
        mean_confocal = np.mean(data)
        std_confocal = np.std(data)
        
        min_scale = mean_confocal + lower_scale*std_confocal
        max_scale = mean_confocal + upper_scale*std_confocal

        ls = LightSource(azdeg = 315, altdeg = 45)
        cmap = plt.cm.gray
        M = np.rot90(data[::-1, :], k = 3)
        
        return ls.shade(M, cmap=cmap, vert_exag = 10, blend_mode='hsv', vmin=min_scale, vmax=max_scale)


    def pattern_matching_auto(self,depth=25,leveling=1,levelplotrange=1,):
        try:
            image_data_1 = self.confocal_image_data_1#[::-1, :]
            image_data_2 = self.confocal_image_data_2#[::-1, :]
            
            dataS=np.reshape(image_data_1, (-1))
            dataS.sort()
            
            minval=0.02
            maxval=0.98
            
            lowerBoundCalPos=int(minval*(len(dataS)-1))
            upperBoundCalPos=int(maxval*(len(dataS)-1))
            if not (lowerBoundCalPos>=0):
                lowerBoundCalPos=0
            if not (lowerBoundCalPos<=(len(dataS)-1)):
                lowerBoundCalPos=int((len(dataS)-1))
            
            if not (upperBoundCalPos>=lowerBoundCalPos):
                upperBoundCalPos=lowerBoundCalPos
            if not (upperBoundCalPos<=(len(dataS)-1)):
                upperBoundCalPos=int((len(dataS)-1))
                
            lowerBoundCal=dataS[lowerBoundCalPos]
            upperBoundCal=dataS[upperBoundCalPos]
            
            image_data_1=((image_data_1-lowerBoundCal)/(-lowerBoundCal+upperBoundCal)*256)
            image_data_1[image_data_1>=255]=255
            image_data_1[image_data_1<=0]=0
            image_data_1=image_data_1.astype(np.uint8)
            
            Image.fromarray(image_data_1).save(join('tmp','CLSM_1.png'))
            
            image_data_2=((image_data_2-lowerBoundCal)/(-lowerBoundCal+upperBoundCal)*256)
            image_data_2[image_data_2>=255]=255
            image_data_2[image_data_2<=0]=0
            image_data_2=image_data_2.astype(np.uint8)
            
            Image.fromarray(image_data_2).save(join('tmp','CLSM_2.png'))

            orb = cv.ORB_create()
            kp1, des1 = orb.detectAndCompute(image_data_1,None)
            kp2, des2 = orb.detectAndCompute(image_data_2,None)
            # create BFMatcher object
            bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
            # Match descriptors.
            matches = bf.match(des1,des2)
            matches = sorted(matches, key = lambda x:x.distance)
            matches = matches[:depth]

            # Draw matches of each quadrant.
            img3 = cv.drawMatches(image_data_1,kp1,image_data_2,kp2,matches,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            
            cv.imwrite(join('tmp','ImgComp.png'),img3)
            
            P1=np.array([kp1[j].pt for j in np.array([i.queryIdx for i in matches])]).astype(int)
            P2=np.array([kp2[j].pt for j in np.array([i.trainIdx for i in matches])]).astype(int)
            
            self.save_diff_points(P1, P2)
            
            figdiff, ax1 = plt.subplots(1, 1)
            mng = plt.get_current_fig_manager()
            mng.resize(1500,800)


            ax1.imshow(img3)
            ax1.set_title('CLSM Matching Points Difference Map', fontsize=16)
            
            self.confocal_data_1 = np.flipud(np.rot90(self.confocal_data_1))
            self.confocal_data_2 = np.flipud(np.rot90(self.confocal_data_2))
            
            textstr1 = 'Enter: resume difference calculation'
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    
            # place a text box in upper left in axes coords
            ax1.text(-0.12, -0.05, textstr1, transform=ax1.transAxes, fontsize=12,
                              va='top',ha='center', bbox=props)
            
            _ = figdiff.ginput(32, timeout=-1)
            
            plt.close(figdiff)
            
            self.affine_transformation(P1, P2, leveling, levelplotrange)

        except:
            print('CLSM data difference failed')


    def load_confocal_data_diff_plt(self,leveling=1,levelplotrange=1,):
        try:
            figdiff, ((ax1), (ax2)) = plt.subplots(1, 2)
            mng = plt.get_current_fig_manager()
            mng.resize(1800,800)
            ax2.imshow(self.confocal_image_data_1)
            ax2.set_title('CLSM data2', fontsize=16)
            ax1.imshow(self.confocal_image_data_2)
            ax1.set_title('CLSM data1', fontsize=16)
            self.confocal_data_1 = np.flipud(np.rot90(self.confocal_data_1))
            self.confocal_data_2 = np.flipud(np.rot90(self.confocal_data_2))
            
            # textstr2 = '\n'.join((r'Left click: Select a point  -----  Right click: Delete the previously selected point', 
            #                               r'Enter: Complete point selection',))
                        #                  r'Find the position (e.g. in total 4 positions) in both pictures'
            textstr1 =  '\n'.join((r'Please select the points at indentical features. First select point in EBSD data than',
                                   r'in CLSM data. Repeat the same procedure for as many points as needed (4 at least).',
                                   r'',
                                   r'Controls:',
                                   r'Left click: Select a point ----- Right click: Delete previously selected point',
                                   r'Mouse Wheel: Zoom | Refrain from using lens button -> klicks counted as selections', # Every corner of the selected rectangles will be counted as points
                                   r'Enter: Complete point selection',))
            
            # these are matplotlib.patch.Patch properties
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    
            # place a text box in upper left in axes coords
            ax1.text(.5, -0.1, textstr1, transform=ax1.transAxes, #fontsize=12,
                             verticalalignment='top', bbox=props)
    
            # ax2.text(-0.1, -0.05, textstr2, transform=ax2.transAxes, fontsize=12,
            #                  verticalalignment='top', bbox=props)

            Ptmp = figdiff.ginput(32, timeout=-1)
            Ptmp = np.asarray(Ptmp)
            plt.close(figdiff)
            P1 = np.zeros([int(len(Ptmp)/2),2])
            P2 = np.zeros([int(len(Ptmp)/2),2])

            for i in range(int(len(Ptmp)/2)):
                P1[i,:] = Ptmp[int(i*2),:]
                P2[i,:] = Ptmp[int(i*2)+1,:]
            
            self.save_diff_points(P1, P2)
            
            self.affine_transformation(P1, P2, leveling, levelplotrange)

        except:
            print('CLSM data difference failed')
            
    def confocal_diff_from_file(self,file_name = 'tmp/000_selected_points_difference_microscopy.txt',leveling=1,levelplotrange=1):   
        try:
            # mng = plt.get_current_fig_manager()
            # mng.resize(1800,800)

            self.confocal_data_1 = np.flipud(np.rot90(self.confocal_data_1))
            self.confocal_data_2 = np.flipud(np.rot90(self.confocal_data_2))
            
            P1, P2 = self.load_diff_points(file_name)
            
            self.affine_transformation(P1, P2, leveling, levelplotrange)

        except:
            print('CLSM data difference failed')
    
    def affine_transformation(self, P1, P2, leveling = 1,levelplotrange=1):
        if np.max(P1) < np.max(P2):
            m,n,dataouttmp = self.sub_tform(P1, P2, self.confocal_data_1, self.confocal_data_2)
            
        else:
            m,n,dataouttmp = self.sub_tform(P2, P1, self.confocal_data_2, self.confocal_data_1)
        
        X,Y = np.meshgrid(np.linspace(0, n-1, n), np.linspace(0, m-1, m))
        A = np.c_[dataouttmp[:,0], dataouttmp[:,1], np.ones(dataouttmp.shape[0])]
        C,_,_,_ = scipy.linalg.lstsq(A,dataouttmp[:,2])
        
        mean_height= np.mean(dataouttmp[:,2])
        Z = C[1]*X + C[0]*Y 
        
        plt.ioff()            
           
        plt.imshow(self.confocal_data)
        plt.colorbar()
        plt.clim([mean_height-levelplotrange,mean_height+levelplotrange])
        plt.savefig('tmp/000_difference_CLSM1_CLSM2.png')
        plt.close()
        
        if leveling==1:
            print('geometrical leveling..................................')    
            
            plt.imshow(Z)
            plt.colorbar()
            plt.savefig('tmp/000_difference_CLSM1_CLSM2_background.png')
            plt.close()
            
            self.confocal_data -= Z
            plt.imshow(self.confocal_data)
            plt.colorbar()
            plt.clim([mean_height-levelplotrange,mean_height+levelplotrange])
            plt.savefig('tmp/000_difference_CLSM1_CLSM2_leveled.png')
            plt.close()
        
        self.confocal_data = np.flipud(np.rot90(self.confocal_data))
        np.savetxt('tmp/000_difference_CLSM1_CLSM2.dat',self.confocal_data)
    
        plt.ion()
    
        self.confocal_image = self.render_confocal_image(self.confocal_data)
        
        self.P1 = P1
        self.P2 = P2 
        self.view_confocal_data()
        
    def sub_tform(self, P1, P2, data1, data2, leveling = 1):
        tform = PiecewiseAffineTransform()
        tform.estimate(P1, P2)
        dataout = warp(data1 , tform)
        
        if dataout.shape[0] > data2.shape[0]:
            dataout = dataout[:data2.shape[0],:]
        else:
            data2 = data2[:dataout.shape[0],:]
            
        if dataout.shape[1] > data2.shape[1]:
            dataout = dataout[:,:data2.shape[1]]
        else:
            data2 = data2[:,:dataout.shape[1]]
        
        self.confocal_data = data2-dataout
        if leveling==1:  
            m,n = self.confocal_data.shape
            row,column = np.mgrid[:m,:n]
            dataouttmp = np.column_stack((row.ravel(),column.ravel(), dataout.ravel()))
            data2tmp = np.column_stack((row.ravel(),column.ravel(), data2.ravel()))     
       
        data2tmp = data2tmp[dataouttmp[:,2] !=0]
        dataouttmp = dataouttmp[dataouttmp[:,2] !=0] 
        dataouttmp[:,2] = data2tmp[:,2] - dataouttmp[:,2] 
        
        return m,n,dataouttmp
        
            
    def load_confocal_data_diff(self):
        try:
            rotation_data_1 = self.deg_to_rot(int(self.mergeroationCLSM2))
            rotation_data_2 = self.deg_to_rot(int(self.mergeroationCLSM1))
            
            file_name1 =  self.loadCLSM2line
            file_name2 =  self.loadCLSM1line
                
    
            anzahlrows =  np.loadtxt(file_name1, delimiter=',', skiprows=25, usecols=[5])
            anzahlColums = np.loadtxt(file_name1, delimiter=',', dtype='str', skiprows=25+len(anzahlrows)-1)
            self.confocal_data_1 = np.loadtxt(file_name1, delimiter=',', skiprows=25, usecols=list(np.linspace(1,len(anzahlColums)-2,len(anzahlColums)-1,dtype=int)))
            self.confocal_data_1 = self.confocal_data_1[:, 1:]
            
            if self.CLSM2checkBox == 2:
                self.confocal_data_1 = self.confocal_data_1[::-1, :]

            self.confocal_data_1 = np.rot90(self.confocal_data_1, k = rotation_data_1)
            anzahlrows =  np.loadtxt(file_name2, delimiter=',', skiprows=25, usecols=[5])
            anzahlColums = np.loadtxt(file_name2, delimiter=',', dtype='str', skiprows=25+len(anzahlrows)-1)
            self.confocal_data_2 = np.loadtxt(file_name2, delimiter=',', skiprows=25, usecols=list(np.linspace(1,len(anzahlColums)-2,len(anzahlColums)-1,dtype=int)))
            self.confocal_data_2 = self.confocal_data_2[:, 1:]
    
            
            if self.CLSM1checkBox == 2:
                self.confocal_data_2 = self.confocal_data_2[::-1, :]
            
            self.confocal_data_2 = np.rot90(self.confocal_data_2, k = rotation_data_2)
            
            print('Data2_Load................')
            self.confocal_image_data_1 = self.render_confocal_image(self.confocal_data_1)
            print('Render_Data_1_....................')
            self.confocal_image_data_2 = self.render_confocal_image(self.confocal_data_2)
            print('Render_Data2_..........................')
            
        except:
            print('CLSM1 and 2 data load failed')
            
        if self.threading_true: self.finished.emit()    
            
            
    def deg_to_rot(self, rot = 0):
        return int(rot / 90)
            

class mergeThread(parent_merge):
    finished = pyqtSignal()
    deletThread =  pyqtSignal()
    progress = pyqtSignal(int)
    def __init__(self):
        super().__init__()
        self.threading_true = True
        
    #%%
    def load_confocal_data(self):
        try:

            
            file_name = self.loadCLSM1line
            rot = self.deg_to_rot(int(self.mergeroationCLSM1))
                        
            print('Reading file: %s, please wait... this can take several minutes...'%file_name)
            anzahlrows =  np.loadtxt(file_name, delimiter=',', skiprows=25, usecols=[5])
            anzahlColums = np.loadtxt(file_name, delimiter=',', dtype='str', skiprows=25+len(anzahlrows)-1)
            data = np.loadtxt(file_name, delimiter=',', skiprows=25, usecols=list(np.linspace(1,len(anzahlColums)-2,len(anzahlColums)-1,dtype=int)))
            data = data[:, 1:]
            if self.CLSM1checkBox == 2:
                data= np.flipud(data)
                
            if rot>0:
                data= np.rot90(data,k=rot)

            self.confocal_data = data
            

            if self.leveling!=0:
                print('geometrical leveling..................................')    
                m,n = data.shape
                row,column = np.mgrid[:m,:n]
                data2tmp = np.column_stack((row.ravel(),column.ravel(), data.ravel()))     
                
                A = np.c_[data2tmp[:,0], data2tmp[:,1], np.ones(data2tmp.shape[0])]
                C,_,_,_ = scipy.linalg.lstsq(A,data2tmp[:,2])
                           
                X,Y = np.meshgrid(np.linspace(0, n-1, n), np.linspace(0, m-1, m))
                
                self.Zebene = C[1]*X + C[0]*Y 
                self.confocal_data = self.confocal_data-self.Zebene
            
            
            self.confocal_image =  self.render_confocal_image(data = self.confocal_data)
            self.finished.emit()
        except:
            print('CLSM data load failed')
            self.finished.emit()
            
    def loadAFMdata(self):
        try:           
            file_name = self.loadAFM1line
            rot = self.deg_to_rot(int(self.mergeroationAFM1))            

            print('Reading file: %s, please wait... this can take several minutes...'%file_name)
            dataSet = self.dataSetselect
            data = 0
            data = data['wave']['wData'][:,:,dataSet]
            if self.AFM1checkBox == 2:
                data= np.flipud(data)
                
            if rot>0:
                data= np.rot90(data,k=rot)

            self.confocal_data = data

            if self.leveling!=0:
                print('geometrical leveling..................................')    
                m,n = data.shape
                row,column = np.mgrid[:m,:n]
                data2tmp = np.column_stack((row.ravel(),column.ravel(), data.ravel()))     
                A = np.c_[data2tmp[:,0], data2tmp[:,1], np.ones(data2tmp.shape[0])]
                C,_,_,_ = scipy.linalg.lstsq(A,data2tmp[:,2])
                X,Y = np.meshgrid(np.linspace(0, n-1, n), np.linspace(0, m-1, m))
                self.Zebene = C[1]*X + C[0]*Y 
                self.confocal_data = self.confocal_data-self.Zebene
            
            self.confocal_image =  self.render_confocal_image(data = self.confocal_data)
            self.finished.emit()
        except:
            print('AFM data load failed')
            self.finished.emit()

    
    def loadAFMdataDiff(self):
            rotation_data_1 = self.deg_to_rot(int(self.mergeroationAFM2))
            rotation_data_2 = self.deg_to_rot(int(self.mergeroationAFM1))
            
            flip_data_1 = self.AFM2checkBox
            flip_data_2 = self.AFM1checkBox

            dataSet = self.dataSetselect

            dataAFM1 = 0
            self.confocal_data_1 = dataAFM1['wave']['wData'][:,:,dataSet]
    
            
            if flip_data_1 == 2:
                self.confocal_data_1 = self.confocal_data_1[::-1, :]

            self.confocal_data_1 = np.rot90(self.confocal_data_1, k = rotation_data_1)
    
            dataAFM2 = 0
            self.confocal_data_2 = dataAFM2['wave']['wData'][:,:,dataSet]
            
            
            if flip_data_2 == 2:
                self.confocal_data_2 = self.confocal_data_2[::-1, :]
            
            self.confocal_data_2 = np.rot90(self.confocal_data_2, k = rotation_data_2)
            
            print('Data2_Load................')
            self.confocal_image_data_1 = self.render_confocal_image(self.confocal_data_1)
            print('Render_Data_1_....................')
            self.confocal_image_data_2 = self.render_confocal_image(self.confocal_data_2)
            print('Render_Data2_..........................')
            self.finished.emit()


    def loadPicdata(self):
        try:
            file_name = self.loadPicDataline
            rot = self.deg_to_rot(int(self.mergeroationPicData))
           
            print('Reading file: %s, please wait... this can take several minutes...'%file_name)
            data =  io.imread(file_name, as_gray=False)
            if len(data.shape) == 3:
                data = data[:,:,2]
                
            data= np.rot90(data,k=1) 
            data= np.flipud(data)
            data = data[:, 1:]
            
            if self.PicDatacheckBox == 2:
                data= np.flipud(data)
                
            if rot>0:
                data= np.rot90(data,k=rot)

            self.confocal_data = data
            
            print('Rendering picture data, please wait...')
            
            self.confocal_image = np.rot90(data[::-1, :], k=3)  
            self.finished.emit()
        except:
            print('Pic data load failed')
            self.finished.emit()

    #%%
    def load_EBSD_data(self):
        try:
            file_name= self.fileNameEBSD
            phase= self.phaseEBSD
            print('Reading file: %s, please wait... this can take several minutes...'%file_name)

            with open(file_name,"r",errors="replace") as f:
                CRSTdata = f.readlines()
            CRSTdata = CRSTdata[1:]
            num = len(CRSTdata)
            X = -1*np.ones(num,)
            Y = -1*np.ones(num,)
            Orientation = np.zeros((num,3))
            
            XEBSDPicture = -1*np.ones(num,)
            YEBSDPicture = -1*np.ones(num,)
            OrientationEBSDPicture = np.zeros((num,3))
            start_of_data = 0
            

            for i , lines in enumerate(CRSTdata):
                cells = lines.split()
    
                if cells[0] == "XCells":
                    XCells = cells[1]
                    print(f"The Number of XCells is {XCells}")
                    X_grid = int(XCells)
                if cells[0] == "YCells":
                    YCells = cells[1]
                    print(f"The Number of YCells is {YCells}")    
                    Y_grid = int(YCells)
                if cells[0] == 'XStep':
                    XStep = cells[1]
                if cells[0] == 'YStep':
                    YStep = cells[1]
        
        
                if cells[0] == phase:
                    X[i] = float(cells[1]) / float(XStep)
                    Y[i] = float(cells[2]) / float(YStep)
                    Orientation[i,:] = cells[5:8]
        
                if start_of_data == 1:
                    XEBSDPicture[i] = float(cells[1]) / float(XStep)
                    YEBSDPicture[i] = float(cells[2]) / float(YStep)
                    OrientationEBSDPicture[i,:] = cells[5:8]
                if cells[0] == 'Phase':
                    start_of_data = 1
    
            Orientation = Orientation[X != -1, :]
            X = X[X != -1]
            Y = Y[Y != -1]
            Orientation = Orientation.reshape(-1, 3)
            X = X.astype(float)
            Y = Y.astype(float)
            Orientation = Orientation.astype(float)
        ##### For Picture  
            print('Start imaging...')
        
            OrientationEBSDPicture = OrientationEBSDPicture[XEBSDPicture != -1, :]
            XEBSDPicture = XEBSDPicture[XEBSDPicture != -1]
            YEBSDPicture = YEBSDPicture[YEBSDPicture != -1]
            OrientationEBSDPicture = OrientationEBSDPicture.reshape(-1, 3)
               
            NormaEBSDPicture = OrientationEBSDPicture.max(axis=0)
            colorEBSDPicture = OrientationEBSDPicture / NormaEBSDPicture
            colorz = colorEBSDPicture[:,0]+colorEBSDPicture[:,1]+colorEBSDPicture[:,2]
            Z_grid = colorz.reshape((Y_grid, X_grid))
    
            Z_grid = Z_grid[::-1,:]
            
            HKL = np.zeros(Orientation.shape)
            HKL[:,0] = np.sin((Orientation[:, 1]) * np.pi / 180) * np.sin(Orientation[:, 2] * np.pi / 180)
            HKL[:,1] = np.sin((Orientation[:, 1]) * np.pi / 180) * np.cos(Orientation[:, 2] * np.pi / 180)
            HKL[:,2] = np.cos((Orientation[:, 1])* np.pi / 180)
            r = np.sqrt(HKL[:,0] ** 2 + HKL[:,1] ** 2 + HKL[:,2] ** 2)
            HKL[:,0] = HKL[:,0] / r
            HKL[:,1] = HKL[:,1] / r
            HKL[:,2] = HKL[:,2] / r
            
            print('Done.')
            Norma = Orientation.max(axis=0)
            self.color = Orientation / Norma
            self.HKL = HKL
            self.X = X
            self.Y = Y 
            self.Orientation = Orientation
            self.X_grid = X_grid
            self.Y_grid = Y_grid
            self.Z_grid = Z_grid
            self.finished.emit()
            return True
        except:
            print('EBSD data load failed!')
            self.finished.emit()



class mergedata(parent_merge):
#%%
    
    def calculate_new_coordinates(self):
        dst_EBSD = self.Pc
        src_confocal = self.P
        if len(dst_EBSD) != len(src_confocal) :
            print("Destination and source have different number of points...")
            return -1
        tform = PiecewiseAffineTransform()
        tform.estimate(src_confocal,dst_EBSD)
        coordinates = np.ones((len(self.X), 2))
        coordinates[:, 0] = self.X
        coordinates[:, 1] = self.Y
        new_coordinates = tform.inverse(coordinates)
        return new_coordinates

    #%%
    
    def load_confocal_data(self):

        file_name = self.loadCLSM1line
        rot = self.deg_to_rot(int(self.mergeroationCLSM1))
         
        print('Reading file: %s, please wait... this can take several minutes...'%file_name)
        anzahlrows =  np.loadtxt(file_name, delimiter=',', skiprows=25, usecols=[5])
        anzahlColums = np.loadtxt(file_name, delimiter=',', dtype='str', skiprows=25+len(anzahlrows)-1)
        data = np.loadtxt(file_name, delimiter=',', skiprows=25, usecols=list(np.linspace(1,len(anzahlColums)-2,len(anzahlColums)-1,dtype=int)))
        data = data[:, 1:]
        if self.CLSM1checkBox != 0:
            data= np.flipud(data)
            
        if rot>0:
            data= np.rot90(data,k=rot)
            
        self.confocal_data = data
        if self.leveling!=0:
            print('geometrical leveling..................................')    
            m,n = data.shape
            row,column = np.mgrid[:m,:n]
            data2tmp = np.column_stack((row.ravel(),column.ravel(), data.ravel()))     
            A = np.c_[data2tmp[:,0], data2tmp[:,1], np.ones(data2tmp.shape[0])]
            C,_,_,_ = scipy.linalg.lstsq(A,data2tmp[:,2])
            X,Y = np.meshgrid(np.linspace(0, n-1, n), np.linspace(0, m-1, m))
            self.Zebene = C[1]*X + C[0]*Y 
            self.confocal_data = self.confocal_data-self.Zebene
        
        self.confocal_image =  self.render_confocal_image(data = self.confocal_data)

       
    def load_confocal_data2(self, file_name=' ', rot=0,flip=0):
            file_name = self.loadCLSM1line
            
            rot = self.deg_to_rot(int(self.mergeroationCLSM1))
            
            print('Reading file: %s, please wait... this can take several minutes...'%file_name)
            anzahlrows =  np.loadtxt(file_name, delimiter=',', skiprows=25, usecols=[5])
            anzahlColums = np.loadtxt(file_name, delimiter=',', dtype='str', skiprows=25+len(anzahlrows)-1)
            data = np.loadtxt(file_name, delimiter=',', skiprows=25, usecols=list(np.linspace(1,len(anzahlColums)-2,len(anzahlColums)-1,dtype=int)))
            data = data[:, 1:]
            if rot>0:
                data= np.rot90(data,k=rot)
            if flip==1:
                data= np.flipud(data)
            self.confocal_data_2 = data
    
    
    
    def load_confocal_data1_rot_flip(self):
        if hasattr(self, 'confocal_data_1'):
            rot = self.deg_to_rot(int(self.mergeroationCLSM1))
                
            if rot>0:
                self.confocal_data_1 = np.rot90(self.confocal_data_1,k=rot)
                self.confocal_data = self.confocal_data_1
                self.confocal_image =  self.render_confocal_image(data = self.confocal_data)
            
        else:
            print('Data not loaded geladen')
            
#%%            

    
    
    
    def save_diff_points(self, P1, P2):
        hString='CLSM1_xcoord CLSM1_ycoord CLSM2_xcoord CLSM2_ycoord'
        np.savetxt('tmp/000_selected_points_difference_microscopy.txt',np.concatenate((P1, P2), axis = 1), header=hString)
    
        # np.savetxt('tmp/000_confocal_data_diff_P1.txt',P1)
        # np.savetxt('tmp/000_confocal_data_diff_P2.txt',P2)
    
    def loadAFMdata(self):
        try:           
            file_name = self.loadAFM1line
            rot = self.deg_to_rot(int(self.mergeroationAFM1))         

            print('Reading file: %s, please wait... this can take several minutes...'%file_name)
            dataSet = self.dataSetselect
            data = 0
            data = data['wave']['wData'][:,:,dataSet]
            if self.AFM1checkBox == 2:
                data= np.flipud(data)
                
            if rot>0:
                data= np.rot90(data,k=rot)

            self.confocal_data = data

            if self.leveling!=0:
                print('geometrical leveling..................................')    
                m,n = data.shape
                row,column = np.mgrid[:m,:n]
                data2tmp = np.column_stack((row.ravel(),column.ravel(), data.ravel()))     
                A = np.c_[data2tmp[:,0], data2tmp[:,1], np.ones(data2tmp.shape[0])]
                C,_,_,_ = scipy.linalg.lstsq(A,data2tmp[:,2])
                X,Y = np.meshgrid(np.linspace(0, n-1, n), np.linspace(0, m-1, m))
                self.Zebene = C[1]*X + C[0]*Y 
                self.confocal_data = self.confocal_data-self.Zebene
            
            self.confocal_image =  self.render_confocal_image(data = self.confocal_data)
        except:
            print('AFM data load failed')

           
    def loadAFMdataDiff(self):
            rotation_data_1 = self.deg_to_rot(int(self.mergeroationAFM2))
            rotation_data_2 = self.deg_to_rot(int(self.mergeroationAFM1))
            
            dataSet = self.dataSetselect

            dataAFM1 = 0
            self.confocal_data_1 = dataAFM1['wave']['wData'][:,:,dataSet]
            
            if self.AFM2checkBox != 0:
                self.confocal_data_1 = self.confocal_data_1[::-1, :]

            self.confocal_data_1 = np.rot90(self.confocal_data_1, k = rotation_data_1)
    
            dataAFM2 = 0
            self.confocal_data_2 = dataAFM2['wave']['wData'][:,:,dataSet]
            
            if self.AFM1checkBox != 0:
                self.confocal_data_2 = self.confocal_data_2[::-1, :]
            
            self.confocal_data_2 = np.rot90(self.confocal_data_2, k = rotation_data_2)
            
            print('Data2_Load................')
            self.confocal_image_data_1 = self.render_confocal_image(self.confocal_data_1)
            print('Render_Data_1_....................')
            self.confocal_image_data_2 = self.render_confocal_image(self.confocal_data_2)
            print('Render_Data2_..........................')
            self.finished.emit()

    #%%
    def load_EBSD_data(self):
            file_name= self.fileNameEBSD
            phase= self.phaseEBSD
            print('Reading file: %s, please wait... this can take several minutes...'%file_name)

            with open(file_name,"r",errors="replace") as f:
                CRSTdata = f.readlines()
            CRSTdata = CRSTdata[1:]
            num = len(CRSTdata)
            X = -1*np.ones(num,)
            Y = -1*np.ones(num,)
            Orientation = np.zeros((num,3))
            
            XEBSDPicture = -1*np.ones(num,)
            YEBSDPicture = -1*np.ones(num,)
            OrientationEBSDPicture = np.zeros((num,3))
            start_of_data = 0
            for i , lines in enumerate(CRSTdata):
                cells = lines.split()
    
                if cells[0] == "XCells":
                    XCells = cells[1]
                    print(f"The Number of XCells is {XCells}")
                    X_grid = int(XCells)
                if cells[0] == "YCells":
                    YCells = cells[1]
                    print(f"The Number of YCells is {YCells}")    
                    Y_grid = int(YCells)
                if cells[0] == 'XStep':
                    XStep = cells[1]
                if cells[0] == 'YStep':
                    YStep = cells[1]
        
                if cells[0] == phase:
                    X[i] = float(cells[1]) / float(XStep)
                    Y[i] = float(cells[2]) / float(YStep)
                    Orientation[i,:] = cells[5:8]
        
                if start_of_data == 1:
                    XEBSDPicture[i] = float(cells[1]) / float(XStep)
                    YEBSDPicture[i] = float(cells[2]) / float(YStep)
                    OrientationEBSDPicture[i,:] = cells[5:8]
                if cells[0] == 'Phase':
                    start_of_data = 1
    
            Orientation = Orientation[X != -1, :]
            X = X[X != -1]
            Y = Y[Y != -1]
            Orientation = Orientation.reshape(-1, 3)
            X = X.astype(float)
            Y = Y.astype(float)
            Orientation = Orientation.astype(float)
        ##### For Picture  
            print('Start imageing............................................................')
        
            OrientationEBSDPicture = OrientationEBSDPicture[XEBSDPicture != -1, :]
            XEBSDPicture = XEBSDPicture[XEBSDPicture != -1]
            YEBSDPicture = YEBSDPicture[YEBSDPicture != -1]
            OrientationEBSDPicture = OrientationEBSDPicture.reshape(-1, 3)
               
            NormaEBSDPicture = OrientationEBSDPicture.max(axis=0)
            colorEBSDPicture = OrientationEBSDPicture / NormaEBSDPicture
            colorz = colorEBSDPicture[:,0]+colorEBSDPicture[:,1]+colorEBSDPicture[:,2]
            Z_grid = colorz.reshape((Y_grid, X_grid))
    
            Z_grid = Z_grid[::-1,:]
            
            HKL = np.zeros(Orientation.shape)
            HKL[:,0] = np.sin((Orientation[:, 1]) * np.pi / 180) * np.sin(Orientation[:, 2] * np.pi / 180)
            HKL[:,1] = np.sin((Orientation[:, 1]) * np.pi / 180) * np.cos(Orientation[:, 2] * np.pi / 180)
            HKL[:,2] = np.cos((Orientation[:, 1])* np.pi / 180)
            r = np.sqrt(HKL[:,0] ** 2 + HKL[:,1] ** 2 + HKL[:,2] ** 2)
            HKL[:,0] = HKL[:,0] / r
            HKL[:,1] = HKL[:,1] / r
            HKL[:,2] = HKL[:,2] / r
            
            print('Done...')
            Norma = Orientation.max(axis=0)
            self.color = Orientation / Norma
            self.HKL = HKL
            self.X = X
            self.Y = Y 
            self.Orientation = Orientation
            self.X_grid = X_grid
            self.Y_grid = Y_grid
            self.Z_grid = Z_grid
            return True

    #%%       
    def rotate_confocal_data(self, n=1):
        self.confocal_data = np.rot90(self.confocal_data, k = n)
        self.confocal_image = self.render_confocal_image(data=self.confocal_data)    
    
     #%%   
    def remove_confocal_data(self, number_of_removals):
        global confocal_data, confocal_image
        print('Waiting for user input...')
        for number in range(0,number_of_removals):
            self.view_confocal_data()
            P = plt.ginput(2, timeout=-1)
            P = np.asarray(P)
            if P[0][0] > P[1][0]:
                tmp = P[0][0]
                P[0][0] = P[1][0]
                P[1][0] = tmp
            if P[0][1] > P[1][1]:
                tmp = P[0][1]
                P[0][1] = P[1][1]
                P[1][1] = tmp
            
            self.confocal_data[int(P[0][0]):int(P[1][0]), int(P[0][1]):int(P[1][1])] = -1
            plt.close('all')
        self.confocal_image = self.render_confocal_image(self.confocal_data)
        
        return True
      
    #%%    
    def view_confocal_data(self, pause=0):
        self.fig_clsm = plt.figure(figsize=(10, 10))
        self.ax_clsm = self.fig_clsm.add_subplot(1, 1, 1)
        
        self.ax_clsm.imshow(self.confocal_image, cmap='gray')
        self.ax_clsm.set_title('Reconstruction of confocal data', fontsize=16)
        self.fig_clsm.tight_layout()
        if self.P.shape != (2,):
            plt.plot(self.P[:,0],self.P[:,1],'or', linewidth=3, markersize=12, alpha=0.5)
        self.ax_clsm.set_xlim(xmin=0, xmax= self.confocal_data.shape[0])
        self.ax_clsm.set_ylim(ymin=0, ymax= self.confocal_data.shape[1])
        self.fig_clsm.gca().invert_yaxis()

        print('Ready.')
        if (pause==1):
            print('CLSM input')
        else:
            self.fig_clsm.show()

    #%%
    
    def find_EBSD_boundaries(width = 3, tolerance = 0.3):
        global X,Y,HKL
        boundaries = np.zeros(X.shape, dtype = bool)
        mask1 = np.zeros(X.shape, dtype = bool)
        mask2 = np.zeros(X.shape, dtype = bool)
        mask = np.zeros(X.shape, dtype = bool)
        for i in range(len(X)):
            mask1 = np.abs(X - X[i]) < width
            mask2 = np.abs(Y - Y[i]) < width
            mask = np.logical_and(mask1,mask2)
            estimator = np.median( np.abs( HKL[mask,:] - HKL[i,:] ), axis=0 )
            if estimator.max() / HKL[i,estimator.argmax()] > tolerance:
                boundaries[i] = True
        return boundaries
    #%%   
    
    def view_EBSD_data(self):
        color = self.color
        print('Rendereing EBSD data, please wait...')
        fig_ebsd = plt.figure(figsize=(10, 10))
        ax_ebsd = fig_ebsd.add_subplot(1, 1, 1)
        Norma = self.Orientation.max(axis=0)
        color = self.Orientation / Norma
        ax_ebsd.imshow(self.Z_grid, extent=(np.min(self.X), np.max(self.X), np.min(self.Y), np.max(self.Y)),  cmap=plt.get_cmap('tab20b'), aspect = np.max(self.Y)/np.max(self.X))
        fig_ebsd.gca().invert_yaxis()
        ax_ebsd.set_title('Reconstruction of EBSD data', fontsize=16)
        fig_ebsd.tight_layout()
        if self.Pc.shape != (2,):
            plt.plot(self.Pc[:,0],self.Pc[:,1],'or', linewidth=3, markersize=12, alpha=0.5)
        
        self.color = color
        fig_ebsd.show()
        print('Ready.')
        
#%%
    def loadPicdataForDelete(self,file_name):         
            print('Reading file: %s, please wait... this can take several minutes...'%file_name)
            data =  io.imread(file_name, as_gray=False)
            if len(data.shape) == 3:
                data = data[:,:,2]
            data= np.flipud(data)    
            data= np.rot90(data,k=1) 
         
            data = data[:, 1:]
            self.imageForDeleteData = data
  
        
    def loadPicdata(self):
        try:
            file_name = self.loadPicDataline
            rot = self.deg_to_rot(int(self.mergeroationPicData))            
           
            print('Reading file: %s, please wait... this can take several minutes...'%file_name)
            data =  io.imread(file_name, as_gray=False)
            if len(data.shape) == 3:
                data = data[:,:,2]
                
            data= np.rot90(data,k=1) 
            data= np.flipud(data)
            data = data[:, 1:]
            if self.PicDatacheckBox != 0:
                data= np.flipud(data)
                
            if rot>0:
                data= np.rot90(data,k=rot)

            self.confocal_data = data
            
            print('Rendereing picture data, please wait...')           
            self.confocal_image = np.rot90(data[::-1, :], k=3)  

        except:
            print('Pic data load failed')

    
    def view_pic_data(self):
        print('Rendereing Tiff data, please wait...')
        plt.figure(figsize=(10, 10))
           
        plt.imshow(self.Ztiff_grid, extent=(np.min(self.Xtiff), np.max(self.Xtiff), 
                                            np.min(self.Ytiff), np.max(self.Ytiff)), cmap = plt.cm.gray, aspect = np.max(self.Ytiff)/np.max(self.Xtiff))
        plt.colorbar()
        plt.xlim(xmin=0, xmax=self.Xtiff.max())
        plt.ylim(ymin=0, ymax=self.Ytiff.max())
        plt.gca().invert_yaxis()
        plt.title('Reconstruction of Contrast data', fontsize=16)
        plt.tight_layout()
        if self.Ptiff.shape != (2,):
            plt.plot(self.Ptiff[:,0],self.Ptiff[:,1],'or', linewidth=3, markersize=12, alpha=0.5)
        plt.show()
        print('Ready.')
        
    #%%    
    def calibrate_confocal_and_EBSD_data_image(self):
        fig_point, ((ax1), (ax2)) = plt.subplots(1, 2)
        mng = plt.get_current_fig_manager()
        mng.resize(1800,800)
        ax1.set_title('EBSD data', fontsize=16)
        ax2.set_title('CLSM data', fontsize=16)
        
        ax1.imshow(self.Z_grid, extent=(np.min(self.X), np.max(self.X), np.min(self.Y), np.max(self.Y)),  cmap=plt.get_cmap('tab20b'), aspect = np.max(self.Y)/np.max(self.X))
        if self.Pc.shape != (2,):
            ax1.plot(self.Pc[:,0],self.Pc[:,1],'ob', linewidth=3, markersize=12, alpha=1)

            for i in range(len(self.Pc)):
                ax1.text(self.Pc[i,0],self.Pc[i,1],f'  P{i}', fontsize=12)
        ax1.invert_yaxis()
        ax2.imshow(self.confocal_image, cmap='gray')
        if self.P.shape != (2,):
            ax2.plot(self.P[:,0],self.P[:,1],'ob', linewidth=3, markersize=14, alpha=1)
            for i in range(len(self.P)):
                ax2.text(self.P[i,0],self.P[i,1],f'  P{i}', fontsize=14)
        
        fig_point.show()

    
    def zoom_factory(self,fig, base_scale = 1.3):
        def zoom_fun(event):
            # get the current x and y limits
            ax = event.inaxes
            cur_xlim = ax.get_xlim()
            cur_ylim = ax.get_ylim()
            xdata = event.xdata # get event x location
            ydata = event.ydata # get event y location
            if event.button == 'down':
                # deal with zoom in
                scale_factor = 1/base_scale
            elif event.button == 'up':
                # deal with zoom out
                scale_factor = base_scale
            else:
                # deal with something that should never happen
                scale_factor = 1
                # print event.button
            # set new limits
            ax.set_xlim([xdata - (xdata-cur_xlim[0]) / scale_factor, 
                         xdata + (cur_xlim[1]-xdata) / scale_factor]) 
            ax.set_ylim([ydata - (ydata-cur_ylim[0]) / scale_factor, 
                         ydata + (cur_ylim[1]-ydata) / scale_factor])
            plt.draw() # force re-draw
    
        # attach the call back
        fig.canvas.mpl_connect('scroll_event',zoom_fun)
    
        #return the function
        return zoom_fun
    
    def zoom_reset(self,fig, ax1, ax2):
        def reset_fun(event):
            # get the current x and y limits
            ax = event.inaxes
            if event.key == 'r':
                
                if ax == ax1:
                    ax.set_xlim([np.min(self.X), np.max(self.X)]) 
                    ax.set_ylim([np.min(self.Y), np.max(self.Y)])
                
                elif ax == ax2:
                    ax.set_xlim([0, len(self.confocal_image[0,:])])
                    ax.set_ylim([len(self.confocal_image[:,0]), 0]) 
            plt.draw() # force re-draw
    
        # attach the call back
        fig.canvas.mpl_connect('key_press_event',reset_fun)
    
        #return the function
        return reset_fun
    
    def calibrate_confocal_and_EBSD_data(self,P_keep=0,Pc_keep=0):
        print('Waiting for user input...')
        P = self.P
        Pc = self.Pc
        
        fig, ((ax1), (ax2)) = plt.subplots(1, 2)
        mng = plt.get_current_fig_manager()
        mng.resize(1800,800)
        ax1.set_title('EBSD data', fontsize=16)
        ax2.set_title('CLSM data', fontsize=16)
        
        zoomer = self.zoom_factory(fig) # Zooming with event mouse wheel
        reset_zoomer = self.zoom_reset(fig, ax1, ax2)

        ax1.imshow(self.Z_grid, extent=(np.min(self.X), np.max(self.X), np.min(self.Y), np.max(self.Y)),  cmap=plt.get_cmap('tab20b'), aspect = np.max(self.Y)/np.max(self.X))
        if self.Pc.shape != (2,):
            ax1.plot(self.Pc[:,0],self.Pc[:,1],'or', linewidth=3, markersize=12, alpha=0.5)
            for i in range(len(self.Pc)):
                ax1.text(self.Pc[i,0],self.Pc[i,1],f'  P{i}', fontsize=12)

        ax1.invert_yaxis()
        
        ax2.imshow(self.confocal_image, cmap='gray')
        if self.P.shape != (2,):
            ax2.plot(self.P[:,0],self.P[:,1],'or', linewidth=3, markersize=12, alpha=0.5)
            for i in range(len(self.P)):
                ax2.text(self.P[i,0],self.P[i,1],f'  P{i}', fontsize=14)
                
                
        if type(P_keep) != int and type(Pc_keep)!= int:
            ax2.plot(P_keep[:,0],P_keep[:,1],'ob', linewidth=3, markersize=6, alpha=0.7)
            for i in range(len(P_keep)):
                ax2.text(P_keep[i,0],P_keep[i,1],'        selected', fontsize=12)

            ax1.plot(Pc_keep[:,0],Pc_keep[:,1],'ob', linewidth=3, markersize=6, alpha=0.7)
            for i in range(len(Pc_keep)):
                ax1.text(Pc_keep[i,0],Pc_keep[i,1],'       selected', fontsize=12)

        # textstr2 = '\n'.join((r'Left click: Select a point  -----  Right click: Delete the previously selected point', 
        #                               r'Enter: Complete point selection',))
                    #                  r'Find the position (e.g. in total 4 positions) in both pictures'
        textstr1 =  '\n'.join((r'Please select the points at indentical features. First select point in EBSD data than',
                               r'in CLSM data. Repeat the same procedure for as many points as needed (4 at least).',
                               r'',
                               r'Controls:',
                               r'Left click: Select a point ----- Right click: Delete previously selected point',
                               r'Mouse Wheel: Zoom | Refrain from using lens button -> klicks counted as selections', # Every corner of the selected rectangles will be counted as points
                               r'R: Aspect reset | WARNING: Will create additional point! Remove with Right click'
                               r'Enter: Complete point selection',))
        # these are matplotlib.patch.Patch properties
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        
        # place a text box in upper left in axes coords
        ax1.text(.5, -0.1, textstr1, transform=ax1.transAxes, #fontsize=16,
        verticalalignment='top', horizontalalignment='center', bbox=props)
        
        # ax2.text(.5, -0.07, textstr2, transform=ax2.transAxes, fontsize=16,
        # verticalalignment='top', horizontalalignment='center', bbox=props)
        
        

        
        Ptmp = plt.ginput(64, timeout=520)
        Ptmp = np.asarray(Ptmp)
        P = -1*np.ones([int(len(Ptmp)+64),2])
        Pc = -1*np.ones([int(len(Ptmp)+64),2])
    
        for i in range(int(len(Ptmp)/2)):
            Pc[i,:] = Ptmp[int(i*2),:]
            P[i,:] = Ptmp[int(i*2)+1,:]
        if type(P_keep) != int and type(Pc_keep)!= int:           
            Pc[int(len(Ptmp)/2):int(len(Ptmp)/2)+len(Pc_keep),] = Pc_keep
            P[int(len(Ptmp)/2):int(len(Ptmp)/2)+len(P_keep),] = P_keep
        
        P = P[P[:,0] != -1,:]
        Pc = Pc[Pc[:,0] != -1,:]
        Pt=np.append(Pc,P, axis=1)
        hString='EBSD_xcoord EBSD_ycoord CLSM_xcoord CLSM_ycoord'
        np.savetxt('tmp/000_selected_points_merge.txt',Pt, header=hString)
        self.P = P
        self.Pc = Pc
        print('Ready for merging the data.')
        plt.close('all')
        return True

    def load_points_merge(self, filename):
        print(f'Loading matching points from {filename}')
        Pt=np.loadtxt(filename)
        P=Pt[:,2:]
        Pc=Pt[:,:2]
        self.P = P
        self.Pc = Pc
        print('Ready for merging the data.')
        plt.close('all')
        return True
    
    def load_diff_points(self, filename):
        print(f'Loading matching points from {filename}')
        P=np.loadtxt(filename)
        # print(P[:,:2],P[:,2:])
        return P[:,:2],P[:,2:]
    #%%
    def calculate_superposition(self):
        X = self.X
        Y = self.Y
        Heights = self.Heights
        color = self.color
        HKL = self.HKL
        Orientation = self.Orientation
        confocal_data = self.confocal_data
        print('Calculating transformation...')
        if self.P.shape == (2,):
            print('Reference points are loaded')
            Pt=np.loadtxt('tmp/000_selected_points.txt')
            P=Pt[:,1:]
            Pc=Pt[:,:2]
            self.P = P
            self.Pc = Pc
        self.new_coordinates = self.calculate_new_coordinates()
        print('Calculating height interpolation...')
        M = np.rot90(confocal_data[::-1, :], k=3)
        x = np.linspace(1, confocal_data.shape[0], confocal_data.shape[0])
        y = np.linspace(1, confocal_data.shape[1], confocal_data.shape[1])
        f = interpolate.interp2d(x, y.T, M, kind='linear', bounds_error=False, fill_value=-1)
        Heights = np.zeros((len(self.new_coordinates),1))
        for i in range(len(self.new_coordinates)):
            Heights[i] = f(self.new_coordinates[i, 0], self.new_coordinates[i, 1])
        print('Mask start')
        mask = np.logical_not(self.new_coordinates[:,0] == -1)
        print('Mask finished')
        self.new_coordinates = self.new_coordinates[mask,:]
        self.Heights_result = Heights[mask]
        self.color_result = color[mask]
        self.X_result = X[mask]
        self.Y_result = Y[mask]
        self.HKL_result = HKL[mask,:]
        self.Orientation_result = Orientation[mask,:]

        self.XY_result= np.zeros((self.HKL_result.shape[0],2))
        for k in range(0,self.HKL_result.shape[0]):
            self.XY_result[k,:] =ipfzXY.poleA2(self.HKL_result[k,0],self.HKL_result[k,1],self.HKL_result[k,2])
    
    def calculate_superposition_view(self):
        print('Rendering data, please wait...')
        figcalc = plt.figure(figsize=(10, 10))
        axcalc = figcalc.add_subplot(1, 1, 1)
        axcalc.imshow(self.confocal_image, cmap='gray')
        figcalc.tight_layout()
        axcalc.scatter(self.new_coordinates[:, 0], self.new_coordinates[:, 1], c=self.color_result, marker='o', s=10, edgecolors='face',
                    cmap='terrain', alpha=0.1) #KS_set alpha 0.2-> 0.05
        
        axcalc.set_xlim(xmin=0, xmax= self.confocal_data.shape[0])
        axcalc.set_ylim(ymin=0, ymax= self.confocal_data.shape[1])
        
        figcalc.gca().invert_yaxis()
        figcalc.show()
        print('Ready.')
        return 1

    
    def delteDataThroughImage(self, CLSM_Data=1):
        if CLSM_Data==1:
            M = np.rot90(self.imageForDeleteData[::-1, :], k=1)
            x = np.linspace(1, self.imageForDeleteData.shape[1], self.imageForDeleteData.shape[1])
            y = np.linspace(1, self.imageForDeleteData.shape[0], self.imageForDeleteData.shape[0])
            f = interpolate.interp2d(y, x, M, kind='linear', bounds_error=False, fill_value=-1)
            pic_contrast_del = np.zeros((len(self.new_coordinates),1))
            for i in range(len(self.new_coordinates)): 
                pic_contrast_del[i] = f(self.new_coordinates[i, 0], self.new_coordinates[i, 1])
            mask = np.logical_not(pic_contrast_del[:,0] >= 10)

            self.Heights_result = self.Heights_result[mask]
            self.color_result = self.color_result[mask]
            self.X_result = self.X_result[mask]
            self.Y_result = self.Y_result[mask]
            self.HKL_result = self.HKL_result[mask,:]
            self.Orientation_result = self.Orientation_result[mask,:]
            self.XY_result = self.XY_result[mask,:]
        else:
            M = np.rot90(self.imageForDeleteData[:, :], k=2)
            pic_contrast_del = np.zeros((len(self.X_result),1))
            for i in range(len(self.X_result)):
                pic_contrast_del[i] = M[int(self.X_result[i]),int(self.Y_result[i])]      
            mask = np.logical_not(pic_contrast_del[:,0] >= 10)
            self.Heights_result = self.Heights_result[mask]
            self.color_result = self.color_result[mask]
            self.X_result = self.X_result[mask]
            self.Y_result = self.Y_result[mask]
            self.HKL_result = self.HKL_result[mask,:]
            self.Orientation_result = self.Orientation_result[mask,:]
            self.XY_result = self.XY_result[mask,:]
            
    def delte_Data_Through_Image_Plot(self):
           
        figcalc2 = plt.figure(figsize=(18, 9))
        axcalc2 = figcalc2.add_subplot(1, 2, 1)
        axcalc2.scatter(self.X_result, self.Y_result, c=self.color_result, marker='o', s=10, edgecolors='face',
                cmap='terrain', alpha=0.1)
        figcalc2.gca().invert_yaxis()
        axcalc3 = figcalc2.add_subplot(1, 2, 2)
        axcalc3.scatter(self.X_result, self.Y_result , c=self.Heights_result.reshape(len(self.Heights_result)),
                        marker='o', s=10, cmap=plt.cm.gray, alpha=0.1)
        figcalc2.gca().invert_yaxis()

        figcalc2.show()

           
    #%%
    def data_zero_level(self, loopamount):
        global P
        Zerowerte = []
        self.view_confocal_data(pause=1)
        textstr1 = '\n'.join((r'Please select a reference height with the left mouse button',
                    r' --> Two clicks are necessary to select a rectangle'))
        
        # these are matplotlib.patch.Patch properties
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        
        # place a text box in upper left in axes coords
        self.ax_clsm.text(0.05, 0.05, textstr1, transform=self.ax_clsm.transAxes, fontsize=12,
                          verticalalignment='top', bbox=props)
        print('Waiting for user input...')
        for i in range(0,loopamount):
            
            P = plt.ginput(2, timeout=-1)
            P = np.asarray(P)
            if P[0][0] > P[1][0]:
                tmp = P[0][0]
                P[0][0] = P[1][0]
                P[1][0] = tmp
            if P[0][1] > P[1][1]:
                tmp = P[0][1]
                P[0][1] = P[1][1]
                P[1][1] = tmp
         
            self.fig_clsm.gca().add_patch(Rectangle((P[0][0], P[0][1]), P[1][0] -P[0][0] , P[1][1] -P[0][1] , fill=None ,alpha=1))
            Zerowerte.extend([np.mean(self.confocal_data[int(P[0][0]):int(P[1][0]), int(P[0][1]):int(P[1][1])])])
        plt.pause(0.2)
        print('Ready.')
        plt.close('all')
        self.Zerowerte_all = Zerowerte
            
        return True
    
       
    def confocal_data_conc(self):
        X = self.X_result
        Y = self.Y_result
        XY = self.XY_result
        Heights = self.Heights_result
        Orientation = self.Orientation_result
        HKL = self.HKL_result
        self.dataclsm = np.concatenate((X.reshape((len(X),1)),Y.reshape((len(Y),1)),Heights,Orientation,HKL,XY),axis=1)
        
        return 1    
    
    #%%
    def save_confocal_data(self, Samplename='sample'):
        self.confocal_data_conc()
        np.savetxt(Samplename+'.dat',self.dataclsm,fmt='%.5e',header=' X Y Height PHI PSI2 PSI1 H K L X Y   ___datamerge_1_32')
        return 1
    
    def save_tiff_data (Samplename='sample'):
        global X
        global Y
        global XY
        global ContrastShape
        global Orientation
        global HKL
        tosave = np.concatenate((X.reshape((len(X),1)),Y.reshape((len(Y),1)),ContrastShape,Orientation,HKL,XY),axis=1)
        np.savetxt(Samplename+'_merge_Contrast_data_1_31.dat',tosave,fmt='%.5e',header=' X Y Contrast PHI PSI2 PSI1 H K L IPFz_X_Coordinate IPFz_Y_Coordinate')
        return 1