# -*- coding: utf-8 -*-

# the following code was partially checked and sorted by GPT-3.5

# importing core packages
import os
import sys
import gc
import time
from datetime import datetime
import pytz
import threading
import multiprocessing as mp

# importing packages for GUI usage 
from PyQt5 import  QtWidgets
from PyQt5.QtCore import QObject, QThread, QTimer, QRect, pyqtSignal, Qt
from PyQt5.QtWidgets import QFileDialog, QMainWindow, QDialog, QLabel
import ebsd_merge_evaluate.qt5_oberflaeche as qt5_oberflaeche # use this specific GUI

# importing data processing and visalization packages
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.interpolate import griddata
from functools import partial

# importing self-written data processing packages
import ebsd_merge_evaluate.datamerge_threadingQT as merge
import ebsd_merge_evaluate.cubicipf_qt as ks
import ebsd_merge_evaluate.ipfzXY as ipfzXY

os.chdir(os.path.dirname(os.path.abspath(__file__)))    # setting file directory as working directory, mainly to keep the tmp folder working
mpl.use('Qt5Agg')                                       # setting the mpl backend to Qt5

# create the logger for in-GUI printouts of information on working status

        
    def tabMergeCLSM(self): 
        self.mergeloadCLSM1.clicked.connect(self.browse_button_CLSM_1)
        self.mergeloadCLSM2.clicked.connect(self.browse_button_CLSM_2)
        self.mergesubstractCLSM12.clicked.connect(self.browse_CLSM_substract_norm)
        self.autoSubstract.clicked.connect(self.browse_CLSM_substract_auto)
        self.loadsubstractCLSM12.clicked.connect(self.browse_CLSM_substract_file)
        self.mergeviewCLSM.clicked.connect(self.browse_view_CLSM12)
        
        self.mergeloadEBSD.clicked.connect(self.browse_button_EBSD)
        self.phaseEBSD.textChanged.connect(self.ebsd_phase_changed)
        self.mergeviewEBSD.clicked.connect(self.load_ebsd_view)
        self.loadPointsMergeButton.clicked.connect(self.browse_load_points_merge)
        self.mergeSelectPoints.clicked.connect(self.select_points)
        self.mergeCalculateMerge.clicked.connect(self.merge_calc)
        
        self.mergeviewCLSM.setEnabled(False)
        self.autoSubstract.setEnabled(False)
        self.mergesubstractCLSM12.setEnabled(False)
        self.loadsubstractCLSM12.setEnabled(False)
        self.mergeCalculateMerge.setEnabled(False)
        
        self.mergeSelectArea.clicked.connect(self.area_zero_level_button)
        self.mergeDeleteDataPic.clicked.connect(self.browse_load_pic_data)
        self.mergeSave.clicked.connect(self.merge_save_data)
        
        self.mergeDeleteCLSMcheckBox.stateChanged.connect(self.delete_data_CLSM_checkbox)
        self.mergeDeleteEBSDcheckBox.stateChanged.connect(self.delete_data_EBSD_checkbox)
        self.mergeDeleteDataPicExecButton.clicked.connect(self.browse_delete_pic_data)
        
        # self.mergeloadAFM1.clicked.connect(self.browse_button_AFM_1)
        # self.mergeloadAFM2.clicked.connect(self.browse_button_AFM_2)
        # self.mergeviewAFM.clicked.connect(self.browse_merge_view_AFM)
        # self.mergesubstractAFM12.clicked.connect(self.browse_merge_substract_AFM_12)
        
        # self.mergeloadPicData.clicked.connect(self.browse_pic_data)
        # self.mergeviewPic.clicked.connect(self.browse_view_pic_data)
        
        self.mergeviewAFM.clicked.connect(self.show_warning_save)
    
    def show_warning_save(self): # Create Popup
        message = 'Please remember to save your temporary data in tmp folder.'
        print(message)
        
        self.popup = popupWindow(message)
        self.popup.setGeometry(1200, 600, 240, 70)
        self.popup.show()

    def browse_view_CLSM12(self):
        self.load_clsm_data_thread()#followup_selection = False
        if self.loadCLSM1line.text() != '' and self.loadCLSM2line.text() != '':
            0
        elif (len(self.mergedata.confocal_data) == 2 or not self.data_merge_clsm_single):
            self.thread.finished.connect(self.mergedata.view_confocal_data)

        else:
            self.mergedata.view_confocal_data()           
        
    def load_clsm_data_thread(self): #, followup_selection = True
        #self.followup_selection = followup_selection
        
        self.dataMergeProgressBar.setMaximum(0)
        
        self.worker = merge.mergeThread()
        self.thread = QThread()
        self.worker.moveToThread(self.thread)
        self.worker.finished.connect(self.thread.quit)

        self.worker.leveling = self.mergeLevelingcheckBox.checkState()
        
        self.worker.mergeroationCLSM1 = self.mergeroationCLSM1.currentText()[:-1]
        self.worker.mergeroationCLSM2 = self.mergeroationCLSM2.currentText()[:-1]
        self.worker.CLSM1checkBox = self.CLSM1checkBox.checkState()
        self.worker.CLSM2checkBox = self.CLSM2checkBox.checkState()
        self.worker.CLSM1_rendered = self.CLSM1_rendered
        self.worker.CLSM2_rendered = self.CLSM2_rendered

        if (self.loadCLSM1line.text() == '' or self.loadCLSM2line.text() == ''):
            if (len(self.mergedata.confocal_data) == 2 or not self.data_merge_clsm_single):
                if not self.loadCLSM1line.text():
                    self.worker.loadCLSM1line =  self.loadCLSM2line.text()
                    self.worker.mergeroationCLSM1 = self.mergeroationCLSM2.currentText()[:-1]
                    self.worker.CLSM1checkBox = self.CLSM2checkBox.checkState()
                else:
                    self.worker.loadCLSM1line =  self.loadCLSM1line.text()
                    self.worker.mergeroationCLSM1 = self.mergeroationCLSM1.currentText()[:-1]
                    self.worker.CLSM1checkBox = self.CLSM1checkBox.checkState()
                    
                
                self.thread.started.connect(self.worker.load_confocal_data)
                self.thread.finished.connect(self.load_clsm_data_thread_finished)
                self.thread.start()
                
                self.data_merge_clsm_single = True
                self.mergeviewCLSM.setEnabled(False)
                self.autoSubstract.setEnabled(False)
                self.mergesubstractCLSM12.setEnabled(False)
                self.loadsubstractCLSM12.setEnabled(False)
                
                # self.mergeviewCLSM.setEnabled(False)
                # self.autoSubstract.setEnabled(False)
                # self.mergesubstractCLSM12.setEnabled(False)
                # self.loadsubstractCLSM12.setEnabled(False)
                
                # self.load_clsm_data_thread_finished()


            else:
                self.mergedata.view_confocal_data()
                self.dataMergeProgressBar.setMaximum(100)
        else:
            if len(self.mergedata.confocal_data) == 2 or self.data_merge_clsm_single:
                self.worker.loadCLSM1line =  self.loadCLSM1line.text()
                self.worker.loadCLSM2line =  self.loadCLSM2line.text()
                
                self.thread.started.connect(self.worker.load_confocal_data_diff)
                self.thread.start()
                
                self.CLSM1_rendered = self.worker.CLSM1_rendered 
                self.CLSM2_rendered = self.worker.CLSM2_rendered
                
                self.data_merge_clsm_single = False
                self.select_point_for_diff = True
                
                self.mergeviewCLSM.setEnabled(False)
                self.autoSubstract.setEnabled(False)
                self.mergesubstractCLSM12.setEnabled(False)
                self.loadsubstractCLSM12.setEnabled(False)


            else:
                self.mergedata.view_confocal_data()
                self.dataMergeProgressBar.setMaximum(100)


    def  load_clsm_data_thread_finished(self):
        self.dataMergeProgressBar.setMaximum(100)
        
        self.check_CLSM_availability()
        
        if self.worker.leveling != 0:
            plt.imshow(np.flipud(np.rot90(self.worker.Zebene, 1)))
            plt.colorbar()
            plt.savefig('tmp/000_Ebene.png')
            plt.close()

            Zdiff = self.worker.confocal_data + self.worker.Zebene
            self.plot_confocal(Zdiff, '000_Confocal_before_leveling.png')
    
            Z2 = self.worker.confocal_data
            self.plot_confocal(Z2, '000_Confocal_leveled(now).png')
        print(8)
        self.mergedata.confocal_data = self.worker.confocal_data 
        self.mergedata.confocal_image = self.worker.confocal_image
        self.clean_up_thread()

    
    def plot_confocal(self, data, file_name):
        plt.imshow(np.flipud(np.rot90(data, 1)))
        plt.clim([np.mean(data) + np.var(data), np.mean(data) - np.var(data)])
        plt.colorbar()
        plt.savefig(os.path.join('tmp', file_name))
        plt.close()
    
    def clean_up_thread(self):
        self.worker.deleteLater()
        self.thread.deleteLater()
        self.thread = None
        
    def rendering_clsm1_2_data_thread_finished(self):
        self.check_CLSM_availability()
            
        self.dataMergeProgressBar.setMaximum(100)
        
        # self.mergedata.confocal_data_1 = self.worker.confocal_data_1 
        # self.mergedata.confocal_data_2 = self.worker.confocal_data_2
        # self.mergedata.confocal_image_data_1 =  self.worker.confocal_image_data_1
        # self.mergedata.confocal_image_data_2 =  self.worker.confocal_image_data_2
                    
        self.clean_up_thread()  

        
    # def load_clsm1_2_data_thread_finished(self):
    #     self.check_CLSM_availability()
            
    #     self.dataMergeProgressBar.setMaximum(100)
        
    #     self.mergedata.confocal_data_1 = self.worker.confocal_data_1 
    #     self.mergedata.confocal_data_2 = self.worker.confocal_data_2
    #     self.mergedata.confocal_image_data_1 =  self.worker.confocal_image_data_1
    #     self.mergedata.confocal_image_data_2 =  self.worker.confocal_image_data_2
        
    #     self.mergedata.load_confocal_data_diff_plt(leveling = self.mergeLevelingcheckBox.checkState(), followup_selection = self.followup_selection)      
            
    #     self.clean_up_thread()  

        
    def  load_auto_clsm_data_thread_finished(self):
        self.check_CLSM_availability()
            
        self.dataMergeProgressBar.setMaximum(100)
        
        self.mergedata.confocal_data_1 = self.worker.confocal_data_1 
        self.mergedata.confocal_data_2 = self.worker.confocal_data_2
        self.mergedata.confocal_image_data_1 =  self.worker.confocal_image_data_1
        self.mergedata.confocal_image_data_2 =  self.worker.confocal_image_data_2
        
        self.mergedata.pattern_matching_auto(leveling = self.mergeLevelingcheckBox.checkState())      
            
        self.clean_up_thread() 

                
    def  load_auto_clsm_data_thread_finished_from_file(self):
        file_name = self.browse_button_master('Matching Points .txt File', 'CLSM Matching Points File (*.txt)', tmp_true=True)
        
        self.check_CLSM_availability()
        
        self.dataMergeProgressBar.setMaximum(100)
        self.mergedata.confocal_data_1 = self.worker.confocal_data_1 
        self.mergedata.confocal_data_2 = self.worker.confocal_data_2
        self.mergedata.confocal_image_data_1 =  self.worker.confocal_image_data_1
        self.mergedata.confocal_image_data_2 =  self.worker.confocal_image_data_2
        self.mergedata.confocal_diff_from_file(file_name, leveling = self.mergeLevelingcheckBox.checkState())      
            
        self.clean_up_thread() 
        # self.select_points_status = False
        
    def browse_button_master(self, cap, fil, save = False, tmp_true = False, name_suggestion = ''):
        if tmp_true:
            directory = directory=os.path.join(os.getcwd(),'tmp')
        else:
            directory = os.getcwd()
        
        directory = os.path.join(directory, name_suggestion)
        
        options = QFileDialog.Options()
        
        if save:
            file_name, _ = QFileDialog.getSaveFileName(directory = directory, caption= cap, filter=fil, options=options)
        else:
            file_name, _ = QFileDialog.getOpenFileName(directory = directory, caption= cap, filter=fil, options=options)
        
        return file_name
    
    def browse_button_EBSD(self):
        file_name = self.browse_button_master("EBSD CTF File", 'CTF File (*.ctf)')
        # file_name = "C:/Users/Robin/FAUbox/Uni/Garching_HiWi/data_rb/EBSD_data.ctf"
        self.loadEBSDline.setText(file_name)
        
        if file_name:
            self.load_ebsd_data()
            self.mergeloadEBSD.setStyleSheet(self.green)
            self.ebsd_loaded = True
        else:
            print('No file selected for input of EBSD data')
            self.mergeloadEBSD.setStyleSheet(self.color_click_on)
            self.ebsd_loaded = False

    def browse_button_CLSM_1(self):
        file_name = self.browse_button_master('CLSM csv File', 'CLSM CSV File (*.csv)')
        
        self.CLSM1_rendered = False
        self.loadCLSM1line.setText(file_name)

        if file_name:
            self.render_clsm_data(CLSM_render_set = 0)
            self.mergeloadCLSM1.setStyleSheet(self.green)
            self.mergeSelectPoints.setStyleSheet(self.color_click_on)
            self.mergeCalculateMerge.setStyleSheet(self.color_click_on)
        else:
            self.mergeloadCLSM1.setStyleSheet(self.color_click_on)
        self.check_CLSM_availability()    
            
    def browse_button_CLSM_2(self):
        file_name = self.browse_button_master('CLSM csv File', 'CLSM CSV File (*.csv)')
        
        self.CLSM2_rendered = False
        self.loadCLSM2line.setText(file_name)
        
        if file_name:
            self.render_clsm_data(CLSM_render_set = 1)
            self.mergeloadCLSM2.setStyleSheet(self.green)
        else:
            self.mergeloadCLSM2.setStyleSheet(self.color_optional)
        self.check_CLSM_availability()
        
    def check_CLSM_availability(self):
        self.mergeviewCLSM.setEnabled(True)
        if self.loadCLSM1line.text() != '' and self.loadCLSM2line.text() != '':
            self.autoSubstract.setEnabled(True)
            self.mergesubstractCLSM12.setEnabled(True)
            self.loadsubstractCLSM12.setEnabled(True)
        else:
            self.autoSubstract.setEnabled(False)
            self.mergesubstractCLSM12.setEnabled(False)
            self.loadsubstractCLSM12.setEnabled(False)
        
    def browse_load_points_merge(self):
        file_name = self.browse_button_master('Matching Points .txt File', 'EBSD-CLSM Matching Points File (*.txt)', tmp_true = True)
        
        self.loading_points = True
        self.loading_pointsFileName = file_name
        
        if file_name:
            self.select_points()
        else:
            print('No file selected for input of matching points')
    
    
    def browse_load_points_diff(self):
        file_name = self.browse_button_master('Matching Points .txt File', 'CLSM Matching Points File (*.txt)', tmp_true = True)
        
        if file_name:
            self.select_points_diff()
        else:
            print('No file selected for input of matching points')
    
    
    def browse_load_pic_data(self):
        if type(self.mergedata.dataclsm) != int:
            file_name = self.browse_button_master('CLSM csv File', 'Picture data (*.bmp *.jpeg *.jpg *.tif *.tiff);; JPEG (*.jpeg)')
            
            self.loadDeletDataline.setText(file_name)
            self.mergedata.loadPicdataForDelete(file_name)
        
            self.mergeDeleteDataPicExecButton.setEnabled(True)

        else:
            print('Merging was not applied')
    
    
    def merge_save_data(self):
        file_name = self.browse_button_master('Save .dat file', 'File *.dat (*.dat)', save = True, tmp_true=True, name_suggestion = '000_merge.dat')
        
        print('Merged data saved as '+file_name)
        self.show_warning_save()
        self.dataMergeProgressBar.setMaximum(100)
        if file_name[-4:] == '.dat':
            file_name = file_name[:-4]
        
        try:
            if file_name:
                self.mergedata.save_confocal_data(file_name)
                self.merge_save_path=file_name
                self.mergeSave.setStyleSheet(self.green)
                self.dataMergeProgressBar.setMaximum(100)
            self.createLogMergeSave()
            if (self.mergedata.Zerowerte_all != 0):
                np.savetxt(f'{file_name}-Zerolevel-{np.round(np.mean(self.mergedata.Zerowerte_all),3)}.dat', self.mergedata.Zerowerte_all)
                
        except:
            print('Data save failed')


    def browse_sim_load_MD(self):
        file_name = self.browse_button_master('MD data File', 'File .dat (*.dat)')
        
        self.simDataPath.setText(file_name)
    
        if file_name:
            self.simLoadMDpushButton.setEnabled(False)
            self.simLoadBCApushButton.setEnabled(False)
            self.simProgressBar.setMaximum(0)
            
            self.threadCalculateIPFsim = threading.Thread(target=self.browse_sim_load_MD_thread, daemon=True)
            self.threadCalculateIPFsim.start()
            
            QTimer.singleShot(500, self.threadCalculateIPFUpdateSim)
        else:
            print('No file selected for input of MD data')


    def browse_sim_load_BCA(self):
            file_name = self.browse_button_master('BCA data File', 'File .yield (*.yield)')
            
            self.simDataPath.setText(file_name)
            
            if file_name:
                self.simLoadMDpushButton.setEnabled(False)
                self.simLoadBCApushButton.setEnabled(False)
                self.simProgressBar.setMaximum(0)
                
                self.threadCalculateIPFsimBCA = threading.Thread(target=self.browse_sim_load_BCA_thread, daemon=True)
                self.threadCalculateIPFsimBCA.start()
                
                QTimer.singleShot(500, self.threadCalculateIPFUpdateSimBCA)
            else:
                print('No file selected for input of BCA data')
            
            
    def browseOptLoadDataButton(self):
        file_name = self.browse_button_master('Comparison IPF file', 'Merged IPF file .dat (*.dat)')
        
        self.optLoadDataButtonLineEdit.setText(file_name)   
            
        
    def load_clsm1_2_data_thread_finished_finished(self):
        self.rendering_clsm1_2_data_thread_finished()
        self.mergedata.load_confocal_data_diff_plt(leveling = self.mergeLevelingcheckBox.checkState())
        
        
    def load_auto_clsm1_2_data_thread_finished_finished(self):
        self.rendering_clsm1_2_data_thread_finished()
        self.mergedata.pattern_matching_auto(leveling = self.mergeLevelingcheckBox.checkState())
    
    
    def load_auto_clsm_data_thread_finished_from_file_finished(self):
        self.rendering_clsm1_2_data_thread_finished()
        file_name = self.browse_button_master('Matching Points .txt File', 'CLSM Matching Points File (*.txt)', tmp_true=True)
        self.mergedata.confocal_diff_from_file(file_name, leveling = self.mergeLevelingcheckBox.checkState())      
    
    
    def browse_CLSM_substract_norm(self):
        self.load_clsm_data_thread()
        self.thread.finished.connect(self.load_clsm1_2_data_thread_finished_finished)


    def browse_CLSM_substract_auto(self):
        self.load_clsm_data_thread()
        self.thread.finished.connect(self.load_auto_clsm1_2_data_thread_finished_finished)
        
        
    def browse_CLSM_substract_file(self):
        self.load_clsm_data_thread()
        self.thread.finished.connect(self.load_auto_clsm_data_thread_finished_from_file_finished)
        
        
    def ebsd_phase_changed(self):
        if self.ebsd_loaded and (self.phaseEBSD.text() !=''):
             self.load_ebsd_data()
        
        
    def load_ebsd_data(self):
        self.workerEBSD = merge.mergeThread()
        self.threadEBSD = QThread()
        self.workerEBSD.moveToThread(self.threadEBSD)
        self.workerEBSD.finished.connect(self.threadEBSD.quit)
        self.workerEBSD.fileNameEBSD =  self.loadEBSDline.text()
        self.workerEBSD.phaseEBSD = self.phaseEBSD.text()
        if self.workerEBSD.phaseEBSD=='':
            file_name= self.workerEBSD.fileNameEBSD
            
            
            with open(file_name,"r",errors="replace") as f:
                CRSTdata = f.readlines()
            CRSTdata = CRSTdata[1:]   
            zeroColumn= [lines.split()[0] for i , lines in enumerate(CRSTdata)]
            zeroColumnIntegers=[col for col in zeroColumn if (col.isdigit() and col!='0')]
            phase=str(most_frequent(zeroColumnIntegers))
            self.EBSD_phase=phase
            print(f"Automated phase detection used phase = {phase}")
            self.workerEBSD.phaseEBSD=phase
        
        
        
        self.workerEBSD.X_grid = 0
        self.workerEBSD.Y_grid = 0
        self.workerEBSD.Z_grid = 0
        self.workerEBSD.mergeReductionFactor = 2
        
        self.threadEBSD.started.connect(self.workerEBSD.load_EBSD_data)
        self.threadEBSD.finished.connect(self.load_ebsd_data_thread_finished)
        self.threadEBSD.start()
        
        self.mergeviewEBSD.setEnabled(False)
        self.phaseEBSD.setEnabled(False)
        self.dataMergeProgressBar.setMaximum(0)
            
        
    def render_clsm_data(self, CLSM_render_set):
        self.workerCLSM2 = merge.mergeThread()
        self.workerCLSM2.CLSM_render_set = CLSM_render_set
        
        self.threadCLSM2 = QThread()
        self.workerCLSM2.moveToThread(self.threadCLSM2)
        self.workerCLSM2.finished.connect(self.threadCLSM2.quit)
        self.workerCLSM2.fileNameEBSD =  self.loadEBSDline.text()
        self.workerCLSM2.phaseEBSD = self.phaseEBSD.text()
        
        self.workerCLSM2.loadCLSM1line =  self.loadCLSM1line.text()
        self.workerCLSM2.loadCLSM2line =  self.loadCLSM2line.text()
        self.workerCLSM2.mergeroationCLSM1 = self.mergeroationCLSM1.currentText()[:-1]
        self.workerCLSM2.mergeroationCLSM2 = self.mergeroationCLSM2.currentText()[:-1]
        self.workerCLSM2.CLSM1checkBox = self.CLSM1checkBox.checkState()
        self.workerCLSM2.CLSM2checkBox = self.CLSM2checkBox.checkState()
        
        
        self.workerCLSM2.CLSM1_rendered = self.CLSM1_rendered
        self.workerCLSM2.CLSM2_rendered = self.CLSM2_rendered
        
        self.threadCLSM2.started.connect(self.workerCLSM2.render_confocal_data)
        self.threadCLSM2.finished.connect(self.render_clsm_data_thread_finished)
        self.threadCLSM2.start()
        
        self.mergeviewCLSM.setEnabled(False)
        self.autoSubstract.setEnabled(False)
        self.mergesubstractCLSM12.setEnabled(False)
        self.loadsubstractCLSM12.setEnabled(False)
        
        self.dataMergeProgressBar.setMaximum(0)
        
        self.CLSM_render_set = CLSM_render_set
    
    def render_clsm_data_thread_finished(self):
        
        self.mergedata.confocal_image_data_2 = self.workerCLSM2.confocal_image_data_2
        
        if (self.workerCLSM2.CLSM_render_set == 0):
            self.mergedata.confocal_image_data_1 = self.workerCLSM2.confocal_image_data_1
            self.mergedata.confocal_data_1 = self.workerCLSM2.confocal_data_1
        elif(self.workerCLSM2.CLSM_render_set == 1):
            self.mergedata.confocal_image_data_2 = self.workerCLSM2.confocal_image_data_2
            self.mergedata.confocal_data_2 = self.workerCLSM2.confocal_data_2
            
        
        self.mergedata.confocal_image_data_2 = self.workerCLSM2.confocal_image_data_2
        
        self.CLSM1_rendered = self.workerCLSM2.CLSM1_rendered
        self.CLSM2_rendered = self.workerCLSM2.CLSM2_rendered
        
        if self.thread == 0:
            self.dataMergeProgressBar.setMaximum(100)
        
        self.check_CLSM_availability()
        
        if self.CLSM_render_set==0:
            self.CLSM1_rendered = True
            
        elif self.CLSM_render_set==1:
            self.CLSM2_rendered = True
            
        self.workerCLSM2.CLSM_render_set = None
        
        
    def load_ebsd_data_thread_finished(self):
        if self.thread == 0:
            self.dataMergeProgressBar.setMaximum(100)
       
        self.mergeviewEBSD.setEnabled(True)
        self.phaseEBSD.setEnabled(True)
        if type(self.workerEBSD.Z_grid) != int:
            self.mergedata.color = self.workerEBSD.color 
            self.mergedata.HKL = self.workerEBSD.HKL 
            self.mergedata.X =  self.workerEBSD.X
            self.mergedata.Y = self.workerEBSD.Y 
            self.mergedata.Orientation = self.workerEBSD.Orientation 
            self.mergedata.X_grid =  self.workerEBSD.X_grid
            self.mergedata.Y_grid = self.workerEBSD.Y_grid 
            self.mergedata.Z_grid = self.workerEBSD.Z_grid 
            self.workerEBSD.deleteLater
            self.threadEBSD.deleteLater
            self.mergeloadEBSD.setStyleSheet(self.green)
            self.mergeSelectPoints.setStyleSheet(self.color_click_on)
            self.mergeCalculateMerge.setStyleSheet(self.color_click_on)
              
    def load_ebsd_view(self):
        if self.ebsd_loaded:
            self.mergedata.view_EBSD_data()
        else:
            print('No EBSD data')

    def select_points(self):
        if self.ebsd_loaded:
            if len(self.mergedata.confocal_data) == 2:
                self.load_clsm_data_thread()
                self.select_points_status = 1
            elif len(self.mergedata.confocal_data) != 2 :
                if self.mergedata.P.shape == (2,):
                    self.dataMergeProgressBar.setMaximum(100)
                    if(self.loading_points==1):
                        self.mergedata.load_points_merge(self.loading_pointsFileName)
                        self.manual_selection = False
                        self.read_in_selection = True
                        self.loading_points=0
                    else:
                        self.mergedata.calibrate_confocal_and_EBSD_data()
                    
                    if self.mergedata.P.shape != (2,):
                        self.mergeSelectPoints.setStyleSheet(self.green)
                        self.mergeCalculateMerge.setStyleSheet(self.color_click_on)
                        self.mergeCalculateMerge.setEnabled(True)
                    
    
                else:
                    self.dataMergeProgressBar.setMaximum(100)
                    self.select_points_window()
                
        else:
            print('No EBSD data')
            self.dataMergeProgressBar.setMaximum(100)

    

    def select_points_finished(self):
        self.mergedata.calibrate_confocal_and_EBSD_data()
        self.mergeSelectPoints.setStyleSheet(self.green)
        self.mergeCalculateMerge.setStyleSheet(self.color_click_on)     
            
    def select_points_window(self):
        self.dialog = SecondWindow(self)
        self.dialog.checkBox = []
        self.dialog.labelEBSDcoordinate = []
        self.dialog.labelCLSMcoordinate = []
        for i in range(len(self.mergedata.P)): #This loop creates one tickable box for each preselected point

            self.dialog.checkBox.append(QtWidgets.QCheckBox(self.dialog))
            self.dialog.checkBox[i].setGeometry(QRect(40, 70+40*i, 61, 23))
            self.dialog.checkBox[i].setObjectName(f'checkBox{i}')
            self.dialog.checkBox[i].setText(f"P{i}")
            self.dialog.checkBox[i].setChecked(True)
            self.dialog.checkBox[i].clicked.connect(self.check_all_boxes_select)
            
            self.dialog.labelEBSDcoordinate.append(QtWidgets.QLabel(self.dialog))
            self.dialog.labelEBSDcoordinate[i].setGeometry(QRect(150, 70+40*i, 180, 23))
            self.dialog.labelEBSDcoordinate[i].setObjectName(f"EBSDcoordinate{i}")
            self.dialog.labelCLSMcoordinate.append(QtWidgets.QLabel(self.dialog))
            self.dialog.labelCLSMcoordinate[i].setGeometry(QRect(350, 70+40*i, 180, 23))
            self.dialog.labelCLSMcoordinate[i].setObjectName(f"CLSMcoordinate{i}")

            self.dialog.labelEBSDcoordinate[i].setText(f'x ={int(self.mergedata.P[i,0])} / y= {int(self.mergedata.P[i,1])}')
            self.dialog.labelCLSMcoordinate[i].setText(f'x ={int(self.mergedata.Pc[i,0])} / y= {int(self.mergedata.Pc[i,1])}')
            
        self.mergedata.calibrate_confocal_and_EBSD_data_image()
        self.dialog.pushButton.clicked.connect(self.select_points_window_select)
        self.dialog.pushButton_2.clicked.connect(self.select_points_window_save)
        time.sleep(1)
        self.dialog.show()
    
    def check_all_boxes_select(self):
        mask = np.zeros(len(self.dialog.checkBox))
        for i in range(len(self.dialog.checkBox)):                
            mask[i] = self.dialog.checkBox[i].checkState()
        if all(mask):
            self.dialog.pushButton.setEnabled(True)
        else:
            self.dialog.pushButton.setEnabled(False)

        
    def select_points_window_save(self):
        mask = np.zeros(len(self.dialog.checkBox))
        for i in range(len(self.dialog.checkBox)):                
            mask[i] = self.dialog.checkBox[i].checkState()
        self.mergedata.P = self.mergedata.P[mask != 0]
        self.mergedata.Pc = self.mergedata.Pc[mask != 0]
        plt.close('all')
        self.dialog.destroy()
        if self.mergedata.P.shape != (2,):
            self.mergeSelectPoints.setStyleSheet(self.green)
            self.mergeCalculateMerge.setStyleSheet(self.color_click_on)
            self.mergeCalculateMerge.setEnabled(True)
        else:
            self.mergeSelectPoints.setStyleSheet(self.color_click_on)
            self.mergeCalculateMerge.setStyleSheet(self.green)
            self.mergeCalculateMerge.setEnabled(False)
            
    def select_points_window_select(self):

        mask = np.zeros(len(self.dialog.checkBox))
        for i in range(len(self.dialog.checkBox)):                
            mask[i] = self.dialog.checkBox[i].checkState()
        P_keep = self.mergedata.P[mask != 0]
        Pc_keep = self.mergedata.Pc[mask != 0]
        plt.close('all')
        self.dialog.destroy()
        self.mergedata.calibrate_confocal_and_EBSD_data(P_keep,Pc_keep)
        if self.mergedata.P.shape != (2,):
                    self.mergeSelectPoints.setStyleSheet(self.green)
                    self.mergeCalculateMerge.setStyleSheet(self.color_click_on)
                    self.mergeCalculateMerge.setEnabled(True)
        else:
            self.mergeSelectPoints.setStyleSheet(self.color_click_on)
            self.mergeCalculateMerge.setStyleSheet(self.green)
            self.mergeCalculateMerge.setEnabled(False)
        
    def merge_calc(self):
        
        self.dataMergeProgressBar.setMaximum(0)
        
        self.thread_merge_calc = threading.Thread(target=self.mergeCalcThread, daemon=True)
        self.thread_merge_calc.start()
        
        self.mergeCalculateMerge.setEnabled(False)
        QTimer.singleShot(500, self.mergeCalcUpdate)
    
    def mergeCalcUpdate(self):
        if self.thread_merge_calc.is_alive():
            QTimer.singleShot(500, self.mergeCalcUpdate)
        else:
            self.mergedata.calculate_superposition_view()
            self.mergeCalculateMerge.setEnabled(True)
            self.dataMergeProgressBar.setMaximum(100)
            self.mergeCalculateMerge.setStyleSheet(self.green)
            self.mergeSave.setStyleSheet(self.color_click_on)
            
    def mergeCalcThread(self):
        self.mergedata.calculate_superposition()
        self.mergedata.confocal_data_conc()
        
        print('Thread finished')
        
    def area_zero_level_button(self):
        self.mergeSave.setStyleSheet(self.color_click_on)
        if len(self.mergedata.confocal_data) == 2:
            self.load_clsm_data_thread()
            self.thread.finished.connect(self.area_zero_level)
        else:
            self.area_zero_level()
            
    def area_zero_level(self):
        try:
            self.mergedata.data_zero_level(int(self.mergeAmountSelect.text()))
        except:
            print('Error')
       

        
    def browse_delete_pic_data(self):
        global  EBSD_coordinates,   CLSM_coordinates
        if self.mergeDeleteCLSMcheckBox.checkState():
            print('Delete data with CLSM coordinate system')
            self.mergedata.delteDataThroughImage(CLSM_Data=1)
        else:
            self.mergedata.delteDataThroughImage(CLSM_Data=0)
            print('Delete data with ESBD coordinate system')
        self.mergedata.delte_Data_Through_Image_Plot()
            
    def delete_data_CLSM_checkbox(self):
        if self.mergeDeleteCLSMcheckBox.checkState():
            self.mergeDeleteEBSDcheckBox.setChecked(False)
        else:
            self.mergeDeleteEBSDcheckBox.setChecked(True)
 
    def delete_data_EBSD_checkbox(self):
        if self.mergeDeleteEBSDcheckBox.checkState():
            self.mergeDeleteCLSMcheckBox.setChecked(False)
        else:
            self.mergeDeleteCLSMcheckBox.setChecked(True)
    
    def logNewHead(self, file, title):
        linewidth=40
        file.writelines("-"*(linewidth-int(len(title)/2))+' '+title+' '+"-"*(linewidth-int(len(title)/2))+'\n')
        
    def logNewLine(self, file, text):
        file.writelines(text+'\n')
    
    def createLogMergeSave(self):
        self.logfile_merge = open(os.path.join('tmp', 'logfile_merging.log'), 'w')
        
        EBSD_file = self.loadEBSDline.text()
        
        self.logNewHead(self.logfile_merge, 'Data Merging')
        self.logNewLine(self.logfile_merge, 'EBSD filepath:\t'+EBSD_file)
        self.logNewLine(self.logfile_merge, 'Used phase:\t'+self.EBSD_phase)
        self.logNewLine(self.logfile_merge, '')

        CLSM_file_1 = self.loadCLSM1line.text()
        CLSM_file_1_rotation = self.mergeroationCLSM1.currentText()
        CLSM_file_1_mirrorCheck = bool(self.CLSM1checkBox.checkState())
        
        CLSM_file_2 = self.loadCLSM2line.text()
        CLSM_file_2_rotation = self.mergeroationCLSM2.currentText()
        CLSM_file_2_mirrorCheck = bool(self.CLSM2checkBox.checkState())
        
        CLSM_file_leveling = bool(self.mergeLevelingcheckBox.checkState())
        
        
        
        self.logNewHead(self.logfile_merge, 'Confocal Data')
        differenceMicroscopyUsed=False
        if ((CLSM_file_1 != '') and (CLSM_file_2 != '')):
            differenceMicroscopyUsed=True
        self.logNewLine(self.logfile_merge, 'CLSM 1 filepath:\t'+CLSM_file_1)
        self.logNewLine(self.logfile_merge, 'CLSM 1 used rotaion:\t'+CLSM_file_1_rotation)
        self.logNewLine(self.logfile_merge, 'CLSM 1 was mirrored:\t'+str(CLSM_file_1_mirrorCheck))
        self.logNewLine(self.logfile_merge, '')
        self.logNewLine(self.logfile_merge, 'Difference Microscopy applied: '+str(differenceMicroscopyUsed))
        if differenceMicroscopyUsed:
            self.logNewLine(self.logfile_merge, 'CLSM 2 filepath:\t'+CLSM_file_2)
            self.logNewLine(self.logfile_merge, 'CLSM 2 used rotaion:\t'+CLSM_file_2_rotation)
            self.logNewLine(self.logfile_merge, 'CLSM 2 was mirrored:\t'+str(CLSM_file_2_mirrorCheck))
        self.logNewLine(self.logfile_merge, '')
        self.logNewLine(self.logfile_merge, 'Leveling was applied:\t'+str(CLSM_file_leveling))
        self.logNewLine(self.logfile_merge, '')
        
        self.logNewHead(self.logfile_merge, 'Merging Options')
        
        if self.EBSD_CLSM_manual_selection and self.EBSD_CLSM_read_in:
            self.logNewLine(self.logfile_merge, 'Points partially manually selected and read in from file')
            self.logNewLine(self.logfile_merge, f'Read from path:\t{self.loading_pointsFileName}')
        elif self.EBSD_CLSM_manual_selection:
            self.logNewLine(self.logfile_merge, 'Points manually selected')
        elif self.EBSD_CLSM_read_in:
            self.logNewLine(self.logfile_merge, 'Points read in from file')
            self.logNewLine(self.logfile_merge, f'Read from path:\t{self.loading_pointsFileName}')
        self.logNewLine(self.logfile_merge, f'Final selection written to path:\t{self.mergedata.save_selected_points_merge}')
        
        self.logNewLine(self.logfile_merge, '')
        self.logNewLine(self.logfile_merge, f'Merged savepath:\t{self.merge_save_path}.dat')
        
        self.logfile_merge.close()
        
#%%
    def tabEvaluateCLSM(self): 
        
        self.simLoadMDpushButton.clicked.connect(self.browse_sim_load_MD)
        self.simLoadBCApushButton.clicked.connect(self.browse_sim_load_BCA)
        self.simHKLplotpushButton.clicked.connect(self.browsePlotHKLsim)
        self.simErosionPlotButton.clicked.connect(self.browseErosionDepthPlotsim)
        
        self.evaluateLoadDataButton.clicked.connect(self.browse_evaluate_load_data)
        self.evaluatePlotEBSDbutton.clicked.connect(self.browse_plot_EBSD_Data)
        self.evaluateClosePlotsButton.clicked.connect(lambda: plt.close('all'))
        self.evaluatePlotCLSMbutton.clicked.connect(self.browsePlotDataHeight)
        self.evaluateRefereceLevelLineEdit.textChanged.connect(self.browse_calculate_mean_erosion)
        self.evaluateLevelDataButton.clicked.connect(self.leveling_data)
        self.evaluateLevelingPushButton.clicked.connect(self.levelingOxidData)
        self.evaluateApplyFilterButton.clicked.connect(self.browseLevelingDataFilters)
        self.evaluateCalculateIPFbutton.clicked.connect(self.browseCalculateIPF)
        self.evaluateSaveDataButton.clicked.connect(self.evaluateSaveData)
        self.evaluateSYplotButton.clicked.connect(self.browse_sputter_yield_plot)
        self.evaluateErosionPlotButton.clicked.connect(self.browseErosionDepthPlot)
        self.evaluateSDplotButton.clicked.connect(self.browseSDplot)
        self.evalualteChangeTabOpti.clicked.connect(lambda: self.evaluatSubtabWidget.setCurrentIndex(2))
        self.evaluateCountplotButton.clicked.connect(self.browseCountsplot)
        self.evaluateHKLplotpushButton.clicked.connect(self.browsePlotHKL)
        
        
        self.optiResolutionIPFLineEdit.textChanged.connect(lambda: 
            self.optiResolutionIPFLineEdit_2.setText(self.optiResolutionIPFLineEdit.text()))
        self.optiResolutionIPFLineEdit_2.textChanged.connect(lambda: 
            self.optiResolutionIPFLineEdit.setText(self.optiResolutionIPFLineEdit_2.text()))
        self.optCalculateButton.clicked.connect(self.browse_optimisation_calculate)
        self.optLoadDataButton.clicked.connect(self.browseOptLoadDataButton)
        self.optReduceDataPushButton.clicked.connect(self.browseBinningData)
        self.optSelectDataPointspushButton.clicked.connect(self.optSelctPoint)
        self.evaluateMergeDataButton.clicked.connect(self.browseMergeLeveldData)
        self.evaluateRotationMatrixCalcbutton.clicked.connect(self.browseDataRotationMatrix)
        
     
    def threadCalculateIPFUpdateSim(self):
        if self.threadCalculateIPFsim.is_alive():
            QTimer.singleShot(500, self.threadCalculateIPFUpdateSim)
        else:
            self.simLoadMDpushButton.setEnabled(True)
            self.simLoadBCApushButton.setEnabled(True)
            self.simErosionPlotButton.setStyleSheet(self.green)
            self.simLoadMDpushButton.setStyleSheet(self.green)
            self.simLoadBCApushButton.setStyleSheet(self.color_click_on)
            self.simProgressBar.setMaximum(100)
  
    def browse_sim_load_MD_thread(self):
        try:
            self.evaluateSim.dataPathFile = self.simDataPath.text()
            self.evaluateSim.euler1Index = int(self.simMDEuler1LineEdit.text())
            self.evaluateSim.euler2Index = int(self.simMDEuler2LineEdit.text())
            self.evaluateSim.dataIndex = int(self.simMDDataLineEdit.text())
            self.evaluateSim.loadMDData()
            self.evaluateSim.resolution = 2
            self.evaluateSim.smoothing = 2
            self.evaluateSim.cutOffFilter = 1
            self.evaluateSim.medianDataXYZ()
        except:
            print('Data load failed! -------------- ??Column wrong??')
            
         
    def threadCalculateIPFUpdateSimBCA(self):
        if self.threadCalculateIPFsimBCA.is_alive():
            QTimer.singleShot(500, self.threadCalculateIPFUpdateSimBCA)
        else:
            self.simLoadMDpushButton.setEnabled(True)
            self.simLoadBCApushButton.setEnabled(True)
            self.simErosionPlotButton.setStyleSheet(self.green)
            self.simLoadMDpushButton.setStyleSheet(self.color_click_on)
            self.simLoadBCApushButton.setStyleSheet(self.green)
            self.simProgressBar.setMaximum(100)
        
    def browse_sim_load_BCA_thread(self):
        try:
            self.evaluateSim.dataPathFile = self.simDataPath.text()
            self.evaluateSim.loadBCAData()
            self.evaluateSim.resolution = 2
            self.evaluateSim.smoothing = 2
            self.evaluateSim.cutOffFilter = 1
            self.evaluateSim.medianDataXYZ()
        except:
            print('Data load failed')

    def browseErosionDepthPlotsim(self):
        
        if len(self.evaluateSim.heightIPF) == 1:
            print('No data')
        else:
            b = self.simErosionTicksLineEdit.text()
            c = c=b.split(',')
            f = np.asarray(c,dtype=float)
            if self.simErosionCheckBox.checkState() != 0:
                contourplot1 = 1
            else:
                contourplot1 = 0
            if self.simMultiplyCheckBox.checkState() != 0:
                erosion = np.multiply(self.evaluateSim.heightIPF,-1)
            else:
                erosion = self.evaluateSim.heightIPF
                
            erosion = np.multiply(erosion, float(self.simMultiplyLineEdit.text()))
            saveipf = self.simDataPath.text()[:-4]+'_IPF.png'
            self.evaluateSim.plot_inter_IPFz(self.evaluateSim.xIPFcoordinate, self.evaluateSim.yIPFcoordinate, erosion, float(2), 
                               cbarrange= [float(self.simErosionMinLineEdit.text()),float(self.simErosionMaxLineEdit.text())], 
                               Title='IPFz', cbarlabel=self.simErosionLabelLineEdit.text(), ticks = f, contourplot=contourplot1,savepath = saveipf)  
        
    def browsePlotHKLsim(self):
        try:
            h = float(self.simHlineEdit.text())
            k = float(self.simKlineEdit.text())
            l = float(self.simLlineEdit.text())
            ipf = [h,k,l]
            self.evaluateSim.plot_IPF_plane(IPF=ipf,testsize=20, 
                                            text=self.simPlotHKLtextlineEdit.text(), 
                                            yTextoffset =float(self.simPlotYoffsetLtextlineEdit.text())/100 )
        except:
            print('No figure avalible')
    
    def evaluate_load_data_IPF_update(self):
        if self.thread_evaluate_loadData_IPF.is_alive():
            QTimer.singleShot(500, self.evaluate_load_data_IPF_update)
        else:

            self.evaluateApplyFilterButton.setEnabled(True)
            self.evaluateProgressBar.setMaximum(100)
        
        
    def evaluate_load_data_update(self):
        if self.thread_evaluate_loadData.is_alive():
            QTimer.singleShot(500, self.evaluate_load_data_update)
        else:
            self.evaluateLoadDataButton.setEnabled(True)
            self.evaluateLevelDataButton.setEnabled(True)
            self.evaluateSaveDataButton.setStyleSheet(self.color_click_on)
            self.evaluateCalculateIPFbutton.setStyleSheet(self.color_click_on)
            self.evaluateLoadDataButton.setStyleSheet(self.green)
            self.evaluateMeanEroisionlineEdit.setText( str(np.round(np.mean(self.evaluate.mergeDataSet[:,2]),4)))
            if self.evaluate.mergeDataSet.shape[1] == 11:
                self.thread_evaluate_loadData_IPF = threading.Thread(target=self.evaluate.IPFYandIPFX, daemon=True)
                self.thread_evaluate_loadData_IPF.start()    
                
                QTimer.singleShot(500, self.evaluate_load_data_IPF_update)
            else:

                self.evaluateApplyFilterButton.setEnabled(True)
                self.evaluateProgressBar.setMaximum(100)

    def browse_evaluate_load_data(self):
        file_name = self.browse_button_master('EBSD CTF File', 'Merged file .dat (*.dat)')
        
        self.evaluateLoadDataButtonLineEdit.setText(file_name)
        self.evaluate.dataPathFile = file_name
        
        if file_name:
            self.evaluateProgressBar.setMaximum(0)
            self.evaluateLoadDataButton.setEnabled(False)
            self.evaluateLevelDataButton.setEnabled(False)
            self.evaluateApplyFilterButton.setEnabled(False)
            
            self.thread_evaluate_loadData = threading.Thread(target=self.evaluate_load_data, daemon=True)
            self.thread_evaluate_loadData.start()
            
            QTimer.singleShot(500, self.evaluate_load_data_update)
        
    def evaluate_load_data(self):
        self.evaluate.mergeDataSet = np.loadtxt(self.evaluate.dataPathFile)
        # k = self.evaluate_load_reduction_factor
        k = int(self.spinBoxMergedReduce.cleanText())
        #k = 12
        if k != 1:
            mask = np.array([i % k == 0 for i in range(len(self.evaluate.mergeDataSet))])
        
        self.evaluate.mergeDataSet = self.evaluate.mergeDataSet[mask]
        # print(self.evaluate.mergeDataSet[mask].shape)
        # print(mask.shape)
    
    def browse_plot_EBSD_Data(self):
        try:
            self.evaluate.plotEBSD_Data()
        except:
            print('No data loaded')

    def browsePlotDataHeight(self):
        try:
            self.evaluate.plotDataHeight()
        except:
            print('No data loaded')
    def browse_calculate_mean_erosion(self):
        global reference_level
        try:
            self.evaluate.refereceHeight  = np.round(np.mean(self.evaluate.mergeDataSet[:,2]) - float(self.evaluateRefereceLevelLineEdit.text()),5)
            self.evaluateMeanEroisionlineEdit.setText(str(self.evaluate.refereceHeight ))
        except:
            pass
        
    def leveling_data(self):
        if type(self.evaluate.mergeDataSet)== int:
            print('No data')
        else:
            self.evaluateCalculateIPFbutton.setStyleSheet(self.color_click_on)
            self.evaluateSaveDataButton.setStyleSheet(self.color_click_on)
            self.evaluate.relativeHeighttoAbsHeight()
            self.evaluateMeanEroisionlineEdit.setText( str(np.round(np.mean(self.evaluate.mergeDataSet[:,2]),4)))
            self.evaluateRefereceLevelLineEdit.setText(str(0 ))
        
    def levelingOxidData(self):
        if type(self.evaluate.mergeDataSet)== int:
            print('No data')
        else:
            self.evaluateLevelingPushButton.setEnabled(False)
            self.evaluateCalculateIPFbutton.setStyleSheet(self.color_click_on)
            self.evaluateSaveDataButton.setStyleSheet(self.color_click_on)
            self.evaluate.DepthgrowFactor = float(self.evaluateDepthGrowlineEdit.text())
            self.evaluate.ABSHeightOxide = float(self.evaluateOxideHeightLineEdit.text())
            
            if self.evaluate100OxideHeightRadioButton.isChecked():
                self.evaluate.relativeHeighttoAbsOxidTungstenHeight100()
            elif self.evaluateMeanOxideHeightRadioButton.isChecked():
                self.evaluate.relativeHeighttoAbsOxidTungstenHeightmean()
                
            self.evaluateMeanEroisionlineEdit.setText( str(np.round(np.mean(self.evaluate.mergeDataSet[:,2]),4)))
            print('Leveling button is disable. Only one time the depth growth should be multiply to the data')
        
        
    def threadlevelingDataFiltersUpdate(self):
        if self.thread_levelingDataFilters.is_alive():
            self.evaluateProgressBar2.setValue(self.evaluate.progressUpdate2)
            QTimer.singleShot(500, self.threadlevelingDataFiltersUpdate)
        else:
            self.evaluateApplyFilterButton.setEnabled(True)
            self.evaluateProgressBar.setMaximum(100)
            self.evaluateProgressBar2.setValue(0)
        
        
    def browseLevelingDataFilters(self):
        if type(self.evaluate.mergeDataSet)== int:
            print('No data')
        else:
            self.evaluateApplyFilterButton.setEnabled(False)
            self.evaluateProgressBar.setMaximum(0)
            
            self.thread_levelingDataFilters = threading.Thread(target=self.levelingDataFilters, daemon=True)
            self.thread_levelingDataFilters.start()
            
            QTimer.singleShot(500, self.threadlevelingDataFiltersUpdate)

    def levelingDataFilters(self):
        if type(self.evaluate.mergeDataSet)== int:
            print('No data')
        else:
            self.evaluate.mergeDataSetcopy =self.evaluate.mergeDataSet.copy()
            if self.evaluateNoiseFilterCheckBox.checkState() != 0:
                self.evaluateSaveDataButton.setStyleSheet(self.color_click_on)
                print('---------------- Noise filter ----------------')
                self.evaluate.grainboundaryOrNois = 0
                self.evaluate.pixelCluster = float(self.evaluateNoiseFilterSize.text())
                self.evaluate.grainBounderies()
            if self.evaluateGBfilterCheckBox.checkState() != 0:
                self.evaluateSaveDataButton.setStyleSheet(self.color_click_on)
                print('---------------- Grain boundary filter ----------------')
                self.evaluate.grainboundaryOrNois = 1
                self.evaluate.pixelCluster = float(self.evaluateGBfilterSize.text())
                self.evaluate.grainBounderies()
            
                
            if len(self.evaluate.mergeDataSetcopy[:,0]) < 30:
                print("----------------------------Warning----------------------------","Probably to small grains or filter size to big! Filter is not applied! ")
            else:
                print('Copy Data')
                self.evaluate.mergeDataSet = self.evaluate.mergeDataSetcopy.copy()
                self.evaluateCalculateIPFbutton.setStyleSheet(self.color_click_on)
                self.evaluateSaveDataButton.setStyleSheet(self.color_click_on)
            
            
    def browseMergeLeveldData(self):
        file_name = self.browse_button_master('EBSD CTF File', 'Merged file .dat (*.dat)')
        
        self.evaluate.dataPathFiles = file_name
        if file_name:
            self.evaluateProgressBar.setMaximum(0)
            self.evaluateLoadDataButton.setEnabled(False)
            self.evaluateLevelDataButton.setEnabled(False)
            self.evaluateLevelingPushButton.setEnabled(False)
            self.evaluateApplyFilterButton.setEnabled(False)
    
            self.mergedLeveldDataThread = threading.Thread(target=self.browse_merged_leveld_data, daemon=True)
            self.mergedLeveldDataThread.start()
            
            QTimer.singleShot(500, self.mergedLeveldDataThreadUpdate)

            
    def browse_merged_leveld_data(self):
        try:
            self.evaluate.browse_merged_leveld_data()
        except:
            print('Error')
            self.evaluateProgressBar.setMaximum(100)
            
    def mergedLeveldDataThreadUpdate(self):
        if self.mergedLeveldDataThread.is_alive():
            QTimer.singleShot(500, self.mergedLeveldDataThreadUpdate)
        else:
            self.evaluateProgressBar.setMaximum(100)
            self.evaluateLoadDataButton.setEnabled(True)
            self.evaluateLevelDataButton.setEnabled(True)
            self.evaluateLevelingPushButton.setEnabled(True)
            self.evaluateApplyFilterButton.setEnabled(True)
            try:
                self.evaluateMeanEroisionlineEdit.setText( str(np.round(np.mean(self.evaluate.mergeDataSet[:,2]),4)))
                self.evaluateSaveDataButton.setStyleSheet(self.color_click_on)
                self.evaluateCalculateIPFbutton.setStyleSheet(self.color_click_on)
                self.evaluateLoadDataButton.setStyleSheet(self.green)
                
                if self.evaluate.mergeDataSet.shape[1] == 11:
                    self.thread_evaluate_loadData_IPF = threading.Thread(target=self.evaluate.IPFYandIPFX, daemon=True)
                    self.thread_evaluate_loadData_IPF.start()
                    
                    QTimer.singleShot(500, self.evaluate_load_data_IPF_update)
                else:
    
                    self.evaluateApplyFilterButton.setEnabled(True)
            except:
                 print('Loding Error')
            
    def browseCalculateIPF(self):
        try:
            self.evaluate.resolution = float(self.evaluateIPFresolutionLineEdit.text())
            self.evaluate.smoothing = self.evaluateSmoothingCheckBox.checkState()
            self.evaluate.cutOffFilter = float(self.evaluateCuttOffFilterLineEdit.text())
            
            self.evaluateCalculateIPFbutton.setEnabled(False)
            self.evaluateProgressBar.setMaximum(0)
            
            self.threadCalculateIPF = threading.Thread(target=self.evaluate.medianDataXYZ, daemon=True)
            self.threadCalculateIPF.start()
            
            QTimer.singleShot(500, self.threadCalculateIPFUpdate)
        except:
            print('Error')
            self.evaluateCalculateIPFbutton.setEnabled(True)
     
    def threadCalculateIPFUpdate(self):
        if self.threadCalculateIPF.is_alive():
            QTimer.singleShot(500, self.threadCalculateIPFUpdate)
        else:
            self.evaluateCalculateIPFbutton.setEnabled(True)
            self.evaluateProgressBar.setMaximum(100)
            if len(self.evaluate.heightIPF) == 1:
                print('No data')
            else:
                self.evaluateCalculateIPFbutton.setEnabled(True)
                self.evaluateCalculateIPFbutton.setStyleSheet(self.green)
                self.evaluateSYplotButton.setStyleSheet(self.green)
                self.evaluateErosionPlotButton.setStyleSheet(self.green)
                self.evaluateSDplotButton.setStyleSheet(self.green)
                self.evaluateCountplotButton.setStyleSheet(self.green)
                
                self.saveIPFData()
            
    def saveIPFData(self):
        0        
    
    def browseDataRotationMatrix(self):
       
        self.evaluateCalculateIPFbutton.setEnabled(False)
        self.evaluateRotationMatrixCalcbutton.setEnabled(False)
        self.evaluateProgressBar.setMaximum(0)
        
        self.threadDataRotationMatrix = threading.Thread(target=self.dataRotationMatrix, daemon=True)
        self.threadDataRotationMatrix.start()
        
        QTimer.singleShot(500, self.dataRotationMatrixUpdate) 
        
    def dataRotationMatrix(self):
        if type(self.evaluate.mergeDataSet) == int:
            print('No data')
        else:
            if self.evaluateRotAngleLineEdit.text() == '' or self.evaluateTiltAngleLineEdit.text() == '':
                print('Enter rotation angles')
            else:
                try:
                    rotationAngle = float(self.evaluateRotAngleLineEdit.text())
                    tiltAngel = float(self.evaluateTiltAngleLineEdit.text())
                    self.evaluate.mergeDataSet = rotation_ebsd(self.evaluate.mergeDataSet, rotationAngle, tiltAngel,0) 
                except:
                    print('Only numbers. No Strings!')
                     
    def dataRotationMatrixUpdate(self):
        if self.threadDataRotationMatrix.is_alive():
            QTimer.singleShot(500, self.dataRotationMatrixUpdate)
        else:
            self.evaluateCalculateIPFbutton.setEnabled(True)
            self.evaluateRotationMatrixCalcbutton.setEnabled(True)
            self.evaluateSaveDataButton.setStyleSheet(self.color_click_on)
            self.evaluateCalculateIPFbutton.setStyleSheet(self.color_click_on)
            self.evaluateProgressBar.setMaximum(100)
            self.evaluateSYplotButton.setStyleSheet(self.color_click_on)
            self.evaluateErosionPlotButton.setStyleSheet(self.color_click_on)
            self.evaluateSDplotButton.setStyleSheet(self.color_click_on)
            self.evaluateCountplotButton.setStyleSheet(self.color_click_on)
        
    def browse_sputter_yield_plot(self):
        if len(self.evaluate.heightIPF) == 1:
            print('No data')
        else:
            b = self.evaluateSYticksLineEdit.text()
            c = c=b.split(',')
            f = np.asarray(c,dtype=float)
            
            if self.evaluateSYcheckBox.checkState() != 0:
                contourplot1 = 1
            else:
                contourplot1 = 0
    
            
            saveipf = self.evaluateLoadDataButtonLineEdit.text()[:-4]+'_IPF_sputter_yield.png'
            self.evaluate.atomicDensity = float(self.evaluateSYatomicDensityLineEdit.text())
            self.evaluate.fluence = float(self.evaluateSYfluenceLineEdit.text())
            self.evaluate.heighttoSputterYield()
            self.evaluate.plot_inter_IPFz(self.evaluate.xIPFcoordinate, self.evaluate.yIPFcoordinate, self.evaluate.sputterYield, float(self.evaluateIPFresolutionLineEdit.text()), 
                               cbarrange= [float(self.evaluateSYminLineEdit.text()),float(self.evaluateSYmaxLineEdit.text())], 
                               Title='IPFz', cbarlabel=self.evaluateSYlabelLineEdit.text(), ticks = f, contourplot=contourplot1,savepath = saveipf)
        
    def browseErosionDepthPlot(self):
        
        if len(self.evaluate.heightIPF) == 1:
            print('No data')
        else:
            b = self.evaluateErosionTicksLineEdit.text()
            c = c=b.split(',')
            f = np.asarray(c,dtype=float)
            if self.evaluateErosioncheckBox.checkState() != 0:
                contourplot1 = 1
            else:
                contourplot1 = 0
            if self.evaluateMultiplycheckBox.checkState() != 0:
                erosion = np.multiply(self.evaluate.heightIPF,-1)
            else:
                erosion = self.evaluate.heightIPF
            saveipf = self.evaluateLoadDataButtonLineEdit.text()[:-4]+'_IPF_erosion.png'
            self.evaluate.plot_inter_IPFz(self.evaluate.xIPFcoordinate, self.evaluate.yIPFcoordinate, erosion, float(self.evaluateIPFresolutionLineEdit.text()), 
                               cbarrange= [float(self.evaluateErosionMinLineEdit.text()),float(self.evaluateErosionMaxLineEdit.text())], 
                               Title='IPFz', cbarlabel=self.evaluateErosionLabelLineEdit.text(), ticks = f, contourplot=contourplot1,savepath = saveipf)
            
    def browseSDplot(self):
        
        if len(self.evaluate.heightIPF) == 1:
            print('No data')
        else:
            b = self.evaluateSDticksLineEdit.text()
            c = c=b.split(',')
            f = np.asarray(c,dtype=float)
            
            saveipf = self.evaluateLoadDataButtonLineEdit.text()[:-4]+'_IPF_SD.png'
            self.evaluate.plot_inter_IPFz(self.evaluate.xIPFcoordinate, self.evaluate.yIPFcoordinate, self.evaluate.heightSD, float(self.evaluateIPFresolutionLineEdit.text()), 
                               cbarrange= [float(self.evaluateSDminLineEdit.text()),float(self.evaluateSDmaxLineEdit.text())], 
                               Title='IPFz', cbarlabel=self.evaluateSDlabelLineEdit.text(), ticks = f, contourplot=0,savepath = saveipf)

    def browseCountsplot(self):
        if len(self.evaluate.heightIPF) == 1:
            print('No data')
        else:       
            saveipf = self.evaluateLoadDataButtonLineEdit.text()[:-4]+'_IPF_Counts.png'
            self.evaluate.plot_inter_IPFz(self.evaluate.xIPFcoordinate, self.evaluate.yIPFcoordinate, np.log10(self.evaluate.heightCounts), float(self.evaluateIPFresolutionLineEdit.text()), 
                               cbarrange= [np.log10(float(self.evaluateCountMinLineEdit.text())),np.log10(float(self.evaluateCountMaxLineEdit.text()))], 
                               Title='IPFz', cbarlabel=self.evaluateCountLabelLineEdit.text(), ticks = (0,1,2,3,4,5,6), log=1, contourplot=0,savepath = saveipf)

    def browsePlotHKL(self):
        try:
            h = float(self.evaluateHlineEdit.text())
            k = float(self.evaluateKlineEdit.text())
            l = float(self.evaluateLlineEdit.text())
            ipf = [h,k,l]
            self.evaluate.plot_IPF_plane(IPF=ipf,testsize=20, 
                                            text=self.evaluatePlotHKLtextlineEdit.text(),
                                            yTextoffset =float(self.simPlotYoffsetLtextlineEdit_2.text())/100)
        except:
            print('Error')
        
    def evaluateSaveData(self):
        if type(self.evaluate.mergeDataSet) == int:
            print('No data')
        else:
            file_name = self.browse_button_master('Save .dat file', 'File *.dat (*.dat)', save = True)
            
            print(file_name)
            if file_name[-4:] == '.dat':
                file_name = file_name[:-4]
            if file_name:
                np.savetxt(f'{file_name}.dat', self.evaluate.mergeDataSet,fmt='%.5e',
                           header=' X Y Contrast PHI PSI2 PSI1 H K L IPFz_X_Coordinate IPFz_Y_Coordinate IPFx_X_Coordinate IPFx_Y_Coordinate IPFy_X_Coordinate IPFy_Y_Coordinate')
                self.evaluateLoadDataButtonLineEdit.setText(f'{file_name}.dat')
                self.evaluateSaveDataButton.setStyleSheet(self.green)
            self.evaluateProgressBar.setMaximum(100)

    def browseBinningData(self):
        if type(self.evaluate.mergeDataSet)== int:
            print('No data')
        else:
            self.evaluate.pixelClusterBinning = self.optBinningSizeLineEdit.text()
            self.evaluate.grainBounderies_Reduce_Data()
            self.evaluate.plotEBSDDataBinning()
    
    
    def browse_optimisation_calculate(self):
        if type(self.evaluate.mergeDataSet)== int:
            print('No data')
        else:
    
            print('calculate')
            self.optCalculateButton.setEnabled(False)
    
        
            rotation_angle_list = []
            tilt_angle_list =[]
            angle_list = []
            self.result_optimisation = []
            self.progressbarindex = 0
            
          
            if self.optCompareDataCheckBox.checkState() != 0 and self.optLoadDataButtonLineEdit.text() != '':
                file_name = self.optLoadDataButtonLineEdit.text()
                print('Load Compare data')
                dataToCompare = np.loadtxt(file_name)
                print('Calculate IPF')
                resultDataCompare = meanDataXYZ(dataToCompare,float(self.optiResolutionIPFLineEdit.text()), data_less=3)
                print('Calculate IPF finished')
        
                Xcompare = resultDataCompare[0]
                Ycompare = resultDataCompare[1]
                Zcompare = resultDataCompare[2]
            else:
                Xcompare = [0,1]
                Ycompare = [0,1]
                Zcompare = [0,1]
            
            
            if  int(self.optReduceDataCheckBox.checkState()) !=0:
                data_resulttmp = self.evaluate.mergeDataSetBinning
                print('Use reduced data set')
            else:
                data_resulttmp = self.evaluate.mergeDataSet
            
            gc.disable()
            
            for tilt_angle in np.arange(float(self.optiTiltAngleMinLineEdit.text()), 
                                        float(self.optiTiltAngleMaxLineEdit.text()) + float(self.optiStepSizeLineEdit.text()),
                                        float(self.optiStepSizeLineEdit.text())):
                if tilt_angle == 0:
                    stepsize_rotation = 360
                else:
                    stepsize_rotation = 90*float(self.optiStepSizeLineEdit.text())/tilt_angle
                for rotation_angle in np.arange(float(self.optiRotAngleMinLineEdit.text()), 
                                                float(self.optiRotAngleMaxLineEdit.text()), stepsize_rotation):
                    self.progressbarindex +=1
               
            print(f'Start pool with {self.progressbarindex} processes')
            for tilt_angle in np.arange(float(self.optiTiltAngleMinLineEdit.text()), 
                                        float(self.optiTiltAngleMaxLineEdit.text()) + float(self.optiStepSizeLineEdit.text()),
                                        float(self.optiStepSizeLineEdit.text())):
                
                if tilt_angle == 0:
                    stepsize_rotation = 360
                else:
                    stepsize_rotation = 90*float(self.optiStepSizeLineEdit.text())/tilt_angle
                
                
                for rotation_angle in np.arange(float(self.optiRotAngleMinLineEdit.text()), 
                                                float(self.optiRotAngleMaxLineEdit.text()), stepsize_rotation):
                    rotation_angle_list.extend([rotation_angle])
                    tilt_angle_list.extend([tilt_angle])
                    angle_list.append([rotation_angle,tilt_angle])
                    func2 = partial(optimization_calculate, data_resulttmp, Xcompare, Ycompare, Zcompare, float(self.optiResolutionIPFLineEdit.text()),
                                       int(self.optUseDataEditLine.text()),self.evaluate.opt_points1_xy , self.optSelectedDataCheckBox.checkState(),  )
                    self.pool.apply_async(func=func2, args=([rotation_angle,tilt_angle],), callback=self.result_optimisation.append)
            QTimer.singleShot(2000, self.optimisation_result)
        
    def optimisation_result(self):
        
        if (len(self.result_optimisation) < self.progressbarindex):
            self.optimizeProgressBar.setValue(int(len(self.result_optimisation)/self.progressbarindex *100)+1)
            QTimer.singleShot(2000, self.optimisation_result)
        else:
            Angle1 = []
            Angle2 = []
            Zvariation = []
            ErrorZ = []
            DiffVariation = []
            FitParameter = []
            FitParameterError = []
            print('Multiprocessing finished')
            if self.evaluateLoadDataButtonLineEdit.text() == '':
                path1 = os.getcwd()
            else:
                path1=os.path.split(self.evaluateLoadDataButtonLineEdit.text() )[0]
            file_name=os.path.split(self.evaluateLoadDataButtonLineEdit.text() )[1][0:-4]
            if (file_name == ''):
                file_name = 'optimise_No_File_Name'
            path2=path1+'/'+file_name
            if os.path.exists(path2):
                print('Path result already exist')
            else:
                os.mkdir(path2)
                
            plot_number = 1
            statetmp = 0
            for result_index in range(len(self.result_optimisation)):   
               
                save_file_as = f'{plot_number}__rotation_{np.round(self.result_optimisation[result_index][5],2)}_deg__tilt_{np.round(self.result_optimisation[result_index][4],2)}_deg.png'
                if plot_number < 10:
                    save_file_as = '000' + save_file_as
                elif plot_number < 100:
                    save_file_as = '00' + save_file_as
                elif plot_number < 1000:
                    save_file_as = '0' + save_file_as
        
        
                ErrorZ.extend([np.mean(self.result_optimisation[result_index][3])])
                Zvariation.extend([np.std(self.result_optimisation[result_index][2])])
                Angle1.extend(np.round([self.result_optimisation[result_index][5]],1))
                Angle2.extend(np.round([self.result_optimisation[result_index][4]],1))
                DiffVariation.extend([self.result_optimisation[result_index][7]])
                FitParameter.extend([self.result_optimisation[result_index][8]])
                FitParameterError.extend([self.result_optimisation[result_index][9]])
                if self.result_optimisation[result_index][6] == 1 and statetmp == 0:
                    print('')
                    print('----------------------------------------------------------------------------------------------------------------------------------------')
                    print('Attention: Different amount of data is used during the optimisation!!!')
                    print('Information: Cut off filter is set to 3, because at least 3 data values are needed for a SD.')
                    print('For different rotation and tilte angle new cluster arangments are build and the cut off filter is more or less used')
                    print('')
                    print('For having the same amount of data for the "PCA" or "Min error result": Please reduce the value "Used data in %"')
                    print('')
                    print('Hint: By using "select data points" reduces the amount of data by the IPF selection. Please set used data to e.g. 20%')
                    print('----------------------------------------------------------------------------------------------------------------------------------------')
                    print('')
                    statetmp = 1
                
                plot_number += 1
        
            Angle11 = np.asarray(Angle1)
            Angle21 = np.asarray(Angle2)
            ErrorZ1 = np.asarray(ErrorZ)
            Zvariation1 = np.asarray(Zvariation)
            IPFdiffVariation = np.asarray(DiffVariation)
            FitParameter2 = np.asarray(FitParameter)
            FitParameterError2 = np.asarray(FitParameterError)
            self.evaluate.data_optimize = np.concatenate((Angle11.reshape((len(Angle11),1)), Angle21.reshape((len(Angle21),1)), 
                                            ErrorZ1.reshape((len(Angle21),1)), Zvariation1.reshape((len(Angle21),1)),
                                            IPFdiffVariation.reshape((len(Angle21),1)), 
                                            FitParameter2.reshape((len(Angle21),1)),
                                            FitParameterError2.reshape((len(Angle21),1))  ) ,axis=1)    
            np.savetxt(f'{path2}/{file_name}_Angle1_Angle2_ErrorZ_PCA.txt', self.evaluate.data_optimize)           
            gc.enable()
            
            a = np.where(self.evaluate.data_optimize[:,2]==np.min(self.evaluate.data_optimize[:,2]))
            b = np.where(self.evaluate.data_optimize[:,3]==np.max(self.evaluate.data_optimize[:,3]))
            
            self.optMinRotLineEdit.setText(str(self.evaluate.data_optimize[a[0][0],0]))
            self.optMinTiltLineEdit.setText(str(self.evaluate.data_optimize[a[0][0],1]))
            self.optPCArotLineEdit.setText(str(self.evaluate.data_optimize[b[0][0],0]))
            self.optPCAtiltLineEdit.setText(str(self.evaluate.data_optimize[b[0][0],1]))
            if self.optCompareDataCheckBox.checkState() != 0 and self.optLoadDataButtonLineEdit.text() != '':
                c = np.where(self.evaluate.data_optimize[:,4]==np.min(self.evaluate.data_optimize[:,4]))
                self.optComRotLineEdit.setText(str(self.evaluate.data_optimize[c[0][0],0]))
                self.optComTiltLineEdit.setText(str(self.evaluate.data_optimize[c[0][0],1]))
            
            self.evaluateRotAngleLineEdit.setText(self.optPCArotLineEdit.text())
            self.evaluateTiltAngleLineEdit.setText(self.optPCAtiltLineEdit.text())
            self.optimizeProgressBar.setValue(0)
            self.optCalculateButton.setEnabled(True)
            self.browse_optimisation_calculate_plot(path2,file_name)
                       
#%%
    def browse_optimisation_calculate_plot(self,path2,file_name):        
        self.plot_Angle1Angel2(self.evaluate.data_optimize[:,0],self.evaluate.data_optimize[:,1],self.evaluate.data_optimize[:,3],label='SD_PCA', name=f'{path2}/{file_name}_PCA')
        self.plot_Angle1Angel2(self.evaluate.data_optimize[:,0],self.evaluate.data_optimize[:,1],self.evaluate.data_optimize[:,2],label='mean(SD)', name=f'{path2}/{file_name}_meanError')
        if self.optCompareDataCheckBox.checkState() != 0 and self.optLoadDataButtonLineEdit.text() != '':
            self.plot_Angle1Angel2(self.evaluate.data_optimize[:,0],self.evaluate.data_optimize[:,1],self.evaluate.data_optimize[:,4],label='Difference IPF variation', name=f'{path2}/{file_name}_DifferenceIPFvariation')
            self.plot_Angle1Angel2(self.evaluate.data_optimize[:,0],self.evaluate.data_optimize[:,1],self.evaluate.data_optimize[:,5],label='Difference Fit Parameter', name=f'{path2}/{file_name}_DifferenceFitParameter')
            self.plot_Angle1Angel2(self.evaluate.data_optimize[:,0],self.evaluate.data_optimize[:,1],self.evaluate.data_optimize[:,6],label='SD of the fit Parameter', name=f'{path2}/{file_name}_DifferenceErrorOfTheFitParameter')

    def plot_Angle1Angel2(self,Angle1,Angle2,Z,label='std(Z)', name='name',interpolation=1):
    
        fig1 = plt.figure(figsize=(12,8))
        ax1 = fig1.add_subplot(111, xlabel='x', ylabel='y')
        im1 = ax1.scatter(np.asarray(Angle1), np.asarray(Angle2), c = np.asarray(Z), s=300, marker="s",   cmap=plt.get_cmap('jet'))
        cbar = fig1.colorbar(im1)
        
        cbar.set_label(f'{label} / µm', size=12, labelpad=14)
        
        ax1.set_xlabel('$φ$ / $ \degree$', fontsize=14)
        ax1.set_ylabel('$ϕ$ / $ \degree$', fontsize=14)
        for item in ([ax1.title, ax1.xaxis.label, ax1.yaxis.label] +
                     ax1.get_xticklabels() + ax1.get_yticklabels()):
            item.set_fontsize(12)
         
        for l in cbar.ax.yaxis.get_ticklabels():
            l.set_size(12)
            
        fig1.subplots_adjust(bottom=0.16)
        fig1.savefig(f'{name}.png')
        plt.close(fig1)
        xpolar = np.asarray(Angle2) * np.cos(np.asarray(Angle1)/180*np.pi)
        ypolar = np.asarray(Angle2) * np.sin(np.asarray(Angle1)/180*np.pi)

        
        fig2 = plt.figure(figsize=(14,8))
        ax2 = fig2.add_subplot(111, xlabel='x', ylabel='y')
        if interpolation ==1:
            xi, yi = np.mgrid[-1*max(Angle2):max(Angle2):100j,-1*max(Angle2):max(Angle2):100j]
            zi = griddata((xpolar, ypolar), np.asarray(Z), (xi, yi))
            im2 = ax2.scatter(xi, yi, c = zi, s=100, marker="s",  cmap=plt.get_cmap('jet'))
        else:
            im2 = ax2.scatter(xpolar, ypolar, c = np.asarray(Z), s=300, marker="s",   cmap=plt.get_cmap('jet'))
        
        cbar = fig2.colorbar(im2)
        ax2.plot([0,0],[min(Angle2),max(Angle2)],color='black')
        ax2.plot([min(Angle2),max(Angle2)],[0,0],color='black')
        grad360 = np.arange(0,360,1)
        for k in range(int(max(Angle2))+1):
            ax2.plot(k*np.cos(grad360/180*np.pi), k*np.sin(grad360/180*np.pi),linewidth=1, color='black' )
            ax2.text(0.3,k,f'{k}°')
        
        cbar.set_label(f'{label} / µm', size=12, labelpad=14)
        ax2.set_xlabel('$φ$ / deg', fontsize=14)
        ax2.set_ylabel('$ϕ$ / deg', fontsize=14)
        for item in ([ax2.title, ax2.xaxis.label, ax2.yaxis.label] +
                     ax2.get_xticklabels() + ax2.get_yticklabels()):
            item.set_fontsize(12)
         
        for l in cbar.ax.yaxis.get_ticklabels():
            l.set_size(12)
        plt.savefig(f'{name}_polar.png')
        fig2.show()
        
    def optSelctPoint(self):
        if type(self.evaluate.mergeDataSet)== int:
            print('No data')
        else:
            self.evaluate.optReduceData = self.optReduceDataCheckBox.checkState()
            self.evaluate.resolutionOptimisation = self.optiResolutionIPFLineEdit_2.text()
            self.evaluate.optimisationSelctPointsIPF()
        
    def createLogEvalSave(self):
        LoadedMergedData = self.evaluateLoadDataButtonLineEdit.text()
        self.levelingNumber = 0
        self.logNewHead(self.logfile_eval, 'Data Evaluating')
        self.logNewLine(self.logfile_eval, 'Merged Data filepath:\t'+LoadedMergedData)

    def appendLogEvalLeveling(self, logfile):
        self.levelingNumber += 1
        self.logNewHead(logfile, f'Sputter_Erosion_Leveling #{self.levelingNumber}')
        self.logNewLine(logfile, '')
        self.logNewLine(logfile, 'Sputter_Erosion_Leveling was applied with parameters:')
        self.logNewLine(logfile, f'Reference level:\t{self.evaluateRefereceLevelLineEdit.text()}')
        self.logNewLine(logfile, 'Mean Value:\t')
        
    # def browse_button_AFM_1(self):
    #     file_name = self.browse_button_master('AFM ibw File', 'AFM ibw File (*.ibw)')

    #     self.loadAFM1line.setText(file_name)
    #     self.loadPicDataline.setText('')
    #     self.loadCLSM1line.setText('')
    #     self.loadCLSM2line.setText('')
    #     if file_name:
    #         self.mergeloadPicData.setStyleSheet(self.color_click_on)
    #         self.mergeloadCLSM1.setStyleSheet(self.color_click_on)
    #         self.mergeSelectPoints.setStyleSheet(self.color_click_on)
    #         self.mergeCalculateMerge.setStyleSheet(self.color_click_on)
    #         self.mergeloadAFM1.setStyleSheet(self.green)
    #     else:
    #         self.mergeloadAFM1.setStyleSheet(self.color_click_on)
        
    # def browse_button_AFM_2(self):
    #     file_name = self.browse_button_master('AFM ibw File', 'AFM ibw File (*.ibw)')
        
    #     self.loadAFM2line.setText(file_name)
    #     self.loadPicDataline.setText('')
    #     self.loadCLSM1line.setText('')
    #     self.loadCLSM2line.setText('')
    #     if file_name:
    #         self.mergeloadAFM2.setStyleSheet(self.green)
    #     else:
    #         self.mergeloadAFM2.setStyleSheet(self.color_optional)
    
    # def browse_pic_data(self):
    #     file_name = self.browse_button_master('Picture', 'Picture data (*.bmp *.jpeg *.jpg *.tif *.tiff);; JPEG (*.jpeg)')

    #     self.loadPicDataline.setText(file_name)
    #     self.loadCLSM1line.setText('')
    #     self.loadCLSM2line.setText('')
  
    #     if file_name:
    #         self.mergeloadPicData.setStyleSheet(self.green)
    #         self.mergeloadCLSM1.setStyleSheet(self.color_click_on)
    #         self.mergeSelectPoints.setStyleSheet(self.color_click_on)
    #         self.mergeCalculateMerge.setStyleSheet(self.color_click_on)  
    #     else:
    #         self.mergeloadPicData.setStyleSheet(self.color_click_on)
        
    # def browse_merge_view_AFM(self):
    #     if self.loadAFM1line.text() != '' and self.loadAFM2line.text() != '' :
    #         self.loadAFMdataThread()
    #     elif (len(self.mergedata.confocal_data) == 2 or self.data_merge_afm_single == 0 ):
    #         self.loadAFMdataThread()
    #         self.threadAFM.finished.connect(self.mergedata.view_confocal_data)
            
    # def loadAFMdataThread(self):     
    #     self.workerAFM = merge.mergeThread()
    #     self.threadAFM = QThread()
    #     self.workerAFM.moveToThread(self.threadAFM)

    #     self.workerAFM.finished.connect(self.threadAFM.quit)
        
    #     self.workerAFM.leveling = self.mereAFMLevelingcheckBox.checkState()
    #     if self.mergeAFMradioButton1.isChecked():
    #         self.workerAFM.dataSetselect = 0
    #     elif self.mergeAFMradioButton2.isChecked():
    #         self.workerAFM.dataSetselect = 1
    #     elif self.mergeAFMradioButton3.isChecked():
    #         self.workerAFM.dataSetselect = 2
    #     elif self.mergeAFMradioButton4.isChecked():
    #         self.workerAFM.dataSetselect = 3
    #     elif self.mergeAFMradioButton5.isChecked():
    #         self.workerAFM.dataSetselect = 4
            
    #     print(f'Data select {self.workerAFM.dataSetselect}')    
    #     if (self.loadAFM1line.text() == '' or self.loadAFM2line.text() == ''):
    #         if (len(self.mergedata.confocal_data) == 2 or self.data_merge_afm_single == 0 ):
    #             if (self.loadAFM1line.text()== ''):
    #                 self.workerAFM.loadAFM1line =  self.loadAFM2line.text()
    #                 self.workerAFM.mergeroationAFM1 = self.mergeroationAFM2.currentText()[:-1]
    #                 self.workerAFM.AFM1checkBox = self.AFM2checkBox.checkState()
                    
    #             else:                                       
    #                 self.workerAFM.loadAFM1line =  self.loadAFM1line.text()
    #                 self.workerAFM.AFM1checkBox = self.AFM1checkBox.checkState()
    #                 self.workerAFM.mergeroationAFM1 = self.mergeroationAFM1.currentText()[:-1]
                    
    #             self.threadAFM.started.connect(self.workerAFM.loadAFMdata)
    #             self.threadAFM.start()                
    #             self.data_merge_afm_single = 1
    #             self.mergeviewAFM.setEnabled(False)
    #             self.dataMergeProgressBar.setMaximum(0)
    #             self.threadAFM.finished.connect(self.loadAFMdataThreadFinished)
    #         else:
    #             self.mergedata.view_confocal_data()
    #     else:
    #         if (len(self.mergedata.confocal_data) == 2 or self.data_merge_afm_single == 1 ):
    #             self.workerAFM.loadAFM1line =  self.loadAFM1line.text()
    #             self.workerAFM.loadAFM2line =  self.loadAFM2line.text()
    #             self.workerAFM.mergeroationAFM1 = self.mergeroationAFM1.currentText()[:-1]
    #             self.workerAFM.mergeroationAFM2 = self.mergeroationAFM2.currentText()[:-1]
    #             self.workerAFM.AFM1checkBox = self.AFM1checkBox.checkState()
    #             self.workerAFM.AFM2checkBox = self.AFM2checkBox.checkState()
                
    #             self.threadAFM.started.connect(self.workerAFM.loadAFMdataDiff)
    #             self.threadAFM.start()

    #             self.data_merge_afm_single = 0
    #             self.select_point_for_diff = True
    #             self.mergeviewAFM.setEnabled(False)
    #             self.dataMergeProgressBar.setMaximum(0)
    #             self.threadAFM.finished.connect(self.loadAFM12DataThreadFinished)
    #         else:
    #             self.mergedata.view_confocal_data()

    # def loadAFM12DataThreadFinished(self):
    #     self.mergedata.confocal_data_1 = self.workerAFM.confocal_data_1 
    #     self.mergedata.confocal_data_2 = self.workerAFM.confocal_data_2
    #     self.mergedata.confocal_image_data_1 =  self.workerAFM.confocal_image_data_1
    #     self.mergedata.confocal_image_data_2 =  self.workerAFM.confocal_image_data_2
    #     self.mergeviewAFM.setEnabled(True)
    #     self.mergedata.load_confocal_data_diff_plt()      
    #     self.dataMergeProgressBar.setMaximum(100)
    #     self.workerAFM.deleteLater
    #     self.threadAFM.deleteLater
    #     self.thread = 0
    #     if self.select_points_status:
    #         self.select_points_status = False
    #         self.select_points()
            
    # def loadAFMdataThreadFinished(self):
    #     self.dataMergeProgressBar.setMaximum(100)
    #     self.mergeviewAFM.setEnabled(True)
    #     if self.workerAFM.leveling != 0:
    #         plt.imshow(np.flipud(np.rot90(self.workerAFM.Zebene,1)))
    #         plt.colorbar()
    #         plt.savefig('tmp/000_Ebene.png')
    #         plt.close()

    #         Zdiff = self.workerAFM.confocal_data+self.workerAFM.Zebene
    #         plt.imshow(np.flipud(np.rot90(self.workerAFM.confocal_data+self.workerAFM.Zebene,1)))
    #         plt.clim([np.mean(Zdiff)+np.var(Zdiff),np.mean(Zdiff)-np.var(Zdiff)])
    #         plt.colorbar()
    #         plt.savefig('tmp/000_Confocal_before_leveling.png')
    #         plt.close()
    #         Z2 = self.workerAFM.confocal_data
    #         plt.imshow(np.flipud(np.rot90(Z2,1)))
    #         plt.clim([np.mean(Z2)+np.var(Z2),np.mean(Z2)-np.var(Z2)])
    #         plt.colorbar()
    #         plt.savefig('tmp/000_Confocal_leveled(now).png')
    #         plt.show()


    #     self.mergedata.confocal_data = self.workerAFM.confocal_data 
    #     self.mergedata.confocal_image =  self.workerAFM.confocal_image
    #     self.workerAFM.deleteLater
    #     self.threadAFM.deleteLater
    #     self.threadAFM = 0
    #     if self.select_points_status:
    #         self.select_points_status = False
    #         self.select_points()
        
        
    # def loadPicDataThread(self):     
    #     self.workerPicData = merge.mergeThread()
    #     self.threadPicData= QThread()
    #     self.workerPicData.moveToThread(self.threadPicData)
    #     self.workerPicData.finished.connect(self.threadPicData.quit)
    

    #     self.workerPicData.loadPicDataline =  self.loadPicDataline.text()
    #     self.workerPicData.mergeroationPicData = self.mergeroationPicData.currentText()[:-1]
    #     self.workerPicData.PicDatacheckBox = self.PicDatacheckBox.checkState()
    #     self.threadPicData.started.connect(self.workerPicData.loadPicdata)
    #     self.threadPicData.start()

    #     self.mergeviewPic.setEnabled(False)
    #     self.dataMergeProgressBar.setMaximum(0)
    #     self.threadPicData.finished.connect(self.PicDataThreadFinished)

    # def PicDataThreadFinished(self):
    #     self.dataMergeProgressBar.setMaximum(100)
    #     self.mergeviewPic.setEnabled(True)
    #     self.mergedata.confocal_data = self.workerPicData.confocal_data 
    #     self.mergedata.confocal_image =  self.workerPicData.confocal_image
    #     self.mergedata.view_confocal_data()
    #     self.workerPicData.deleteLater
    #     self.threadPicData.deleteLater
    #     self.threadPicData = 0
    #     if self.select_points_status:
    #         self.select_points_status = False
    #         self.select_points()
                   
    # def browse_view_pic_data(self):        
    #         if (len(self.mergedata.confocal_data) == 2):
    #             self.loadPicDataThread()
    #         else:
    #             self.mergedata.view_confocal_data() 

    # def browse_merge_substract_AFM_12(self):
    #     self.loadAFMdataThread() 
#%%
if __name__ == "__main__":
    print('Program has launched!')
    global ui
    mp.freeze_support()

    app = QtWidgets.QApplication([])
    app.setStyle('Fusion')
    MainWindow = QtWidgets.QMainWindow()
    ui = connectButton()

    MainWindow.show()
    
    sys.exit(app.exec_())