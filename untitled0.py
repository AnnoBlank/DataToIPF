# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 22:04:48 2024

@author: Robin
"""


    def select_points(self):
        print(not self.read_in_selection)
        print(self.ebsd_loaded)
        print(len(self.mergedata.confocal_data) == 2)
        print(len(self.mergedata.confocal_data) != 2 )
        print(self.mergedata.P.shape == (2,))
        print(self.loading_points==1)
        if not self.read_in_selection:
            self.manual_selection=True
        self.dataMergeProgressBar.setMaximum(0)
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
        
    def merge_calc(self):
        
        self.dataMergeProgressBar.setMaximum(0)
        self.mergeCalculateMerge.setEnabled(False)
        self.thread_merge_calc = threading.Thread(target=self.mergeCalcThread, daemon=True)
        self.thread_merge_calc.start()
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
        if self.manual_selection:
            self.logNewLine(self.logfile_merge, 'Points manually selected')
            self.logNewLine(self.logfile_merge, f'Written to path:\t{self.mergedata.save_selected_points_merge}')
        elif self.read_in_selection:
            self.logNewLine(self.logfile_merge, 'Points read in from file')
            self.logNewLine(self.logfile_merge, f'Read from path:\t{self.loading_pointsFileName}')
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

