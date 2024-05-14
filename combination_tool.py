# -*- coding: utf-8 -*-
"""
Created on Thu May  2 10:30:55 2024

@author: Robin
"""

import numpy as np
from PyQt5 import  QtWidgets
from PyQt5.QtWidgets import QFileDialog, QMainWindow, QDialog
import os
import sys
# import ebsd_merge_evaluate.qt5_oberflaeche as qt5_oberflaeche # use this specific GUI

os.chdir(os.path.dirname(os.path.abspath(__file__)))

class connectButton(QMainWindow):
    def __init__(self, parent=None):
        self.path_1 = self.browse_button_master('Merged Dataset .dat File', 'Merge File (*.dat)')
        self.path_2 = self.browse_button_master('Merged Dataset .dat File', 'Merge File (*.dat)')
        dataset_1 = np.loadtxt(self.path_1)
        dataset_2 = np.loadtxt(self.path_2)
        # dataset_2 = np.loadtxt(path_2)
        
        # print(dataset_1[:,0])
        values_1 = np.array([[self.get_ext(dataset_1, True, 0), 
                              self.get_ext(dataset_1, False, 0)],
                             [self.get_ext(dataset_1, True, 1),
                              self.get_ext(dataset_1, False, 1)]])
        
        values_2 = np.array([[self.get_ext(dataset_2, True, 0), 
                              self.get_ext(dataset_2, False, 0)],
                             [self.get_ext(dataset_2, True, 1),
                              self.get_ext(dataset_2, False, 1)]])

        
        X_offset = values_1[0, 1] - values_2[0, 0] + 20
        # X_offset = 500
        # Y_offset = values_2[1, 0] - values_1[1, 1] + 20
        Y_offset = 0

        dataset_2[:, 0] = dataset_2[:, 0] + X_offset
        dataset_2[:, 1] = dataset_2[:, 1] + Y_offset

        dataset_final = np.concatenate((dataset_1, dataset_2)) 

        np.savetxt('combined.dat',dataset_final,fmt='%.5e',header=' X Y Height PHI PSI2 PSI1 H K L X Y   ___datamerge_1_32')
        
        sys.exit()
    
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
    
    def get_ext(self, dataset, minimum = True, column = 0):
        if minimum:
            ext = np.amin(dataset[:,column])
        else: 
            ext = np.amax(dataset[:,column])
        return ext

if __name__ == "__main__":
    print('Program has launched!')
    global ui

    app = QtWidgets.QApplication([])
    MainWindow = QtWidgets.QMainWindow()
    ui = connectButton()

    MainWindow.show()
    
    sys.exit(app.exec_())
