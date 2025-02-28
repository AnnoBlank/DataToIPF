# -*- coding: utf-8 -*-

# Standard library imports
import os
import sys
import gc
import time
from datetime import datetime
import threading
import multiprocessing as mp

# Third-party package imports
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.interpolate import griddata
import scipy.linalg
from functools import partial

# PyQt5 imports
from PyQt5 import QtWidgets
from PyQt5.QtCore import QObject, QThread, QTimer, QRect, pyqtSignal, Qt
from PyQt5.QtWidgets import QFileDialog, QMainWindow, QDialog, QLabel

# Project-specific imports
import ebsd_merge_evaluate.qt5_interface_level as qt5_interface
import ebsd_merge_evaluate.datamerge_threadingQT as merge
import ebsd_merge_evaluate.cubicipf_qt as cubic_ipf
import ebsd_merge_evaluate.ipfzXY as ipfzXY

# Constants
WORKING_DIRECTORY = os.path.dirname(os.path.abspath(__file__))


# Utility function
def setup_environment():
    """Set up the working directory and matplotlib backend for the environment."""
    os.chdir(WORKING_DIRECTORY)
    mpl.use("Qt5Agg")  # Use Qt5Agg backend for Matplotlib


# Initialize environment
setup_environment()


# create the logger for in-GUI printouts of information on working status
class PrintLogger(QObject):
    """
    PrintLogger is a PyQt QObject subclass acting as a custom logging mechanism.

    The class is designed to direct the output text to a specified PyQt signal,
    allowing integration with a GUI application. It formats text to include
    time stamps in a specific format and emits the processed text to the
    connected PyQt signal.

    Attributes:
        newText (pyqtSignal): Signal used to emit the formatted text to connected
        components.

    Public Methods:
        write(text: str) -> None: Writes formatted text, alternating between
        including time information or not, and emits it via the associated signal.
        flush() -> None: Placeholder method required for compatibility with
        stream-like objects.
    """

    newText = pyqtSignal(str)

    def __init__(self):  # pass reference to text widget
        super().__init__()
        self.n = 1  # for self.n functionality

    def write(
        self,
        text,
    ):  # write the input text in the following way
        if self.n == 1:
            # print new line in format 'time_now: new_line'
            self.newText.emit(str(datetime.now().time())[:-7] + ":  " + str(text))
            self.n = 0
        else:
            self.n = 1

    def flush(self):
        pass


# create second interactive window for the case of point selection options
class SecondWindow(
    QtWidgets.QMainWindow,
):
    def __init__(
        self,
        parent=None,
    ):
        super().__init__()
        self.resize(640, 640)
        self.label = QtWidgets.QLabel(self)
        self.label.setGeometry(QRect(200, 20, 89, 23))
        self.label.setObjectName("EBSDcolumn")
        self.label.setText("EBSD")

        self.label_2 = QtWidgets.QLabel(self)
        self.label_2.setGeometry(QRect(350, 20, 89, 23))
        self.label_2.setObjectName("CLSMcolumn")
        self.label_2.setText("CLSM")

        self.pushButton = QtWidgets.QPushButton(self)
        self.pushButton.setGeometry(QRect(20, 580, 181, 41))
        self.pushButton.setObjectName("pushButton")
        self.pushButton.setText("Select new points")

        self.pushButton_2 = QtWidgets.QPushButton(self)
        self.pushButton_2.setGeometry(QRect(340, 580, 181, 41))
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_2.setText("Save")


class popupWindow(
    QDialog,
):
    def __init__(
        self,
        name,
        parent=None,
    ):
        super().__init__(parent)
        self.name = name
        self.label = QLabel(self.name, self)
        self.label.setGeometry(QRect(0, 0, 240, 70))
        self.label.setTextFormat(Qt.AutoText)
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setWordWrap(True)
        self.label.setObjectName("label")
        self.setWindowTitle("Hint")


def linear_fit(
    x,
    a,
    b,
):
    return x[0] - x[1] * a + b


# function for finding most frequent entry within a list (used for determining the phase of the material)
def most_frequent(
    lst,
):
    unique, counts = np.unique(lst, return_counts=True)
    index_of_max_count = np.argmax(counts)
    return unique[index_of_max_count]


def meanDataXYZ(
    matrix_data,
    resolution=2,
    data_less=10,
):
    X, Y, Z, Zstd, Zcounts, h, k1, l = [], [], [], [], [], [], [], []

    scaled_resolution = int(resolution * 4)
    matrix = np.full(
        (40 * scaled_resolution, 38 * scaled_resolution), fill_value=None, dtype=object
    )

    print("Sort Data")

    for i in range(len(matrix_data[:, 9])):
        tmp = matrix_data[i : i + 1, :]
        row_index, col_index = int(matrix_data[i, 9] * resolution), int(
            matrix_data[i, 10] * resolution
        )
        if matrix[row_index, col_index] is None:
            matrix[row_index, col_index] = tmp * resolution
        else:
            matrix[row_index, col_index] = np.concatenate(
                (matrix[row_index, col_index], tmp * resolution)
            )

    print("Sort Data finished")
    less_data_count = 0
    valid_data_count = 0

    for k in range(38 * scaled_resolution):
        for o in range(35 * scaled_resolution):
            tmp = matrix[k, o]
            if tmp is None:
                continue
            elif len(tmp) < data_less:
                less_data_count += 1
            else:
                valid_data_count += 1
                X.append(int(np.mean(tmp[:, 9])))
                Y.append(int(np.mean(tmp[:, 10])))
                Z.append(np.mean(tmp[:, 2]) / resolution)
                Zstd.append(np.std(tmp[:, 2] / resolution))
                Zcounts.append(len(tmp[:, 2]))
                h.append(np.mean(tmp[:, 6]))
                k1.append(np.mean(tmp[:, 7]))
                l.append(np.mean(tmp[:, 8]))

    print("Cut off filter used: ", less_data_count)
    print("IPF data point: ", valid_data_count)

    return X, Y, Z, Zstd, Zcounts, h, k1, l


# def optimization_calculate(matrix_data, Xcompare, Ycompare, Zcompare, ipf_resolution,  prozentdata, opt_points1_xy, optimisation_selct_point, tilt_angleall,):
#     print(tilt_angleall)

#     data_resulttmp = rotation_ebsd(matrix_data, tilt_angleall[0], tilt_angleall[1], 0)
#     data_count_max = int(len(data_resulttmp) * prozentdata / 100)
#     state = 0
#     data_resulttmp2 = data_resulttmp.copy()

#     if optimisation_selct_point !=0 and len(opt_points1_xy) != 4:
#         mask = (
#         (data_resulttmp2[:, 9] < opt_points1_xy[1][0] * 150) &
#         (data_resulttmp2[:, 10] < opt_points1_xy[1][1] * 150) &
#         (data_resulttmp2[:, 9] > opt_points1_xy[0][0] * 150) &
#         (data_resulttmp2[:, 10] > opt_points1_xy[0][1] * 150)
#     )
#     data_resulttmp2 = data_resulttmp2[mask]

#     result = meanDataXYZ(data_resulttmp2, ipf_resolution, data_less = 3)

#     if len(Zcompare) != 2:
#         Z = (result[2] - np.min(result[2])) / np.ptp(result[2])
#         Z2 = (Zcompare - np.min(Zcompare)) / np.ptp(Zcompare)

#         xi, yi = np.mgrid[0:1:200j, 0:1:200j]

#         zi = griddata((result[0] / (150 * ipf_resolution), result[1] / (150 * ipf_resolution)), Z, (xi, yi))
#         zi2 = griddata((Xcompare / (150 * ipf_resolution), Ycompare / (150 * ipf_resolution)), Z2, (xi, yi))

#         mask = ~np.isnan(zi) & ~np.isnan(zi2)
#         zi, zi2 = zi[mask], zi2[mask]

#         data = np.vstack([zi, zi2 - np.mean(zi2) * 0.45])
#         null = np.zeros(len(zi))
#         popt, pcov = curve_fit(linear_fit, data, null)

#         zi -= popt[1]
#         zi2 *= 0.45
#         zdiff = zi - zi2
#         Zdfiivar = np.var(zdiff)
#     else:
#         Zdfiivar = 0
#         popt = [0, 0]
#         pcov = [[0, 0], [0, 0]]

#     result2 = np.asarray(result)
#     result2 = result2[ :, result2[4].argsort()]

#     datacount = 0
#     print('test')

#     for count in reversed(result2[4]):
#         datacount += count
#         if datacount > data_count_max:
#             break
#     if datacount < data_count_max:
#         state = 1

#     result2 = result2[:, -count:]

#     X, Y, Z, Zstd = result[0], result[1], result[2], result[3]

#     return [X, Y, Z, Zstd, tilt_angleall[1], tilt_angleall[0], state, Zdfiivar, popt[0], np.sqrt(pcov[0][0])]


def optimization_calculate(
    matrix_data,
    Xcompare,
    Ycompare,
    Zcompare,
    ipf_resolution,
    prozentdata,
    opt_points1_xy,
    optimisation_selct_point,
    tilt_angleall,
):
    # ipf_resolution, optimisation_selct_point , opt_points1_xy, prozentdata, tilt_angle,rotation_angle):
    print(tilt_angleall)
    data_resulttmp = rotation_ebsd(matrix_data, tilt_angleall[0], tilt_angleall[1], 0)
    data_count_max = int(len(data_resulttmp) / 100 * prozentdata)
    state = 0
    data_resulttmp2 = data_resulttmp.copy()
    if optimisation_selct_point != 0 and len(opt_points1_xy) != 4:
        data_resulttmp2 = data_resulttmp2[
            data_resulttmp2[:, 9] < opt_points1_xy[1][0] * 150
        ]
        data_resulttmp2 = data_resulttmp2[
            data_resulttmp2[:, 10] < opt_points1_xy[1][1] * 150
        ]
        data_resulttmp2 = data_resulttmp2[
            data_resulttmp2[:, 9] > opt_points1_xy[0][0] * 150
        ]
        data_resulttmp2 = data_resulttmp2[
            data_resulttmp2[:, 10] > opt_points1_xy[0][1] * 150
        ]

    result = meanDataXYZ(data_resulttmp2, ipf_resolution, data_less=3)
    if len(Zcompare) != 2:
        Z = np.subtract(result[2], np.min(result[2]))
        Z = Z / np.max(Z)
        Z2 = np.subtract(Zcompare, np.min(Zcompare))
        Z2 = Z2 / np.max(Z2)
        xi, yi = np.mgrid[0:1:200j, 0:1:200j]

        zi = griddata(
            (
                np.divide(result[0], 150 * ipf_resolution),
                np.divide(result[1], 150 * ipf_resolution),
            ),
            Z,
            (xi, yi),
        )
        zi2 = griddata(
            (
                np.divide(Xcompare, 150 * ipf_resolution),
                np.divide(Ycompare, 150 * ipf_resolution),
            ),
            Z2,
            (xi, yi),
        )
        #        zdiff = zi-zi2
        mask = np.logical_not(np.isnan(zi))
        zi = zi[mask]
        zi2 = zi2[mask]
        mask = np.logical_not(np.isnan(zi2))
        zi = zi[mask]
        zi2 = zi2[mask]
        data = np.ones((2, len(zi)))
        zi -= np.mean(zi)
        zi2 -= np.mean(zi2)
        data[0, :] = zi
        data[1, :] = zi2
        null = np.zeros(len(zi))
        popt, pcov = curve_fit(linear_fit, data, null)
        zi = np.subtract(zi, popt[1])
        zi2 = np.multiply(zi2, 0.45)
        zdiff = zi - zi2
        Zdfiivar = np.var(zdiff)
    else:
        Zdfiivar = 0
        popt = [0, 0]
        pcov = [[0, 0], [0, 0]]
    result2 = np.asarray(result)
    result2 = result2[:, result2[4].argsort()]
    i = 0
    datacount = 0
    print("test")
    for i in range(0, len(result2[4])):
        datacount += result2[4, -i]
        if datacount > data_count_max:
            break
    if datacount < data_count_max:
        state = 1
    result2 = result2[:, -i:]
    X = result[0]
    Y = result[1]
    Z = result[2]
    Zstd = result[3]

    return [
        X,
        Y,
        Z,
        Zstd,
        tilt_angleall[1],
        tilt_angleall[0],
        state,
        Zdfiivar,
        popt[0],
        np.sqrt(pcov[0][0]),
    ]


def rotation_ebsd(
    matrix,
    phi1,
    phi2,
    phi3,
):

    position_rotation = np.array([[0], [0], [1]])
    R_1 = ipfzXY.rotation(phi1, phi2, phi3)

    for i in range(0, len(matrix)):
        R_2 = ipfzXY.rotation(matrix[i, 3], matrix[i, 4], matrix[i, 5])
        R = np.dot(np.linalg.inv(R_1), R_2)
        xx, yy, zz = np.dot(np.transpose(R), position_rotation)
        matrix[i, 6:9] = xx[0], yy[0], zz[0]
        r = np.linalg.norm(matrix[i, 6:9])
        matrix[i, 6:9] /= r

    for k in range(0, len(matrix)):
        matrix[k, 9:11] = ipfzXY.poleA2(matrix[k, 6], matrix[k, 7], matrix[k, 8])

    return matrix


class connectButton(
    qt5_interface.Ui_MainWindow,
    QMainWindow,
):
    def __init__(self):
        super().__init__()

        # Initialize class attributes
        # self.select_points_status = False
        self.data_merge_clsm_single = False
        self.data_merge_afm_single = 0
        self.select_point_for_diff = False
        self.ebsd_loaded = False
        self.loading_points = False
        self.CLSM1_rendered = False
        self.CLSM2_rendered = False

        self.setupUi(MainWindow)
        self.thread = 0

        self.EBSD_CLSM_manual_selection = False
        self.EBSD_CLSM_read_in = False
        self.EBSDPhaseQuiet = False

        self.CLSM12_manual_selection = False
        self.CLSM12_read_in = False

        self.loading_pointsFileName = "tmp/000_selected-points.txt"
        self.merge_save_path = ""

        self.read_in = False

        # Initialize mergedata and evaluate objects
        self.mergedata = merge.mergedata()
        self.tabMergeCLSM()

        self.evaluate = cubic_ipf.EBSDdataEvaluation()
        self.evaluateSim = cubic_ipf.EBSDdataEvaluation()
        self.tabEvaluateCLSM()
        self.evaluate.opt_points1_xy = np.array([0, 0, 0, 0])

        # Initialize PrintLogger
        self.PrintLogger = PrintLogger()
        sys.stdout = self.PrintLogger  # Set standard text output to logger window
        self.PrintLogger.newText.connect(self.addTextprint_logger)

        # Define color constants
        self.green = "background-color: rgb(138, 226, 52);"
        self.grey = "background-color: rgb(155, 155, 155);"
        self.color_optional = "background-color: rgb(252, 233, 79);"
        self.color_click_on = "background-color: rgb(252, 175, 62);"

        self.pool = mp.Pool(mp.cpu_count())

        # Create tmp directory if it doesn't exist
        os.makedirs("tmp", exist_ok=True)

        # Initialize timestamp
        self.initTimestamp()
        self.mergedata.now = self.now
        self.mergedata.nowstr = self.nowstr

        # Initialize logfiles
        self.createLogFileMerge()
        self.createLogFileEval()

    def addTextprint_logger(
        self,
        new_text,
    ):
        self.textBrowser.append(new_text)

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

        self.checkBoxParams.clicked.connect(self.checkBoxFileUncheck)
        self.checkBoxFileInput.clicked.connect(self.checkBoxParamsUncheck)
        self.calcByEdge.clicked.connect(self.calcSlopeByEdge)
        self.LevelingReadIn.clicked.connect(self.browseLevelingFile)
        self.LevelingMain.clicked.connect(self.levelingMainBrowse)

        self.mergeviewCLSM.setEnabled(False)
        self.autoSubstract.setEnabled(False)
        self.mergesubstractCLSM12.setEnabled(False)
        self.loadsubstractCLSM12.setEnabled(False)
        self.mergeCalculateMerge.setEnabled(False)
        self.calcByEdge.setEnabled(False)
        self.mergeSelectArea.setEnabled(False)

        self.mergeSelectArea.clicked.connect(self.area_zero_level_button)
        self.mergeDeleteDataPic.clicked.connect(self.browse_load_pic_data)
        self.mergeSave.clicked.connect(self.merge_save_data)

        self.mergeDeleteCLSMcheckBox.stateChanged.connect(
            self.delete_data_CLSM_checkbox
        )
        self.mergeDeleteEBSDcheckBox.stateChanged.connect(
            self.delete_data_EBSD_checkbox
        )
        self.mergeDeleteDataPicExecButton.clicked.connect(self.browse_delete_pic_data)

        # self.mergeloadAFM1.clicked.connect(self.browse_button_AFM_1)
        # self.mergeloadAFM2.clicked.connect(self.browse_button_AFM_2)
        # self.mergeviewAFM.clicked.connect(self.browse_merge_view_AFM)
        # self.mergesubstractAFM12.clicked.connect(self.browse_merge_substract_AFM_12)

        # self.mergeloadPicData.clicked.connect(self.browse_pic_data)
        # self.mergeviewPic.clicked.connect(self.browse_view_pic_data)

        self.mergeviewAFM.clicked.connect(self.show_warning_save)

    def checkBoxFileUncheck(self):
        if self.checkBoxParams.isChecked():
            self.checkBoxFileInput.setChecked(False)
        else:
            self.checkBoxFileInput.setChecked(True)

    def checkBoxParamsUncheck(self):
        if self.checkBoxFileInput.isChecked():
            self.checkBoxParams.setChecked(False)
        else:
            self.checkBoxParams.setChecked(True)

    def calcSlopeByEdge(self):
        data = self.mergedata.confocal_data
        m, n = data.shape
        row, column = np.mgrid[:m, :n]
        data2tmp = np.column_stack((row.ravel(), column.ravel(), data.ravel()))

        A = np.c_[data2tmp[:, 0], data2tmp[:, 1], np.ones(data2tmp.shape[0])]
        C, _, _, _ = scipy.linalg.lstsq(A, data2tmp[:, 2])

        X, Y = np.meshgrid(np.linspace(0, n - 1, n), np.linspace(0, m - 1, m))

        self.XSlopeLine.setText(str(round(C[1], 10)))
        self.YSlopeLine.setText(str(round(C[0], 10)))

    def area_zero_level_button(self):
        self.mergeSave.setStyleSheet(self.color_click_on)
        if len(self.mergedata.confocal_data) == 2:
            self.load_clsm_data_thread()
            print("Calculating CLSM Data first, try once again.")
            # self.thread.finished.connect(self.area_zero_level)
        else:
            try:
                self.mergedata.data_zero_level(1)
                print(
                    f"Mean height of selected Area: {round(self.mergedata.Zerowerte_all[0], 10)}"
                )
                self.Offset00Line.setText(
                    str(round(self.mergedata.Zerowerte_all[0], 10))
                )
            except:
                print("Error")

    def levelingMainBrowse(self):
        if self.checkBoxParams.checkState():
            self.levelingByParams()
        elif self.checkBoxFileInput.checkState():
            self.levelingByFile()

    def levelingByFile(self):
        filepath = self.lineEdit_4.text()

        Zebene = np.loadtxt(filepath, delimiter=",")

        self.mergedata.confocal_data = self.mergedata.confocal_data - Zebene

        print(f"Substracted heights through file {filepath}")

    def browseLevelingFile(self):
        file_name = self.browse_button_master(
            "Plane CSV File",
            "CSV File (*.csv)",
        )
        self.lineEdit_4.setText(file_name)

    def levelingByParams(self):
        data = self.mergedata.confocal_data
        m, n = data.shape
        X, Y = np.meshgrid(np.linspace(0, n - 1, n), np.linspace(0, m - 1, m))

        XSlope = float(self.XSlopeLine.text())
        YSlope = float(self.YSlopeLine.text())
        if self.Offset00Line.text():
            Offset = float(self.Offset00Line.text())
        else:
            Offset = 0

        Zebene = XSlope * X + YSlope * Y + Offset

        np.savetxt(
            os.path.join("tmp", "Affine_plane.csv"), Zebene, fmt="%.7e", delimiter=","
        )

        self.mergedata.confocal_data = self.mergedata.confocal_data - Zebene

        print(f"X_Slope: {round(XSlope, 10)}")
        print(f"Y_Slope: {round(YSlope, 10)}")
        print(f"Offset: {round(Offset, 10)}")

    def show_warning_save(self):  # Create Popup
        message = "Please remember to save your temporary data in tmp folder."
        print(message)

        self.popup = popupWindow(message)
        self.popup.setGeometry(1200, 600, 240, 70)
        self.popup.show()

    def browse_view_CLSM12(self):
        self.load_clsm_data_thread()  # followup_selection = False
        if self.loadCLSM1line.text() != "" and self.loadCLSM2line.text() != "":
            # self.thread.finished.connect(self.mergedata.view_confocal_data)
            0
        elif len(self.mergedata.confocal_data) == 2 or not self.data_merge_clsm_single:
            self.thread.finished.connect(self.mergedata.view_confocal_data)

        else:
            self.mergedata.view_confocal_data()
        self.calcByEdge.setEnabled(True)
        self.mergeSelectArea.setEnabled(True)

    def load_clsm_data_thread(self):

        self.dataMergeProgressBar.setMaximum(0)

        self.worker = merge.mergeThread()
        self.thread = QThread()
        self.worker.moveToThread(self.thread)
        self.worker.finished.connect(self.thread.quit)

        self.worker.leveling = 0  # self.mergeLevelingcheckBox.checkState()

        self.worker.mergeroationCLSM1 = self.mergeroationCLSM1.currentText()[:-1]
        self.worker.mergeroationCLSM2 = self.mergeroationCLSM2.currentText()[:-1]
        self.worker.CLSM1checkBox = self.CLSM1checkBox.checkState()
        self.worker.CLSM2checkBox = self.CLSM2checkBox.checkState()
        self.worker.CLSM1_rendered = self.CLSM1_rendered
        self.worker.CLSM2_rendered = self.CLSM2_rendered

        if self.loadCLSM1line.text() == "" or self.loadCLSM2line.text() == "":
            if (
                len(self.mergedata.confocal_data) == 2
                or not self.data_merge_clsm_single
            ):
                if not self.loadCLSM1line.text():
                    self.worker.loadCLSM1line = self.loadCLSM2line.text()
                    self.worker.mergeroationCLSM1 = (
                        self.mergeroationCLSM2.currentText()[:-1]
                    )
                    self.worker.CLSM1checkBox = self.CLSM2checkBox.checkState()
                else:
                    self.worker.loadCLSM1line = self.loadCLSM1line.text()
                    self.worker.mergeroationCLSM1 = (
                        self.mergeroationCLSM1.currentText()[:-1]
                    )
                    self.worker.CLSM1checkBox = self.CLSM1checkBox.checkState()

                self.thread.started.connect(self.worker.load_confocal_data)
                self.thread.finished.connect(self.load_clsm_data_thread_finished)
                self.thread.start()

                self.check_CLSM_availability()

                # self.load_clsm_data_thread_finished()

            else:
                self.mergedata.view_confocal_data()
                self.dataMergeProgressBar.setMaximum(100)
        else:
            if len(self.mergedata.confocal_data) == 2 or self.data_merge_clsm_single:
                self.worker.loadCLSM1line = self.loadCLSM1line.text()
                self.worker.loadCLSM2line = self.loadCLSM2line.text()

                self.thread.started.connect(self.worker.load_confocal_data_diff)
                self.thread.start()

                self.CLSM1_rendered = self.worker.CLSM1_rendered
                self.CLSM2_rendered = self.worker.CLSM2_rendered

                self.data_merge_clsm_single = False
                self.select_point_for_diff = True

                self.check_CLSM_availability()

            else:
                self.mergedata.view_confocal_data()
                self.dataMergeProgressBar.setMaximum(100)

    def load_clsm_data_thread_finished(self):
        self.dataMergeProgressBar.setMaximum(100)

        self.check_CLSM_availability()

        self.worker.leveling = 0  # self.mergeLevelingcheckBox.checkState()

        if self.worker.leveling != 0:
            # plt.imshow(np.flipud(np.rot90(self.worker.Zebene, 1)))
            # plt.colorbar()
            # plt.savefig('tmp/000_Ebene.png')
            # plt.close()

            # self.plot_confocal(self.worker.Zebene, '000_Ebene.png')

            Zdiff = self.worker.confocal_data + self.worker.Zebene
            self.plot_confocal(Zdiff, "000_Confocal_before_leveling.png")

            # Z2 = self.worker.confocal_data
            # self.plot_confocal(Z2, '000_Confocal_leveled(now).png')
            self.plot_confocal(
                self.worker.confocal_data, "000_Confocal_leveled(now).png"
            )

            self.plot_confocal(self.worker.Zebene, "000_Ebene.png")
        # print(8)
        self.mergedata.confocal_data = self.worker.confocal_data
        self.mergedata.confocal_image = self.worker.confocal_image
        self.clean_up_thread()

    def plot_confocal(
        self,
        data,
        file_name,
    ):
        plt.imshow(np.flipud(np.rot90(data, 1)))
        plt.clim([np.mean(data) + np.var(data), np.mean(data) - np.var(data)])
        plt.colorbar()
        plt.savefig(os.path.join("tmp", file_name))
        # plt.waitforbuttonpress()
        # plt.show(block=True)
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

    def load_auto_clsm_data_thread_finished(self):
        self.check_CLSM_availability()

        self.dataMergeProgressBar.setMaximum(100)

        self.mergedata.confocal_data_1 = self.worker.confocal_data_1
        self.mergedata.confocal_data_2 = self.worker.confocal_data_2
        self.mergedata.confocal_image_data_1 = self.worker.confocal_image_data_1
        self.mergedata.confocal_image_data_2 = self.worker.confocal_image_data_2

        self.mergedata.pattern_matching_auto(
            leveling=0
        )  # self.mergeLevelingcheckBox.checkState())

        self.clean_up_thread()

    def load_auto_clsm_data_thread_finished_from_file(self):
        file_name = self.browse_button_master(
            "Matching Points .txt File",
            "CLSM Matching Points File (*.txt)",
            tmp_true=True,
        )

        self.check_CLSM_availability()

        self.dataMergeProgressBar.setMaximum(100)
        self.mergedata.confocal_data_1 = self.worker.confocal_data_1
        self.mergedata.confocal_data_2 = self.worker.confocal_data_2
        self.mergedata.confocal_image_data_1 = self.worker.confocal_image_data_1
        self.mergedata.confocal_image_data_2 = self.worker.confocal_image_data_2
        self.mergedata.confocal_diff_from_file(
            file_name, leveling=0
        )  # self.mergeLevelingcheckBox.checkState())

        self.clean_up_thread()
        # self.select_points_status = False

    def browse_button_master(
        self,
        cap,
        fil,
        save=False,
        tmp_true=False,
        name_suggestion="",
    ):
        if tmp_true:
            directory = directory = os.path.join(os.getcwd(), "tmp")
        else:
            directory = os.getcwd()

        directory = os.path.join(directory, name_suggestion)

        options = QFileDialog.Options()

        if save:
            file_name, _ = QFileDialog.getSaveFileName(
                directory=directory,
                caption=cap,
                filter=fil,
                options=options,
            )
        else:
            file_name, _ = QFileDialog.getOpenFileName(
                directory=directory,
                caption=cap,
                filter=fil,
                options=options,
            )

        return file_name

    def browse_button_EBSD(self):
        file_name = self.browse_button_master(
            "EBSD CTF File",
            "CTF File (*.ctf)",
        )
        self.loadEBSDline.setText(file_name)

        if file_name:
            # Logfile entry
            self.logNewHead(self.logfile_merge_path, "New Data Merging")
            _, relativePath = os.path.split(file_name)
            self.logNewLine(self.logfile_merge_path, "Load_EBSD: " + relativePath)
            self.logNewSubline(
                self.logfile_merge_path, "Full EBSD filepath: " + file_name
            )

            # Continuation of processing the file
            self.load_ebsd_data()
            self.mergeloadEBSD.setStyleSheet(self.green)
            self.ebsd_loaded = True
        else:
            print("No file selected for input of EBSD data")
            self.mergeloadEBSD.setStyleSheet(self.color_click_on)
            self.ebsd_loaded = False

    def browse_button_CLSM_1(self):
        file_name = self.browse_button_master(
            "CLSM csv File",
            "CLSM CSV File (*.csv)",
        )

        self.CLSM1_rendered = False
        self.loadCLSM1line.setText(file_name)

        if file_name:
            self.render_clsm_data(CLSM_render_set=0)
            self.mergeloadCLSM1.setStyleSheet(self.green)
            self.mergeSelectPoints.setStyleSheet(self.color_click_on)
            self.mergeCalculateMerge.setStyleSheet(self.color_click_on)

            # Logfile entry
            _, relativePath = os.path.split(file_name)
            self.logNewLine(self.logfile_merge_path, "Load_CLSM_1: " + relativePath)
            self.logNewSubline(
                self.logfile_merge_path, "Full CLSM_1 filepath: " + file_name + "\n"
            )
        else:
            self.mergeloadCLSM1.setStyleSheet(self.color_click_on)
        self.check_CLSM_availability()

    def browse_button_CLSM_2(self):
        file_name = self.browse_button_master(
            "CLSM csv File",
            "CLSM CSV File (*.csv)",
        )

        self.CLSM2_rendered = False
        self.loadCLSM2line.setText(file_name)

        if file_name:
            self.render_clsm_data(CLSM_render_set=1)
            self.mergeloadCLSM2.setStyleSheet(self.green)

            # Logfile entry
            _, relativePath = os.path.split(file_name)
            self.logNewLine(self.logfile_merge_path, "Load_CLSM_2: " + relativePath)
            self.logNewSubline(
                self.logfile_merge_path, "Full CLSM_2 filepath: " + file_name + "\n"
            )
        else:
            self.mergeloadCLSM2.setStyleSheet(self.color_optional)
        self.check_CLSM_availability()

    def check_CLSM_availability(self):
        self.mergeviewCLSM.setEnabled(True)
        if self.loadCLSM1line.text() != "" and self.loadCLSM2line.text() != "":
            self.autoSubstract.setEnabled(True)
            self.mergesubstractCLSM12.setEnabled(True)
            self.loadsubstractCLSM12.setEnabled(True)
        else:
            self.autoSubstract.setEnabled(False)
            self.mergesubstractCLSM12.setEnabled(False)
            self.loadsubstractCLSM12.setEnabled(False)

    def browse_load_points_merge(self):
        file_name = self.browse_button_master(
            "Matching Points .txt File",
            "EBSD-CLSM Matching Points File (*.txt)",
            tmp_true=True,
        )

        self.loading_points = True
        self.loading_pointsFileName = file_name

        if file_name:
            self.select_points()

            # Logfile entry
            # _, relativePath = os.path.split(file_name)
            # self.logNewLine(self.logfile_merge_path, 'Load_Matching_Points_Difference_Microscopy: ' + relativePath)
            # self.logNewSubline(self.logfile_merge_path, 'Full Matching Points filepath: ' + file_name)

        else:
            print("No file selected for input of matching points")

            # Logfile entry
            self.logNewLine(
                self.logfile_merge_path,
                "Load_Matching_Points_Difference_Microscopy Failed: No input file given!",
            )

    # def browse_load_points_diff(self):
    #     file_name = self.browse_button_master('Matching Points .txt File', 'CLSM Matching Points File (*.txt)', tmp_true = True)

    #     if file_name:
    #         self.select_points_diff()
    #     else:
    #         print('No file selected for input of matching points')

    def browse_load_pic_data(self):
        if type(self.mergedata.dataclsm) != int:
            file_name = self.browse_button_master(
                "CLSM File",
                "Picture data (*.bmp *.jpeg *.jpg *.tif *.tiff);; JPEG (*.jpeg)",
            )

            if file_name:
                self.loadDeletDataline.setText(file_name)
                self.mergedata.loadPicdataForDelete(file_name)

                self.mergeDeleteDataPicExecButton.setEnabled(True)

                # Logfile entry
                _, relativePath = os.path.split(file_name)
                self.logNewLine(
                    self.logfile_merge_path,
                    "Load_mask_from_Picture_data: " + relativePath,
                )
                self.logNewSubline(
                    self.logfile_merge_path, "Full Picture data filepath: " + file_name
                )
            else:
                print("Subtraction of mask was not applied: No mask file given!")

                # Logfile entry
                self.logNewLine(
                    self.logfile_merge_path,
                    "Load_mask_from_Picture_data Failed: No mask file given!",
                )

        else:
            print("Subtraction of mask was not applied: No CLSM Data!")

            # Logfile entry
            self.logNewLine(
                self.logfile_merge_path,
                "Load_mask_from_Picture_data Failed: No CLSM Data!",
            )

    def merge_save_data(self):
        file_name = self.browse_button_master(
            "Save .dat file",
            "File *.dat (*.dat)",
            save=True,
            tmp_true=True,
            name_suggestion="000_merge.dat",
        )

        self.show_warning_save()
        self.dataMergeProgressBar.setMaximum(100)
        if file_name[-4:] == ".dat":
            file_name = file_name[:-4]

        try:
            if file_name:
                self.mergedata.save_confocal_data(file_name)
                self.merge_save_path = file_name
                self.mergeSave.setStyleSheet(self.green)
                self.dataMergeProgressBar.setMaximum(100)
                print("Merged data saved as: " + file_name + ".dat")

                # Logfile entry
                self.logNewLine(
                    self.logfile_merge_path, f"Merged data saved as: {file_name}.dat"
                )

            # self.createLogMergeSave()
            if self.mergedata.Zerowerte_all != 0:
                zerolevelName = f"{file_name}-Zerolevel-{np.round(np.mean(self.mergedata.Zerowerte_all),3)}.dat"
                np.savetxt(zerolevelName, self.mergedata.Zerowerte_all)
                print(f"Zerolevel data saved as: {zerolevelName}")

                # Logfile entry
                self.logNewLine(
                    self.logfile_merge_path, f"Zerolevel data saved as: {zerolevelName}"
                )

        except:
            print("Data save failed")

            # Logfile entry
            self.logNewLine(
                self.logfile_merge_path, "Data save failed: Unknown reason!"
            )

    def browse_sim_load_MD(self):
        file_name = self.browse_button_master(
            "MD data File",
            "File .dat (*.dat)",
        )

        self.simDataPath.setText(file_name)

        if file_name:
            self.simLoadMDpushButton.setEnabled(False)
            self.simLoadBCApushButton.setEnabled(False)
            self.simProgressBar.setMaximum(0)

            self.threadCalculateIPFsim = threading.Thread(
                target=self.browse_sim_load_MD_thread, daemon=True
            )
            self.threadCalculateIPFsim.start()

            QTimer.singleShot(500, self.threadCalculateIPFUpdateSim)
        else:
            print("No file selected for input of MD data")

    def browse_sim_load_BCA(self):
        file_name = self.browse_button_master(
            "BCA data File",
            "File .yield (*.yield)",
        )

        self.simDataPath.setText(file_name)

        if file_name:
            self.simLoadMDpushButton.setEnabled(False)
            self.simLoadBCApushButton.setEnabled(False)
            self.simProgressBar.setMaximum(0)

            self.threadCalculateIPFsimBCA = threading.Thread(
                target=self.browse_sim_load_BCA_thread, daemon=True
            )
            self.threadCalculateIPFsimBCA.start()

            QTimer.singleShot(500, self.threadCalculateIPFUpdateSimBCA)
        else:
            print("No file selected for input of BCA data")

    def browseOptLoadDataButton(self):
        file_name = self.browse_button_master(
            "Comparison IPF file",
            "Merged IPF file .dat (*.dat)",
        )

        self.optLoadDataButtonLineEdit.setText(file_name)

    def load_clsm1_2_data_thread_finished_finished(self):
        self.rendering_clsm1_2_data_thread_finished()
        self.mergedata.load_confocal_data_diff_plt(
            leveling=0
        )  # self.mergeLevelingcheckBox.checkState())

    def load_auto_clsm1_2_data_thread_finished_finished(self):
        self.rendering_clsm1_2_data_thread_finished()
        self.mergedata.pattern_matching_auto(
            leveling=0
        )  # self.mergeLevelingcheckBox.checkState())

    def load_auto_clsm_data_thread_finished_from_file_finished(self):
        self.rendering_clsm1_2_data_thread_finished()
        file_name = self.browse_button_master(
            "Matching Points .txt File",
            "CLSM Matching Points File (*.txt)",
            tmp_true=True,
        )
        self.mergedata.confocal_diff_from_file(
            file_name, leveling=0
        )  # self.mergeLevelingcheckBox.checkState())

    def browse_CLSM_substract_norm(self):
        self.load_clsm_data_thread()
        self.thread.finished.connect(self.load_clsm1_2_data_thread_finished_finished)

    def browse_CLSM_substract_auto(self):
        self.load_clsm_data_thread()
        self.thread.finished.connect(
            self.load_auto_clsm1_2_data_thread_finished_finished
        )

    def browse_CLSM_substract_file(self):
        self.load_clsm_data_thread()
        self.thread.finished.connect(
            self.load_auto_clsm_data_thread_finished_from_file_finished
        )

    def ebsd_phase_changed(self):
        if (
            self.ebsd_loaded
            and (self.phaseEBSD.text() != "")
            and not self.EBSDPhaseQuiet
        ):
            # Logfile entry
            self.logNewLine(
                self.logfile_merge_path,
                "Manual phase Change to: " + self.phaseEBSD.text() + "\n",
            )
            print("Phase changed to: " + self.phaseEBSD.text())

            # Continue
            self.load_ebsd_data()

    def load_ebsd_data(self):
        self.workerEBSD = merge.mergeThread()
        self.threadEBSD = QThread()
        self.workerEBSD.moveToThread(self.threadEBSD)
        self.workerEBSD.finished.connect(self.threadEBSD.quit)
        self.workerEBSD.fileNameEBSD = self.loadEBSDline.text()
        self.workerEBSD.phaseEBSD = self.phaseEBSD.text()
        if self.workerEBSD.phaseEBSD == "":
            file_name = self.workerEBSD.fileNameEBSD

            with open(file_name, "r", errors="replace") as f:
                CRSTdata = f.readlines()
            CRSTdata = CRSTdata[1:]
            zeroColumn = [lines.split()[0] for i, lines in enumerate(CRSTdata)]
            zeroColumnIntegers = [
                col for col in zeroColumn if (col.isdigit() and col != "0")
            ]
            phase = str(most_frequent(zeroColumnIntegers))
            self.EBSD_phase = phase
            print(f"Automated phase detection used phase = {phase}")
            self.workerEBSD.phaseEBSD = phase
            self.EBSDPhaseQuiet = True
            self.phaseEBSD.setText(str(phase))
            self.EBSDPhaseQuiet = False

            # Entry for logfile
            self.logNewSubline(
                self.logfile_merge_path,
                "Automated detection found phase: " + phase + "\n",
            )

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

    def render_clsm_data(
        self,
        CLSM_render_set,
    ):
        self.workerCLSM2 = merge.mergeThread()
        self.workerCLSM2.CLSM_render_set = CLSM_render_set

        self.threadCLSM2 = QThread()
        self.workerCLSM2.moveToThread(self.threadCLSM2)
        self.workerCLSM2.finished.connect(self.threadCLSM2.quit)
        self.workerCLSM2.fileNameEBSD = self.loadEBSDline.text()
        self.workerCLSM2.phaseEBSD = self.phaseEBSD.text()

        self.workerCLSM2.loadCLSM1line = self.loadCLSM1line.text()
        self.workerCLSM2.loadCLSM2line = self.loadCLSM2line.text()
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

        if self.workerCLSM2.CLSM_render_set == 0:
            self.mergedata.confocal_image_data_1 = (
                self.workerCLSM2.confocal_image_data_1
            )
            self.mergedata.confocal_data_1 = self.workerCLSM2.confocal_data_1
        elif self.workerCLSM2.CLSM_render_set == 1:
            self.mergedata.confocal_image_data_2 = (
                self.workerCLSM2.confocal_image_data_2
            )
            self.mergedata.confocal_data_2 = self.workerCLSM2.confocal_data_2

        self.mergedata.confocal_image_data_2 = self.workerCLSM2.confocal_image_data_2

        self.CLSM1_rendered = self.workerCLSM2.CLSM1_rendered
        self.CLSM2_rendered = self.workerCLSM2.CLSM2_rendered

        if self.thread == 0:
            self.dataMergeProgressBar.setMaximum(100)

        self.check_CLSM_availability()

        if self.CLSM_render_set == 0:
            self.CLSM1_rendered = True

        elif self.CLSM_render_set == 1:
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
            self.mergedata.X = self.workerEBSD.X
            self.mergedata.Y = self.workerEBSD.Y
            self.mergedata.Orientation = self.workerEBSD.Orientation
            self.mergedata.X_grid = self.workerEBSD.X_grid
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
            print("No EBSD data")

    def select_points(self):
        if self.ebsd_loaded:
            if len(self.mergedata.confocal_data) == 2:
                self.load_clsm_data_thread()
                self.select_points_status = 1

                print("CLSM data was rendered")
                self.logNewLine(
                    self.logfile_merge_path, "CLSM data was rendered" + "\n"
                )

            elif len(self.mergedata.confocal_data) != 2:

                if self.mergedata.P.shape == (2,):
                    self.dataMergeProgressBar.setMaximum(100)
                    if self.loading_points == 1:
                        # print("Loading matching points from file")

                        self.mergedata.load_points_merge(self.loading_pointsFileName)
                        self.manual_selection = False
                        self.read_in_selection = True
                        self.loading_points = 0

                    else:

                        self.mergedata.calibrate_confocal_and_EBSD_data()

                    if self.mergedata.P.shape != (2,):
                        self.mergeSelectPoints.setStyleSheet(self.green)
                        self.mergeCalculateMerge.setStyleSheet(self.color_click_on)
                        self.mergeCalculateMerge.setEnabled(True)

                else:
                    self.dataMergeProgressBar.setMaximum(100)
                    print(
                        "Matching points already given, proceeding to selection window"
                    )
                    self.logNewLine(
                        self.logfile_merge_path, "Selection window opened" + "\n"
                    )

                    self.select_points_window()

        else:
            print("No EBSD data")
            self.dataMergeProgressBar.setMaximum(100)

    # def select_points_finished(self):
    #     self.mergedata.calibrate_confocal_and_EBSD_data()
    #     self.mergeSelectPoints.setStyleSheet(self.green)
    #     self.mergeCalculateMerge.setStyleSheet(self.color_click_on)

    def select_points_window(self):
        self.dialog = SecondWindow(self)
        self.dialog.checkBox = []
        self.dialog.labelEBSDcoordinate = []
        self.dialog.labelCLSMcoordinate = []
        for i in range(
            len(self.mergedata.P)
        ):  # This loop creates one tickable box for each preselected point

            self.dialog.checkBox.append(QtWidgets.QCheckBox(self.dialog))
            self.dialog.checkBox[i].setGeometry(QRect(40, 70 + 40 * i, 61, 23))
            self.dialog.checkBox[i].setObjectName(f"checkBox{i}")
            self.dialog.checkBox[i].setText(f"P{i}")
            self.dialog.checkBox[i].setChecked(True)
            self.dialog.checkBox[i].clicked.connect(self.check_all_boxes_select)

            self.dialog.labelEBSDcoordinate.append(QtWidgets.QLabel(self.dialog))
            self.dialog.labelEBSDcoordinate[i].setGeometry(
                QRect(150, 70 + 40 * i, 180, 23)
            )
            self.dialog.labelEBSDcoordinate[i].setObjectName(f"EBSDcoordinate{i}")
            self.dialog.labelCLSMcoordinate.append(QtWidgets.QLabel(self.dialog))
            self.dialog.labelCLSMcoordinate[i].setGeometry(
                QRect(350, 70 + 40 * i, 180, 23)
            )
            self.dialog.labelCLSMcoordinate[i].setObjectName(f"CLSMcoordinate{i}")

            self.dialog.labelEBSDcoordinate[i].setText(
                f"x ={int(self.mergedata.P[i,0])} / y= {int(self.mergedata.P[i,1])}"
            )
            self.dialog.labelCLSMcoordinate[i].setText(
                f"x ={int(self.mergedata.Pc[i,0])} / y= {int(self.mergedata.Pc[i,1])}"
            )

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
        plt.close("all")
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
        plt.close("all")
        self.dialog.destroy()
        self.mergedata.calibrate_confocal_and_EBSD_data(
            self.mergedata.P, self.mergedata.Pc
        )
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

        self.thread_merge_calc = threading.Thread(
            target=self.mergeCalcThread, daemon=True
        )
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
            print("Merging of EBSD and CLSM data was applied!")
            self.logNewLine(
                self.logfile_merge_path,
                "Transformation of CLSM coordinates to EBSD framework was executed succesfully"
                + "\n",
            )

    def mergeCalcThread(self):
        self.mergedata.calculate_superposition()
        self.mergedata.confocal_data_conc()

        print("Thread finished")

    # def area_zero_level_button(self):
    #     self.mergeSave.setStyleSheet(self.color_click_on)
    #     if len(self.mergedata.confocal_data) == 2:
    #         self.load_clsm_data_thread()
    #         self.thread.finished.connect(self.area_zero_level)
    #     else:
    #         try:
    #             self.mergedata.data_zero_level(1)
    #             print(f"Mean height of selected Area: {self.mergedata.Zerowerte_all[0]}")
    #             self.Offset00Line.setText(str(self.mergedata.Zerowerte_all[0]))
    #         except:
    #             print('Error')

    # def area_zero_level_button(self):
    #     self.mergeSave.setStyleSheet(self.color_click_on)
    #     if len(self.mergedata.confocal_data) == 2:
    #         self.load_clsm_data_thread()
    #         self.thread.finished.connect(self.area_zero_level)
    #     else:
    #         self.area_zero_level()

    # def area_zero_level(self):
    #     try:
    #         self.mergedata.data_zero_level(int(self.mergeAmountSelect.text()))
    #     except:
    #         print('Error')

    def browse_delete_pic_data(self):
        global EBSD_coordinates, CLSM_coordinates
        if self.mergeDeleteCLSMcheckBox.checkState():
            print("Delete data with CLSM coordinate system")
            self.mergedata.delteDataThroughImage(CLSM_Data=1)
        else:
            self.mergedata.delteDataThroughImage(CLSM_Data=0)
            print("Delete data with ESBD coordinate system")
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

    def logNewHead(
        self,
        filepath,
        title,
    ):
        if not os.path.isfile(filepath):
            print("Error: Logfile not initialized!")
        else:
            linewidth = 40
            logfile = open(filepath, "a")
            logfile.writelines(
                "-" * (linewidth - int(len(title) / 2))
                + " "
                + title
                + " "
                + "-" * (linewidth - int(len(title) / 2))
                + "\n\n"
            )
            logfile.close()

    def logNewLine(
        self,
        filepath,
        text,
    ):
        if not os.path.isfile(filepath):
            print("Error: Logfile not initialized!")
        else:
            logfile = open(filepath, "a")
            logfile.writelines(text + "\n")
            logfile.close()

    def logNewSubline(
        self,
        filepath,
        text,
    ):
        self.logNewLine(filepath, " - " + text)

    def initTimestamp(self):
        self.now = datetime.now()
        self.nowstr = (
            str(self.now.date())
            + "_"
            + "{:02}".format(self.now.hour)
            + "h_"
            + "{:02}".format(self.now.minute)
            + "m_"
            + "{:02}".format(self.now.second)
            + "s"
        )

    def createLogFileMerge(self):
        self.logfile_merge_path = os.path.join(
            "tmp", self.nowstr + "_logfile_merging.log"
        )
        self.mergedata.logfile_merge_path = self.logfile_merge_path
        logfile = open(self.logfile_merge_path, "w")
        # logfile.writelines('_Header_\n')
        logfile.writelines(
            "Logfile created: "
            + str(self.now.date())
            + " at "
            + str(self.now.hour)
            + ":"
            + str(self.now.minute)
            + ":"
            + str(self.now.second)
            + "\n\n"
        )
        logfile.close()

    def createLogFileEval(self):
        self.logfile_eval_path = os.path.join(
            "tmp", self.nowstr + "_logfile_evaluation.log"
        )
        self.mergedata.logfile_eval_path = self.logfile_eval_path
        logfile = open(self.logfile_eval_path, "w")
        # logfile.writelines('_Header_\n')
        logfile.writelines(
            "Logfile created: "
            + str(self.now.date())
            + " at "
            + str(self.now.hour)
            + ":"
            + str(self.now.minute)
            + ":"
            + str(self.now.second)
            + "\n\n"
        )
        logfile.close()

    # def createLogMergeSave(self):
    #     self.logfile_merge = open(os.path.join('tmp', 'logfile_merging.log'), 'w')

    #     EBSD_file = self.loadEBSDline.text()

    #     self.logNewHead(self.logfile_merge, 'Data Merging')
    #     self.logNewLine(self.logfile_merge, 'EBSD filepath:\t'+EBSD_file)
    #     self.logNewLine(self.logfile_merge, 'Used phase:\t'+self.EBSD_phase)
    #     self.logNewLine(self.logfile_merge, '')

    #     CLSM_file_1 = self.loadCLSM1line.text()
    #     CLSM_file_1_rotation = self.mergeroationCLSM1.currentText()
    #     CLSM_file_1_mirrorCheck = bool(self.CLSM1checkBox.checkState())

    #     CLSM_file_2 = self.loadCLSM2line.text()
    #     CLSM_file_2_rotation = self.mergeroationCLSM2.currentText()
    #     CLSM_file_2_mirrorCheck = bool(self.CLSM2checkBox.checkState())

    #     CLSM_file_leveling = bool(0)#self.mergeLevelingcheckBox.checkState())

    #     self.logNewHead(self.logfile_merge, 'Confocal Data')
    #     differenceMicroscopyUsed=False
    #     if ((CLSM_file_1 != '') and (CLSM_file_2 != '')):
    #         differenceMicroscopyUsed=True
    #     self.logNewLine(self.logfile_merge, 'CLSM 1 filepath:\t'+CLSM_file_1)
    #     self.logNewLine(self.logfile_merge, 'CLSM 1 used rotaion:\t'+CLSM_file_1_rotation)
    #     self.logNewLine(self.logfile_merge, 'CLSM 1 was mirrored:\t'+str(CLSM_file_1_mirrorCheck))
    #     self.logNewLine(self.logfile_merge, '')
    #     self.logNewLine(self.logfile_merge, 'Difference Microscopy applied: '+str(differenceMicroscopyUsed))
    #     if differenceMicroscopyUsed:
    #         self.logNewLine(self.logfile_merge, 'CLSM 2 filepath:\t'+CLSM_file_2)
    #         self.logNewLine(self.logfile_merge, 'CLSM 2 used rotaion:\t'+CLSM_file_2_rotation)
    #         self.logNewLine(self.logfile_merge, 'CLSM 2 was mirrored:\t'+str(CLSM_file_2_mirrorCheck))
    #     self.logNewLine(self.logfile_merge, '')
    #     self.logNewLine(self.logfile_merge, 'Leveling was applied:\t'+str(CLSM_file_leveling))
    #     self.logNewLine(self.logfile_merge, '')

    #     self.logNewHead(self.logfile_merge, 'Merging Options')

    #     if self.EBSD_CLSM_manual_selection and self.EBSD_CLSM_read_in:
    #         self.logNewLine(self.logfile_merge, 'Points partially manually selected and read in from file')
    #         self.logNewLine(self.logfile_merge, f'Read from path:\t{self.loading_pointsFileName}')
    #     elif self.EBSD_CLSM_manual_selection:
    #         self.logNewLine(self.logfile_merge, 'Points manually selected')
    #     elif self.EBSD_CLSM_read_in:
    #         self.logNewLine(self.logfile_merge, 'Points read in from file')
    #         self.logNewLine(self.logfile_merge, f'Read from path:\t{self.loading_pointsFileName}')
    #     self.logNewLine(self.logfile_merge, f'Final selection written to path:\t{self.mergedata.save_selected_points_merge}')

    #     self.logNewLine(self.logfile_merge, '')
    #     self.logNewLine(self.logfile_merge, f'Merged savepath:\t{self.merge_save_path}.dat')

    #     self.logfile_merge.close()

    # %%
    def tabEvaluateCLSM(self):

        self.simLoadMDpushButton.clicked.connect(self.browse_sim_load_MD)
        self.simLoadBCApushButton.clicked.connect(self.browse_sim_load_BCA)
        self.simHKLplotpushButton.clicked.connect(self.browsePlotHKLsim)
        self.simErosionPlotButton.clicked.connect(self.browseErosionDepthPlotsim)

        self.evaluateLoadDataButton.clicked.connect(self.browse_evaluate_load_data)
        self.evaluatePlotEBSDbutton.clicked.connect(self.browse_plot_EBSD_Data)
        self.evaluateClosePlotsButton.clicked.connect(lambda: plt.close("all"))
        self.evaluatePlotCLSMbutton.clicked.connect(self.browsePlotDataHeight)
        self.evaluateRefereceLevelLineEdit.textChanged.connect(
            self.browse_calculate_mean_erosion
        )
        self.evaluateLevelDataButton.clicked.connect(self.leveling_data)
        self.evaluateLevelingPushButton.clicked.connect(self.levelingOxidData)
        self.evaluateApplyFilterButton.clicked.connect(self.browseLevelingDataFilters)
        self.evaluateCalculateIPFbutton.clicked.connect(self.browseCalculateIPF)
        self.evaluateSaveDataButton.clicked.connect(self.evaluateSaveData)
        self.evaluateSYplotButton.clicked.connect(self.browse_sputter_yield_plot)
        self.evaluateErosionPlotButton.clicked.connect(self.browseErosionDepthPlot)
        self.evaluateSDplotButton.clicked.connect(self.browseSDplot)
        self.evalualteChangeTabOpti.clicked.connect(
            lambda: self.evaluatSubtabWidget.setCurrentIndex(2)
        )
        self.evaluateCountplotButton.clicked.connect(self.browseCountsplot)
        self.evaluateHKLplotpushButton.clicked.connect(self.browsePlotHKL)

        self.optiResolutionIPFLineEdit.textChanged.connect(
            lambda: self.optiResolutionIPFLineEdit_2.setText(
                self.optiResolutionIPFLineEdit.text()
            )
        )
        self.optiResolutionIPFLineEdit_2.textChanged.connect(
            lambda: self.optiResolutionIPFLineEdit.setText(
                self.optiResolutionIPFLineEdit_2.text()
            )
        )
        self.optCalculateButton.clicked.connect(self.browse_optimisation_calculate)
        self.optLoadDataButton.clicked.connect(self.browseOptLoadDataButton)
        self.optReduceDataPushButton.clicked.connect(self.browseBinningData)
        self.optSelectDataPointspushButton.clicked.connect(self.optSelctPoint)
        self.evaluateMergeDataButton.clicked.connect(self.browseMergeLeveledData)
        self.evaluateRotationMatrixCalcbutton.clicked.connect(
            self.browseDataRotationMatrix
        )

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
            print("Data load failed! -------------- ??Column wrong??")

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
            print("Data load failed")

    def browseErosionDepthPlotsim(self):

        if len(self.evaluateSim.heightIPF) == 1:
            print("No data")
        else:
            b = self.simErosionTicksLineEdit.text()
            c = c = b.split(",")
            f = np.asarray(c, dtype=float)
            if self.simErosionCheckBox.checkState() != 0:
                contourplot1 = 1
            else:
                contourplot1 = 0
            if self.simMultiplyCheckBox.checkState() != 0:
                erosion = np.multiply(self.evaluateSim.heightIPF, -1)
            else:
                erosion = self.evaluateSim.heightIPF

            erosion = np.multiply(erosion, float(self.simMultiplyLineEdit.text()))
            saveipf = self.simDataPath.text()[:-4] + "_IPF.png"
            self.evaluateSim.plot_inter_IPFz(
                self.evaluateSim.xIPFcoordinate,
                self.evaluateSim.yIPFcoordinate,
                erosion,
                float(2),
                cbarrange=[
                    float(self.simErosionMinLineEdit.text()),
                    float(self.simErosionMaxLineEdit.text()),
                ],
                Title="IPFz",
                cbarlabel=self.simErosionLabelLineEdit.text(),
                ticks=f,
                contourplot=contourplot1,
                savepath=saveipf,
            )

    def browsePlotHKLsim(self):
        try:
            h = float(self.simHlineEdit.text())
            k = float(self.simKlineEdit.text())
            l = float(self.simLlineEdit.text())
            ipf = [h, k, l]
            self.evaluateSim.plot_IPF_plane(
                IPF=ipf,
                testsize=20,
                text=self.simPlotHKLtextlineEdit.text(),
                yTextoffset=float(self.simPlotYoffsetLtextlineEdit.text()) / 100,
            )
        except:
            print("No figure avalible")

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
            self.evaluateMeanEroisionlineEdit.setText(
                str(np.round(np.mean(self.evaluate.mergeDataSet[:, 2]), 4))
            )
            mean = self.evaluateMeanEroisionlineEdit.text()
            self.logNewSubline(
                self.logfile_eval_path, f"Calculated mean height of data: {mean}\n"
            )

            if self.evaluate.mergeDataSet.shape[1] == 11:
                self.thread_evaluate_loadData_IPF = threading.Thread(
                    target=self.evaluate.IPFYandIPFX, daemon=True
                )
                self.thread_evaluate_loadData_IPF.start()

                QTimer.singleShot(500, self.evaluate_load_data_IPF_update)
            else:

                self.evaluateApplyFilterButton.setEnabled(True)
                self.evaluateProgressBar.setMaximum(100)

    def browse_evaluate_load_data(self):
        file_name = self.browse_button_master(
            "EBSD CTF File",
            "Merged file .dat (*.dat)",
        )

        self.evaluateLoadDataButtonLineEdit.setText(file_name)
        self.evaluate.dataPathFile = file_name

        if file_name:
            self.logNewHead(self.logfile_eval_path, "New Evaluation")
            _, relativePath = os.path.split(file_name)
            self.logNewLine(self.logfile_eval_path, "Load merge data: " + relativePath)
            self.logNewSubline(
                self.logfile_eval_path, "Full mergefile filepath: " + file_name
            )
            reduction_factor = int(self.spinBoxMergedReduce.cleanText())
            self.logNewSubline(
                self.logfile_eval_path,
                f"Compression factor applied: {reduction_factor}^2 = {reduction_factor**2}",
            )

            self.evaluateProgressBar.setMaximum(0)
            self.evaluateLoadDataButton.setEnabled(False)
            self.evaluateLevelDataButton.setEnabled(False)
            self.evaluateApplyFilterButton.setEnabled(False)

            self.thread_evaluate_loadData = threading.Thread(
                target=self.evaluate_load_data, daemon=True
            )
            self.thread_evaluate_loadData.start()

            QTimer.singleShot(500, self.evaluate_load_data_update)

    def evaluate_load_data(self):
        self.evaluate.mergeDataSet = np.loadtxt(self.evaluate.dataPathFile)
        # k = self.evaluate_load_reduction_factor
        k = int(self.spinBoxMergedReduce.cleanText())
        # k = 12
        if k != 1:
            mask = np.array(
                [i % k == 0 for i in range(len(self.evaluate.mergeDataSet))]
            )
            self.evaluate.mergeDataSet = self.evaluate.mergeDataSet[mask]
        # print(self.evaluate.mergeDataSet[mask].shape)
        # print(mask.shape)

    def browse_plot_EBSD_Data(self):
        try:
            self.evaluate.plotEBSD_Data()
        except:
            print("No data loaded")

    def browsePlotDataHeight(self):
        try:
            self.evaluate.plotDataHeight()
        except:
            print("No data loaded")

    def browse_calculate_mean_erosion(self):
        global reference_level
        try:
            self.evaluate.refereceHeight = np.round(
                np.mean(self.evaluate.mergeDataSet[:, 2])
                - float(self.evaluateRefereceLevelLineEdit.text()),
                4,
            )
            self.evaluateMeanEroisionlineEdit.setText(
                str(np.round(np.mean(self.evaluate.mergeDataSet[:, 2]), 4))
            )
            self.lineEditAfterLeveling.setText(str(self.evaluate.refereceHeight))
        except:
            pass

    def leveling_data(self):
        if type(self.evaluate.mergeDataSet) == int:
            print("No data")
        else:
            self.logNewLine(
                self.logfile_eval_path, "Sputter/Ersoion leveling was applied"
            )
            self.logNewSubline(
                self.logfile_eval_path,
                f"Mean height before leveling: {self.evaluateMeanEroisionlineEdit.text()}",
            )
            self.logNewSubline(
                self.logfile_eval_path,
                f"Reference height given: {self.evaluateRefereceLevelLineEdit.text()}",
            )
            self.logNewSubline(
                self.logfile_eval_path,
                f"Mean height after leveling: {self.lineEditAfterLeveling.text()}\n",
            )

            self.evaluateCalculateIPFbutton.setStyleSheet(self.color_click_on)
            self.evaluateSaveDataButton.setStyleSheet(self.color_click_on)
            self.evaluate.relativeHeighttoAbsHeight()
            self.evaluateMeanEroisionlineEdit.setText(
                str(np.round(np.mean(self.evaluate.mergeDataSet[:, 2]), 4))
            )
            self.lineEditAfterLeveling.setText(str(0))
            self.evaluateRefereceLevelLineEdit.setText(str(0))

    def levelingOxidData(self):
        if type(self.evaluate.mergeDataSet) == int:
            print("No data")
        else:
            self.logNewLine(self.logfile_eval_path, "Oxidation leveling was applied")
            self.logNewSubline(
                self.logfile_eval_path,
                f"Height of oxide layer: {self.evaluateOxideHeightLineEdit.text()} micrometers",
            )
            self.logNewSubline(
                self.logfile_eval_path,
                f"Compensate grow in depth: {self.evaluateDepthGrowlineEdit.text()}",
            )
            if self.evaluate100OxideHeightRadioButton.isChecked():
                option = "<100> grains"
            elif self.evaluateMeanOxideHeightRadioButton.isChecked():
                option = "mean oxide heights"
            self.logNewSubline(self.logfile_eval_path, f"Chosen option: {option}\n")

            self.evaluateLevelingPushButton.setEnabled(False)
            self.evaluateCalculateIPFbutton.setStyleSheet(self.color_click_on)
            self.evaluateSaveDataButton.setStyleSheet(self.color_click_on)
            self.evaluate.DepthgrowFactor = float(self.evaluateDepthGrowlineEdit.text())
            self.evaluate.ABSHeightOxide = float(
                self.evaluateOxideHeightLineEdit.text()
            )

            if self.evaluate100OxideHeightRadioButton.isChecked():
                self.evaluate.relativeHeighttoAbsOxidTungstenHeight100()
            elif self.evaluateMeanOxideHeightRadioButton.isChecked():
                self.evaluate.relativeHeighttoAbsOxidTungstenHeightmean()

            self.evaluateMeanEroisionlineEdit.setText(
                str(np.round(np.mean(self.evaluate.mergeDataSet[:, 2]), 4))
            )
            print(
                "Leveling button is disabled. Only one time the depth growth should be multiply to the data"
            )

    def threadlevelingDataFiltersUpdate(self):
        if self.thread_levelingDataFilters.is_alive():
            self.evaluateProgressBar2.setValue(self.evaluate.progressUpdate2)
            QTimer.singleShot(500, self.threadlevelingDataFiltersUpdate)
        else:
            self.evaluateApplyFilterButton.setEnabled(True)
            self.evaluateProgressBar.setMaximum(100)
            self.evaluateProgressBar2.setValue(0)

    def browseLevelingDataFilters(self):
        if type(self.evaluate.mergeDataSet) == int:
            print("No data")
        else:
            self.logNewLine(self.logfile_eval_path, "Filter was applied")
            if self.evaluateNoiseFilterCheckBox.isChecked():
                self.logNewSubline(
                    self.logfile_eval_path,
                    f"Noise filter size: {self.evaluateNoiseFilterSize.text()}",
                )
            if self.evaluateNoiseFilterCheckBox.isChecked():
                self.logNewSubline(
                    self.logfile_eval_path,
                    f"Noise filter size: {self.evaluateNoiseFilterSize.text()}",
                )
            self.evaluateApplyFilterButton.setEnabled(False)
            self.evaluateProgressBar.setMaximum(0)

            self.thread_levelingDataFilters = threading.Thread(
                target=self.levelingDataFilters, daemon=True
            )
            self.thread_levelingDataFilters.start()

            QTimer.singleShot(500, self.threadlevelingDataFiltersUpdate)

    def levelingDataFilters(self):
        if type(self.evaluate.mergeDataSet) == int:
            print("No data")
        else:
            self.evaluate.mergeDataSetcopy = self.evaluate.mergeDataSet.copy()
            if self.evaluateNoiseFilterCheckBox.checkState() != 0:
                self.evaluateSaveDataButton.setStyleSheet(self.color_click_on)
                print("---------------- Noise filter ----------------")
                self.evaluate.grainboundaryOrNois = 0
                self.evaluate.pixelCluster = float(self.evaluateNoiseFilterSize.text())
                self.evaluate.grainBounderies()
            if self.evaluateGBfilterCheckBox.checkState() != 0:
                self.evaluateSaveDataButton.setStyleSheet(self.color_click_on)
                print("---------------- Grain boundary filter ----------------")
                self.evaluate.grainboundaryOrNois = 1
                self.evaluate.pixelCluster = float(self.evaluateGBfilterSize.text())
                self.evaluate.grainBounderies()

            if len(self.evaluate.mergeDataSetcopy[:, 0]) < 30:
                print(
                    "----------------------------Warning----------------------------",
                    "Probably to small grains or filter size to big! Filter is not applied! ",
                )
            else:
                print("Copy Data")
                self.evaluate.mergeDataSet = self.evaluate.mergeDataSetcopy.copy()
                self.evaluateCalculateIPFbutton.setStyleSheet(self.color_click_on)
                self.evaluateSaveDataButton.setStyleSheet(self.color_click_on)

    def browseMergeLeveledData(self):
        file_name = self.browse_button_master(
            "EBSD CTF File",
            "Merged file .dat (*.dat)",
        )

        self.evaluate.dataPathFiles = file_name
        if file_name:
            self.evaluateProgressBar.setMaximum(0)
            self.evaluateLoadDataButton.setEnabled(False)
            self.evaluateLevelDataButton.setEnabled(False)
            self.evaluateLevelingPushButton.setEnabled(False)
            self.evaluateApplyFilterButton.setEnabled(False)

            self.mergedLeveldDataThread = threading.Thread(
                target=self.browse_merged_leveld_data, daemon=True
            )
            self.mergedLeveldDataThread.start()

            QTimer.singleShot(500, self.mergedLeveldDataThreadUpdate)

    def browse_merged_leveld_data(self):
        try:
            self.evaluate.browse_merged_leveld_data()
        except:
            print("Error")
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
                self.evaluateMeanEroisionlineEdit.setText(
                    str(np.round(np.mean(self.evaluate.mergeDataSet[:, 2]), 4))
                )
                self.evaluateSaveDataButton.setStyleSheet(self.color_click_on)
                self.evaluateCalculateIPFbutton.setStyleSheet(self.color_click_on)
                self.evaluateLoadDataButton.setStyleSheet(self.green)

                if self.evaluate.mergeDataSet.shape[1] == 11:
                    self.thread_evaluate_loadData_IPF = threading.Thread(
                        target=self.evaluate.IPFYandIPFX, daemon=True
                    )
                    self.thread_evaluate_loadData_IPF.start()

                    QTimer.singleShot(500, self.evaluate_load_data_IPF_update)
                else:

                    self.evaluateApplyFilterButton.setEnabled(True)
            except:
                print("Loding Error")

    def browseCalculateIPF(self):
        try:
            self.evaluate.resolution = float(self.evaluateIPFresolutionLineEdit.text())
            self.evaluate.smoothing = self.evaluateSmoothingCheckBox.checkState()
            self.evaluate.cutOffFilter = float(
                self.evaluateCuttOffFilterLineEdit.text()
            )

            self.evaluateCalculateIPFbutton.setEnabled(False)
            self.evaluateProgressBar.setMaximum(0)

            self.threadCalculateIPF = threading.Thread(
                target=self.evaluate.medianDataXYZ, daemon=True
            )
            self.threadCalculateIPF.start()

            QTimer.singleShot(500, self.threadCalculateIPFUpdate)
        except:
            print("Error")
            self.evaluateCalculateIPFbutton.setEnabled(True)

    def threadCalculateIPFUpdate(self):
        if self.threadCalculateIPF.is_alive():
            QTimer.singleShot(500, self.threadCalculateIPFUpdate)
        else:
            self.evaluateCalculateIPFbutton.setEnabled(True)
            self.evaluateProgressBar.setMaximum(100)
            if len(self.evaluate.heightIPF) == 1:
                print("No data")
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

        self.threadDataRotationMatrix = threading.Thread(
            target=self.dataRotationMatrix, daemon=True
        )
        self.threadDataRotationMatrix.start()

        QTimer.singleShot(500, self.dataRotationMatrixUpdate)

    def dataRotationMatrix(self):
        if type(self.evaluate.mergeDataSet) == int:
            print("No data")
        else:
            if (
                self.evaluateRotAngleLineEdit.text() == ""
                or self.evaluateTiltAngleLineEdit.text() == ""
            ):
                print("Enter rotation angles")
            else:
                try:
                    rotationAngle = float(self.evaluateRotAngleLineEdit.text())
                    tiltAngel = float(self.evaluateTiltAngleLineEdit.text())
                    self.evaluate.mergeDataSet = rotation_ebsd(
                        self.evaluate.mergeDataSet, rotationAngle, tiltAngel, 0
                    )
                except:
                    print("Only numbers. No Strings!")

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
            print("No data")
        else:
            b = self.evaluateSYticksLineEdit.text()
            c = c = b.split(",")
            f = np.asarray(c, dtype=float)

            if self.evaluateSYcheckBox.checkState() != 0:
                contourplot1 = 1
            else:
                contourplot1 = 0

            saveipf = (
                self.evaluateLoadDataButtonLineEdit.text()[:-4]
                + "_IPF_sputter_yield.png"
            )
            self.evaluate.atomicDensity = float(
                self.evaluateSYatomicDensityLineEdit.text()
            )
            self.evaluate.fluence = float(self.evaluateSYfluenceLineEdit.text())
            self.evaluate.heighttoSputterYield()
            self.evaluate.plot_inter_IPFz(
                self.evaluate.xIPFcoordinate,
                self.evaluate.yIPFcoordinate,
                self.evaluate.sputterYield,
                float(self.evaluateIPFresolutionLineEdit.text()),
                cbarrange=[
                    float(self.evaluateSYminLineEdit.text()),
                    float(self.evaluateSYmaxLineEdit.text()),
                ],
                Title="IPFz",
                cbarlabel=self.evaluateSYlabelLineEdit.text(),
                ticks=f,
                contourplot=contourplot1,
                savepath=saveipf,
            )

    def browseErosionDepthPlot(self):

        if len(self.evaluate.heightIPF) == 1:
            print("No data")
        else:
            b = self.evaluateErosionTicksLineEdit.text()
            c = c = b.split(",")
            f = np.asarray(c, dtype=float)
            if self.evaluateErosioncheckBox.checkState() != 0:
                contourplot1 = 1
            else:
                contourplot1 = 0
            if self.evaluateMultiplycheckBox.checkState() != 0:
                erosion = np.multiply(self.evaluate.heightIPF, -1)
            else:
                erosion = self.evaluate.heightIPF
            saveipf = (
                self.evaluateLoadDataButtonLineEdit.text()[:-4] + "_IPF_erosion.png"
            )
            self.evaluate.plot_inter_IPFz(
                self.evaluate.xIPFcoordinate,
                self.evaluate.yIPFcoordinate,
                erosion,
                float(self.evaluateIPFresolutionLineEdit.text()),
                cbarrange=[
                    float(self.evaluateErosionMinLineEdit.text()),
                    float(self.evaluateErosionMaxLineEdit.text()),
                ],
                Title="IPFz",
                cbarlabel=self.evaluateErosionLabelLineEdit.text(),
                ticks=f,
                contourplot=contourplot1,
                savepath=saveipf,
            )

    def browseSDplot(self):

        if len(self.evaluate.heightIPF) == 1:
            print("No data")
        else:
            b = self.evaluateSDticksLineEdit.text()
            c = c = b.split(",")
            f = np.asarray(c, dtype=float)

            saveipf = self.evaluateLoadDataButtonLineEdit.text()[:-4] + "_IPF_SD.png"
            self.evaluate.plot_inter_IPFz(
                self.evaluate.xIPFcoordinate,
                self.evaluate.yIPFcoordinate,
                self.evaluate.heightSD,
                float(self.evaluateIPFresolutionLineEdit.text()),
                cbarrange=[
                    float(self.evaluateSDminLineEdit.text()),
                    float(self.evaluateSDmaxLineEdit.text()),
                ],
                Title="IPFz",
                cbarlabel=self.evaluateSDlabelLineEdit.text(),
                ticks=f,
                contourplot=0,
                savepath=saveipf,
            )

    def browseCountsplot(self):
        if len(self.evaluate.heightIPF) == 1:
            print("No data")
        else:
            saveipf = (
                self.evaluateLoadDataButtonLineEdit.text()[:-4] + "_IPF_Counts.png"
            )
            self.evaluate.plot_inter_IPFz(
                self.evaluate.xIPFcoordinate,
                self.evaluate.yIPFcoordinate,
                np.log10(self.evaluate.heightCounts),
                float(self.evaluateIPFresolutionLineEdit.text()),
                cbarrange=[
                    np.log10(float(self.evaluateCountMinLineEdit.text())),
                    np.log10(float(self.evaluateCountMaxLineEdit.text())),
                ],
                Title="IPFz",
                cbarlabel=self.evaluateCountLabelLineEdit.text(),
                ticks=(0, 1, 2, 3, 4, 5, 6),
                log=1,
                contourplot=0,
                savepath=saveipf,
            )

    def browsePlotHKL(self):
        try:
            h = float(self.evaluateHlineEdit.text())
            k = float(self.evaluateKlineEdit.text())
            l = float(self.evaluateLlineEdit.text())
            ipf = [h, k, l]
            self.evaluate.plot_IPF_plane(
                IPF=ipf,
                testsize=20,
                text=self.evaluatePlotHKLtextlineEdit.text(),
                yTextoffset=float(self.simPlotYoffsetLtextlineEdit_2.text()) / 100,
            )
        except:
            print("Error")

    def evaluateSaveData(self):
        if type(self.evaluate.mergeDataSet) == int:
            print("No data")
        else:
            file_name = self.browse_button_master(
                "Save .dat file",
                "File *.dat (*.dat)",
                save=True,
            )

            print(file_name)
            if file_name[-4:] == ".dat":
                file_name = file_name[:-4]
            if file_name:
                np.savetxt(
                    f"{file_name}.dat",
                    self.evaluate.mergeDataSet,
                    fmt="%.5e",
                    header=" X Y Contrast PHI PSI2 PSI1 H K L IPFz_X_Coordinate IPFz_Y_Coordinate IPFx_X_Coordinate IPFx_Y_Coordinate IPFy_X_Coordinate IPFy_Y_Coordinate",
                )
                self.evaluateLoadDataButtonLineEdit.setText(f"{file_name}.dat")
                self.evaluateSaveDataButton.setStyleSheet(self.green)
            self.evaluateProgressBar.setMaximum(100)

    def browseBinningData(self):
        if type(self.evaluate.mergeDataSet) == int:
            print("No data")
        else:
            self.evaluate.pixelClusterBinning = self.optBinningSizeLineEdit.text()
            self.evaluate.grainBounderies_Reduce_Data()
            self.evaluate.plotEBSDDataBinning()

    def browse_optimisation_calculate(self):
        if type(self.evaluate.mergeDataSet) == int:
            print("No data")
        else:

            print("calculate")
            self.optCalculateButton.setEnabled(False)

            rotation_angle_list = []
            tilt_angle_list = []
            angle_list = []
            self.result_optimisation = []
            self.progressbarmax = 0

            # Reads in data set for comparison, if applicable, else comparison to trivial values
            if (
                self.optCompareDataCheckBox.checkState() != 0
                and self.optLoadDataButtonLineEdit.text() != ""
            ):
                file_name = self.optLoadDataButtonLineEdit.text()
                print("Load Compare data")
                dataToCompare = np.loadtxt(file_name)
                print("Calculate IPF")
                resultDataCompare = meanDataXYZ(
                    dataToCompare,
                    float(self.optiResolutionIPFLineEdit.text()),
                    data_less=3,
                )
                print("Calculate IPF finished")

                Xcompare = resultDataCompare[0]
                Ycompare = resultDataCompare[1]
                Zcompare = resultDataCompare[2]
            else:
                Xcompare = [0, 1]
                Ycompare = [0, 1]
                Zcompare = [0, 1]

            if int(self.optReduceDataCheckBox.checkState()) != 0:
                data_resulttmp = self.evaluate.mergeDataSetBinning
                print("Use reduced data set")
            else:
                data_resulttmp = self.evaluate.mergeDataSet

            gc.disable()

            for tilt_angle in np.arange(
                float(self.optiTiltAngleMinLineEdit.text()),
                float(self.optiTiltAngleMaxLineEdit.text())
                + float(self.optiStepSizeLineEdit.text()),
                float(self.optiStepSizeLineEdit.text()),
            ):
                if tilt_angle == 0:
                    stepsize_rotation = 360
                else:
                    stepsize_rotation = (
                        90 * float(self.optiStepSizeLineEdit.text()) / tilt_angle
                    )
                for rotation_angle in np.arange(
                    float(self.optiRotAngleMinLineEdit.text()),
                    float(self.optiRotAngleMaxLineEdit.text()),
                    stepsize_rotation,
                ):
                    self.progressbarmax += 1
            # print(f'Progress bar index = {self.progressbarmax}')

            print(f"Start pool with {self.progressbarmax} processes")
            for tilt_angle in np.arange(
                float(self.optiTiltAngleMinLineEdit.text()),
                float(self.optiTiltAngleMaxLineEdit.text())
                + float(self.optiStepSizeLineEdit.text()),
                float(self.optiStepSizeLineEdit.text()),
            ):
                # print(f'Calculation for tilt angle: {tilt_angle}')
                if tilt_angle == 0:
                    stepsize_rotation = 360
                else:
                    stepsize_rotation = (
                        90 * float(self.optiStepSizeLineEdit.text()) / tilt_angle
                    )

                for rotation_angle in np.arange(
                    float(self.optiRotAngleMinLineEdit.text()),
                    float(self.optiRotAngleMaxLineEdit.text()),
                    stepsize_rotation,
                ):
                    # print(f'Calculation for rotation angle: {rotation_angle}')
                    rotation_angle_list.extend([rotation_angle])
                    tilt_angle_list.extend([tilt_angle])
                    angle_list.append([rotation_angle, tilt_angle])
                    self.func2 = partial(
                        optimization_calculate,
                        data_resulttmp,
                        Xcompare,
                        Ycompare,
                        Zcompare,
                        float(self.optiResolutionIPFLineEdit.text()),
                        int(self.optUseDataEditLine.text()),
                        self.evaluate.opt_points1_xy,
                        self.optSelectedDataCheckBox.checkState(),
                    )
                    print(f"Controll message: Output of self.func2: {self.func2}")

                    # result_optimisation_current = self.func2([rotation_angle,tilt_angle])
                    if self.result_optimisation == None:
                        self.result_optimisation = [
                            self.func2([rotation_angle, tilt_angle])
                        ]
                    else:
                        self.pool.apply_async(
                            func=self.func2,
                            args=([rotation_angle, tilt_angle],),
                            callback=self.result_optimisation.append,
                        )

                    # print(f'Status result optimization: {self.result_optimisation}')

            QTimer.singleShot(2000, self.optimisation_result)
            # print('Finished Calculation!')

    def optimisation_result(self):

        if len(self.result_optimisation) < self.progressbarmax:
            self.optimizeProgressBar.setValue(
                int(len(self.result_optimisation) / self.progressbarmax * 100) + 1
            )
            print(
                f"Still working! Progress: {int(len(self.result_optimisation)/self.progressbarmax *100)}"
            )
            QTimer.singleShot(2000, self.optimisation_result)
        else:
            Angle1 = []
            Angle2 = []
            Zvariation = []
            ErrorZ = []
            DiffVariation = []
            FitParameter = []
            FitParameterError = []
            print("Multiprocessing finished")
            if self.evaluateLoadDataButtonLineEdit.text() == "":
                path1 = os.getcwd()
            else:
                path1 = os.path.split(self.evaluateLoadDataButtonLineEdit.text())[0]
            file_name = os.path.split(self.evaluateLoadDataButtonLineEdit.text())[1][
                0:-4
            ]
            if file_name == "":
                file_name = "optimise_No_File_Name"
            path2 = path1 + "/" + file_name
            if os.path.exists(path2):
                print("Path result already exist")
            else:
                os.mkdir(path2)

            plot_number = 1
            statetmp = 0
            for result_index in range(len(self.result_optimisation)):

                save_file_as = f"{plot_number}__rotation_{np.round(self.result_optimisation[result_index][5],2)}_deg__tilt_{np.round(self.result_optimisation[result_index][4],2)}_deg.png"
                if plot_number < 10:
                    save_file_as = "000" + save_file_as
                elif plot_number < 100:
                    save_file_as = "00" + save_file_as
                elif plot_number < 1000:
                    save_file_as = "0" + save_file_as

                ErrorZ.extend([np.mean(self.result_optimisation[result_index][3])])
                Zvariation.extend([np.std(self.result_optimisation[result_index][2])])
                Angle1.extend(np.round([self.result_optimisation[result_index][5]], 1))
                Angle2.extend(np.round([self.result_optimisation[result_index][4]], 1))
                DiffVariation.extend([self.result_optimisation[result_index][7]])
                FitParameter.extend([self.result_optimisation[result_index][8]])
                FitParameterError.extend([self.result_optimisation[result_index][9]])
                if self.result_optimisation[result_index][6] == 1 and statetmp == 0:
                    print("")
                    print(
                        "----------------------------------------------------------------------------------------------------------------------------------------"
                    )
                    print(
                        "Attention: Different amount of data is used during the optimisation!!!"
                    )
                    print(
                        "Information: Cut off filter is set to 3, because at least 3 data values are needed for a SD."
                    )
                    print(
                        "For different rotation and tilte angle new cluster arangments are build and the cut off filter is more or less used"
                    )
                    print("")
                    print(
                        'For having the same amount of data for the "PCA" or "Min error result": Please reduce the value "Used data in %"'
                    )
                    print("")
                    print(
                        'Hint: By using "select data points" reduces the amount of data by the IPF selection. Please set used data to e.g. 20%'
                    )
                    print(
                        "----------------------------------------------------------------------------------------------------------------------------------------"
                    )
                    print("")
                    statetmp = 1

                plot_number += 1

            Angle11 = np.asarray(Angle1)
            Angle21 = np.asarray(Angle2)
            ErrorZ1 = np.asarray(ErrorZ)
            Zvariation1 = np.asarray(Zvariation)
            IPFdiffVariation = np.asarray(DiffVariation)
            FitParameter2 = np.asarray(FitParameter)
            FitParameterError2 = np.asarray(FitParameterError)
            self.evaluate.data_optimize = np.concatenate(
                (
                    Angle11.reshape((len(Angle11), 1)),
                    Angle21.reshape((len(Angle21), 1)),
                    ErrorZ1.reshape((len(Angle21), 1)),
                    Zvariation1.reshape((len(Angle21), 1)),
                    IPFdiffVariation.reshape((len(Angle21), 1)),
                    FitParameter2.reshape((len(Angle21), 1)),
                    FitParameterError2.reshape((len(Angle21), 1)),
                ),
                axis=1,
            )
            np.savetxt(
                f"{path2}/{file_name}_Angle1_Angle2_ErrorZ_PCA.txt",
                self.evaluate.data_optimize,
            )
            gc.enable()

            a = np.where(
                self.evaluate.data_optimize[:, 2]
                == np.min(self.evaluate.data_optimize[:, 2])
            )
            b = np.where(
                self.evaluate.data_optimize[:, 3]
                == np.max(self.evaluate.data_optimize[:, 3])
            )

            self.optMinRotLineEdit.setText(str(self.evaluate.data_optimize[a[0][0], 0]))
            self.optMinTiltLineEdit.setText(
                str(self.evaluate.data_optimize[a[0][0], 1])
            )
            self.optPCArotLineEdit.setText(str(self.evaluate.data_optimize[b[0][0], 0]))
            self.optPCAtiltLineEdit.setText(
                str(self.evaluate.data_optimize[b[0][0], 1])
            )
            if (
                self.optCompareDataCheckBox.checkState() != 0
                and self.optLoadDataButtonLineEdit.text() != ""
            ):
                c = np.where(
                    self.evaluate.data_optimize[:, 4]
                    == np.min(self.evaluate.data_optimize[:, 4])
                )
                self.optComRotLineEdit.setText(
                    str(self.evaluate.data_optimize[c[0][0], 0])
                )
                self.optComTiltLineEdit.setText(
                    str(self.evaluate.data_optimize[c[0][0], 1])
                )

            self.evaluateRotAngleLineEdit.setText(self.optPCArotLineEdit.text())
            self.evaluateTiltAngleLineEdit.setText(self.optPCAtiltLineEdit.text())
            self.optimizeProgressBar.setValue(0)
            self.optCalculateButton.setEnabled(True)
            self.browse_optimisation_calculate_plot(path2, file_name)

    # %%
    def browse_optimisation_calculate_plot(
        self,
        path2,
        file_name,
    ):
        self.plot_Angle1Angel2(
            self.evaluate.data_optimize[:, 0],
            self.evaluate.data_optimize[:, 1],
            self.evaluate.data_optimize[:, 3],
            label="SD_PCA",
            name=f"{path2}/{file_name}_PCA",
        )
        self.plot_Angle1Angel2(
            self.evaluate.data_optimize[:, 0],
            self.evaluate.data_optimize[:, 1],
            self.evaluate.data_optimize[:, 2],
            label="mean(SD)",
            name=f"{path2}/{file_name}_meanError",
        )
        if (
            self.optCompareDataCheckBox.checkState() != 0
            and self.optLoadDataButtonLineEdit.text() != ""
        ):
            self.plot_Angle1Angel2(
                self.evaluate.data_optimize[:, 0],
                self.evaluate.data_optimize[:, 1],
                self.evaluate.data_optimize[:, 4],
                label="Difference IPF variation",
                name=f"{path2}/{file_name}_DifferenceIPFvariation",
            )
            self.plot_Angle1Angel2(
                self.evaluate.data_optimize[:, 0],
                self.evaluate.data_optimize[:, 1],
                self.evaluate.data_optimize[:, 5],
                label="Difference Fit Parameter",
                name=f"{path2}/{file_name}_DifferenceFitParameter",
            )
            self.plot_Angle1Angel2(
                self.evaluate.data_optimize[:, 0],
                self.evaluate.data_optimize[:, 1],
                self.evaluate.data_optimize[:, 6],
                label="SD of the fit Parameter",
                name=f"{path2}/{file_name}_DifferenceErrorOfTheFitParameter",
            )

    def plot_Angle1Angel2(
        self,
        Angle1,
        Angle2,
        Z,
        label="std(Z)",
        name="name",
        interpolation=1,
    ):

        fig1 = plt.figure(figsize=(12, 8))
        ax1 = fig1.add_subplot(111, xlabel="x", ylabel="y")
        im1 = ax1.scatter(
            np.asarray(Angle1),
            np.asarray(Angle2),
            c=np.asarray(Z),
            s=300,
            marker="s",
            cmap=plt.get_cmap("jet"),
        )
        cbar = fig1.colorbar(im1)

        cbar.set_label(f"{label} / µm", size=12, labelpad=14)

        ax1.set_xlabel("$φ$ / $ \degree$", fontsize=14)
        ax1.set_ylabel("$ϕ$ / $ \degree$", fontsize=14)
        for item in (
            [ax1.title, ax1.xaxis.label, ax1.yaxis.label]
            + ax1.get_xticklabels()
            + ax1.get_yticklabels()
        ):
            item.set_fontsize(12)

        for l in cbar.ax.yaxis.get_ticklabels():
            l.set_size(12)

        fig1.subplots_adjust(bottom=0.16)
        fig1.savefig(f"{name}.png")
        plt.close(fig1)
        xpolar = np.asarray(Angle2) * np.cos(np.asarray(Angle1) / 180 * np.pi)
        ypolar = np.asarray(Angle2) * np.sin(np.asarray(Angle1) / 180 * np.pi)

        fig2 = plt.figure(figsize=(14, 8))
        ax2 = fig2.add_subplot(111, xlabel="x", ylabel="y")
        if interpolation == 1:
            xi, yi = np.mgrid[
                -1 * max(Angle2) : max(Angle2) : 100j,
                -1 * max(Angle2) : max(Angle2) : 100j,
            ]
            zi = griddata((xpolar, ypolar), np.asarray(Z), (xi, yi))
            im2 = ax2.scatter(xi, yi, c=zi, s=100, marker="s", cmap=plt.get_cmap("jet"))
        else:
            im2 = ax2.scatter(
                xpolar,
                ypolar,
                c=np.asarray(Z),
                s=300,
                marker="s",
                cmap=plt.get_cmap("jet"),
            )

        cbar = fig2.colorbar(im2)
        ax2.plot([0, 0], [min(Angle2), max(Angle2)], color="black")
        ax2.plot([min(Angle2), max(Angle2)], [0, 0], color="black")
        grad360 = np.arange(0, 360, 1)
        for k in range(int(max(Angle2)) + 1):
            ax2.plot(
                k * np.cos(grad360 / 180 * np.pi),
                k * np.sin(grad360 / 180 * np.pi),
                linewidth=1,
                color="black",
            )
            ax2.text(0.3, k, f"{k}°")

        cbar.set_label(f"{label} / µm", size=12, labelpad=14)
        ax2.set_xlabel("$φ$ / deg", fontsize=14)
        ax2.set_ylabel("$ϕ$ / deg", fontsize=14)
        for item in (
            [ax2.title, ax2.xaxis.label, ax2.yaxis.label]
            + ax2.get_xticklabels()
            + ax2.get_yticklabels()
        ):
            item.set_fontsize(12)

        for l in cbar.ax.yaxis.get_ticklabels():
            l.set_size(12)
        plt.savefig(f"{name}_polar.png")
        fig2.show()

    def optSelctPoint(self):
        if type(self.evaluate.mergeDataSet) == int:
            print("No data")
        else:
            self.evaluate.optReduceData = self.optReduceDataCheckBox.checkState()
            self.evaluate.resolutionOptimisation = (
                self.optiResolutionIPFLineEdit_2.text()
            )
            self.evaluate.optimisationSelctPointsIPF()

    # def createLogEvalSave(self):
    #     LoadedMergedData = self.evaluateLoadDataButtonLineEdit.text()
    #     self.levelingNumber = 0
    #     self.logNewHead(self.logfile_eval, 'Data Evaluating')
    #     self.logNewLine(self.logfile_eval, 'Merged Data filepath:\t'+LoadedMergedData)

    # def appendLogEvalLeveling(self, logfile,):
    #     self.levelingNumber += 1
    #     self.logNewHead(logfile, f'Sputter_Erosion_Leveling #{self.levelingNumber}')
    #     self.logNewLine(logfile, '')
    #     self.logNewLine(logfile, 'Sputter_Erosion_Leveling was applied with parameters:')
    #     self.logNewLine(logfile, f'Reference level:\t{self.evaluateRefereceLevelLineEdit.text()}')
    #     self.logNewLine(logfile, 'Mean Value:\t')

    # def browse_button_AFM_1(self):
    #     file_name = self.browse_button_master('AFM ibw File', 'AFM ibw File (*.ibw)',)

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
    #     file_name = self.browse_button_master('AFM ibw File', 'AFM ibw File (*.ibw)',)

    #     self.loadAFM2line.setText(file_name)
    #     self.loadPicDataline.setText('')
    #     self.loadCLSM1line.setText('')
    #     self.loadCLSM2line.setText('')
    #     if file_name:
    #         self.mergeloadAFM2.setStyleSheet(self.green)
    #     else:
    #         self.mergeloadAFM2.setStyleSheet(self.color_optional)

    # def browse_pic_data(self):
    #     file_name = self.browse_button_master('Picture', 'Picture data (*.bmp *.jpeg *.jpg *.tif *.tiff);; JPEG (*.jpeg)',)

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
    #             self.workerAFM.mergeroationAFM2 =loadAFM12DataThreadFinished self.mergeroationAFM2.currentText()[:-1]
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


# %%
if __name__ == "__main__":
    print("Program has launched!")
    global ui
    mp.freeze_support()

    app = QtWidgets.QApplication([])
    app.setStyle("Fusion")
    MainWindow = QtWidgets.QMainWindow()
    ui = connectButton()

    MainWindow.show()

    sys.exit(app.exec_())
