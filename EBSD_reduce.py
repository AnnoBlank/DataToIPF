# -*- coding: utf-8 -*-
"""
Created on Tue May 21 09:24:05 2024

@author: Robin
"""
import numpy as np
import argparse, textwrap
argParser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

# input path (required)
argParser.add_argument("-i", "--input", help="input file name and its full path", required=True)

# # output path
# argParser.add_argument("-o", "--output", nargs='?', help="output file")

# reduction Factor (required)
argParser.add_argument("-f", "--reduction_factor", nargs='?', default=2, type=int, help=textwrap.dedent('''\
                        input reduction factor as int
                        factor is used along both axis 
                        -> effective reduction: factor^2 
                        ! using factor = 1 may cause problems !
                        '''))

# get parameters
args = argParser.parse_args()
file_name = args.input #"C:/Users/Robin/FAUbox/Uni/Garching_HiWi/data_rb/EBSD_data.ctf"
reduction_factor = args.reduction_factor

# read dataset to be reduced
with open(file_name,"r",errors="replace") as f:
    CRSTdata = f.readlines()
    f.close()

# get information from header (XCells, YCells, end of header)
i = 0
X_pos = 0
Y_pos = 0
for line in CRSTdata:
    i += 1
    if line.split()[0] == 'XCells':
        X_len = int(line.split()[1])
        X_pos = i
    if line.split()[0] == 'YCells':
        Y_len = int(line.split()[1])
        Y_pos = i
    if line.split()[0] == 'Phase':
        break
    
# error massage if no X- or Y-length could be read from header
if X_pos == 0 or Y_pos == 0:
    print(f'An error occurred! Length of X-dimension found in line {X_pos}, length of Y-dimension found in line {Y_pos}.')

# convert data to numpy array for easier reduction
data = np.array(CRSTdata[i:])
shape = (Y_len, X_len)
data = np.reshape(data, shape)

# reduce data along the two dimensions
if reduction_factor != 1:
    mask = np.array([i % reduction_factor == 0 for i in range(len(data))])
data = data[mask]

if reduction_factor != 1:
    mask = np.array([i % reduction_factor == 0 for i in range(len(data[0,:]))])
data = np.array([line[mask] for line in data])

# get shape of data
Y_len, X_len = data.shape
     
# restore original shape and type of data column
data = data.flatten().tolist()

# adjustment of header (XCells, YCells)
header = CRSTdata[:i]
header[X_pos-1] = header[X_pos-1].split()[0] + '\t' + str(X_len) + '\n'
header[Y_pos-1] = header[Y_pos-1].split()[0] + '\t' + str(Y_len) + '\n'

# saving reduced data in original format with appended reduction factor
savefile_name = file_name[:-4] + '_reduced_by_' + str(reduction_factor) + '.ctf'
with open(savefile_name, 'w+') as file:
    file.writelines(header)
    file.writelines(data)
    file.close()
