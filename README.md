# DataToIPF

------------  Start Program  -----------------

These ways are tested:

Linux with console:
Installation of python is required + moduls e.g. opencv

python3.8 DataToIPF.py


Windows:
Installation of anaconda is requiered + moduls e.g. opencv

For Use on E2M_Auriga:

Search for Anaconda Prompt

Execute "python D:\Balden-Data\Merge-program_Barth\DataToIPF.py" (insert via Shift + Einf)



------------  Data sets for testing -----------
Folder: data

-------  re-evaluation on at 11.07.2023  ------
tested on
anaconda 	== 2.4.0
python		== 3.9.17
spyder		== 5.4.3
CMD		== 0.1.1

Required packages, installed with pip in (Anaconda):
numpy 		== 1.25.1
matplotlib 	== 3.7.2
scipy		== 1.10.1
opencv-python	== 4.8.0.74
scikit-image	== 0.21.0
scikit-learn	== 1.3.0

---------------  Documentation  ---------------
3 evaluation-tabs + 1 info-tab:
Tab "Data Merging":

Reading EBSD-data from .ctf-files 
-> option for several material types through input line; fails if wrong 
	type inserted; for further information go into .ctf-header, 
	if prepared carefully, 1 will be sufficient
-> option to view data input
-> CAUTION: If program is started from command line, the file select 
	dialogs should not be closed without selecting any file, this
	leads to crashing

Data can be combined with other data sets
#1 CLSM_Merge:
Heights from CLSM data can be merged to EBSD data
Read in via "File_CLSM1 (experiment)"
-> difference microscopy possible
 -> difference data read in via "File_CLSM2 (optional)"
 -> substraction via affine transformation, marking of matching features needed

  -> manual selection "Manual Substract":
   -> mark first feature on CLSM1, then same feature on CLSM2, repeat for 4 features
   -> press "Enter" for affirmation
  -> automated selection "Automated Substract":
   -> feature matching with open-cv2
   -> mapped features shown
   -> press "Enter" for affirmation
  -> the point-pairs selected in both ways will be saved to file
     "000_selected_points_difference_microscopy.pts"
  -> for future reuse remember to rename file before next run!

  -> "*.pts"-files can also be read in for faster merging run "Load Points"
   -> press "Enter" for affirmation
 
 -> after each merging process, the differenced data is saved as 
    "000-difference_CLSM1_CLSM2.dat" for fututre reuse
 -> for visualization, PNGs "000_difference_CLSM1_CLSM2.png", -"_leveled.png", 
    -"_background.png" will be saved additionally, where "background" is a least-
    square-fit of a plane to the data and "leveled" has this plane removed
    

#2 AFM_Merge:

#3 Image_Merge:

------------------  Merging  ------------------
Matching points between EBSD and reference data can be selected manually
or read in from file. The manually selected points get saved to 
'tmp/000_selected-points.dat' for future reuse.
When saving the data as an .dat file, an additional logfile gets created
as 'logfile_merging.log', carrying all the inputs made in the GUI.

Current possibility to improve selection points:
For an already already established set of points, one can delete single 
points and replace them:
1. Press 'Select points for merging' and uncheck to be deleted points, 
	then press 'Save'
2. Press 'Select points for merging', press 'Select new points' and 
	continue normally, e.g with zoom function 
3. Press 'Select points for merging' and uncheck to be deleted additional
	bug point, then press 'Save'

----------------  Evaluating  -----------------

