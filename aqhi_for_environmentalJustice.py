# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 16:15:23 2017
Author: Vikram Ravi
Description: (a) Grab information about different census tracts and (row, col) pairs
             (b) Reads CMAQ output - NO2, O3 and PM2.5
             (c) Calculate the three hour rolling averages for O3, PM2.5 and NO2
             (d) Calculate AQHI category
             (e) Assign AQHI value for each census tract (as is for smaller than 16 km2 tracts and 
                 an average of all (row,col) pairs in that tract if tract area is larger than 16 km2)
                 
"""

from netCDF4 import Dataset
import pandas as pd
import numpy as np
from numba import jit
import time
#%%

cases = ['fire2015', 'ozone2012']
case  = cases[0]

# define and read files
waTractsFile = r"C:\Users\vik\Documents\Projects\environmentalHealthJustice\WA_tracts.csv"
orTractsFile = r"C:\Users\vik\Documents\Projects\environmentalHealthJustice\OR_tracts.csv"
idTractsFile = r"C:\Users\vik\Documents\Projects\environmentalHealthJustice\ID_tracts.csv"

waGridFile = r"C:\Users\vik\Documents\Projects\environmentalHealthJustice\WAgrid.csv"
orGridFile = r"C:\Users\vik\Documents\Projects\environmentalHealthJustice\ORgrid.csv"
idGridFile = r"C:\Users\vik\Documents\Projects\environmentalHealthJustice\IDgrid.csv"


waTractsAll = pd.read_csv(waTractsFile)
orTractsAll = pd.read_csv(orTractsFile)
idTractsAll = pd.read_csv(idTractsFile)

waGrid = pd.read_csv(waGridFile)
orGrid = pd.read_csv(orGridFile)
idGrid = pd.read_csv(idGridFile)

waTracts  = waTractsAll.loc[:, ['GEOID']]
orTracts  = orTractsAll.loc[:, ['GEOID']]
idTracts  = idTractsAll.loc[:, ['GEOID']]

waGrid  = waGrid.loc[:, ['I', 'J', 'GEOID']]
orGrid  = orGrid.loc[:, ['I', 'J', 'GEOID']]
idGrid  = idGrid.loc[:, ['I', 'J', 'GEOID']]

#%%
# define the various input files
inputDir    = r"C:\Users\vik\Documents\Projects\environmentalHealthJustice"
outputDir   = r"C:\Users\vik\Documents\Projects\environmentalHealthJustice"
gridFileDir = "C:/Users/vik/Documents/Projects/MCIP_data/2011111100/MCIP/"

cctmFile = '{d}/POST_CCTM_M3XTRACT_NCRCAT_{c}_2011_L01.ncf'.format(d=inputDir, c=case)# inputDir + "/" + "POST_CCTM_M3XTRACT_NCRCAT_fire2015_2011_L01.ncf"
gridFile = gridFileDir + "GRIDCRO2D"

#%%
def readNETCDF(infile, chemSpecies):
    variable_conc  = infile.variables[chemSpecies][:,0,:,:]
    # if the concentration is in ppm, convert it to ppb
    if infile.variables[chemSpecies].units.strip() == 'ppm':
        print ('concentration for {} in input file is in ppm, converting to ppb...'.format(chemSpecies))
        variable_conc = 1000.0*variable_conc

    return variable_conc

#%%
# calculate three hour average as required by AQHI function
# for a description of np.ma.average function used below, see:
# http://docs.scipy.org/doc/numpy-1.10.1/reference/generated/numpy.ma.average.html
# also, the weights list created below is for assigning weight of 1 for the three 
# elements used for calculating 3 hour mean, and weight for all other elements is set to zero
@jit
def threeHourAverage(inVar):
    outVar = np.zeros((inVar.shape[0]-3, inVar.shape[1], inVar.shape[2]))
    for itime in np.arange(inVar.shape[0]-3):
        if ( itime%10 == 0 ):
            print (itime)
        weights = []
        for ix in np.arange(itime):
            weights.append(0)
        for ix in np.arange(3):
            weights.append(1)
        for ix in np.arange(itime+3,inVar.shape[0]):
            weights.append(0)
        t3average = np.ma.average(inVar, axis=0, weights=weights)
        outVar[itime, :, :] = t3average
    #return weights, outVar
    return outVar

#%%
def getMaxAtCells(inVar):
    outVar = np.zeros((inVar.shape[1], inVar.shape[2]))
    for itime in np.arange(inVar.shape[0]):
        for row in np.arange(inVar.shape[1]):
            for col in np.arange(inVar.shape[2]):
                if (inVar[itime, row, col] > outVar[row, col]):
                    outVar[row, col] = inVar[itime, row, col]
    return outVar
#%%
# convert aqhi > 10 to 10.5  for plotting purpose    
def greaterThanTenAQHI(inVar):
    outVar = np.zeros((inVar.shape[0], inVar.shape[1], inVar.shape[2]))
    for itime in np.arange(inVar.shape[0]):
        for row in np.arange(inVar.shape[1]):
            for col in np.arange(inVar.shape[2]):
                if (inVar[itime, row, col] > 10.0):
                    outVar[itime, row, col] = 10.75
                elif (inVar[itime, row, col] <= 10.0):
                    outVar[itime, row, col] = inVar[itime, row, col]
    return outVar    

#%%
# read the variables    
grd = Dataset(gridFile,'r')
lat = grd.variables['LAT'][0,0,:,:]
lon = grd.variables['LON'][0,0,:,:]
ht  = grd.variables['HT'][0,0,:,:]
w = (grd.NCOLS)*(grd.XCELL)
h = (grd.NROWS)*(grd.YCELL)
lat_0 = grd.YCENT
lon_0 = grd.XCENT
nrows = grd.NROWS
ncols = grd.NCOLS

fileACONC = Dataset(cctmFile, 'r')
pm  = readNETCDF(fileACONC, 'PMIJ')
o3  = readNETCDF(fileACONC, 'O3')
nox = readNETCDF(fileACONC, 'NOX')

no2 = 0.9*nox
grd.close()
fileACONC.close()    
#%%
# apply the three hour averaging function
begin_time = time.time()
pm_3hr  = threeHourAverage(pm)
no2_3hr = threeHourAverage(no2)
o3_3hr  = threeHourAverage(o3)
end_time = time.time()
print ("time taken =%s"%(end_time-begin_time))
#%%
no2CoeffCool  = 0.000457
pm25CoeffCool = 0.000462

no2CoeffWarm  = 0.00101
pm25CoeffWarm = 0.000621
o3CoeffWarm   = 0.00104

aqhi_warm_base = (10.0/12.80)*(100*(np.exp(no2CoeffWarm*no2_3hr) -1 + np.exp(pm25CoeffWarm*pm_3hr) -1 + np.exp(o3CoeffWarm*o3_3hr) -1))
aqhi_warm_base = aqhi_warm_base.mean(axis=0)

#%%
outFileName = {0:'WA_tracts_mean_aqhi', 1:'OR_tracts_mean_aqhi', 2:'ID_tracts_mean_aqhi'}
for i, df in enumerate(list([waGrid, orGrid, idGrid])):
    for idx in df.index:
        aqhi_at_IJ = aqhi_warm_base[df.at[idx, 'J'], df.at[idx, 'I']]
        df.at[idx, 'AQHI'] = aqhi_at_IJ
    df = df.set_index('GEOID', drop=True)
    df = df.groupby(by=df.index).mean()
    df.to_csv('{directory}/{fileName}_{case}.csv'.format(directory=outputDir, fileName=outFileName[i], case=case))
#%%    
