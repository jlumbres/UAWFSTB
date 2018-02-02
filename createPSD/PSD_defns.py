import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy import signal
import h5py
import copy


# Function: SpatFreqMap
# Description: Builds spatial frequency map for PSD generation (pulled from magaox_poppy code)
# Input Parameters:
#   optic_radius    - radius of primary mirror  
#   num_pix         - side length of test region (512 is passed in for MagAO-X)
# Output:
#    magK           - spatial frequency map with proper circular pattern
def SpatFreqMap(optic_radius, num_pix):
    sample_rate = (optic_radius*2)/num_pix
    
    FT_freq = np.fft.fftfreq(num_pix,d=sample_rate)
    kx = np.resize(FT_freq,(FT_freq.size, FT_freq.size))
    
    # Build ky the slow way
    y_val=np.reshape(FT_freq,(FT_freq.size,1))
    ky=y_val
    for m in range (0,y_val.size-1):
        ky=np.hstack((ky,y_val))
    magK = np.fft.fftshift(np.sqrt(kx*kx + ky*ky))
    return magK

# Function: calcPower
# Description: Calculate the power of a wavefront (needs better description)
# Input Parameters:
#   wf_data     - wavefront data
#   wavelength  - wavelength used to measure
# Output:
#   power_data  - REAL PART modulous square of the fourier transform with the wavefront data
def calcPower(wf_data, wavelength):
    wavefront = np.exp((2*np.pi/wavelength)*1j*wf_data)
    FT_wf = np.fft.fftshift(np.fft.fft2(wavefront))
    power_data = np.real(FT_wf*np.conjugate(FT_wf))
    return power_data

# Function: calcPower
# Description: Calculate the power of a wavefront, without putting it in exponential (needs better description)
# Input Parameters:
#   wf_data     - wavefront data
# Output:
#   power_data  - REAL PART modulous square of the fourier transform with the wavefront data
def calcPower2(wf_data):
    FT_wf = np.fft.fftshift(np.fft.fft2(wf_data))
    power_data = np.real(FT_wf*np.conjugate(FT_wf))
    return power_data

# Function: getSampSide
# Description: Calculates the sample side based on the largest side of the image
# Input Parameters:
#   optic_data  - 2D array of the data
# Output:
#   samp        - Sample side value
def getSampSide(optic_data):
    # Choose the larger side first
    if np.shape(optic_data)[0] > np.shape(optic_data)[1]:
        samp_side = np.shape(optic_data)[0]
    else:
        samp_side = np.shape(optic_data)[1]
    
    # Choosing a sample size
    if samp_side < 512:
        if samp_side < 256:
            samp = 256
        else:
            samp = 512
    else:
        samp = 1024
    
    return samp

# Function: zeroPadSquare
# Description: Zero pad a non-symmetric array into a square array
# Input Parameters:
#   optic_data  - 2D array of data, non-symmetric in size
# Output:
#   sym_data    - zero-padded symmetric array (size is of the larger side)
def zeroPadSquare(optic_data):
    squareMat = copy.copy(optic_data)
    n_row = np.shape(optic_data)[0]
    n_col = np.shape(optic_data)[1]
    side_diff = np.abs(n_row - n_col)
    #print('side difference: %d' % side_diff)
    
    # check if the difference is odd or even
    if side_diff % 2 == 0:
        odd_diff = False
        #print('The difference is even.')
    else:
        odd_diff = True
        #print('The difference is odd.')
    
    # count how many times to add row/col on both sides (hence divide by 2)
    count = np.int(np.floor(side_diff/2))
    #print('count = %d' % count)
    
    # fill in the matrix
    if n_row > n_col: # if too many rows over columns, fill in more columns both sides
        #print('There are more columns than rows')
        filler_row = np.zeros(n_row)[np.newaxis]
        for c in range(0,count):
            #print('entered the filler loop')
            squareMat = np.hstack((filler_row.T,np.hstack((squareMat,filler_row.T))))
        if odd_diff == True: # add one more column on left if odd difference
            squareMat = np.hstack((filler_row.T,squareMat))
        #print('This is the new matrix dimensions: %d, %d' % (np.shape(squareMat)[0], np.shape(squareMat)[1]))
    
    elif n_col > n_row: # too many columns than rows
        #print('There are more rows than columns')
        filler_col = np.zeros(n_col)
        for c in range(0,count):
            #print('entered the filler loop')
            squareMat = np.vstack((filler_col,np.vstack((squareMat,filler_col))))
        if odd_diff == True:
            squareMat = np.vstack((filler_col,squareMat))
        #print('This is the new matrix dimensions: %d, %d' % (np.shape(squareMat)[0], np.shape(squareMat)[1]))
    
    return squareMat

# Function: zeroPadOversample
# Description: makes a zero pad based on some oversampling requirements
# Input Parameters:
#   optic_data  - 2D array of the data
#   oversamp    - oversampling multiplier
# Output:
#   zp_wf       - zero padded wavefront
def zeroPadOversample(optic_data,oversamp):
    n_row = np.shape(optic_data)[0]
    n_col = np.shape(optic_data)[1]
    
    if n_row != n_col: # check if a symmetric matrix is being passed in
        # zero pad data into symmetric matrix
        data = zeroPadSquare(optic_data)
        # recalibrate the number of rows and columns
        n_row = np.shape(data)[0]
        n_col = np.shape(data)[1]
    else:
        data = np.copy(optic_data)
    # Sample the matrix as some 2-factor value
    samp = getSampSide(data)
    
    # This is the oversampled side size
    side = samp * oversamp
    # NOTE: This will not work for an odd symmetric matrix! If you get an error, this is why.
    row_pad = np.int((side - n_row)/2)
    zp_wf = np.pad(data, (row_pad,row_pad), 'constant')
                
    return zp_wf

# Function: doCenterCrop
# Description: crops image into square from center of image
# Input Parameters:
#   optic_data  - original 2D optic data, assumes SQUARE data
#   shift       - how many pixels to shift on each side
# Output:
#   crop_data   - cropped data
def doCenterCrop(optic_data,shift):
    side = np.shape(optic_data)[0]
    center = np.int(side/2)
    crop_data = optic_data[center-shift:center+shift,center-shift:center+shift]
    return crop_data

# Function: makeHannWindow
# Description: Builds a 2D Hann window
# Input Paramenters:
#   sideLen     - length of side for Hann window
# Output:
#   hannWin     - 2D (sideLen, sideLen) Hann window
def makeHannWindow(sideLen):
    hannSide = signal.hann(sideLen)
    hannArray = np.tile(hannSide,(sideLen,1)) # tile the rows
    hannWin = hannArray * hannSide[:, np.newaxis] # multiply across each column in tiled row array
    return hannWin

# Function: makeRingMask
# Description: Makes the radial median mask, which looks like a ring.
# Input Parameters:
#   y           - meshgrid vertical values (pixel count units)
#   x           - meshgrid horizontal values (pixel count units)
#   inner_r     - inner radial value (pixel count units)
#   dr          - ring thickness (pixel count units)
# Output:
#   ringMask    - ring mask (boolean type)
def makeRingMask(y,x,inner_r,dr):
    inside_mask = x**2+y**2 <= inner_r**2
    outside_mask = x**2+y**2 <= (inner_r+dr)**2
    ringMask = outside_mask - inside_mask
    return ringMask
    
# Function: makeRingMaskBin
# Description: Returns the bin of values from the ring mask
# Input Parameters:
#   power_data  - wavefront power data, must be square matrix
#   ringMask    - ring mask, must be same size as power_data and boolean type
# Output:
#   ringMaskBin - vector of values that survived through the mask
def makeRingMaskBin(power_data, ringMask):
    ringMaskBin = np.extract(ringMask, power_data)
    return ringMaskBin

# Function: getRadialSpatFreq
# Description: Determines the spatial frequency value at a radial distance
# Input Parameters:
#   radialFreqVector    - radial frequency in vector format (can do vector since it's radially symmetric
#   r                   - index value for inner radial distance
#   dr                  - radial thickness value
# Output:
#   radialFreq          - radial frequency value
def getRadialSpatFreq(radialFreqVector, r, dr):
    radialFreq = ((radialFreqVector[r+dr] - radialFreqVector[r])/2)+radialFreqVector[r]
    return radialFreq

# Function:calcPSD
# Description: calculates the PSD from a given wavefront data
# Input Parameters:
#   wf_loc              - wavefront data location (string)
#   oversamp            - how big to oversample with zero padding
#   shift               - half length of side to crop (so returned data will be sized 2*shift x 2*shift
# Output:
#   psd_out             - output PSD
def calcPSD(wf_loc, oversamp, shift):
    # Call in wavefront data
    optic_data, optic_header = fits.getdata(wf_loc, header=True)
    # Calculate the PSD value
    optic = zeroPadSquare(optic_data)
    hannWin = makeHannWindow(np.shape(optic)[0])
    optic_win = optic * hannWin
    optic_oversamp = zeroPadOversample(optic_win,oversamp)
    zygo_wavelen = optic_header['WAVELEN']
    power_optic = doCenterCrop(calcPower2(optic_oversamp),shift)
    return power_optic

# Function: getAvgPSD
# Description: Returns the average PSD from a set of PSDs
# Input Parameters:
#   psd_set             - list-type of 2D PSD's (so a 3D list)
# Output:
#   avg_psd             - average PSD
def getAvgPSD(psd_set):
    # convert to numpy array to work numpy from list
    np_psd = np.asarray(psd_set)
    
    # determine size of each set (each PSD slice should be the same size)
    row_side = np.shape(psd_set)[1]
    col_side = np.shape(psd_set)[2]

    # initialize variable
    avg_psd = np.zeros((row_side, col_side))
    
    # pixel-by-pixel averaging until Jhen becomes a better programmer
    for row in range(0,row_side):
        for col in range(0,col_side):
            avg_psd[row][col] = np.mean(np_psd[:,row,col])
    
    return avg_psd

# Function: median_radial_PSD
# Description: Returns 2 lists: median PSD value, and the spatial frequency for that median PSD value
# Input Parameters:
#   psd_set             - list-type of 2D PSD's (so a 3D list)
# Output:
#   avg_psd             - average PSD
def get_median_radial_PSD(avg_psd, shift, dr, radialFreq):
    # Build the radial mask grid
    maskY,maskX = np.ogrid[-shift:shift, -shift:shift]
    r = 1 # skip the center pixel
    # initialize empty lists
    median_val = [] # initialize empty list of median power values
    k_val = [] # initialize empty list of frequencies
    # Calculate through radial mask grid to get median PSD value
    while((r+dr)<(shift)):
        radial_mask = makeRingMask(maskY,maskX,r,dr)
        radial_bin = makeRingMaskBin(avg_psd, radial_mask)
        median_val.append(np.median(radial_bin))
        k_val.append(getRadialSpatFreq(radialFreq,r,dr))
        r = r+1
    
    return median_val, k_val
