import sys
sys.path.append('/usr/local/lib/python2.7/site-packages')

import math

import cv2
import numpy as np
from matplotlib import pyplot as plt
from pprint import pprint


img = cv2.imread('mp2.tif',0)
print "img"
print img
img = np.float32(img)

dft = cv2.dft(img, flags = cv2.DFT_COMPLEX_OUTPUT)
print "dft"
print dft
print dft.shape


phase_angle = cv2.phase(dft[:,:,0], dft[:,:,1])
phase_angle = np.float32(phase_angle)
print "phase_angle"
print phase_angle

dft_shift = np.fft.fftshift(dft)
magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))


height,width = img.shape
one_matrix = np.ones(shape=(height, width))
one_matrix = np.float32(one_matrix)*1


one_pa = cv2.polarToCart(one_matrix, phase_angle)
print "one_pa"
print one_pa


one_pa_merge = cv2.merge(one_pa[0],one_pa[1])
print "one_pa_merge"
print one_pa_merge
print one_pa_merge.shape


remove_middle = one_pa_merge[:,0,:]


reconpa = cv2.idft(remove_middle, flags = cv2.DFT_REAL_OUTPUT)
print "=============="
print "reconpa"
print reconpa
print reconpa.shape

reconpa = np.uint8(reconpa)
print "=============="
print "reconpa uint8"
print reconpa


one_mag = cv2.polarToCart(magnitude_spectrum, one_matrix)
print "one_mag"
print one_mag


one_mag_merge = cv2.merge(one_mag[0],one_mag[1])
print "one_mag_merge"
print one_mag_merge
print one_mag_merge.shape


remove_middle = one_mag_merge[:,0,:]

reconmag = cv2.idft(remove_middle, flags = cv2.DFT_REAL_OUTPUT)
print "=============="
print "reconmag"
print reconmag
print reconmag.shape

reconmag = np.uint8(reconmag)
print "=============="
print "reconmag uint8"
print reconmag




'''
rec = cv2.imread('rectangle.tif',0)
rec = rec.resize((height,width), Image.ANTIALIAS)
print "rec"
print rec
print rec.shape
rec = np.float32(rec)

dft_rec = cv2.dft(rec, flags = cv2.DFT_COMPLEX_OUTPUT)
print "dft_rec"
print dft_rec
print dft_rec.shape


rec_pa = cv2.phase(dft[:,:,0], dft[:,:,1])
rec_pa = np.float32(rec_pa)
print "rec_pa"
print rec_pa

dft_rec_shift = np.fft.fftshift(dft_rec)
rec_mag = 20*np.log(cv2.magnitude(dft_rec_shift[:,:,0],dft_rec_shift[:,:,1]))


rec_mag_pa = cv2.polarToCart(rec_mag, phase_angle)
print "rec_mag_pa"
print rec_mag_pa


rec_mag_pa_merge = cv2.merge(rec_mag_pa[0],rec_mag_pa[1])
print "rec_mag_pa_merge"
print rec_mag_pa_merge
print rec_mag_pa_merge.shape


remove_middle = rec_mag_pa_merge[:,0,:]


reconpa_rec = cv2.idft(remove_middle, flags = cv2.DFT_REAL_OUTPUT)
print "=============="
print "reconpa_rec"
print reconpa_rec
print reconpa_rec.shape

reconpa_rec = np.uint8(reconpa_rec)
print "=============="
print "reconpa_rec uint8"
print reconpa_rec
'''


'''
f = np.fft.fft2(img)
#print "f"
#print f
phase_angle = cv2.phase(f.real, f.imag)
phase_angle = np.float32(phase_angle)
print "phase_angle"
print phase_angle


fshift = np.fft.fftshift(f)
magnitude_spectrum = 20*np.log(np.abs(fshift))

#print phase_angle
#print phase_angle.shape
#print (phase_angle*180/(np.pi))

height,width = img.shape
one_matrix = np.ones(shape=(height, width))
one_matrix = np.float32(one_matrix)*1


test = cv2.polarToCart(one_matrix, phase_angle)
print "test"
print test

test_merge = cv2.merge(test[0],test[1])
print "test_merge"
print test_merge
print test_merge.shape

remove_middle = test_merge[:,0,:]

reconpa = np.fft.ifft2(remove_middle)
print "=============="
print "reconpa"
print reconpa
print reconpa.shape

reconpa = np.uint8(reconpa)
print "=============="
print "reconpa"
print reconpa
'''


plt.subplot(231),plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(232),plt.imshow(phase_angle, cmap = 'gray')
plt.title('Phase Angle'), plt.xticks([]), plt.yticks([])
plt.subplot(233),plt.imshow(magnitude_spectrum, cmap = 'gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.subplot(234),plt.imshow(reconpa, cmap = 'gray')
plt.title('Reconstruct using PA'), plt.xticks([]), plt.yticks([])
plt.subplot(235),plt.imshow(reconmag, cmap = 'gray')
plt.title('Reconstruct using MS'), plt.xticks([]), plt.yticks([])
#plt.subplot(236),plt.imshow(reconpa_rec, cmap = 'gray')
#plt.title('Reconstruct using PA_rec'), plt.xticks([]), plt.yticks([])
plt.show()
    
    
cv2.waitKey(0)
cv2.destroyAllWindows()
