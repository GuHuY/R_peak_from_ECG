from scipy import io
import numpy as np
import matplotlib.pyplot as plt
import time
import preprocessing
from preprocessing import image_test
import struct
import wfdb
import sys
import R_refine1
from R_detection import generate_interval_list
from preprocessing import B


mat_pos = '/Users/rex/z_thesis/apnea_ECG_data/a02'

m_plot = [[3700, 3900], ['ECG', 'EVA', 'ANA', 'EVA2']]

#, 'DER_S3', 'DER_S7'  image_test(R, A, [[3000, 4000], ['ECG', 'A', 'DER']])

(result, R, A, e, f) = preprocessing.synthesize(mat_pos)
np.savetxt('/Users/rex/z_thesis/a8.txt', result, fmt='%.5f')
# np.savetxt("a8.txt", round_result)


print(len(f))
# #
E = [A[0], e]

F = [f*(1000/100)]


RIS_after_first_refine = [f*(1000/100), generate_interval_list(f*(1000/100))]
G = B(RIS_after_first_refine)

image_test(R, A, m_plot, E, F, G)



