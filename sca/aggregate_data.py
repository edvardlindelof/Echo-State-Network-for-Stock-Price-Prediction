from collections import deque
import numpy as np

omx_file = open('OMX')
omx_lines = omx_file.readlines()[1:]  # first line is documentation
omx_file.close()

sca_file = open('SCAA')
sca_lines = sca_file.readlines()[1:]  # first line is documentation
sca_file.close()

matrix_list = []
steps_looked_back = 20
omx_dq = deque(maxlen=steps_looked_back+1)
sca_dq = deque(maxlen=steps_looked_back+1)
line = []
for i in range(len(omx_lines)):
    omx_close = float(omx_lines[i].split(',')[4])
    omx_dq.append(omx_close)
    omx_dql = list(omx_dq)
    sca_close = float(sca_lines[i].split(',')[4])
    sca_dq.append(sca_close)
    sca_dql = list(sca_dq)
    if len(omx_dql) == (steps_looked_back+1):
        matrix_list.append([
            sca_dql[-1],  # next sca close (to be predicted)
            sca_dql[-2],  # last close
            sca_dql[-2] - sca_dql[-3],  # derivative
            (sca_dql[-2] - sca_dql[-3]) - (sca_dql[-4] - sca_dql[-5]),  # 2nd derivative
            sum(sca_dql[-6:-1]) / 5.,  # rolling avg
            (sum(sca_dql[-6:-1]) / 5.) / (sum(sca_dql[-21:-1]) / 20.),  # rolling avg quote
            omx_dql[-2],  # last close
            omx_dql[-2] - omx_dql[-3],  # derivative
            (omx_dql[-2] - omx_dql[-3]) - (omx_dql[-4] - omx_dql[-5]),  # 2nd derivative
            sum(omx_dql[-6:-1]) / 5.,  # rolling avg
            (sum(omx_dql[-6:-1]) / 5.) / (sum(omx_dql[-21:-1]) / 20.)  # rolling avg quote
        ])

destination_file = open('aggregated_data', 'w')
for line in matrix_list:
    destination_file.write(str(line).strip('[]') + '\n')
destination_file.close()
