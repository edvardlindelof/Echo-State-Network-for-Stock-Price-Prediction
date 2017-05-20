from collections import deque
import numpy as np

original_file = open('SML_data_1')
raw_lines = original_file.readlines()[1:]  # first line is documentation
original_file.close()

matrix_list = []
steps_looked_back = 5
dq = deque(maxlen=steps_looked_back+1)
line = []
for raw_line in raw_lines:
    last_line = line
    line = raw_line.split(' ')[2:-1]  # first 2 elements non-numeric, last one \n
    dq.append(float(line[0]))
    if len(dq) == (steps_looked_back+1):
        ## aggregated features
        dql = list(dq)
        matrix_line = [
            dql[-1],  # next temperature (to be predicted)
            dql[-2],  # temperature now
            dql[-2] - dql[-3],  # some sorta derivative
            dql[-2] - dql[-6],  # some sorta derivative
            (dql[-2] - dql[-3]) - (dql[-2] - dql[-6]),  # some sorta second derivative?
            sum(dql[-6:-1]) / 5.  # rolling avg
        ]
        ## features from original dataset
        for feature in last_line[1:]:
            matrix_line.append(float(feature))

        matrix_list.append(matrix_line)

'''
print(matrix_list[5])
print(len(matrix_list[5]))  # 26
print(len(matrix_list))  # 2759
'''

destination_file = open('SML_data_1_aggregated', 'w')
for line in matrix_list:
    destination_file.write(str(line).strip('[]') + '\n')
destination_file.close()
