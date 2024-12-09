import numpy as np
# T_matrix_RAS = [[0,0,1,-4.6],
#                 [1,0,0,-39.7],
#                 [0,1,0,85.5],
#                 [0,0,0,1]]

# T_matrix_Z = [[-1,0,0,226],
#                 [0,-1,0,0.2529],
#                 [0,0,1,0.2529],
#                 [0,0,0,1]]

# T_Zframe2RAS = np.matmul(T_matrix_RAS, np.linalg.inv(T_matrix_Z))
# print(T_Zframe2RAS)
T_1=np.array([-266.74,93.41,185.88])
T_2 = np.array([-268.94, 91.11, 175.68])
dir = T_1-T_2

norm_dir =dir/ np.linalg.norm(dir)

T_3 = T_1 + norm_dir *50
print(T_3)

# T_4 = T_3 + -norm_dir * 60
# print(T_4)

# dir_l = np.sqrt(dir[0]*dir[0]+dir[1]*dir[1]+dir[2]*dir[2])

# jia = 2/dir_l

# poi = T_2+(T_2-T_1)*(1+jia)

# print(poi)
     