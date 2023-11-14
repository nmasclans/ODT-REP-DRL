"""
Check barycentric mapping: direct & inverse transformation

Direct mapping:
  x = B*eigval + c, where B = 2x2, eigval = 2x1, c = 2x1
Inverse mapping:
  eigval = Binv * (x-c)

Used known coordinates and eigenvalues of the 3 barycentric map corners
"""


import numpy as np

np.set_printoptions(precision=3)

# Barycentric map corners: known coordinates & eigenvalues                
x1c = np.array([1.0, 0.0]);                 eigvalx1c = np.array([2/3, -1/3, -1/3])
x2c = np.array([0.0, 0.0]);                 eigvalx2c = np.array([1/6, 1/6, -1/3])
x3c = np.array([0.5, np.sqrt(3.0)/2.0]);    eigvalx3c = np.array([0,0,0])                

# Operador lineal
B      = np.zeros((2,2))
B[:,0] =  x1c + 2*x2c - 3*x3c
B[:,1] = -x1c + 4*x2c - 3*x3c
Binv   = np.linalg.inv(B)
print("B = \n", B)
print("Binv = \n", Binv)
# Terme independent
c      = x3c

# Mapping eigval -> x
def mapping_B_eigval_to_x(eigval):
    eigval_red = eigval[:2]
    return B.dot(eigval_red) + c

# Mapping x -> eigval
def mapping_Binv_x_to_eigval(x):
    eigval = np.zeros(3)
    eigval[:2] = Binv.dot(x-c)
    eigval[2]  = 1 - eigval[0] - eigval[1] # constrain sum(eigval) = 0 
    return eigval

# According to Emory2013-A, the eigenvalues of the barycentric map corners are:
# eigval = (2/3, -1/3, -1/3) = eigvalx1c for x = x1c
# eigval = (1/6, 1/6, -1/3)  = eigvalx2c for x = x2c
# eigval = (0,0,0)           = eigvalx3c for x = x3c

print("\nCheck direct and inverse barycentric mapping")

print("\n1st corner x1c: coord x = ", x1c, " & eigval = ", eigvalx1c)
print("(Direct Mapping)   eigval  = ", eigvalx1c, " --> coord x = ", mapping_B_eigval_to_x(eigvalx1c))
print("(Inverse Mapping)  coord x = ", x1c, " --> eigval = ",        mapping_Binv_x_to_eigval(x1c))

print("\n2nd corner x2c: coord x = ", x2c, " & eigval = ", eigvalx2c)
print("(Direct Mapping)   eigval  = ", eigvalx2c, " --> coord x = ", mapping_B_eigval_to_x(eigvalx2c))
print("(Inverse Mapping)  coord x = ", x2c, " --> eigval = ",        mapping_Binv_x_to_eigval(x2c))

print("\n3rd corner x3c: coord x = ", x3c, " & eigval = ", eigvalx3c)
print("(Direct Mapping)   eigval  = ", eigvalx3c, " --> coord x = ", mapping_B_eigval_to_x(eigvalx3c))
print("(Inverse Mapping)  coord x = ", x3c, " --> eigval = ",        mapping_Binv_x_to_eigval(x3c))

print("Compare previous results to check direct & inverse mapping!")
