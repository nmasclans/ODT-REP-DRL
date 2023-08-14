import numpy as np 
import matplotlib.pyplot as plt

# Latex figures
plt.rc( 'text',       usetex = True )
plt.rc( 'font',       size = 18 )
plt.rc( 'axes',       labelsize = 18)
plt.rc( 'legend',     fontsize = 18)
plt.rc( 'text.latex', preamble = r'\usepackage{amsmath} \usepackage{amssymb} \usepackage{color}')


# Parameters
y0 = 0.5; l = 1 
f1 = 1/3
f2 = 2/3
npoints = 200

# Axis grid
yn = np.linspace(0.1,2,npoints)

# Profile y vs u:
def func_u(y_profile):
    return np.log(y_profile) - np.log(np.min(y_profile)) 

# Triplet map
def triplet_map(y, y0, l, f1, f2, npoints):
    y_triplet = np.zeros(npoints)
    for i in range(npoints):
        yi = y[i]
        if yi > y0 and yi < y0 + f1*l:
            y_triplet[i] = y0 + (1/f1) * (yi - y0)
        elif yi > y0 + f1*l and yi < y0 + f2*l:
            y_triplet[i] = y0 - (1/f1)*(yi-y0) + (f2/f1)*l
        elif yi > y0 + f2*l and yi < y0 + l:
            y_triplet[i] = y0 + (1/f1)*(yi-y0) - (f2/f1)*l
        else:
            y_triplet[i] = yi
    return y_triplet


# Original profile, before triple map
y_beforeTriplet = yn
u_beforeTriplet = func_u(yn)

# Profile after triple map
y_tripletMap = triplet_map(yn, y0, l, f1, f2, npoints)
u_tripletMap = func_u(y_tripletMap)

# print("\nBefore triple mapping:")
# print("y:",y_beforeTriplet)
# print("u:",u_beforeTriplet)
# print("\nAfter triple mapping:")
# print("y:",y_tripletMap)
# print("u:",u_tripletMap)

plt.figure(1)
plt.plot(y_beforeTriplet, u_beforeTriplet, label='before triplet map')
plt.plot(y_beforeTriplet, u_tripletMap,    label='after triplet map')
plt.legend(loc = 'lower right')
plt.xlabel(r'$y$'); plt.ylabel(r'$\phi=f(y)$')
plt.xticks([])
plt.yticks([])
plt.savefig('triplet_map_y_vs_phi.jpg', dpi=600)

plt.figure(2)
plt.plot(y_beforeTriplet, y_tripletMap)
plt.xlabel(r'$y$ before triplet map')
plt.ylabel(r'$y$ after triplet map')
plt.xticks([])
plt.yticks([])
plt.savefig('triplet_map_y_before_vs_after.jpg', dpi=600)