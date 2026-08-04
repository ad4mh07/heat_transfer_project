import numpy as np
import scipy
import pandas as pd
import timeit


#Modified to do everyhting but print
def time_heat_cube(f_left, f_top, f_right, f_bottom, f_front, f_back, n=11, f_bounds=10, zoom=0, slice_axis=None, slice_index=None):

    ##Constructing b, the forcing vector
    b = np.zeros((n, n, n))
    N=n-1

    #Calculating faces
    a=np.linspace(-abs(f_bounds), abs(f_bounds), n)
    for x_index, x in enumerate(a):
        for y_index, y in enumerate(a):
            b[0, x_index, y_index]  = f_top(x,y)
            b[N, x_index, y_index] = f_bottom(x,y)
            b[x_index, 0, y_index]  = f_left(x,y)
            b[x_index, N, y_index] = f_right(x,y)
            b[x_index, y_index, 0]  = f_front(x,y)
            b[x_index, y_index, N] = f_back(x,y)

    #Calculating edges
    for k in range(n): #depth
        b[0, 0, k] = (f_top(a[0],a[k])  + f_left(a[0],a[k]))  / 2
        b[0, N, k] = (f_top(a[N],a[k]) + f_right(a[0],a[k])) / 2
        b[N, 0, k] = (f_bottom(a[0],a[k])  + f_left(a[N],a[k]))  / 2
        b[N, N, k] = (f_bottom(a[N],a[k]) + f_right(a[N],a[k])) / 2

    for j in range(n): #width
        b[0, j, 0] = (f_top(a[j],a[0])  + f_front(a[0], a[j])) / 2
        b[0, j, N] = (f_top(a[j],a[N]) + f_back(a[0], a[j]))  / 2
        b[N, j, 0] = (f_bottom(a[j],a[0])  + f_front(a[N],a[j])) / 2
        b[N, j, N] = (f_bottom(a[j],a[N]) + f_back(a[N],a[j]))  / 2

    for i in range(n): #height

        b[i, 0, 0] = (f_left(a[i],a[0])  + f_front(a[i],a[0])) / 2
        b[i, 0, N] = (f_left(a[i],a[N]) + f_back(a[i],a[0]))  / 2
        b[i, N, 0] = (f_right(a[i],a[0])  + f_front(a[i],a[N])) / 2
        b[i, N, N] = (f_right(a[i],a[N]) + f_back(a[i],a[N]))  / 2

    #Calculating corners
    b[0, 0, 0]    = (f_top(a[0], a[0])  + f_left(a[0], a[0])  + f_front(a[0], a[0])) / 3
    b[0, N, 0]   = (f_top(a[N], a[0]) + f_right(a[0], a[0]) + f_front(a[0], a[N])) / 3
    b[N, 0, 0]   = (f_bottom(a[0], a[0])  + f_left(a[N], a[0])  + f_front(a[N], a[0])) / 3
    b[N, N, 0]  = (f_bottom(a[N], a[0]) + f_right(a[N], a[0]) + f_front(a[N], a[N])) / 3
    b[0, 0, N]   = (f_top(a[0], a[N])  + f_left(a[0], a[N])  + f_back(a[0], a[0])) / 3
    b[0, N, N]  = (f_top(a[N], a[N]) + f_right(a[0], a[N]) + f_back(a[0], a[N])) / 3
    b[N, 0, N]  = (f_bottom(a[0], a[N])  + f_left(a[N], a[N])  + f_back(a[N], a[0])) / 3
    b[N, N, N] = (f_bottom(a[N], a[N]) + f_right(a[N], a[N]) + f_back(a[N], a[N])) / 3

    b = b.flatten()

    ##Constructing the (sparse) coeffecint matrix, A
    A=np.zeros((n**3, n**3)) #Making the LHS, each row is a flattened 11x11 grid for easy workings

    for i in range(1,N): #For the 'inside' points, using the fact that a points equals the mean of its 4 neighbors
        for j in range(1,N):
            for k in range(1,N):
                A_builder=np.zeros((n,n,n))
                A_builder[i][j][k]=6

                A_builder[i-1][j][k]=-1
                A_builder[i+1][j][k]=-1

                A_builder[i][j-1][k]=-1
                A_builder[i][j+1][k]=-1

                A_builder[i][j][k-1]=-1
                A_builder[i][j][k+1]=-1

                A[i*(n**2)+j*n+k] = A_builder.flatten()
    
    for i in range(n):
        for j in range(n):
            for k in range(n):
                if i == 0 or i == N or j == 0 or j == N or k==0 or k==N:
                    A[i*(n**2)+j*n+k, i*(n**2)+j*n+k] = 1

    #Solving the equation using a scipy module
    A_sparse = scipy.sparse.csr_matrix(A)
    X = scipy.sparse.linalg.spsolve(A_sparse, b)
    X=X.reshape((n,n,n))

    return X


##Restate the example

# Top: radial Gaussian hot spot at the origin
f_top = lambda x, y: 200 + 800 * np.exp(-(x**2 + y**2) / 20)

# Bottom: cold ring around the origin (low in center, hot at edges)
f_bottom = lambda x, y: 50 + 400 * (np.sqrt(x**2 + y**2) / 14.14)

# Left: linear gradient through the origin, increasing with x
f_left = lambda x, y: 500 + 40 * x

# Right: linear gradient through the origin, increasing with y
f_right = lambda x, y: 500 + 40 * y

# Front: saddle shape centered at origin
f_front = lambda x, y: 750

# Back: sinusoidal ripple centered at origin
f_back = lambda x, y: 500 + 300 * np.sin(np.pi * x / 10) * np.cos(np.pi * y / 10)


##Time the simulation

sizes = range(35, 37, 2)   # n = 1, 3, 5, 7
trials = 1
times = []

for n in sizes:
    t = timeit.timeit(lambda: time_heat_cube(f_top, f_bottom, f_left, f_right, f_front, f_back, n=n),number=trials
                      )
    times.append(round(t / trials, 5))

df = pd.DataFrame({"Cube size (n)": list(sizes),"Average runtime (s)": times})
print(df)