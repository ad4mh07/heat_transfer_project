import numpy as np
import scipy
import matplotlib.pyplot as plt
import plotly.express as px 

def solve_heat_cube(f_left, f_top, f_right, f_bottom, f_front, f_back, n=11, f_bounds=10, zoom=0, slice_axis=None, slice_index=None):

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


    ##Post-processing and plotting

    #Argument checking
    if zoom > 0 and (slice_axis is not None or slice_index is not None):
        raise ValueError("Cannot zoom and slice simultaneously.")
    elif zoom<0:
        raise ValueError("Zoom must be positive")

    #Deciding what to plot 
    else:
        #Zoomed in plotting
        if zoom > 0:
            sl = slice(zoom, -zoom)
            X_plot = X[sl, sl, sl]
            x, y, z = np.meshgrid(range(zoom, n - zoom),range(zoom, n - zoom),range(zoom, n - zoom),indexing="ij")

        #Normal plotting 
        elif slice_axis is None and slice_index is None:
            X_plot = X
            x, y, z = np.meshgrid(range(n), range(n), range(n), indexing="ij")

        #Sliced plotting
        elif slice_axis is not None and slice_index is not None:
            if slice_axis == "x":
                X_plot = X[slice_index, :, :]
            elif slice_axis == "y":
                X_plot = X[:, slice_index, :]
            elif slice_axis == "z":
                X_plot = X[:, :, slice_index]
            else:
                raise ValueError("slice_axis must be 'x', 'y', or 'z'.")

            # Plot the 2D slice here
            plt.imshow(X_plot, origin="lower", cmap="coolwarm")
            plt.colorbar(label="Temperature")
            plt.show()
            return
        else:
            raise ValueError("Specify both slice_axis and slice_index, or neither.")

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        sc = ax.scatter(x.flatten(), y.flatten(), z.flatten(), c=X_plot.flatten(), cmap="coolwarm", s=100,)

        plt.colorbar(sc, label="Temperature")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        plt.title("3D Heat Distribution")
        plt.show()

        return X
