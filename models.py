import numpy as np
import matplotlib.pyplot as plt
import scipy

#==================================================
# 0: 11x11 Square
#==================================================
def temp_analysis_square(left_t, top_t, right_t, bottom_t, v=11):
    b = np.zeros((11, 11))

    #sides
    b[:, 0]  = left_t
    b[:, 10] = right_t
    b[10, :]  = top_t
    b[0, :] = bottom_t

    #corners
    b[0, 0]   = (top_t + left_t) / 2
    b[0, 10]  = (top_t + right_t) / 2
    b[10, 0]  = (bottom_t + left_t) / 2
    b[10, 10] = (bottom_t + right_t) / 2

    b = b.flatten()
    
    A=np.zeros((121,121)) #Making the LHS, each row is a flattened 11x11 grid for easy workings

    for i in range(1,10): #For the 'inside' points, using the fact that a points equals the mean of its 4 neighbors
        for j in range(1,10):
            a=np.zeros((11,11))
            a[i][j]=4
            a[i-1][j]=-1
            a[i][j-1]=-1
            a[i+1][j]=-1
            a[i][j+1]=-1
            A[i*11+j]=a.flatten()
    
    for i in range(11):
        for j in range(11):
            if i == 0 or i == 10 or j == 0 or j == 10:
                A[i*11+j, i*11+j] = 1
    
    x=np.linalg.solve(A,b)
    x=x.reshape((11,11))
    return x

#==================================================
# 1: Cube (constant face temps)
#==================================================
def temp_analysis_cube_1(left_t, top_t, right_t, bottom_t, front_t, back_t, v=11):
    b = np.zeros((11, 11, 11))
    """B is an 11x11x11 cube which we manually assign the known elements, ie faces, edges and corners.
    We then flatten it to geth the RHS of our equation, i.e every point in our cube in a collumn vector
    This is a lot easier to work with than assign face and edge etc values to a collumn vector."""

    #faces
    b[:, 0,:]  = left_t
    b[:, 10, :] = right_t
    b[0, :, :]  = top_t
    b[10, :, :] = bottom_t
    b[:, :, 0] = front_t
    b[:, :, 10] = back_t

    #Edges
    b[:,0,0] = (left_t + front_t ) / 2
    b[:,0,10] = (left_t + back_t) / 2
    b[:,10,0] = (right_t + front_t ) / 2
    b[:,10,10] =  (right_t + back_t ) / 2

    b[0,:,0] = (top_t +front_t ) / 2
    b[0,:,10] = (top_t + back_t ) / 2
    b[10,:,0] = (bottom_t + front_t) / 2
    b[10,:,10] = (bottom_t + back_t) / 2

    b[0,0,:] = (top_t + left_t) / 2
    b[0,10,:] = (top_t + right_t) / 2
    b[10,0,:] = (bottom_t + left_t) / 2
    b[10,10,:] = ( bottom_t+ right_t) / 2

    #corners
    b[0, 0, 0]   = (top_t + left_t + front_t) / 3
    b[0, 10, 0]  = (top_t + right_t + front_t) / 3
    b[10, 0, 0]  = (bottom_t + left_t + front_t) / 3
    b[10, 10, 0] = (bottom_t + right_t + front_t) / 3
    b[0, 0, 10]   = (top_t + left_t + back_t) / 3
    b[0, 10, 10]  = (top_t + right_t + back_t) / 3
    b[10, 0, 10]  = (bottom_t + left_t + back_t) / 3
    b[10, 10, 10] = (bottom_t + right_t + back_t) / 3

    b = b.flatten()
    
    A=np.zeros((11**3, 11**3)) #Making the LHS, each row is a flattened 11x11 grid for easy workings

    for i in range(1,10): #For the 'inside' points, using the fact that a points equals the mean of its 4 neighbors
        for j in range(1,10):
            for k in range(1,10):
                a=np.zeros((11,11,11))
                a[i][j][k]=6

                a[i-1][j][k]=-1
                a[i+1][j][k]=-1

                a[i][j-1][k]=-1
                a[i][j+1][k]=-1

                a[i][j][k-1]=-1
                a[i][j][k+1]=-1

                A[i*121+j*11+k] = a.flatten()
    
    for i in range(11):
        for j in range(11):
            for k in range(11):
                if i == 0 or i == 10 or j == 0 or j == 10 or k==0 or k==10:
                    A[i*121+j*11+k, i*121+j*11+k] = 1
    
    x=np.linalg.solve(A,b)
    x=x.reshape((11,11,11))
    return x

#==================================================
# 2: Cube (variable face temps), hardcoded 11-cube
#==================================================

def temp_analysis_cube_2(f_left, f_top, f_right, f_bottom, f_front, f_back):
    b = np.zeros((11, 11, 11))
    """B is an 11x11x11 cube which we manually assign the known elements, ie faces, edges and corners.
    We then flatten it to geth the RHS of our equation, i.e every point in our cube in a collumn vector
    This is a lot easier to work with than assign face and edge etc values to a collumn vector.

    i is height, j is width and k is depth"""

    #faces
    for x in range(11):
        for y in range(11):
            b[0, x, y]  = f_top(x,y)
            b[10, x, y] = f_bottom(x,y)
            b[x, 0, y]  = f_left(x,y)
            b[x, 10, y] = f_right(x,y)
            b[x, y, 0]  = f_front(x,y)
            b[x, y, 10] = f_back(x,y)

    #Edges
    for k in range(11): #depth
        b[0, 0, k] = (f_top(0,k)  + f_left(0,k))  / 2
        b[0, 10, k] = (f_top(10,k) + f_right(0,k)) / 2
        b[10, 0, k] = (f_bottom(0,k)  + f_left(10,k))  / 2
        b[10, 10, k] = (f_bottom(10,k) + f_right(10,k)) / 2

    for j in range(11): #width
        b[0, j, 0] = (f_top(j,0)  + f_front(0, j)) / 2
        b[0, j, 10] = (f_top(j,10) + f_back(0, j))  / 2
        b[10, j, 0] = (f_bottom(j,0)  + f_front(10,j)) / 2
        b[10, j, 10] = (f_bottom(j,10) + f_back(10,j))  / 2

    for i in range(11): #height

        b[i, 0, 0] = (f_left(i,0)  + f_front(i,0)) / 2
        b[i, 0, 10] = (f_left(i,10) + f_back(i,0))  / 2
        b[i, 10, 0] = (f_right(i,0)  + f_front(i,10)) / 2
        b[i, 10, 10] = (f_right(i,10) + f_back(i,10))  / 2

    #Corners
    b[0, 0, 0]    = (f_top(0, 0)  + f_left(0, 0)  + f_front(0, 0)) / 3
    b[0, 10, 0]   = (f_top(10, 0) + f_right(0, 0) + f_front(0, 10)) / 3
    b[10, 0, 0]   = (f_bottom(0, 0)  + f_left(10, 0)  + f_front(10, 0)) / 3
    b[10, 10, 0]  = (f_bottom(10, 0) + f_right(10, 0) + f_front(10, 10)) / 3
    b[0, 0, 10]   = (f_top(0, 10)  + f_left(0, 10)  + f_back(0, 0)) / 3
    b[0, 10, 10]  = (f_top(10, 10) + f_right(0, 10) + f_back(0, 10)) / 3
    b[10, 0, 10]  = (f_bottom(0, 10)  + f_left(10, 10)  + f_back(10, 0)) / 3
    b[10, 10, 10] = (f_bottom(10, 10) + f_right(10, 10) + f_back(10, 10)) / 3

    b = b.flatten()
    
    A=np.zeros((11**3, 11**3)) #Making the LHS, each row is a flattened 11x11 grid for easy workings

    for i in range(1,10): #For the 'inside' points, using the fact that a points equals the mean of its 4 neighbors
        for j in range(1,10):
            for k in range(1,10):
                a=np.zeros((11,11,11))
                a[i][j][k]=6

                a[i-1][j][k]=-1
                a[i+1][j][k]=-1

                a[i][j-1][k]=-1
                a[i][j+1][k]=-1

                a[i][j][k-1]=-1
                a[i][j][k+1]=-1

                A[i*121+j*11+k] = a.flatten()
    
    for i in range(11):
        for j in range(11):
            for k in range(11):
                if i == 0 or i == 10 or j == 0 or j == 10 or k==0 or k==10:
                    A[i*121+j*11+k, i*121+j*11+k] = 1
    
    x=scipy.sparse.linalg.spsolve(A,b)
    x=x.reshape((11,11,11))
    return x


#====================================================
# 3: Cube (variable face temps), variable sized cube
#====================================================

def temp_analysis_cube_3(f_left, f_top, f_right, f_bottom, f_front, f_back, n=11):
    b = np.zeros((n, n, n))
    """B is an 11x11x11 cube which we manually assign the known elements, ie faces, edges and corners.
    We then flatten it to geth the RHS of our equation, i.e every point in our cube in a collumn vector
    This is a lot easier to work with than assign face and edge etc values to a collumn vector.

   i is height, j is width and k is depth"""

    N=n-1

    #faces
    for x in range(n):
        for y in range(n):
            b[0, x, y]  = f_top(x,y)
            b[N, x, y] = f_bottom(x,y)
            b[x, 0, y]  = f_left(x,y)
            b[x, N, y] = f_right(x,y)
            b[x, y, 0]  = f_front(x,y)
            b[x, y, N] = f_back(x,y)

    #Edges
    for k in range(n): #depth
        b[0, 0, k] = (f_top(0,k)  + f_left(0,k))  / 2
        b[0, N, k] = (f_top(N,k) + f_right(0,k)) / 2
        b[N, 0, k] = (f_bottom(0,k)  + f_left(N,k))  / 2
        b[N, N, k] = (f_bottom(N,k) + f_right(N,k)) / 2

    for j in range(n): #width
        b[0, j, 0] = (f_top(j,0)  + f_front(0, j)) / 2
        b[0, j, N] = (f_top(j,N) + f_back(0, j))  / 2
        b[N, j, 0] = (f_bottom(j,0)  + f_front(N,j)) / 2
        b[N, j, N] = (f_bottom(j,N) + f_back(N,j))  / 2

    for i in range(n): #height

        b[i, 0, 0] = (f_left(i,0)  + f_front(i,0)) / 2
        b[i, 0, N] = (f_left(i,N) + f_back(i,0))  / 2
        b[i, N, 0] = (f_right(i,0)  + f_front(i,N)) / 2
        b[i, N, N] = (f_right(i,N) + f_back(i,N))  / 2

    #Corners
    b[0, 0, 0]    = (f_top(0, 0)  + f_left(0, 0)  + f_front(0, 0)) / 3
    b[0, N, 0]   = (f_top(N, 0) + f_right(0, 0) + f_front(0, N)) / 3
    b[N, 0, 0]   = (f_bottom(0, 0)  + f_left(N, 0)  + f_front(N, 0)) / 3
    b[N, N, 0]  = (f_bottom(N, 0) + f_right(N, 0) + f_front(N,N)) / 3
    b[0, 0, N]   = (f_top(0, N)  + f_left(0, N)  + f_back(0, 0)) / 3
    b[0, N, N]  = (f_top(N, N) + f_right(0, N) + f_back(0, N)) / 3
    b[N, 0, N]  = (f_bottom(0, N)  + f_left(N, N)  + f_back(N, 0)) / 3
    b[N, N, N] = (f_bottom(N, N) + f_right(N, N) + f_back(N, N)) / 3

    b = b.flatten()
    
    A=np.zeros((n**3, n**3)) #Making the LHS, each row is a flattened 11x11 grid for easy workings

    for i in range(1,N): #For the 'inside' points, using the fact that a points equals the mean of its 4 neighbors
        for j in range(1,N):
            for k in range(1,N):
                a=np.zeros((n,n,n))
                a[i][j][k]=6

                a[i-1][j][k]=-1
                a[i+1][j][k]=-1

                a[i][j-1][k]=-1
                a[i][j+1][k]=-1

                a[i][j][k-1]=-1
                a[i][j][k+1]=-1

                A[i*(n**2)+j*n+k] = a.flatten()
    
    for i in range(n):
        for j in range(n):
            for k in range(n):
                if i == 0 or i == N or j == 0 or j == N or k==0 or k==N:
                    A[i*(n**2)+j*n+k, i*(n**2)+j*n+k] = 1
    
    x=scipy.sparse.linalg.spsolve(A,b)
    x=x.reshape((n,n,n))
    return x

#====================================================
# 4: Cube (variable face temps), variable sized cube, functions centered at the origin
#====================================================

def temp_analysis_cube_4(f_left, f_top, f_right, f_bottom, f_front, f_back, n=11, f_bounds=10, zoom=0):
    b = np.zeros((n, n, n))
    """B is an 11x11x11 cube which we manually assign the known elements, ie faces, edges and corners.
    We then flatten it to geth the RHS of our equation, i.e every point in our cube in a collumn vector
    This is a lot easier to work with than assign face and edge etc values to a collumn vector.

    i is height, j is width and k is depth"""

    N=n-1

    #faces
    a=np.linspace(-abs(f_bounds), abs(f_bounds), n)
    for x_index, x in enumerate(a):
        for y_index, y in enumerate(a):
            b[0, x_index, y_index]  = f_top(x,y)
            b[N, x_index, y_index] = f_bottom(x,y)
            b[x_index, 0, y_index]  = f_left(x,y)
            b[x_index, N, y_index] = f_right(x,y)
            b[x_index, y_index, 0]  = f_front(x,y)
            b[x_index, y_index, N] = f_back(x,y)

    #Edges
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

    #Corners
    b[0, 0, 0]    = (f_top(a[0], a[0])  + f_left(a[0], a[0])  + f_front(a[0], a[0])) / 3
    b[0, N, 0]   = (f_top(a[N], a[0]) + f_right(a[0], a[0]) + f_front(a[0], a[N])) / 3
    b[N, 0, 0]   = (f_bottom(a[0], a[0])  + f_left(a[N], a[0])  + f_front(a[N], a[0])) / 3
    b[N, N, 0]  = (f_bottom(a[N], a[0]) + f_right(a[N], a[0]) + f_front(a[N], a[N])) / 3
    b[0, 0, N]   = (f_top(a[0], a[N])  + f_left(a[0], a[N])  + f_back(a[0], a[0])) / 3
    b[0, N, N]  = (f_top(a[N], a[N]) + f_right(a[0], a[N]) + f_back(a[0], a[N])) / 3
    b[N, 0, N]  = (f_bottom(a[0], a[N])  + f_left(a[N], a[N])  + f_back(a[N], a[0])) / 3
    b[N, N, N] = (f_bottom(a[N], a[N]) + f_right(a[N], a[N]) + f_back(a[N], a[N])) / 3

    b = b.flatten()
    
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
    
    A_sparse = scipy.sparse.csr_matrix(A)
    X = scipy.sparse.linalg.spsolve(A_sparse, b)
    X=X.reshape((n,n,n))

    if zoom==0:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        x, y, z = np.meshgrid(range(n), range(n), range(n))
        sc = ax.scatter(x,y,z, c=X.flatten(), cmap='coolwarm', s=100)

        plt.colorbar(sc, label='Temperature')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.title('3D Heat Distribution')
        plt.show()
    elif zoom>=0:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        x, y, z = np.meshgrid(range(n), range(n), range(n), indexing='ij')

        sl = slice(zoom, -zoom)
        x_inner = x[sl, sl, sl]
        y_inner = y[sl, sl, sl]
        z_inner = z[sl, sl, sl]
        X_inner = X[sl, sl, sl]

        sc = ax.scatter(x_inner, y_inner, z_inner, c=X_inner.flatten(), cmap='coolwarm', s=100)
        plt.colorbar(sc, label='Temperature')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.title('3D Heat Distribution')
        plt.show()
    else:
        raise ValueError("Zoom must be positive")

    return X






