def tridiagonal(A):
    n = A.nrows()
    D = matrix.diagonal(A.diagonal())
    for i in range(n-1):
        D[i,i+1] = A[i,i+1]
        D[i+1,i] = A[i+1,i]
    return A == D


# In[2]:


def copiar_matriz(A):
    n,m = A.dimensions()
    espacio = A.base_ring()
    Ac = matrix(espacio,n,m,0)
    for i in range(n):
        for j in range(m):
            Ac[i,j] = A[i,j]
        
    return Ac


# In[3]:


def permutar_filas(A,f1,f2):
    n = A.nrows()
    if not f1 in range(0,n) or not f2 in range(0,n):
        raise ValueError("Permutacion no valida")
        
    aux = A[f1]
    A[f1] = A[f2]
    A[f2] = aux


# In[4]:


def pivotaje(A,etapa):
    n = A.nrows()
    if not etapa in range(0,n-1):
        raise ValueError("La etapa no es valida")
    bestrow = 0
    for i in range(etapa,n+1):
        bestrow = i
        if not A[bestrow,etapa] == 0:
            break
    
    if A[bestrow,etapa] == 0:
        return -1
    
    permutar_filas(A,etapa,bestrow)
    
    return bestrow


# In[5]:


def radio_espectral(M):
    return max([abs(x) for x in M.eigenvalues()])


# In[6]:


def converge(A):
    return radio_espectral(A) < 1


# In[7]:


def metodo_del_remonte(A, b, espacio=RR):
    if not A.is_square():
        raise ValueError("La matriz A tiene que ser cuadrada")
    
    if b.column().nrows() != A.nrows():
        raise ValueError("El vector b tiene que ser del mismo tamano que la matriz A")
        
    n = A.nrows()
    u = vector(espacio,[0]*n)
    
    for i in range(n-1,-1,-1):
        u[i] = 1/A[i][i] * (b[i] - sum([A[i][j]*u[j] for j in range(i+1,n)]))
    
    return u


# In[8]:


def eliminacion_gaussiana(A,b):
    n,m = A.dimensions()
    for paso in range(0,n-1):
        pivotaje(A,paso)
        for fila in range(paso+1,n):
            A[fila] = A[fila] - (A[fila,paso]/A[paso,paso]) * A[paso]
            b[fila] = b[fila] - (A[fila,paso]/A[paso,paso]) * b[paso]
            
    return A,b


# In[9]:


def fact_doolittle(A,espacio=RR):
    if not A.is_square():
        raise ValueError("A debe ser cuadrada")
        
    n = A.nrows()
    L = matrix(espacio,n,n,1)
    U = copiar_matriz(A)
    P = matrix(QQ,n,n,1)
    
    for paso in range(0,n-1):
        pivrow = pivotaje(U,paso)
        if pivrow != paso:
            permutar_filas(P,paso,pivrow)
            permutar_filas(L,paso,pivrow)
            L[paso,pivrow] = 0
            L[pivrow,paso] = 0
            L[paso,paso] = 1
            L[pivrow,pivrow] = 1
        
        for fila in range(paso+1,n):
            coef = U[fila,paso]/U[paso,paso]
            U[fila] = U[fila] - coef*U[paso]
            L[fila,paso] = coef
            
    return P,L,U


# In[10]:


def fact_crout(A,espacio=RR):
    if not A.is_square():
        raise ValueError("A debe ser cuadrada")
        
    P,L,U = fact_doolittle(A,espacio)
    
    U_diag = U.diagonal()
    
    if 0 in U_diag:
        raise ValueError("Hay un cero en la diagonal de U") # no se si es posible, no lo quiero saber
    
    D2 = matrix.diagonal(espacio,U_diag)
    
    Lp = L*D2
    Up = D2.inverse()*U
    
    return P,Lp,Up


# In[11]:


def fact_cholesky(A,espacio=RR):
    if not A.is_square():
        raise ValueError("A debe ser cuadrada")
        
    if not A.is_hermitian():
        raise ValueError("A debe ser hermitiana (simetrica en R)")
        
    if not A.is_positive_definite():
        raise ValueError("A debe ser definida positiva")
        
    n = A.nrows()
    P,L,U = fact_doolittle(A,espacio)
    D = matrix.diagonal(espacio,[sqrt(x) for x in U.diagonal()])
    B = L*D
    
    return P,B,B.T


# In[12]:


def separacion_DEF(A:sage.matrix,espacio=QQ):
    if not A.is_square():
        raise ValueError("La matriz no es cuadrada")
        
    n = A.nrows()
    
    D = matrix(espacio,n,n,0)
    E = matrix(espacio,n,n,0)
    F = matrix(espacio,n,n,0)
    
    for i in range(0,n):
        D[i,i] = A[i,i]
        
    for i in range(0,n):
        for j in range(0,n):
            if i > j:
                E[i,j] = -A[i,j]
            elif i < j:
                F[i,j] = -A[i,j]
    
    return D,E,F
    


# In[13]:


def jacobi_matriz(A,espacio=QQ):
    D,E,F = separacion_DEF(A,espacio)
    return D.inverse() * (E+F)


# In[14]:


def jacobi_iter(A,b,ini,iters,espacio=QQ):
    if not A.is_square():
        raise ValueError("La matriz debe ser cuadrada")
    
    n = A.dimensions()[0]
    print("Matriz inicial:")
    show(A)
    print("Termino independiente:")
    show(b)
    
    D,E,F = separacion_DEF(A,espacio)
    
    M = D
    N = E + F
    
    if M.determinant() == 0:
        raise ValueError("D debe de ser inversibles (determ. no nulo)")
    
    M_1 = M.inverse()
    J = M_1*N
    new_b = M_1 * b
    
    print(f"Radio espectral de J: {radio_espectral(J)}")
    
    for k in range(iters):
        ini = J*ini + new_b
        print(f"Ciclo {k}: x = {ini}")


# In[15]:


def gauss_seidel_matriz(A,espacio=QQ):
    D,E,F = separacion_DEF(A,espacio)
    return (D-E).inverse() * F


# In[16]:


def gauss_seidel_iter(A,b,ini,iters,espacio=QQ):
    if not A.is_square():
        raise ValueError("La matriz debe ser cuadrada")
    
    n = A.dimensions()[0]
    print("Matriz inicial:")
    show(A)
    print("Termino independiente:")
    show(b)
    
    D,E,F = separacion_DEF(A,espacio)
    
    M = D - E
    N = F
    
    if M.determinant() == 0:
        raise ValueError("D-E debe de ser inversibles (determ. no nulo)")
    
    M_1 = M.inverse()
    L1 = M_1*N
    new_b = M_1 * b
    
    print(f"Radio espectral de L1: {radio_espectral(L1)}")
    
    for k in range(iters):
        ini = L1*ini + new_b
        print(f"Ciclo {k}: x = {ini}")


# In[22]:


def sor_matriz(A,w,espacio=QQ):
    D,E,F = separacion_DEF(A,espacio)
    return (w^(-1)*D - E).inverse() * (F + (w^(-1) - 1)*D)


# In[23]:


def sor_iter(A,b,ini,iters,w,espacio=QQ):
    if not A.is_square():
        raise ValueError("La matriz debe ser cuadrada")
        
    if not w:
        raise ValueError("w (el factor de relajacion) no puede ser nulo")
    
    n = A.dimensions()[0]
    print("Matriz inicial:")
    show(A)
    print("Termino independiente:")
    show(b)
    
    D,E,F = separacion_DEF(A,espacio)
    
    M = w^(-1)*D - E
    N = F + (w^(-1) - 1)*D
    
    if M.determinant() == 0:
        raise ValueError("w^-1 * D - E debe de ser inversibles (determ. no nulo)")
    
    M_1 = M.inverse()
    Lw = M_1*N
    new_b = M_1 * b
    
    print(f"Radio espectral de Lw: {radio_espectral(Lw)}")
    
    for k in range(iters):
        ini = Lw*ini + new_b
        print(f"Ciclo {k}: x = {ini}")


# In[32]:


def parametro_relajacion_optimo(A):
    if not A.is_square():
        raise ValueError("A debe ser cuadrada")
        
    if not A.is_hermitian():
        raise ValueError("A debe ser hermitiana")
        
    if not A.is_positive_definite():
        raise ValueError("A debe ser definida positiva")
        
    if not tridiagonal(A):
        raise ValueError("A no es tridiagonal")
        
    return 2 / (1 + sqrt(1 - radio_espectral(gauss_seidel_matriz(A))))


# In[17]:


def metodo_biseccion(f,a,b,prec=4):
    if f(a)*f(b) >= 0:
        raise ValueError("Los extremos de f no tienen distinto signo")
        
    m = [a]
    n = [b]
    while (n[-1] - m[-1] > 10**(-prec)):
        medio = (m[-1]+n[-1])/2
        if f(m[-1]) * f(medio) < 0:
            m.append(m[-1])
            n.append(medio)
        else:
            m.append(medio)
            n.append(n[-1])
    
    return (N(m[-1]),N(n[-1]))
