#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
    """
    Radio espectral
    M: Matriz cuadrada
    
    Devuelve: radio espectral de M
    """
    if not M.is_square():
        raise ValueError("M debe ser cuadrada")
    return max([abs(x) for x in M.eigenvalues()])


# In[6]:


def metodo_del_remonte(A, b, espacio=RR):
    """
    Metodo del remonte:
    A: matriz triangular superior
    b: termino independiente
    espacio: espacio del vector solucion, por defecto R
    
    Te devuelve el vector solucion de un sistema lineal con matriz del sistema traingular superior.
    """
    if not A.is_square():
        raise ValueError("La matriz A tiene que ser cuadrada")
    
    if b.column().nrows() != A.nrows():
        raise ValueError("El vector b tiene que ser del mismo tamano que la matriz A")
        
    n = A.nrows()
    u = vector(espacio,[0]*n)
    
    for i in range(n-1,-1,-1):
        u[i] = 1/A[i][i] * (b[i] - sum([A[i][j]*u[j] for j in range(i+1,n)]))
    
    return u


# In[7]:


def eliminacion_gaussiana(A,b):
    """
    Eliminacion Gaussiana:
    A: matriz cualquiera
    b: termino independiente
    
    Devuelve: Matriz escalonada y el vector con sus respectivas operaciones aplicadas
    """
    n,m = A.dimensions()
    for paso in range(0,n-1):
        pivotaje(A,paso)
        for fila in range(paso+1,n):
            A[fila] = A[fila] - (A[fila,paso]/A[paso,paso]) * A[paso]
            b[fila] = b[fila] - (A[fila,paso]/A[paso,paso]) * b[paso]
            
    return A,b


# In[8]:


def fact_doolittle(A,espacio=RR):
    """
    Factorizacion de Doolittle
    A: Matriz a factorizar, cuadrada
    espacio: Espacio de las matrices resultantes, por defecto, QQ
    
    Devuelve: Matrices P,L,U tal que P*A = L*U, L trian. inf., U trian. sup. y diag(L) = {1,...,1}
    """
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


# In[9]:


def fact_crout(A,espacio=RR):
    """
    Factorizacion de Crout
    A: Matriz a factorizar, cuadrada
    espacio: Espacio de las matrices resultantes, por defecto, QQ
    
    Devuelve: Matrices P,L,U tal que P*A = L*U, L trian. inf., U trian. sup. y diag(U) = {1,...,1}
    """
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


# In[10]:


def fact_cholesky(A,espacio=RR):
    """
    Factorizacion de Cholesky
    A: Matriz a factorizar, cuadrada, hermitiana y definida positiva
    espacio: Espacio de las matrices resultantes, por defecto, QQ
    
    Devuelve: Matrices P,B,B^T tal que P*A = B*B^T, con B trian. inf.
    """
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


# In[11]:


def separacion_DEF(A,espacio=QQ):
    """
    Separacion DEF
    A: Matriz cuadrada para separar
    espacio: Espacio de las matrices resultantes, por defecto, QQ
    
    Devuelve: Matrices D,E y F tal que A = D-E-F, con D diag., E trian. inf., F trian. sup.
    """
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
    


# In[12]:


def jacobi_matriz(A,espacio=QQ):
    """
    Matriz de Jacobi
    A: Matriz cuadrada
    
    Devuelve: matriz asociada al metodo de Jacobi
    """
    D,E,F = separacion_DEF(A,espacio)
    return D.inverse() * (E+F)


# In[13]:


def jacobi_iter(A,b,ini,iters,espacio=QQ):
    """
    Metodo recursivo de Jacobi
    A: Matriz cuadrada del sistema lineal
    b: Vector de terminos independientes del sistema
    ini: Primer vector de aproximacion para el metodo
    iters: Numero de iteraciones a realizar
    espacio: Espacio del vector aproximacion resultante, por defecto, QQ
    
    Devuelve: Vector aproximacion segun el metodo
    """
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


# In[14]:


def gauss_seidel_matriz(A,espacio=QQ):
    """
    Matriz de Gauss-Seidel
    A: Matriz cuadrada
    
    Devuelve: matriz asociada al metodo de Gauss-Seidel
    """
    D,E,F = separacion_DEF(A,espacio)
    return (D-E).inverse() * F


# In[15]:


def gauss_seidel_iter(A,b,ini,iters,espacio=QQ):
    """
    Metodo recursivo de Gauss-Seidel
    A: Matriz cuadrada del sistema lineal
    b: Vector de terminos independientes del sistema
    ini: Primer vector de aproximacion para el metodo
    iters: Numero de iteraciones a realizar
    espacio: Espacio del vector aproximacion resultante, por defecto, QQ
    
    Devuelve: vector aproximacion segun el metodo
    """
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


# In[16]:


def sor_matriz(A,w,espacio=QQ):
    """
    Matriz de relajacion
    A: Matriz cuadrada
    w: Parametro de relajacion
    
    Devuelve: matriz asociada al metodo de relajacion, con el parametro indicado
    """
    D,E,F = separacion_DEF(A,espacio)
    return (w^(-1)*D - E).inverse() * (F + (w^(-1) - 1)*D)


# In[17]:


def sor_iter(A,b,ini,iters,w,espacio=QQ):
    """
    Metodo recursivo de relajacion
    A: Matriz cuadrada del sistema lineal
    b: Vector de terminos independientes del sistema
    ini: Primer vector de aproximacion para el metodo
    iters: Numero de iteraciones a realizar
    w: Parametro de relajacion
    espacio: Espacio del vector aproximacion resultante, por defecto, QQ
    
    Devuelve: vector aproximacion segun el metodo
    """
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


# In[18]:


def parametro_relajacion_optimo(A):
    """
    Parametro optimo para el metodo de relajacion SOR
    A: Matriz cuadrada, hermitiana, y definida positiva tridiagonal
    
    Devuelve: Parametro de relajacion optimo para la matriz A
    """
    if not A.is_square():
        raise ValueError("A debe ser cuadrada")
        
    if not A.is_hermitian():
        raise ValueError("A debe ser hermitiana")
        
    if not A.is_positive_definite():
        raise ValueError("A debe ser definida positiva")
        
    if not tridiagonal(A):
        raise ValueError("A no es tridiagonal")
        
    return 2 / (1 + sqrt(1 - radio_espectral(gauss_seidel_matriz(A))))


# In[19]:


def acotacion_error_matrices_iter(B,TOL,aprox0,aprox1):
    """
    Acotacion del error en metodos iterativos para stms. lin.
    B: Matriz del metodo iterativo (J en Jacobi, L1 en G-S,...)
    TOL: Tolerancia exigida
    aprox0: Valor inicial de la aproximacion en el metodo
    aprox1: Primer valor obtenido en la iteracion del metodo
    
    Devuelve: Numero de iteraciones necesarias para obtener la tolerancia exigida
    """
    bn = B.norm(Infinity)
    un = (aprox1 - aprox0).norm(Infinity)
    k = 0
    while bn^k / (1 - bn) * un >= 10^(-TOL):
        k += 1
    return k


# In[20]:


def metodo_biseccion(f,a,b,prec=4):
    """
    Metodo de biseccion:
    f: Funcion continua
    a,b: Extremos del intervalo que contiene una raiz
    prec: Precision en digitos decimales del intervalo resultante
    
    Devuelve: Intervalo que contiene una raiz, con una apertura menor que 10^-prec
    """
    if f(a)*f(b) >= 0:
        raise ValueError("Los extremos de f no tienen distinto signo")
        
    m = [a]
    n = [b]
    while abs(n[-1] - m[-1]) > 10**(-prec):
        medio = (m[-1]+n[-1])/2
        if f(m[-1]) * f(medio) < 0:
            m.append(m[-1])
            n.append(medio)
        else:
            m.append(medio)
            n.append(n[-1])
    
    return (N(m[-1]),N(n[-1]))


# In[21]:


def metodo_regla_falsa(f,a,b,prec=3):
    """
    Metodo de la regla falsa:
    f: Funcion continua
    a,b: Extremos del intervalo que contiene una raiz
    prec: Precision en digitos decimales del intervalo resultante
    
    Devuelve: Intervalo que contiene una raiz, con una apertura menor que 10^-prec
    """
    if f(a)*f(b) >= 0:
        raise ValueError("Los extremos a y b deben tener distinto signo")
    
    while abs(b - a) > 10^(-prec):
        c = (f(b)*a - f(a)*b)/(f(b) - f(a))
        if f(a) * f(c) < 0:
            b = c
        else:
            a = c
    return N(a),N(b)


# In[29]:


def metodo_punto_fijo(f,ini,iters):
    """
    Metodo del punto fijo
    f: funcion de la que se desea el punto fijo
    ini: valor inicial
    iters: cantidad de iteraciones
    
    Devuelve: aproximaciones iterativas del punto fijo
    """
    for k in range(iters):
        ini = N(f(ini))
        print(f"Ciclo {k}: x = {ini}")


# In[54]:


def metodo_newton(f,ini,iters):
    """
    Metodo de Newton
    f: funcion para buscar una raiz
    
    Devuelve: aproximaciones mediante el metodo de Newton
    """ 
    
    metodo_punto_fijo(x - (f(x)/derivative(f,x)),ini,iters)


# In[55]:


def ayuda():
    """
    Funcion de ayuda
    Para que cojones miras la ayuda de la funcion de ayuda
    """
    print("""
    Para obtener informacion sobre una funcion, usa help(<funcion>)
    Funciones implementadas:
    radio_espectral(M)
    metodo_del_remonte(A, b, espacio=RR)
    eliminacion_gaussiana(A,b)
    fact_doolittle(A,espacio=RR)
    fact_crout(A,espacio=RR)
    fact_cholesky(A,espacio=RR)
    separacion_DEF(A,espacio=QQ)
    jacobi_matriz(A,espacio=QQ)
    jacobi_iter(A,b,ini,iters,espacio=QQ)
    gauss_seidel_matriz(A,espacio=QQ)
    gauss_seidel_iter(A,b,ini,iters,espacio=QQ)
    sor_matriz(A,w,espacio=QQ)
    sor_iter(A,b,ini,iters,w,espacio=QQ)
    parametro_relajacion_optimo(A)
    acotacion_error_matrices_iter(B,TOL,aprox0,aprox1)
    metodo_biseccion(f,a,b,prec=4)
    metodo_regla_falsa(f,a,b,prec=3)
    metodo_punto_fijo(f,ini,iters)
    metodo_newton(f,ini,iters)
    ayuda()
    """)

