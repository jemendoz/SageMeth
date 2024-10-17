def radio_espectral(M):
    return max([abs(x) for x in M.eigenvalues()])

def converge(A):
    return radio_espectral(A) < 1

def metodo_del_remonte(A, b):
    if A.nrows() != A.ncols():
        raise ValueError("La matriz A tiene que ser cuadrada")
    
    if b.column().nrows() != A.nrows():
        raise ValueError("El vector b tiene que ser del mismo tamano que la matriz A")
        
    n = A.nrows()
    u = vector(SR,[0]*n)
    
    for i in range(n-1,-1,-1):
        u[i] = 1/A[i][i] * (b[i] - sum([A[i][j]*u[j] for j in range(i+1,n)]))
    
    return u

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

def jacobi_iter(A,b,ini,kmax):
    if (A.dimensions()[0] != A.dimensions()[1]):
        print("A no es cuadrada")
        return
    
    n = A.dimensions()[0]
    print("Matriz inicial:")
    show(A)
    print("Termino independiente:")
    show(b)
    D = matrix(RR,n,n,0)
    E = matrix(RR,n,n,0)
    F = matrix(RR,n,n,0)
    for i in range(0,n):
        D[i,i] = A[i,i]
        
    for i in range(0,n):
        for j in range(0,n):
            if i > j:
                E[i,j] = -A[i,j]
            elif i < j:
                F[i,j] = -A[i,j]
                
    D_1 = D.inverse()
    J = D_1*(E+F)
    new_b = D_1 * b
    
    k = 0
    
    while(k < kmax):
        k += 1
        ini = J*ini + new_b
        print(f"Ciclo {k}: x = {ini}")
