
import random
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import time


def svt(mat, tau):

    U, S, V = np.linalg.svd(mat, full_matrices=False)
    num = np.sum(S > tau)
    return U[:, :num] @ np.diag(S[:num]-tau) @ V[:num, :]


def tensor_svt(Y, rho):

    # Initial
    n1, n2, n3 = np.shape(Y)
    Xf = np.zeros((n1, n2, n3), dtype='complex_')
    Yf = np.fft.fft(Y)

    halfn3 = int(np.rint((n3+1)/2))

    for d in range(halfn3):
        Xf[:, :, d] = svt(Yf[:, :, d], rho)

    for d in range(halfn3, n3):
        Xf[:, :, d] = Xf[:, :, n3-d].conjugate()

    return np.fft.ifft(Xf).real


def rmse(mat, mat_hat):
    return np.sqrt(np.sum((mat - mat_hat)**2)/mat.shape[0])


def mat_times_vec(X, A):

    N = X.shape[0]
    zero_col = np.zeros((N, 1))

    start1 = time.time()
    F1 = np.concatenate((zero_col, X[:, 1:]), axis=1)
    F2 = A @ np.concatenate((zero_col, X[:, :-1]), axis=1)
    F3 = A.transpose() @ np.concatenate((X[:, 1:], zero_col), axis=1)
    F4 = A.transpose() @ A @ np.concatenate((X[:, :-1], zero_col), axis=1)
    end1 = time.time()
    # print(end1-start1)

    F = F1 - F2 - F3 + F4
    return np.reshape(F, -1, order='F')  # fortran style order


def CG(f, e, G, A):

    max_iter_cg = 20
    N, TD = G.shape
    # TD = T*D
    g = np.reshape(G, -1, order='F')

    r = g - f
    q = r.copy()

    rtr = np.inner(r, r)

    for ite in range(max_iter_cg):

        # start = time.time()
        Q = np.reshape(q, (N, TD), order='F')

        Wq = mat_times_vec(Q, A)

        epsilon = rtr / np.inner(q, Wq)

        e = e + epsilon*q
        r = r - epsilon*Wq

        rtr_new = np.inner(r, r)

        mu = rtr_new / rtr

        q = r + mu * q

        rtr = rtr_new.copy()

        # end = time.time()
        if np.sqrt(rtr) < 1e-8:
            break

        # print("Running time:{}".format(end-start))
    return np.reshape(e, (N, TD), order='F')


def tc_var(mat_full, mat_missing, mask, alpha, beta, gamma, rho, max_iter):

    M = mat_missing.copy()
    M0 = mat_missing.copy()
    truth = mat_full.copy()

    # parameters
    T = 288  # timestamp length for one day
    N, TD = M.shape
    D = int(TD/T)
    rho_rate = 1.05
    rho_max = 10
    eps = 1e-2

    # initial
    E = np.zeros((N, TD))
    E_tensor = np.zeros((N, T, D))
    Lambda = np.zeros((N, TD))/rho
#     A = np.diag(np.random.randn(N))
#     A = np.random.randn(N,N)
    A = np.identity(N)
    A_old = A.copy()

    X_old = mat_missing.copy()
    norm0 = np.linalg.norm(X_old, 'fro')

    for outer_loop in range(max_iter):

        start = time.time()

        # ADMM start from here
        for inner_loop in range(20):

            # update X
            temp_tensor = np.reshape(M - E - Lambda, (N, T, D), order='F')
            X_tensor = tensor_svt(temp_tensor, 1/rho)
            X = np.reshape(X_tensor, (N, TD), order='F')
            M_tensor = np.reshape(M, (N, T, D), order='F')
            Lambda_tensor = np.reshape(Lambda, (N, T, D), order='F')

            for d in range(D):

                # Right part

                G = rho*(M_tensor[:, :, d] - X_tensor[:, :, d] - Lambda_tensor[:, :, d])

                # Left part
                b = mat_times_vec(E_tensor[:, :, d], A)

                e = np.reshape(E_tensor[:, :, d], -1, order='F')

                f = alpha * b + (rho + beta)*e

                # update E
                E_tensor[:, :, d] = CG(f, e, G, A)

            E = np.reshape(E_tensor, (N, TD), order='F')

            # update M
            M = (X + E + Lambda)*~mask + M0

            # update Lambda
            Lambda = Lambda + (X + E - M)

        # update A
        zero_col = np.zeros((N, 1))
        Al = alpha * np.concatenate((E[:, 1:], zero_col), axis=1) @ E.transpose()
        Ar = np.linalg.inv(alpha * np.concatenate((E[:, :-1], zero_col), axis=1) @ E.transpose() + gamma * np.identity(N))

        A = Al @ Ar

        tol = np.linalg.norm((X - X_old), 'fro')/norm0

        X_old = X.copy()

        RMSE = rmse(M[~mask], truth[~mask])

        rho = min(rho*rho_rate, rho_max)

        end = time.time()

        if outer_loop % 1 == 0:
            print("Iteration:{}, RMSE:{:.4}".format(outer_loop+1, RMSE))
            print("Tol:{:.4}".format(tol))
            print("Running time:{:.4}".format(end-start))
            print()

        if (tol < eps) or (outer_loop > max_iter):
            break

    return X, E, M, A, A_old


# load traffic speed data
data = scipy.io.loadmat('Data/Seattle/speed.mat')
data = data['data']

T = 288
data = data[:, 4*T:9*T]  # change here for more days

N, TD = data.shape

D = int(TD/T)
mat_full = data.copy()

print('The spatial dimension of data(N) is {}, temporal dimension(T) is {} for {} days(D)'.format(N, T, D))


# generate random missing mask

# np.random.seed(500)
# mr = 0.9
# mask = np.random.rand(N,TD) > mr
# mat_missing = mat_full*mask


# generate column missing
random.seed(10)
mr = 0.3
ncol = int(TD*mr)

col_missing = random.sample(range(TD), ncol)

mask = np.ones((N, TD))
mask[:, col_missing] = np.zeros((N, 1))
mask = mask.astype(bool)

mat_missing = mat_full*mask


rho = 1e-4
max_iter = 10

alphas = [0.8]
betas = [0.001]
gammas = [1e-3]

for alpha in alphas:
    for beta in betas:
        for gamma in gammas:
            print("alpha={}, beta={}, gamma={}\n".format(alpha, beta, gamma))

            X, E, M, A, A_old = tc_var(mat_full, mat_missing, mask, alpha, beta, gamma, rho, max_iter)


# 323*288
