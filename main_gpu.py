import argparse
import datetime
import pandas as pd
import random
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import time
import torch
import os


def svt(mat, tau):
    U, S, V = torch.linalg.svd(mat, full_matrices=False)
    num = torch.sum(S > tau)
    out = U[:, :num] @ torch.diag(S[:num]-tau).type(torch.complex64) @ V[:num, :]
    return out


def tensor_svt(Y, rho):

    n1, n2, n3 = Y.shape
    Xf = torch.zeros((n1, n2, n3), dtype=torch.complex64)
    Yf = torch.fft.fft(Y)

    halfn3 = int(np.rint((n3+1)/2))

    for d in range(halfn3):
        Xf[:, :, d] = svt(Yf[:, :, d], rho)

    for d in range(halfn3, n3):
        Xf[:, :, d] = Xf[:, :, n3-d].conj()

    return torch.fft.ifft(Xf).real


def rmse(mat, mat_hat):
    return (((mat-mat_hat)**2).sum() / mat.shape[0]).sqrt()


def mat_times_vec(X, A):

    N, T = X.shape
    zero_col = torch.zeros((N, 1), device=X.device)

    start1 = time.time()
    F1 = torch.concat((zero_col, X[:, 1:]), axis=1)
    F2 = A @ torch.concat((zero_col, X[:, :-1]), axis=1)
    F3 = A.T @ torch.concat((X[:, 1:], zero_col), axis=1)
    F4 = A.T @ A @ torch.concat((X[:, :-1], zero_col), axis=1)
    end1 = time.time()
    # print(end1-start1)

    F = F1 - F2 - F3 + F4
    return torch.reshape(F, (N*T,))


def CG(f, e, G, A):

    max_iter_cg = 20
    N, TD = G.shape
    # TD = T*D
    g = torch.reshape(G, (N*TD,))

    r = g - f
    q = r.clone()

    rtr = torch.inner(r, r)

    for ite in range(max_iter_cg):

        # start = time.time()
        Q = torch.reshape(q, (N, TD))

        Wq = mat_times_vec(Q, A)

        epsilon = rtr / torch.inner(q, Wq)

        e = e + epsilon*q
        r = r - epsilon*Wq

        rtr_new = torch.inner(r, r)

        mu = rtr_new / rtr

        q = r + mu * q

        rtr = rtr_new.clone()

        # end = time.time()
        if torch.sqrt(rtr) < 1e-8:
            break

        # print("Running time:{}".format(end-start))
    return torch.reshape(e, (N, TD))


def tc_var(mat_full, mat_missing, mask, alpha, beta, gamma, rho, max_iter, device):

    mat_full = mat_full.to(device)
    mat_missing = mat_missing.to(device)
    mask = mask.to(device)

    M = mat_missing.clone()
    M0 = mat_missing.clone()
    truth = mat_full.clone()

    # parameters
    T = 288  # timestamp length for one day
    N, TD = M.shape
    D = int(TD/T)
    rho_rate = 1.05
    rho_max = 10
    eps = 1e-2

    # initial
    E = torch.zeros((N, TD), device=device)
    E_tensor = torch.zeros((N, T, D), device=device)
    Lambda = torch.zeros((N, TD), device=device)/rho
    A = torch.eye(N, device=device)
    A_old = A.clone()

    X_old = mat_missing.clone()
    norm0 = torch.linalg.norm(X_old, 'fro')

    # best_RMSE = 1000
    # best_MAPE = 1000

    for outer_loop in range(max_iter):

        start = time.time()

        # ADMM start from here
        for inner_loop in range(20):

            # update X
            temp_tensor = torch.reshape(M - E - Lambda, (N, T, D))

            X_tensor = tensor_svt(temp_tensor, 1/rho)
            X_tensor = torch.FloatTensor(X_tensor).to(device)
            X = torch.reshape(X_tensor, (N, TD))
            M_tensor = torch.reshape(M, (N, T, D))
            Lambda_tensor = torch.reshape(Lambda, (N, T, D))

            # G = rho * (M_tensor - X_tensor - Lambda_tensor)

            for d in range(D):

                # Right part

                G = rho*(M_tensor[:, :, d] - X_tensor[:, :, d] - Lambda_tensor[:, :, d])

                # Left part
                b = mat_times_vec(E_tensor[:, :, d], A)

                e = torch.reshape(E_tensor[:, :, d], (N*T,))

                f = alpha * b + (rho + beta)*e

                # update E
                E_tensor[:, :, d] = CG(f, e, G, A)

            E = torch.reshape(E_tensor, (N, TD))

            # update M
            M = (X + E + Lambda)*~mask + M0

            # update Lambda
            Lambda = Lambda + (X + E - M)

        # update A
        zero_col = torch.zeros((N, 1), device=device)
        Al = alpha * torch.concat((E[:, 1:], zero_col), axis=1) @ E.T
        Ar = torch.linalg.inv(alpha * torch.concat((E[:, :-1], zero_col), axis=1) @ E.T + gamma * torch.eye(N, device=device))

        A = Al @ Ar
        A = A.type(torch.float32)

        tol = torch.linalg.norm((X - X_old), 'fro')/norm0

        X_old = X.clone()

        RMSE = rmse(M[~mask], truth[~mask])
        MAPE = torch.mean(torch.abs(M[~mask] - truth[~mask]) / truth[~mask]) * 100

        # if RMSE < best_RMSE:
        #     best_RMSE = RMSE

        # if MAPE < best_MAPE:
        #     best_MAPE = MAPE

        rho = min(rho*rho_rate, rho_max)

        end = time.time()

        if outer_loop % 1 == 0:
            print("Iteration:{}, RMSE:{:.4}".format(outer_loop+1, RMSE))
            print("Tol:{:.4}".format(tol))
            print("Running time:{:.4}".format(end-start))
            print()

        if (tol < eps) or (outer_loop > max_iter):
            break

    return X, E, M, A, A_old, RMSE.item(), MAPE.item()


if __name__ == "__main__":
    print("Start")
    print(os.getcwd())
    # load traffic speed data
    data = scipy.io.loadmat('./Data/Seattle/speed.mat')
    data = data['data']

    T = 288
    # data = data[:, 4*T:9*T]  # change here for more days

    N, TD = data.shape

    D = int(TD/T)
    data = np.reshape(data, (N, T, D))
    data = np.reshape(data, (N, TD))

    mat_full = data.copy()

    print('The spatial dimension of data(N) is {}, temporal dimension(T) is {} for {} days(D)'.format(N, T, D))

    # generate random missing mask

    # mr_lists = [0.9, 0.6, 0.3]
    alphas = [0.1, 0.5, 1, 1.5, 2]
    betas = [1e-3, 1e-2, 1e-1, 1]
    gammas = [1e-3, 1e-2, 1e-1, 1]
    rho = 1e-4
    max_iter = 30

    # mr_lists = [0.3, 0.6, 0.9]
    result = []

    # for mr in mr_lists:
    # np.random.seed(500)
    # mr = 0.9

    parser = argparse.ArgumentParser()
    parser.add_argument('--mr', type=float, default=0.9)

    args = parser.parse_args()

    mr = args.mr

    mask = np.random.rand(N, TD) > mr
    # mat_missing = mat_full*mask

    # generate column missing
    # random.seed(10)
    # mr = 0.9
    # ncol = int(TD*mr)

    # col_missing = random.sample(range(TD), ncol)

    # mask = np.ones((N, TD))
    # mask[:, col_missing] = np.zeros((N, 1))
    # mask = mask.astype(bool)

    mat_missing = mat_full*mask

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for alpha in alphas:
        for beta in betas:
            for gamma in gammas:
                tick = time.time()
                print(f"mr={mr} alpha={alpha}, beta={beta}, gamma={gamma}")

                mat_missing = torch.FloatTensor(mat_missing)
                mask = torch.BoolTensor(mask)
                mat_full = torch.FloatTensor(mat_full)

                try:
                    X, E, M, A, A_old, RMSE, MAPE = tc_var(mat_full, mat_missing, mask, alpha, beta, gamma, rho, max_iter, device)

                    print("Total running time:{:.4}".format(time.time()-tick))

                    result.append([mr, alpha, beta, gamma, RMSE, MAPE, datetime.datetime.now()])

                    pd.DataFrame(result).to_csv(f"result_{mr}.csv")

                except Exception as e:
                    print(e)
                    print("Total running time:{:.4}".format(time.time()-tick))

                    result.append([mr, alpha, beta, gamma, np.nan, np.nan, datetime.datetime.now()])

                    pd.DataFrame(result).to_csv(f"result_{mr}.csv")

                # save results

    # 323*288
