# python version is 3.6.1
from enum import Enum

import numpy as np
import scipy as sp
from scipy import sparse

from . import rulsif


def sq_puc_tr_rulsif(xp_tr, xu_tr, xu_te, prior, lambda_list=np.logspace(-3, 0, num=11),
                     gamma_list=None, sigma_list=None, n_fold=5, n_basis=200, kertype='gauss'):
    if gamma_list is None:
        gamma_list = [0.01, .05, .25, .5, .75, .95, .99]
    if isinstance(kertype, Enum):
        kertype = kertype.value

    np_tr, d = xp_tr.shape
    nu_tr = xu_tr.shape[0]
    nu_te = xu_te.shape[0]

    is_sparse = sparse.issparse(xp_tr)

    if kertype == 'gauss':
        b = np.minimum(n_basis, nu_te)
        center_index = np.random.permutation(nu_te)
        xc = xu_te[center_index[:b], :]
        dp = squared_dist(xp_tr, xc)
        du = squared_dist(xu_tr, xc)
        if sigma_list is None:
            med = np.median(du.ravel())
            sigma_list = np.sqrt(med)*np.logspace(-1, 1, num=11)
    else:
        sigma_list = [0]
        b = d + 1
        if is_sparse:
            dp = sparse.hstack((xp_tr, sparse.csr_matrix(np.ones((np_tr, 1)))), format='csr')
            du = sparse.hstack((xu_tr, sparse.csr_matrix(np.ones((nu_tr, 1)))), format='csr')
        else:
            dp = np.c_[xp_tr, np.ones(np_tr)]
            du = np.c_[xu_tr, np.ones(nu_tr)]

    n_gamma, n_sigma, n_lambda = len(gamma_list), len(sigma_list), len(lambda_list)

    mix_rate_list = gamma_list
    if 0 not in mix_rate_list:
        mix_rate_list = np.append(mix_rate_list, 0)
    else:
        raise Exception('exception for now')

    wm = rulsif.rulsif_cv(xu_tr, xu_te, mix_rate_list=mix_rate_list)

    wph_list = {}
    wuh_list = {}
    ite_gam = 0
    for ite_mix in range(len(mix_rate_list)):
        if mix_rate_list[ite_mix] == 0:
            wph0 = np.array(rulsif.est_w(xp_tr, wm[ite_mix])).squeeze()
            wuh0 = np.array(rulsif.est_w(xu_tr, wm[ite_mix])).squeeze()
        else:
            wph_list[ite_gam] = np.array(rulsif.est_w(xp_tr, wm[ite_mix])).squeeze()
            wuh_list[ite_gam] = np.array(rulsif.est_w(xu_tr, wm[ite_mix])).squeeze()
            ite_gam += 1

    cv_index_p_tr = (np.arange(np_tr, dtype=np.int_)*n_fold)//np_tr
    cv_index_p_tr = cv_index_p_tr[np.random.permutation(np_tr)]

    cv_index_u_tr = (np.arange(nu_tr, dtype=np.int_)*n_fold)//nu_tr
    cv_index_u_tr = cv_index_u_tr[np.random.permutation(nu_tr)]


    score_cv_fold = np.zeros((n_gamma, n_sigma, n_lambda, n_fold))
    for ite_fold in range(n_fold):
        dp_tr_cvtr = dp[cv_index_p_tr != ite_fold, :]
        dp_tr_cvte = dp[cv_index_p_tr == ite_fold, :]

        du_tr_cvtr = du[cv_index_u_tr != ite_fold, :]
        du_tr_cvte = du[cv_index_u_tr == ite_fold, :]

        for ite_sigma, sigma in enumerate(sigma_list):
            if kertype == 'gauss':
                Kp_tr_cvtr = np.exp(-dp_tr_cvtr/(2*sigma**2))
                Kp_tr_cvte = np.exp(-dp_tr_cvte/(2*sigma**2))

                Ku_tr_cvtr = np.exp(-du_tr_cvtr/(2*sigma**2))
                Ku_tr_cvte = np.exp(-du_tr_cvte/(2*sigma**2))
            else:
                Kp_tr_cvtr = dp_tr_cvtr
                Kp_tr_cvte = dp_tr_cvte

                Ku_tr_cvtr = du_tr_cvtr
                Ku_tr_cvte = du_tr_cvte

            for ite_gamma in range(n_gamma):
                gamma = gamma_list[ite_gamma]

                wph_tr = (wph_list[ite_gamma])[cv_index_p_tr != ite_fold]
                wph_te = (wph0)[cv_index_p_tr == ite_fold]
                wuh_tr = (wuh_list[ite_gamma])[cv_index_u_tr != ite_fold]
                wuh_te = (wuh0)[cv_index_u_tr == ite_fold]

                Hu = Ku_tr_cvtr.T.dot(np.diag(wuh_tr)).dot(Ku_tr_cvtr)/Ku_tr_cvtr.shape[0]
                hp = prior*wph_tr.dot(Kp_tr_cvtr).T/Kp_tr_cvtr.shape[0]
                hu = wuh_tr.dot(Ku_tr_cvtr).T/Ku_tr_cvtr.shape[0]

                for ite_lambda, lam in enumerate(lambda_list):
                    Reg = lam*np.eye(b)
                    if kertype != 'gauss':
                        Reg[b-1, b-1] = 0
                    alpha_cv = sp.linalg.solve(Hu + Reg, 2*hp - hu)
                    score_cv_fold[ite_gamma, ite_sigma, ite_lambda, ite_fold] \
                        = risk_puc_tr(Kp_tr_cvte, Ku_tr_cvte, alpha_cv, prior, wph_te, wuh_te)

    score_cv = np.mean(score_cv_fold, axis=3)
    tmp = np.argmin(score_cv.ravel())
    tmp = np.unravel_index(tmp, score_cv.shape)
    gamma_index, sigma_index, lambda_index = tmp[0], tmp[1], tmp[2]

    gamma = gamma_list[gamma_index]
    sigma = sigma_list[sigma_index]
    lam = lambda_list[lambda_index]
    print("(gamma, sigma, lambda) = ({:.2f}, {:2f}, {:6f})".format(gamma, sigma, lam))

    if kertype == 'gauss':
        Kp_tr = np.exp(-dp/(2*sigma**2))
        Ku_tr = np.exp(-du/(2*sigma**2))
    else:
        Kp_tr = dp
        Ku_tr = du

    wph = wph_list[gamma_index]
    wuh = wuh_list[gamma_index]

    Reg = lam*np.eye(b)
    if kertype != 'gauss':
        Reg[b-1, b-1] = 0
    Hu = Ku_tr.T.dot(np.diag(wuh)).dot(Ku_tr)/Ku_tr.shape[0]
    hp = prior*wph.dot(Kp_tr).T/Kp_tr.shape[0]
    hu = wuh.dot(Ku_tr).T/Ku_tr.shape[0]
    alpha = sp.linalg.solve(Hu + Reg, 2*hp - hu)

    model = dict()
    model['kertype'] = kertype
    model['gamma'] = gamma
    model['sigma'] = sigma
    model['lambda'] = lam
    model['alpha'] = alpha
    for index, gam in enumerate(mix_rate_list):
        if gam == gamma:
            model['wm'] = wm[index]
            break

    if kertype == 'gauss':
        model['center'] = xc
    else:
        model['bias'] = True

    return model


fit = sq_puc_tr_rulsif


def decision_function(model, x_te):
    if model['kertype'] == 'gauss':
        K = gaussian_kernel(squared_dist(x_te, model['center']), model['sigma'])
    else:
        if model['bias']:
            if sparse.issparse(x_te):
                K = sparse.hstack((x_te, np.ones((x_te.shape[0], 1))), format='csr')
            else:
                K = np.c_[x_te, np.ones(x_te.shape[0])]
        else:
            K = x_te

    return K.dot(model['alpha'])


def risk_puc_tr(Kp, Ku, alpha, prior, wp, wu):
    rp_p = np.mean(wp*(Kp.dot(alpha) <= 0))
    rp_n = np.mean(wp*(Kp.dot(alpha) >= 0))
    ru_n = np.mean(wu*(Ku.dot(alpha) >= 0))
    risk = prior*rp_p + np.maximum(0, ru_n - prior*rp_n)
    return risk


def logilos(m):
    return sp.misc.logsumexp(np.c_[np.zeros(len(m)), -m], axis=1)


def squared_dist(x, c):
    n1 = x.shape[0]
    n2 = c.shape[0]
    if sparse.issparse(x):
        dist2 = x.power(2).sum(axis=1).reshape((n1, 1)) \
                + c.power(2).sum(axis=1).reshape((n2, 1)).T - 2*x.dot(c.T)
    else:
        dist2 = np.sum(x**2, axis=1).reshape((n1, 1)) \
                + np.sum(c**2, axis=1).reshape((n2, 1)).T - 2*x.dot(c.T)
    return dist2


def gaussian_kernel(dist2, sigma):
    return np.exp(-dist2/(2*sigma**2))



