#!/usr/bin/env python
import numpy as np
import scipy as sp
from scipy import sparse, optimize, special

from . import rulsif


def linsq_rulsif_simple(xp_tr, xn_tr, xu_tr, xu_te, prior, 
                         lambda_list=np.logspace(-3, 0, num=5), 
                         gamma_list=[.01, .05, .25, .5, .75, .95, 99],
                         n_fold=5, bias=True):
    np_tr, d = xp_tr.shape
    nn_tr = xn_tr.shape[0]

    is_sparse = sparse.issparse(xp_tr)

    if bias:
        d += 1
        if is_sparse:
            dp_tr = sparse.hstack((xp_tr, sparse.csr_matrix(np.ones((np_tr, 1)))), format='csr')
            dn_tr = sparse.hstack((xn_tr, sparse.csr_matrix(np.ones((nn_tr, 1)))), format='csr')
        else:
            dp_tr = np.c_[xp_tr, np.ones(np_tr)]
            dn_tr = np.c_[xn_tr, np.ones(nn_tr)]
    else:
        dp_tr, dn_tr = xp_tr, xn_tr

    n_gamma, n_lambda = len(gamma_list), len(lambda_list)

    cv_index_p_tr = (np.arange(np_tr, dtype=np.int_)*n_fold)//np_tr
    cv_index_p_tr = cv_index_p_tr[np.random.permutation(np_tr)]

    cv_index_n_tr = (np.arange(nn_tr, dtype=np.int_)*n_fold)//nn_tr
    cv_index_n_tr = cv_index_n_tr[np.random.permutation(nn_tr)]

    mix_rate_list = gamma_list
    if 0 not in mix_rate_list:
        mix_rate_list = np.append(mix_rate_list, 0)
    else:
        raise Exception('exception for now')

    wm = rulsif.rulsif_cv(xu_tr, xu_te, mix_rate_list=mix_rate_list)

    wph_list = []
    wnh_list = []
    for ite_mix in range(len(mix_rate_list)):
        if mix_rate_list[ite_mix] == 0:
            wph0 = np.array(rulsif.est_w(xp_tr, wm[ite_mix])).squeeze()
            wnh0 = np.array(rulsif.est_w(xn_tr, wm[ite_mix])).squeeze()
        else:
            wph_list.append(np.array(rulsif.est_w(xp_tr, wm[ite_mix])).squeeze())
            wnh_list.append(np.array(rulsif.est_w(xn_tr, wm[ite_mix])).squeeze())

    score_cv_fold = np.zeros((n_gamma, n_lambda, n_fold))
    for ite_fold in range(n_fold):
        dp_tr_cvtr = dp_tr[cv_index_p_tr != ite_fold, :]
        dp_tr_cvte = dp_tr[cv_index_p_tr == ite_fold, :]

        dn_tr_cvtr = dn_tr[cv_index_n_tr != ite_fold, :]
        dn_tr_cvte = dn_tr[cv_index_n_tr == ite_fold, :]

        wph_te = (wph0)[cv_index_p_tr == ite_fold]
        wnh_te = (wnh0)[cv_index_n_tr == ite_fold]
        for ite_gamma in range(n_gamma):
            gamma = gamma_list[ite_gamma]

            wph_tr = (wph_list[ite_gamma])[cv_index_p_tr != ite_fold]
            wnh_tr = (wnh_list[ite_gamma])[cv_index_n_tr != ite_fold]

            Hp = prior*dp_tr_cvtr.T.dot(np.diag(wph_tr)).dot(dp_tr_cvtr)/dp_tr_cvtr.shape[0]
            Hn = (1-prior)*dn_tr_cvtr.T.dot(np.diag(wnh_tr)).dot(dn_tr_cvtr)/dn_tr_cvtr.shape[0]
            hp = prior*dp_tr_cvtr.T.dot(wph_tr)/dp_tr_cvtr.shape[0]
            hn = (1-prior)*dn_tr_cvtr.T.dot(wnh_tr)/dn_tr_cvtr.shape[0]

#            H = xb_tr.T.dot(np.diag(np.squeeze(w_tr**gamma))).dot(xb_tr)
#            h = xb_tr.T.dot(np.diag(np.squeeze(w_tr**gamma))).dot(y_tr)
            for ite_lambda in range(n_lambda):
                lam = lambda_list[ite_lambda]
                Reg = lam*np.eye(d)
                if bias:
                    Reg[d-1, d-1] = 0
                alpha_cv = sp.linalg.solve(Hp + Hn + Reg, hp - hn)
#                alpha_cv = solve(dp_tr_cvtr, dn_tr_cvtr, prior, lam, wph_tr, wnh_tr)
                score_cv_fold[ite_gamma, ite_lambda, ite_fold] \
                    = risk_pnc(dp_tr_cvte, dn_tr_cvte, alpha_cv, prior, wph_te, wnh_te)

    score_cv = np.mean(score_cv_fold, axis=2)
    tmp = np.argmin(score_cv.ravel())
    tmp = np.unravel_index(tmp, score_cv.shape)
    gamma_index, lambda_index = tmp[0], tmp[1]

    gamma = gamma_list[gamma_index]
    lam = lambda_list[lambda_index]
    print("(gamma, lambda) = ({:.2f}, {:6f})".format(gamma, lam))

    wph = wph_list[gamma_index]
    wnh = wnh_list[gamma_index]

    Hp = prior*dp_tr.T.dot(np.diag(wph)).dot(dp_tr)/dp_tr.shape[0]
    Hn = (1-prior)*dn_tr.T.dot(np.diag(wnh)).dot(dn_tr)/dn_tr.shape[0]
    hp = prior*dp_tr.T.dot(wph)/dp_tr.shape[0]
    hn = (1-prior)*dn_tr.T.dot(wnh)/dn_tr.shape[0]
    Reg = lam*np.eye(d)
    if bias:
        Reg[d-1, d-1] = 0
    
#    print("pnc")
#    print(Hp+Hn)
#    print(hp-hn)

    alpha = sp.linalg.solve(Hp + Hn + Reg, hp - hn)
#    alpha = solve(dp_tr, dn_tr, prior, lam, wph, wnh)

    model = dict()
    model['kertype'] = 'linear'
    model['gamma'] = gamma
    model['lambda'] = lam
    model['alpha'] = alpha
    model['bias'] = bias
#    if kertype == 'gauss':
#        model['center'] = xc

    return model


def predict(model, x_te, quiet=False):
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


def risk_pnc(Kp, Kn, alpha, prior, wp, wn):
    rp = np.mean(wp*(Kp.dot(alpha) <= 0))
    rn = np.mean(wn*(Kn.dot(alpha) >= 0))
    return prior*rp + (1-prior)*rn


def risk_pn(Kp, Kn, alpha, prior):
    rp = np.mean(Kp.dot(alpha) <= 0)
    rn = np.mean(Kn.dot(alpha) >= 0)
    return prior*rp + (1-prior)*rn


def solve(Kp, Kn, prior, lam, wp, wn, alpha0=None, max_itr=200):
    def f(alpha, *args):
        rp = np.mean(wp*logilos(Kp.dot(alpha)))
        rn = np.mean(wn*logilos(-Kn.dot(alpha)))
        return prior*rp + (1-prior)*rn + lam*alpha.dot(alpha)/2


    def grad(alpha, *args):
        gp, gn = Kp.dot(alpha), Kn.dot(alpha)

        gradp = -Kp.T.dot(wp*special.expit(-gp))/gp.shape[0]
        gradn = Kn.T.dot(wn*special.expit(gn))/gn.shape[0]

        return prior*gradp + (1-prior)*gradn + lam*alpha

    if alpha0 is None:
        alpha0 = np.random.randn(Kp.shape[1])

    ret = optimize.fmin_cg(f, alpha0, fprime=grad, disp=False)
    return ret


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


def est_y(xte, model):
        alphah = model['alpha']
        nte    = xte.shape[0]
        return np.c_[xte, np.ones((nte, 1))].dot(alphah)

