# coding: utf-8

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from . import iwlr
from . import pu
from . import rulsif

sns.set_style('ticks')
sns.set_context('paper', font_scale=1.5)


def get_data(pyx, mp, mn, n_p, n_n, I):
    r"""
    Generates the synthetic data

    :param pyx: Labeling function
    :param mp: Mean for the positive class
    :param mn: Mean for the negative class
    :param n_p: Number of positive samples
    :param n_n: Number of negative samples
    :param I: Covariance matrix
    :return:
    """
    x = np.r_[np.random.multivariate_normal(mean=mp, cov=I, size=int(n_p + n_n)),
              np.random.multivariate_normal(mean=mn, cov=I, size=int(n_p + n_n))]
    y = 2 * (pyx(x[:, 0], x[:, 1]) > .5) - 1
    xp = x[y == +1, :]
    xn = x[y == -1, :]
    index_p = np.random.permutation(xp.shape[0])
    index_n = np.random.permutation(xn.shape[0])
    x = np.r_[(xp[index_p, :])[:n_p, :], (xn[index_n, :])[:n_n, :]]
    y = np.r_[np.ones(n_p), -np.ones(n_n)]
    return x, y


def calc_err(yh, yt, prior):
    return prior * np.mean(yh[yt == +1] <= 0) + (1 - prior) * np.mean(yh[yt == -1] >= 0)


def _main():
    #    np.random.seed(0)
    np.random.seed(4)

    os.makedirs('figs', exist_ok=True)

    d = 2
    np_tr = 400
    nu_tr = 700
    nu_te = 700

    prior_tr = .5
    prior_te = .5

    # Only a small shift for negative but a large shift for positive
    # Each row is a mean
    mp = np.array([[1, -3],   # POSITIVE train mean
                   [-3, 1]])  # POSITIVE test mean
    mn = np.array([[2, 3],   # NEGATIVE train mean
                   [2, 2]])  # NEGATIVE test mean

    a = 4
    pyx = lambda x1, x2: (1 - (np.tanh(x1 + a * x2) + 1) * (np.tanh(x2 + a * x1) + 1) / 4)

    s = 2  # Variance of distribution
    Itr = s * np.eye(2)
    Ite = s * np.eye(2)

    # positive data
    x_tr, y_tr = get_data(pyx, mp[0, :], mn[0, :], np_tr, np_tr, Itr)
    xp_tr = x_tr[y_tr == +1, :]
    xn_tr = x_tr[y_tr == -1, :]

    # unlabeled data for training
    nu_tr_p = np.random.binomial(nu_tr, prior_tr)  # Randomly generate # positive in TRAIN
    nu_tr_n = nu_tr - nu_tr_p
    xu_tr, yu_tr = get_data(pyx, mp[0, :], mn[0, :], nu_tr_p, nu_tr_n, Itr)

    # unlabeled data for test
    nu_te_p = np.random.binomial(nu_tr, prior_te)  # Randomly generate # positive in TEST
    nu_te_n = nu_te - nu_te_p
    xu_te, yu_te = get_data(pyx, mp[1, :], mn[1, :], nu_te_p, nu_te_n, Ite)

    dec_x = np.linspace(-8, 8, num=100)
    dec_y = np.maximum(-a * dec_x, -dec_x / 4)

    priorh = prior_tr
    print("priorh: {:3f}".format(priorh))

    gamma_list = np.mgrid[.1:.9:9j]
    lambda_list = np.logspace(-3, 1, 11)

    # beta_list = np.mgrid[.1:.9:9j]

    # =========================================================================== #
    #                               PUc Algorithm                                 #
    # =========================================================================== #

    pucm = pu.linsq_puc_tr_rulsif_simple(xp_tr, xu_tr, xu_te, priorh,
                                         lambda_list=lambda_list,
                                         gamma_list=gamma_list,
                                         bias=False)
    yh = pu.predict(pucm, xu_te)
    print("puc-err: {}".format(calc_err(yh, yu_te, prior_te)))
    w_puc = pucm['alpha']
    wm_puc = pucm['wm']
    print("puc: ", w_puc)
    w1_puc = w_puc[0]
    w2_puc = w_puc[1]
    b_puc = 0  # w_puc[2]
    dec_yh_puc = -w1_puc * dec_x / w2_puc - b_puc / w2_puc

    # =========================================================================== #
    #                            Standard PU Learning                             #
    # =========================================================================== #

    pum = pu.sq_pu(xp_tr, xu_tr, priorh, kertype='linear',
                   lambda_list=lambda_list, bias=False)
    w_pu = pum['alpha']
    print("pu :", w_pu)
    yh = pu.predict(pum, xu_te)
    print("pu-err: {}".format(calc_err(yh, yu_te, prior_te)))
    w1_pu = w_pu[0]
    w2_pu = w_pu[1]
    b_pu = 0  # w_pu[2]
    dec_yh_pu = -w1_pu * dec_x / w2_pu - b_pu / w2_pu

    # =========================================================================== #
    #                     PNU Learning with Covariate Shift                       #
    # =========================================================================== #

    pnm = iwlr.linsq_rulsif_simple(xp_tr, xn_tr, xu_tr, xu_te, prior=priorh,
                                   lambda_list=lambda_list,
                                   gamma_list=gamma_list, bias=False)
    w_pn = pnm['alpha']
    print("PN: ", w_pn)
    yh = iwlr.predict(pnm, xu_te)
    print("PN-err: {}".format(calc_err(yh, yu_te, prior_te)))
    w1_pn = w_pn[0]
    w2_pn = w_pn[1]
    b_pn = 0  # w_pn[2]
    dec_yh_pn = -w1_pn * dec_x / w2_pn - b_pn / w2_pn

    plt.close()
    fig1 = plt.figure(1)
    wph_vis = rulsif.est_w(xp_tr, wm_puc)
    h_ute = plt.scatter(xu_te[:, 0], xu_te[:, 1], marker='.', c='k', alpha=.8)
    thresh = (wph_vis > 2.).squeeze()
    h_wph = plt.scatter(xp_tr[thresh, 0], xp_tr[thresh, 1], marker='^', lw=1.5,
                        alpha=.8, facecolors='None', edgecolors='blue', s=80)
    h_true, = plt.plot(dec_x, dec_y, label='True', ls='-', lw=4, c='k')
    plt.xlim(-8.2, 8.2)
    plt.ylim(-8.2, 8.2)

    plt.legend([h_wph, h_ute], ['Ptr with large weight', 'Ute'])
    plt.xlabel('$x^{(1)}$')
    plt.ylabel('$x^{(2)}$')

    sns.despine()
    plt.savefig('./figs/demo_pu_iwlinear_syndata_wph.png', bbox_inches="tight")
    plt.savefig('./figs/demo_pu_iwlinear_syndata_wph.pdf', bbox_inches="tight")
    plt.close()

    fig1 = plt.figure(1)
    h_ptr = plt.scatter(x_tr[y_tr == +1, 0], x_tr[y_tr == +1, 1], marker='^', lw=1,
                        alpha=.8, facecolors='None', edgecolors='blue', label='pos-tr')
    h_ntr = plt.scatter(x_tr[y_tr == -1, 0], x_tr[y_tr == -1, 1], marker='s', lw=1,
                        alpha=.8, facecolors='None', edgecolors='red', label='neg-tr')
    h_pte = plt.scatter(xu_te[yu_te == +1, 0], xu_te[yu_te == +1, 1], marker='o', lw=1,
                        alpha=.8, facecolors='None', edgecolors='blue', label='pos-te')
    h_nte = plt.scatter(xu_te[yu_te == -1, 0], xu_te[yu_te == -1, 1], marker='x', lw=1,
                        alpha=.8, facecolors='red', edgecolors='red', label='neg-te')

    plt.xlim(-8.2, 8.2)
    plt.ylim(-8.2, 8.2)

    plt.legend([h_ptr, h_ntr, h_pte, h_nte], ['Ptr', 'Ntr', 'Pte', 'Nte'])
    plt.xlabel('$x^{(1)}$')
    plt.ylabel('$x^{(2)}$')

    sns.despine()  # Removes top and right spines from graph (i.e., top and right lines along axes)
    plt.savefig('./figs/demo_pu_iwlinear_syndata_te.png', bbox_inches="tight")
    plt.savefig('./figs/demo_pu_iwlinear_syndata_te.pdf', bbox_inches="tight")

    fig1 = plt.figure(1)
    plt.clf()
    plt.scatter(xu_te[:, 0], xu_te[:, 1], marker='.', c='k', alpha=.8)
    plt.scatter(xu_tr[:, 0], xu_tr[:, 1], marker='+', c='k', alpha=.8)
    h_ptr = plt.scatter(x_tr[y_tr == +1, 0], x_tr[y_tr == +1, 1], marker='^', lw=1,
                        alpha=.8, facecolors='None', edgecolors='blue', label='pos-tr')

    h_true, = plt.plot(dec_x, dec_y, label='True', ls='-', lw=3, c='k')
    h_pn, = plt.plot(dec_x, dec_yh_pn, label='PNc', ls='--', lw=4, c='orange')
    h_pu, = plt.plot(dec_x, dec_yh_pu, label='pu', ls=':', lw=6, c='g')
    h_puc, = plt.plot(dec_x, dec_yh_puc, label='puc', ls='-.', lw=4, c='r')

    plt.xlim(-8.2, 8.2)
    plt.ylim(-8.2, 8.2)

    plt.legend([h_true, h_pn, h_pu, h_puc], ['True', 'PN', 'pu', 'puc'],
               handlelength=4.25)
    plt.xlabel('$x^{(1)}$')
    plt.ylabel('$x^{(2)}$')

    sns.despine()
    plt.savefig('./figs/demo_pu_iwlinear_syndata_tr.png', bbox_inches="tight")
    plt.savefig('./figs/demo_pu_iwlinear_syndata_tr.pdf', bbox_inches="tight")


if __name__ == "__main__":
    _main()
