#!/usr/bin/env python
#-*- coding: utf-8 -*-

import numpy as np
import scipy as sp
from scipy import sparse


def rulsif_cv(xde, xnu, xc=None, sigma_list=None, mix_rate_list=[.5],
              lambda_list=np.logspace(-3, 0, num=11), n_basis=200, n_fold=5):
        nde = xde.shape[0]
        nnu = xnu.shape[0]

        if xc is None:
                n_basis = np.minimum(n_basis, nnu)
                center_index = np.random.permutation(nnu)
                xc = xnu[center_index[0:n_basis], :]
        
        dist2de = squared_dist(xde, xc)
        dist2nu = squared_dist(xnu, xc)

        if sigma_list is None:
                if sparse.issparse(xde):
                        med = np.median(dist2nu.tolist())
                else:
                        med = np.median(dist2nu.ravel())
                sigma_list = np.sqrt(med)*np.logspace(-1, 1, num=11)

        n_sigma  = len(sigma_list)
        n_lambda = len(lambda_list)
        n_mix    = len(mix_rate_list)

        cv_index_de = (np.arange(nde, dtype=np.int32)*n_fold)//nde
        cv_index_de = cv_index_de[np.random.permutation(nde)]
        
        cv_index_nu = (np.arange(nnu, dtype=np.int32)*n_fold)//nnu      
        cv_index_nu = cv_index_nu[np.random.permutation(nnu)]
        
        score_cv = np.zeros((n_sigma, n_lambda, n_mix, n_fold))
        for ite_sigma in range(n_sigma):
                sigma = sigma_list[ite_sigma]
                Kde = gaussian_kernel(dist2de, sigma)
                Knu = gaussian_kernel(dist2nu, sigma)

                for ite_fold in range(n_fold):
                        Kde_tr = Kde[cv_index_de != ite_fold, :]
                        Kde_te = Kde[cv_index_de == ite_fold, :]

                        Knu_tr = Knu[cv_index_nu != ite_fold, :]
                        Knu_te = Knu[cv_index_nu == ite_fold, :]
                        
                        Hde_tr = Kde_tr.T.dot(Kde_tr)/Kde_tr.shape[0]
                        Hnu_tr = Knu_tr.T.dot(Knu_tr)/Knu_tr.shape[0]
                        Hde_te = Kde_te.T.dot(Kde_te)/Kde_te.shape[0]
                        Hnu_te = Knu_te.T.dot(Knu_te)/Knu_te.shape[0]

                        h_tr   = np.mean(Knu_tr, axis=0).T
                        h_te   = np.mean(Knu_te, axis=0).T

                        for ite_mix in range(n_mix):
                                mix_rate = mix_rate_list[ite_mix]
                                H_tr = (1-mix_rate)*Hde_tr + mix_rate*Hnu_tr
                                H_te = (1-mix_rate)*Hde_te + mix_rate*Hnu_te

                                for ite_lambda in range(n_lambda):
                                        lam = lambda_list[ite_lambda]
                                        Reg = lam*np.eye(n_basis)
                                        alpha_cv = sp.linalg.solve(H_tr + Reg, h_tr)
                                        alpha_cv = np.maximum(0, alpha_cv)
                                        score    = alpha_cv.T.dot(H_te).dot(alpha_cv)/2 \
                                                   - alpha_cv.T.dot(h_te)
                                        score_cv[ite_sigma, ite_lambda, ite_mix, ite_fold] \
                                                = score

        score_cv = np.mean(score_cv, axis=3)
        model = []
        for ite_mix in range(n_mix):
                mix_rate = mix_rate_list[ite_mix]
                tmp = np.argmin(score_cv[:, :, ite_mix].ravel())
                tmp = np.unravel_index(tmp, (n_sigma, n_lambda))
                sigma_index  = tmp[0]
                lambda_index = tmp[1]

                sigma = sigma_list[sigma_index]
                lam   = lambda_list[lambda_index]

                Kde = gaussian_kernel(dist2de, sigma)
                Knu = gaussian_kernel(dist2nu, sigma)

                H   = (1-mix_rate)*Kde.T.dot(Kde)/Kde.shape[0] \
                      + mix_rate*Knu.T.dot(Knu)/Knu.shape[0]
                h   = np.mean(Knu, axis=0).T
                Reg = lam*np.eye(n_basis)

                alphah = sp.linalg.solve(H + Reg, h)
                m = dict()
                m['alpha'] = alphah
                m['sigma'] = sigma
                m['center'] = xc

                model.append(m)

        return model


def est_w(xte, model):
        xc     = model['center']
        sigma  = model['sigma']
        alphah = model['alpha']

        nte = xte.shape[0]
        wh = np.exp(-squared_dist(xte, xc)/(2*sigma**2)).dot(alphah).reshape((nte, 1))
        wh = np.maximum(0, wh)
        return wh

def squared_dist(x, c):
        """
        @param x n1-by-d matrix
        @param c n2-by-d matrix
        @return squared distance between x and c of size n1-by-n2
        """
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
        K = np.exp(-dist2/(2*sigma**2))
        return K

