# # python version is 3.6.1
from enum import Enum
# from typing import Union
#
import numpy as np
# import scipy as sp
# from scipy import sparse
#
# from . import rulsif
from .pu import decision_function
#
NEG_LABEL = -1
POS_LABEL = 1
#
# USE_PUC = True


class PUcKernelType(Enum):
    r""" Supported PUc kernels """
    GAUSSIAN = "gauss"
    LINEAR = "linear"


# # noinspection DuplicatedCode
# def fit(xp_tr: np.ndarray, xu_tr: np.ndarray, xu_te: np.ndarray,
#         prior: float, kertype: Union[str, PUcKernelType],
#         lambda_list=np.logspace(-3, 0, num=11),
#         gamma_list=np.mgrid[.1:.9:9j],
#         sigma_list=None, max_n_basis=200, n_fold=5,
#         bias=True) -> dict:
#     r"""
#     Positive-unlabeled-covariate (PUc) algorithm
#
#     :param xp_tr: \p x vector for the positive set from the TRAIN distribution
#     :param xu_tr: \p x vector for the unlabeled set from the TRAIN distribution
#     :param xu_te: \p x vector for the unlabeled set from the TEST distribution
#     :param prior: Training set positive class prior probability
#     :param lambda_list: Regularization parameters for search
#     :param gamma_list:
#     :param sigma_list: List of GAUSSIAN kernel bandwidths to use in grid search.  Must be \p None
#                        for LINEAR kernel
#     :param kertype: Kernel type used by the SVM
#     :param max_n_basis: Maximum number of basis functions for the Gaussian kernel
#     :param n_fold: Cross validation fold count
#     :param bias: If \p True, use a bias for LINEAR kernel.  Must be \p False for the GAUSSIAN
#                  kernel.
#     :return: Dictionary of the learned parameters
#     """
#     if isinstance(kertype, str):
#         # If passed string, convert to enum automatically
#         kertype = PUcKernelType(kertype)
#
#     np_tr = xp_tr.shape[0]
#     nu_tr = xu_tr.shape[0]
#     nu_te = xu_te.shape[0]
#
#     if kertype == PUcKernelType.GAUSSIAN:
#         assert not bias, "Bias not valid for Gaussian Kernel"
#         if USE_PUC:
#             svm_dim = np.minimum(max_n_basis, nu_te)  # Number of biases
#             center_index = np.random.permutation(nu_te)
#             xc = xu_te[center_index[:svm_dim]]
#         else:
#             svm_dim = np.minimum(max_n_basis, nu_tr)  # Number of biases
#             center_index = np.random.permutation(nu_tr)
#             xc = xu_tr[center_index[:svm_dim]]
#         dp_tr = squared_dist(xp_tr, xc)
#         du_tr = squared_dist(xu_tr, xc)
#         if sigma_list is None:
#             med = np.median(du_tr.ravel())  # ravel flattens the input
#             sigma_list = np.sqrt(med) * np.logspace(-1, 1, num=11)
#     elif kertype == PUcKernelType.LINEAR:
#         assert sigma_list is None, "Sigma not valid for linear kernel"
#         is_sparse = sparse.issparse(xp_tr)
#         xc = None
#         sigma_list = [0]
#         svm_dim = xp_tr.shape[1]
#         if bias:
#             svm_dim += 1
#             if is_sparse:
#                 dp_tr = sparse.hstack((xp_tr, sparse.csr_matrix(np.ones((np_tr, 1)))), format='csr')
#                 du_tr = sparse.hstack((xu_tr, sparse.csr_matrix(np.ones((nu_tr, 1)))), format='csr')
#             else:
#                 dp_tr = np.c_[xp_tr, np.ones(np_tr)]
#                 du_tr = np.c_[xu_tr, np.ones(nu_tr)]
#         else:
#             dp_tr, du_tr = xp_tr, xu_tr
#     else:
#         raise ValueError("Unknown kernel")
#
#     n_gamma, n_lambda, n_sigma = len(gamma_list), len(lambda_list), len(sigma_list)
#
#     if USE_PUC:
#         mix_rate_list = gamma_list
#         if 0 not in mix_rate_list:
#             mix_rate_list = np.append(mix_rate_list, 0)
#         else:
#             raise Exception('exception for now')
#
#     # ============================================================== #
#     #              Estimate the importance function w(x)             #
#     # ============================================================== #
#
#     if USE_PUC:
#         # noinspection PyUnboundLocalVariable
#         wm = rulsif.rulsif_cv(xu_tr, xu_te, xc=xc,
#                               sigma_list=sigma_list if kertype == PUcKernelType.GAUSSIAN else None,
#                               mix_rate_list=mix_rate_list, n_basis=svm_dim, lambda_list=lambda_list)
#
#         wph_list = {}
#         wuh_list = {}
#         ite_gam = 0
#         for ite_mix in range(len(mix_rate_list)):
#             if mix_rate_list[ite_mix] == 0:
#                 wph0 = np.array(rulsif.est_w(xp_tr, wm[ite_mix])).squeeze()
#                 wuh0 = np.array(rulsif.est_w(xu_tr, wm[ite_mix])).squeeze()
#             else:
#                 wph_list[ite_gam] = np.array(rulsif.est_w(xp_tr, wm[ite_mix])).squeeze()
#                 wuh_list[ite_gam] = np.array(rulsif.est_w(xu_tr, wm[ite_mix])).squeeze()
#                 ite_gam += 1
#
#         th = .5
#         # noinspection PyUnboundLocalVariable
#         index_wph = wph0 > th
#         dp_tr = dp_tr[index_wph, :]
#         np_tr = dp_tr.shape[0]
#         # noinspection PyUnboundLocalVariable
#         index_wuh = wuh0 > th
#         du_tr = du_tr[index_wuh, :]
#         nu_tr = du_tr.shape[0]
#         print("new_xp_tr.shape[0]: {}".format(np.sum(index_wph)))
#         print("new_xu_tr.shape[0]: {}".format(np.sum(index_wuh)))
#         ite_gam = 0
#         for ite_mix in range(len(mix_rate_list)):
#             if mix_rate_list[ite_mix] == 0:
#                 wph0 = wph0[index_wph]
#                 wuh0 = wuh0[index_wuh]
#             else:
#                 wph_list[ite_gam] = wph_list[ite_gam][index_wph]
#                 wuh_list[ite_gam] = wuh_list[ite_gam][index_wuh]
#                 ite_gam += 1
#
#     # ============================================================== #
#     #                  Perform CV-Based Grid Search                  #
#     # ============================================================== #
#
#     cv_index_p_tr = (np.arange(np_tr, dtype=np.int_) * n_fold) // np_tr
#     cv_index_p_tr = cv_index_p_tr[np.random.permutation(np_tr)]
#
#     cv_index_u_tr = (np.arange(nu_tr, dtype=np.int_) * n_fold) // nu_tr
#     cv_index_u_tr = cv_index_u_tr[np.random.permutation(nu_tr)]
#
#     score_cv_fold = np.full((n_sigma, n_gamma, n_lambda, n_fold), np.inf)
#     for ite_sigma in range(n_sigma):
#         sigma = sigma_list[ite_sigma]
#         if kertype == PUcKernelType.GAUSSIAN:
#             kp_tr = gaussian_kernel(dp_tr, sigma)
#             ku_tr = gaussian_kernel(du_tr, sigma)
#         elif kertype == PUcKernelType.LINEAR:
#             kp_tr = dp_tr
#             ku_tr = du_tr
#         else:
#             raise ValueError("Unknown kernel")
#
#         for ite_fold in range(n_fold):
#             dp_tr_cvtr = kp_tr[cv_index_p_tr != ite_fold]
#             dp_tr_cvte = kp_tr[cv_index_p_tr == ite_fold]
#
#             du_tr_cvtr = ku_tr[cv_index_u_tr != ite_fold]
#             du_tr_cvte = ku_tr[cv_index_u_tr == ite_fold]
#
#             for ite_gamma in range(n_gamma):
#                 if USE_PUC:
#                     # noinspection PyUnboundLocalVariable
#                     wph_tr = (wph_list[ite_gamma])[cv_index_p_tr != ite_fold]
#                     # noinspection PyUnboundLocalVariable
#                     wuh_tr = (wuh_list[ite_gamma])[cv_index_u_tr != ite_fold]
#                     wph_te = wph0[cv_index_p_tr == ite_fold]
#                     wuh_te = wuh0[cv_index_u_tr == ite_fold]
#
#                     qp = prior * wph_tr.dot(dp_tr_cvtr).T / dp_tr_cvtr.shape[0]
#                     # noinspection PyPep8Naming
#                     Hu = du_tr_cvtr.T.dot(np.diag(wuh_tr)).dot(du_tr_cvtr) / du_tr_cvtr.shape[0]
#                     hu = du_tr_cvtr.T.dot(wuh_tr) / du_tr_cvtr.shape[0]
#                 else:
#                     # noinspection PyPep8Naming
#                     Hu = du_tr_cvtr.T.dot(du_tr_cvtr) / du_tr_cvtr.shape[0]
#                     qp = prior * np.mean(dp_tr_cvtr, axis=0).T
#                     hu = np.mean(du_tr_cvtr, axis=0).T
#
#                 for ite_lambda in range(n_lambda):
#                     lam = lambda_list[ite_lambda]
#                     regularizer = lam * np.eye(svm_dim)
#                     if bias:
#                         assert kertype == PUcKernelType.LINEAR, "Bias only valid for linear kernel"
#                         regularizer[-1, -1] = 0
#                     beta_cv = sp.linalg.solve(Hu + regularizer, 2 * qp - hu)
#
#                     if USE_PUC:
#                         # noinspection PyUnboundLocalVariable
#                         val = risk_puc_tr(dp_tr_cvte, du_tr_cvte, beta_cv, prior, wph_te, wuh_te)
#                     else:
#                         val = risk_pu(dp_tr_cvte, du_tr_cvte, beta_cv, prior)
#                     score_cv_fold[ite_sigma, ite_gamma, ite_lambda, ite_fold] = val
#
#     score_cv = np.mean(score_cv_fold, axis=-1)
#     tmp = np.argmin(score_cv.ravel())
#     tmp = np.unravel_index(tmp, score_cv.shape)
#     sigma_index, gamma_index, lambda_index = tmp[0], tmp[1], tmp[2]
#
#     # noinspection PyTypeChecker
#     sigma = sigma_list[sigma_index]
#     gamma = gamma_list[gamma_index]
#     lam = lambda_list[lambda_index]
#     print(f"(sigma, gamma, lambda) = ({sigma:.6f}, {gamma:.2}, {lam:6f})")
#
#     if kertype == PUcKernelType.GAUSSIAN:
#         kp_tr, ku_tr = gaussian_kernel(dp_tr, sigma), gaussian_kernel(du_tr, sigma)
#     elif kertype == PUcKernelType.LINEAR:
#         kp_tr, ku_tr = dp_tr, du_tr
#     else:
#         raise ValueError("Unknown kernel")
#
#     # noinspection PyPep8Naming
#     Hu = ku_tr.T.dot(ku_tr) / ku_tr.shape[0]
#     qp = prior * np.mean(kp_tr, axis=0).T
#     hu = np.mean(ku_tr, axis=0).T
#     regularizer = lam*np.eye(svm_dim)
#     if bias:
#         assert kertype == PUcKernelType.LINEAR, "Bias only supported for the linear kernel"
#         regularizer[-1, -1] = 0
#     beta = sp.linalg.solve(Hu + regularizer, 2 * qp - hu)
#
#     model = dict()
#     model['kertype'] = kertype.value
#     model['gamma'] = gamma
#     model['lambda'] = lam
#     model['alpha'] = model['beta'] = beta
#     model['bias'] = bias
#     if kertype == PUcKernelType.GAUSSIAN:
#         model['sigma'] = sigma
#         model['center'] = xc
#     else:
#         model['center'] = model['sigma'] = None
#
#     return model
#
#
# # noinspection DuplicatedCode,PyPep8Naming
# def decision_function(model: dict, x_te: np.ndarray) -> np.ndarray:
#     r"""
#     Executes decision function on the model
#
#     :param model: PUc model parameters
#     :param x_te: Test vector
#     :return: Decision function output
#     """
#     if model['kertype'] == PUcKernelType.GAUSSIAN.value:
#         K = gaussian_kernel(squared_dist(x_te, model['center']), model['sigma'])
#     elif model['kertype'] == PUcKernelType.LINEAR.value:
#         if model['bias']:
#             if sparse.issparse(x_te):
#                 K = sparse.hstack((x_te, np.ones((x_te.shape[0], 1))), format='csr')
#             else:
#                 K = np.c_[x_te, np.ones(x_te.shape[0])]
#         else:
#             K = x_te
#     else:
#         raise ValueError("Unknown kernel")
#
#     return K.dot(model['alpha'])


def predict(model: dict, x_te: np.ndarray) -> np.ndarray:
    r"""
    Predicts the labels for \p x_te

    :param model: PUc model parameters
    :param x_te: Test vector
    :return: Predicted labels
    """
    dec_scores = decision_function(model, x_te)
    # Specially handle sign to allow custom labels
    preds = np.full([dec_scores.shape[0]], NEG_LABEL, dtype=np.int32)
    preds[dec_scores >= 0] = POS_LABEL
    return preds
