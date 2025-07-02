# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
from explainers.base import ExplainerBase
# from explainers.MILPverifier import FD_recon, DNN_recon
from explainers.gradients import  VanillaGradient, IntegratedGradients, GuidedBackprop

def _get_norm_batch(x, p):
    batch_size = x.size(0)
    return x.abs().pow(p).view(batch_size, -1).sum(dim=1).pow(1. / p)

def _batch_multiply_tensor_by_vector(vector, batch_tensor):
    """Equivalent to the following
    for ii in range(len(vector)):
        batch_tensor.data[ii] *= vector[ii]
    return batch_tensor
    """
    return (
        batch_tensor.transpose(0, -1) * vector).transpose(0, -1).contiguous()

def batch_multiply(float_or_vector, tensor):
    if isinstance(float_or_vector, torch.Tensor):
        assert len(float_or_vector) == len(tensor)
        tensor = _batch_multiply_tensor_by_vector(float_or_vector, tensor)
    elif isinstance(float_or_vector, float):
        tensor *= float_or_vector
    else:
        raise TypeError("Value has to be float or torch.Tensor")
    return tensor


def normalize_by_pnorm(x, p=2, small_constant=1e-6):
    """
    Normalize gradients for gradient (not gradient sign) attacks.
    # TODO: move this function to utils
    :param x: tensor containing the gradients on the input.
    :param p: (optional) order of the norm for the normalization (1 or 2).
    :param small_constant: (optional float) to avoid dividing by zero.
    :return: normalized gradients.
    """
    # loss is averaged over the batch so need to multiply the batch
    # size to find the actual gradient of each input sample

    assert isinstance(p, float) or isinstance(p, int)
    norm = _get_norm_batch(x, p)
    norm = torch.max(norm, torch.ones_like(norm) * small_constant)
    return batch_multiply(1. / norm, x)

def clamp_by_pnorm(x, p, r):
    assert isinstance(p, float) or isinstance(p, int)
    norm = _get_norm_batch(x, p)
    if isinstance(r, torch.Tensor):
        assert norm.size() == r.size()
    else:
        assert isinstance(r, float)
    factor = torch.min(r / norm, torch.ones_like(norm))
    return batch_multiply(factor, x)

def perturb_iterative(xvar, yvar, model, nb_iter, eps, eps_iter, order=2,
                      clip_min=0.0, clip_max=1.0, m_type='FD', grad_sparsity = 90):

    delta = torch.zeros_like(xvar)
    delta.requires_grad_()
    loss_last = -1e8
    for ii in range(nb_iter):

        outputs = model(xvar + delta)
        #     else: # FC
        #         loss = nn.CrossEntropyLoss(reduction="sum")(outputs, yvar)
        if 'FD' in m_type: #FD
            loss = -nn.MSELoss(reduction="sum")(xvar + delta, outputs)
        elif 'FC' in m_type:
            if 'T' in m_type:# FC_T
                loss = -nn.CrossEntropyLoss(reduction="sum")(outputs, yvar)
            else:
                loss = -nn.MSELoss(reduction="sum")(outputs, yvar)

        loss.backward()
        if order == np.inf:
            grad_sign = delta.grad.data.sign()
            delta.data = delta.data + eps_iter * grad_sign
            delta.data = np.clip(delta.data, -eps, eps)
            delta.data = torch.clamp(xvar.data + delta.data, clip_min, clip_max) - xvar.data
            x_adv = torch.clamp(xvar + delta, clip_min, clip_max)

        elif order == 2:
            grad = delta.grad.data
            grad = normalize_by_pnorm(grad, p=2)
            delta.data = delta.data + batch_multiply(eps_iter, grad)
            delta.data = torch.clamp(xvar.data + delta.data, clip_min, clip_max
                               ) - xvar.data
            if eps is not None:
                delta.data = clamp_by_pnorm(delta.data, order, eps)

        elif order == 1:
            grad = torch.Tensor(np.array(delta.grad.data))
            grad_view = grad.view(grad.shape[0], -1)
            abs_grad = torch.abs(grad_view)
            k = int(grad_sparsity / 100.0 * abs_grad.shape[1])
            percentile_value, _ = torch.kthvalue(abs_grad, k, keepdim=True)

            percentile_value = percentile_value.repeat(1, grad_view.shape[1])
            tied_for_max = torch.ge(abs_grad, percentile_value).int().float()
            num_ties = torch.sum(tied_for_max, dim=1, keepdim=True)

            # if sign:
            optimal_perturbation = (torch.sign(grad_view) * tied_for_max) / num_ties
            # else:
            #     optimal_perturbation = (grad_view * tied_for_max)
            #     optimal_perturbation = optimal_perturbation / torch.sum(torch.abs(optimal_perturbation), dim=1, keepdim=True)
            optimal_perturbation = optimal_perturbation.view(grad.shape)
            delta.data =  delta.data + batch_multiply(eps_iter, optimal_perturbation).numpy()
            delta.data = torch.clamp(xvar.data + delta.data, clip_min, clip_max
                               ) - xvar.data
            if eps is not None:
                delta.data = delta.data.renorm(p=1, dim=0, maxnorm=eps)

        elif order == 'rbc':
            grad = eps_iter * delta.grad.data
            delta.data = delta.data + grad
            delta.data = torch.clamp(xvar.data + delta.data, clip_min, clip_max
                               ) - xvar.data
            if (loss - loss_last).detach().numpy() < 1e-6:
                break
            loss_last = loss
        # elif order == 0:
        #      adv_not_found = np.ones(y_nat.shape)
        #      adv = np.zeros(x_nat.shape)

        #     for i in range(attack.num_steps):
        #       if i > 0:
        #         #pred, grad = sess.run([attack.model.correct_prediction, attack.model.grad], feed_dict={attack.model.x_input: x2, attack.model.y_input: y_nat})
        #         pred, grad = get_predictions_and_gradients(attack.model, x2, y_nat)
        #         adv_not_found = np.minimum(adv_not_found, pred.astype(int))
        #         adv[np.logical_not(pred)] = np.copy(x2[np.logical_not(pred)])

        #         grad /= (1e-10 + np.sum(np.abs(grad), axis=(1,2,3), keepdims=True))
        #         x2 = np.add(x2, (np.random.random_sample(grad.shape)-0.5)*1e-12 + attack.step_size * grad, casting='unsafe')

        #       x2 = x_nat + project_L0_box(x2 - x_nat, attack.k, lb, ub)

        delta.grad.data.zero_()
        x_adv = xvar + delta

        if ii % 500 == 0:
            print(f'{ii} iterations, rep distance Loss {loss}')

    return x_adv


def PGD(model, x, repcenter, m_type, eps=None, nb_iter=4001, eps_iter=0.02,
        order=2, clip_min=0.0, clip_max=1.0):
    if not torch.is_tensor(x):
        x = torch.from_numpy(x).float()

    rval = perturb_iterative(
        x, repcenter, model, nb_iter=nb_iter,
        eps=eps, eps_iter=eps_iter,
        order=order, clip_min=clip_min,
        clip_max=clip_max, m_type=m_type
    )

    delta = (rval.detach() - x).numpy()
    return rval.data

class ABIGX(ExplainerBase):
    def __init__(self, model, normality, algo, params):
        name= {'cp_pgd':'','rbc_pgd':'_OneVar',
               'cp_milp':'','rbc_milp':'_OneVar'}
        super(ABIGX, self).__init__(model, 'ABIGX'+name[algo])
        model.eval()
        self.algo = algo
        if self.model_type in ['FD']:
            self.clip_min = -1000
            self.clip_max = 1000
            self.m_radius = 2000
            self.model = model
            q_normal = model.cal_q(normality)
            # self.qlim = model.cal_limit(q_normal) / 10
            self.qlim = 0
        else:
            self.clip_min = -1000
            self.clip_max = 1000
            self.m_radius = 2000
            self.orig_model = model
            self.model = model.get_layer_output

        if not torch.is_tensor(normality):
            normality = torch.from_numpy(normality).float()
        with torch.no_grad():
            self.repcenter = torch.mean(self.model(normality),dim=0)

        self.params = params

    def reconstruct(self, df, norm=2):
        x = df.values.copy()

        if 'milp' in self.algo:
            if self.model_type in ['FD']:
                explainer = FD_recon(self.model, self.qlim, self.clip_min,
                                     self.clip_max, self.m_radius,
                                     norm=norm, mode='min_distance')
                recon_x = []
                for e in x:
                    res = explainer.verify(e)
                    recon_x.append(explainer.cx.value)
                recon_x = np.array(recon_x)

        elif 'pgd' in self.algo:
            if self.model_type == 'FC_T':
                recon_x = PGD(self.orig_model, x, repcenter=torch.zeros(x.shape[0]).long(), m_type = self.model_type,
                             clip_min=self.clip_min, clip_max=self.clip_max, **self.params)
            else:
                recon_x = PGD(self.model, x, repcenter=self.repcenter, m_type = self.model_type,
                              clip_min=self.clip_min, clip_max=self.clip_max, **self.params)

        return recon_x

    def reconstruct_givendirect(self, df):
        x = df.values.copy()
        if 'milp' in self.algo:
            if self.model_type in ['FD']:
                explainer = FD_recon(self.model, self.qlim, self.clip_min,
                                     self.clip_max, self.m_radius, mode='min_SPE')
                recon_x = []
                for e in x:
                    baselines = []
                    for given_direction in range(df.shape[1]):
                        res = explainer.verify(e,given_direction=given_direction)
                        baselines.append(explainer.cx.value)
                    baselines = np.array(baselines)
                    recon_x.append(baselines)
        elif 'pgd' in self.algo:
            recon_x = []
            for given_direction in range(df.shape[1]):
                gd_min = x.copy()
                gd_min[:,given_direction] = self.clip_min
                gd_min = torch.FloatTensor(gd_min)
                gd_max = x.copy()
                gd_max[:,given_direction] = self.clip_max
                gd_max = torch.FloatTensor(gd_max)

                baselines = PGD(self.model, x, repcenter=self.repcenter, m_type=self.model_type,
                              clip_min=gd_min, clip_max=gd_max, order='rbc',
                              **self.params)
                recon_x.append(baselines.numpy())

            rx = []
            for j in range(df.shape[0]):
                baselines = []
                for i in range(len(recon_x)):
                    baselines.append(recon_x[i][j,:])
                baselines = np.array(baselines)
                rx.append(baselines)
            recon_x = rx
        return recon_x

    def explain(self, fault_explicand, y_explicand=None, plot=True):
        if 'FC' in self.model_type:
            IG = IntegratedGradients(self.orig_model, method_name=self.method_name)
        else:
            IG = IntegratedGradients(self.model, method_name=self.method_name)
        if 'rbc' in self.algo: #Reconstruction like RBC
            x = fault_explicand.values.copy()
            rb_baseline_list = self.reconstruct_givendirect(fault_explicand)
            attr_list = []
            baseline_list = []
            for i, baseline in enumerate(rb_baseline_list):
                tile_x = np.tile(x[i,:],(baseline.shape[0],1))
                if y_explicand is not None:
                    y_explicand = y_explicand[:baseline.shape[0]]
                attr_ABIGXrb_x = IG.explain(tile_x, y=y_explicand, baseline=baseline, plot=False, steps=25)
                attr_ABIGXrb_x = np.diagonal(attr_ABIGXrb_x)
                attr_list.append(attr_ABIGXrb_x)
                baseline_list.append(np.diagonal(baseline))
            attr_ABIGX = np.array(attr_list)
            afr_baseline = np.array(baseline_list)
            if plot:
                self.plot_shap(attr_ABIGX,fault_explicand)

        elif 'cp' in self.algo: #Reconstruction like CP
            afr_baseline = self.reconstruct(fault_explicand)
            attr_ABIGX = IG.explain(fault_explicand, y=y_explicand, baseline=afr_baseline, steps=25, plot=plot)
        return attr_ABIGX, afr_baseline


