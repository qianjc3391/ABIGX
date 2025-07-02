import torch
import torch.nn as nn
import numpy as np
from explainers.base import ExplainerBase
import pandas as pd


class VanillaGradient(ExplainerBase):
    def __init__(self,model,method_name='Saliency map'):
        super(VanillaGradient, self).__init__(model,method_name)
        self.model = model
        self.model.eval()
        self.hooks = list()
        
    def gradient(self,x,y=0):      
        if self.model_type in ['FD']:
            x = x.detach().clone()
            x.requires_grad = True
            x_hat = self.model(x)
            SPE = nn.MSELoss(reduction='sum')(x_hat,x)
            self.model.zero_grad()
            SPE.backward()
            g = x.grad.detach()
        else:
            x = x.detach().clone()
            x.requires_grad = True
            output = self.model(x)
            target = torch.zeros_like(output)
            target.scatter_(1,y.view(-1,1),1)
            self.model.zero_grad()
            output.backward(target)
            g = x.grad.detach()
        return g

    def smoothed_gradient(self, x, y=None, samples=25, std=0.15, process=lambda x: x**2):
        std = std * (torch.max(x) - torch.min(x)).detach().cpu().numpy()

        grad_sum = torch.zeros_like(x)
        for sample in range(samples):
            noise = torch.empty(x.size()).normal_(0, std).to(x.device)
            noise_x = x + noise
            grad_sum += torch.pow(self.gradient(noise_x, y),2)
        return grad_sum / samples

    def explain(self,df,y, absolute=True, plot=True):
        if isinstance(df, pd.DataFrame):
            x = df.values.copy()
        x = torch.FloatTensor(x)
        grad = self.gradient(x,y).detach().numpy()
        if absolute:
            grad = abs(grad)
        if plot:
            self.plot_shap(grad,df)
        return grad

class IntegratedGradients(VanillaGradient):
    def __init__(self,model,method_name='IG'):
        super(IntegratedGradients, self).__init__(model,method_name)

    def explain(self, df, y, baseline, steps=25, plot=True):
        if isinstance(df, pd.DataFrame):
            x = df.values.copy()
        else:
            x = df
        x = torch.FloatTensor(x)

        if isinstance(baseline, str):
            if baseline == 'black':
                baseline = torch.ones([1,x.shape[1]]) * torch.min(x).detach().cpu()
            elif baseline == 'white':
                baseline = torch.ones([1,x.shape[1]]) * torch.max(x).detach().cpu()
            elif baseline == 'zeros':
                baseline = torch.zeros([1,x.shape[1]])
        else:
            if ~torch.is_tensor(baseline):
                baseline = torch.FloatTensor(baseline)

        IG_sum = np.zeros_like(x)

        x_diff = x - baseline
        grad_sum =  torch.zeros_like(x)
        for step, alpha in enumerate(np.linspace(0, 1, steps)):
            x_step = baseline + alpha * x_diff
            grad_sum += self.gradient(x_step, y)

        IG_sum += grad_sum.numpy() * x_diff.detach().cpu().numpy() / steps
        if plot:
            self.plot_shap(IG_sum,df)

        return IG_sum

class GuidedBackprop(VanillaGradient):
    def __init__(self, model):
        super(GuidedBackprop, self).__init__(model)
        self.relu_inputs = list()
        self.update_relus()

    def update_relus(self):
        def clip_gradient(module, grad_input, grad_output):
            relu_input = self.relu_inputs.pop()
            return (grad_output[0] * (grad_output[0] > 0.).float() * (relu_input > 0.).float(),)

        def save_input(module, input, output):
            self.relu_inputs.append(input[0])

        for module in self.model.modules():
            if isinstance(module, torch.nn.ReLU):
                self.hooks.append(module.register_forward_hook(save_input))
                self.hooks.append(module.register_backward_hook(clip_gradient))


class IDGI(VanillaGradient):
    def __init__(self,model,method_name='IDGI'):
        super(IDGI, self).__init__(model,method_name)

    def explain(self, df, y, baseline, steps=25, plot=True):
        if isinstance(df, pd.DataFrame):
            x = df.values.copy()
        else:
            x = df
        x = torch.FloatTensor(x)

        if isinstance(baseline, str):
            if baseline == 'black':
                baseline = torch.ones([1,x.shape[1]]) * torch.min(x).detach().cpu()
            elif baseline == 'white':
                baseline = torch.ones([1,x.shape[1]]) * torch.max(x).detach().cpu()
            elif baseline == 'zeros':
                baseline = torch.zeros([1,x.shape[1]])
        else:
            if not torch.is_tensor(baseline):
                baseline = torch.FloatTensor(baseline)

        x_diff = x - baseline
        grad_list =  []
        pred_list = []
        for step, alpha in enumerate(np.linspace(0, 1, steps)):
            x_step = baseline + alpha * x_diff
            pred_list.append(self.prediction(x_step, y).detach().numpy())
            grad_list.append(self.gradient(x_step, y).detach().numpy())

        idgi_list = []
        for n in range(len(x)):
            pred_sample = []
            grad_sample = []
            for i in range(len(pred_list)):
                pred_sample.append(pred_list[i][n])
                grad_sample.append(grad_list[i][n])
            idgi = self.IDGI_on_list(grad_sample,pred_sample)
            idgi_list.append(idgi)
        if plot:
            self.plot_shap(idgi_list,df)

        return np.array(idgi_list)

   
    def IDGI_on_list(self, Gradients, Predictions):
        """
        IDGI algorithm:
        
        The IDGI is compatible with any IG based method, e.g., Integrated gradients (IG), Guided Integrated gradients (GIG), Blur Integrated gradients (BlurIG), ....
        For more detail, please check our paper: 
        Args:
            Gradients (list of np.array or np.array): All the gradients that are computed from the Integraded gradients path.
                                                    For instance, when compute IG, the gradients are needed for each x_j on the path. e.g. df_c(x_j)/dx_j.
                                                    Gradients is the list (or np.array) which contains all the computed gradients for the IG-base method, 
                                                    and each element in Gradients is the type of np.array.
            Predictions (list of float or np.array): List of float numbers.
                                                    Predictions contains all the predicted value for each points on the path of IG-based methods.
                                                    For instance, the value of f_c(x_j) for each x_j on the path.
                                                    Predictions is the list (or np.array) which contains all the computed target values for IG-based method, 
                                                    and each element in Predictions is a float.
        
        Return:
            IDGI result: Same size as the gradient, e.g., Gradients[0]
        """
        assert len(Gradients) == len(Predictions)
        
        idgi_result = np.zeros_like(Gradients[0])
        for i in range(len(Gradients) -1):
            # We ignore the last gradient, e.g., the gradient of on the original image, since IDGI requires the prediction difference component, e.g., d.
            d = Predictions[i+1] - Predictions[i]
            element_product = Gradients[i]**2
            idgi_result += element_product*d/np.sum(element_product)

        return idgi_result
    
    def GetMask(self, x_value, call_model_function, call_model_args=None,
              x_baseline=None, x_steps=25, batch_size=1):
        """Returns an integrated gradients mask.

        Args:
        x_value: Input ndarray.
        call_model_function: A function that interfaces with a model to return
            specific data in a dictionary when given an input and other arguments.
            Expected function signature:
            - call_model_function(x_value_batch,
                                call_model_args=None,
                                expected_keys=None):
            x_value_batch - Input for the model, given as a batch (i.e. dimension
                0 is the batch dimension, dimensions 1 through n represent a single
                input).
            call_model_args - Other arguments used to call and run the model.
            expected_keys - List of keys that are expected in the output. For this
                method (Integrated Gradients), the expected keys are
                INPUT_OUTPUT_GRADIENTS - Gradients of the output being
                explained (the logit/softmax value) with respect to the input.
                Shape should be the same shape as x_value_batch.
        call_model_args: The arguments that will be passed to the call model
            function, for every call of the model.
        x_baseline: Baseline value used in integration. Defaults to 0.
        x_steps: Number of integrated steps between baseline and x.
        batch_size: Maximum number of x inputs (steps along the integration path)
            that are passed to call_model_function as a batch.
        """
        if x_baseline is None:
            x_baseline = np.zeros_like(x_value)

        assert x_baseline.shape == x_value.shape

        x_diff = x_value - x_baseline

        total_gradients = np.zeros_like(x_value, dtype=np.float32)

        x_step_batched = []
        prediction_list = []
        gradient_list = []
        for alpha in np.linspace(0, 1, x_steps):
            x_step = x_baseline + alpha * x_diff
            x_step_batched.append(x_step)
            if len(x_step_batched) == batch_size or alpha == 1:
                x_step_batched = np.asarray(x_step_batched)
                call_model_output = call_model_function(
                    x_step_batched,
                    call_model_args=call_model_args,
                    expected_keys=self.expected_keys)

                self.format_and_check_call_model_output(call_model_output,
                                                        x_step_batched.shape,
                                                        self.expected_keys)
                gradient_list.extend(call_model_output[INPUT_OUTPUT_GRADIENTS])
                prediction_list.extend(call_model_output[MODEL_LOGIT_OUTPUT])
                x_step_batched = []

        return s


class BAE_explainer:
    def __init__(self,model,baseline, nb_iter=2000,eps_iter=0.02,
                  rand_init=True, clip_min=0., clip_max=1., ord=2):
        model.eval()
        self.model = model.get_layer_output
        if not torch.is_tensor(baseline):
            self.baseline_x = torch.from_numpy(baseline).float()
        else:
            self.baseline_x = baseline
        with torch.no_grad():
            self.baseline = self.model(self.baseline_x)
        self.nb_iter = nb_iter
        self.eps_iter = eps_iter
        self.rand_init = rand_init
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.ord = ord

    def explain(self, x):
        if not torch.is_tensor(x):
            self.x = torch.from_numpy(x).float()
        else:
            self.x = x
        x = self.x.detach().clone()

        delta = torch.zeros_like(x)
        delta = nn.Parameter(delta)
        if self.rand_init:
            delta.data.uniform_(-1, 1)
            delta.data= delta.data * self.eps_iter
            delta.data = torch.clamp(
                x + delta.data, min=self.clip_min, max=self.clip_max) - x

        rval = abigx.perturb_iterative(
            x, self.baseline, self.model, nb_iter=self.nb_iter,
            eps=None, eps_iter=self.eps_iter,
              order=self.ord, clip_min=self.clip_min,
            clip_max=self.clip_max, m_type='FC'
        )

        delta = (rval.detach() - x).numpy()
        # epsilon = np.linalg.norm(delta, ord=np.Inf, axis=1)
        # rho = epsilon / np.linalg.norm(x.numpy(), ord=np.Inf, axis=1)

        return rval.data, -delta #negative - :to show the impact on FAULT