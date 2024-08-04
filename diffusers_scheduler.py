# Link: https://huggingface.co/docs/diffusers/using-diffusers/write_own_pipeline
import math
from typing import List, Optional, Union, Callable, Dict

import numpy as np
import torch
from torch import Tensor, FloatTensor

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.schedulers.scheduling_utils import SchedulerMixin


def betas_for_alpha_bar(
    num_diffusion_timesteps,
    max_beta=0.999,
    alpha_transform_type="cosine"):

    if alpha_transform_type == "cosine":

        def alpha_bar_fn(t):
            return math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2

    elif alpha_transform_type == "exp":

        def alpha_bar_fn(t):
            return math.exp(t * -12.0)

    else:
        raise ValueError(f"Unsupported alpha_tranform_type: {alpha_transform_type}")

    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar_fn(t2) / alpha_bar_fn(t1), max_beta))
    return torch.tensor(betas, dtype=torch.float32)


def robust_clipping(x, eps=1e-3):
    x = torch.where(x == 0, eps, x)
    x = torch.where(x.abs() < eps, torch.sign(x) * eps, x)
    return x


class SCBaseScheduler(SchedulerMixin, ConfigMixin):
    order = None

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "linear",
        straight_type: str = "x0_x1_interpolation",
        trained_betas: Optional[Union[np.ndarray, List[float]]] = None,

        prediction_type: str = "epsilon",
    ):

        self.straight_type = straight_type

        self.num_inference_steps = None

        self.init_noise_sigma = 1.0

        # Original betas and alphas_bar from DDPM
        # ------------------------------ #
        # beta_0 = beta_min = 0.0001
        # ...
        # beta_{T-1} = beta_max = 0.02
        if trained_betas is not None:
            self.betas = torch.tensor(trained_betas, dtype=torch.float32)
        elif beta_schedule == "linear":
            self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float32)
        elif beta_schedule == "scaled_linear":
            # This schedule is very specific to the latent diffusion model.
            self.betas = (torch.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps, dtype=torch.float32) ** 2)
        else:
            raise NotImplementedError(f"{beta_schedule} does is not implemented for {self.__class__}")

        # alpha_0 = (1 - beta_min)
        # ...
        # alpha_{T-1} = (1 - beta_max)
        self.alphas = 1.0 - self.betas

        # alpha_bar_0
        # ...
        # alpha_bar_{T-1}
        self.alphas_bar = torch.cumprod(self.alphas, dim=0)
        # ------------------------------ #

        # ------------------------------ #
        # [1/T, 2/T, ..., T/T]
        # times_in_01 = np.arange(1, num_train_timesteps + 1).astype(np.float32) \
        #               * 1.0 / num_train_timesteps
        # self.times_in_01 = torch.from_numpy(times_in_01)
        # print(self.times_in_01)

        # Set time step for inference
        # timesteps is a list of integers indicating
        # which time steps will be used for sampling
        self.set_timesteps(num_train_timesteps-1)
        # print(self.timesteps)
        # ------------------------------ #

        # ------------------------------ #
        # print(f"self.betas[:10]: {self.betas[:10].data.cpu().numpy()}")
        # print(f"self.alphas_bar[:10]: {self.alphas_bar[:10].data.cpu().numpy()}")
        # print()
        # print(f"self.betas[-10:]: {self.betas[-10:].data.cpu().numpy()}")
        # print(f"self.alphas_bar[-10:]: {self.alphas_bar[-10:].data.cpu().numpy()}")
        # ------------------------------ #

    def scale_model_input(self, sample: FloatTensor, timestep: Optional[int] = None) -> FloatTensor:
        """
        Ensures interchangeability with schedulers that need to scale the denoising model input depending on the
        current timestep.

        Args:
            sample (`FloatTensor`):
                The input sample.
            timestep (`int`, *optional*):
                The current timestep in the diffusion chain.

        Returns:
            `FloatTensor`:
                A scaled input sample.
        """
        return sample

    def set_timesteps(self, num_inference_steps: int,
                      device: Union[str, torch.device] = None):

        if num_inference_steps > self.config.num_train_timesteps:
            raise ValueError(
                f"num_inference_steps ({num_inference_steps}) must be < "
                f" {self.config.num_train_timesteps} since the unet model was "
                f"only trained with {self.config.num_train_timesteps} timesteps!"
            )

        self.num_inference_steps = num_inference_steps

        self.timesteps = (
            np.linspace(0, self.config.num_train_timesteps - 1,
                        num_inference_steps + 1)
                .round()[::-1]
                .copy()
                .astype(np.int64)
        )

        # self.timesteps = torch.from_numpy(timesteps).to(device)
        #
        # assert len(self.timesteps) == self.num_inference_steps + 1, \
        #     f"len(self.timesteps)={len(self.timesteps)} " \
        #     f"while self.num_inference_steps={self.num_inference_steps}!"

    def a_t(self, i: Union[int, Tensor]):
        return torch.sqrt(self.alphas_bar[i])

    def s_t(self, i: Union[int, Tensor]):
        return torch.sqrt(1. - self.alphas_bar[i])

    def get_coefficients_SN(
        self, a_t: Tensor, s_t: Tensor,
        straight_type: str = "x0_x1_interpolation"):

        a_t_P_s_t = a_t + s_t

        if straight_type == "x0_x1_interpolation":
            # x_bar_t = a_t / (a_t + s_t) * X_0 + s_t / (a_t + s_t) * X_1
            # v_t_const = x_1_t - x_0_t = (1 + s_t/a_t) * x_1_t - x_t / a_t

            inv_k_t = a_t_P_s_t

        elif straight_type == "x1_scale_only":
            # X_bar_t = X_0 + (s_t / a_t) * X_1
            # v_t_const = x_1_t

            inv_k_t = a_t

        else:
            raise ValueError(f"Invalid straight_flow_type='{straight_type}'!")

        varphi_t = s_t / inv_k_t

        return {
            'inv_k_t': inv_k_t,
            'varphi_t': varphi_t,
        }

    def step_Euler(
        self,
        model_fn: Callable,
        timestep: int,
        sample: FloatTensor,
        next_timestep: int,
        eps: float = 1e-3, model_kwargs: Dict = {},
        return_intermediate_results: bool = False,
    ):

        a_t = robust_clipping(self.a_t(timestep), eps=eps)
        s_t = self.s_t(timestep)

        a_tm1 = robust_clipping(self.a_t(next_timestep), eps=eps)
        s_tm1 = self.s_t(next_timestep)

        coeffs_t = self.get_coefficients_SN(
            a_t, s_t, straight_type=self.straight_type)

        coeffs_tm1 = self.get_coefficients_SN(
            a_tm1, s_tm1, straight_type=self.straight_type)

        inv_k_t = coeffs_t['inv_k_t']
        varphi_t = coeffs_t['varphi_t']

        inv_k_tm1 = coeffs_tm1['inv_k_t']
        varphi_tm1 = coeffs_tm1['varphi_t']

        x_t = sample
        x_t_SN = x_t / inv_k_t

        # v_t_SC
        # ------------------------------- #
        x_1_t = model_fn(sample, timestep, **model_kwargs)

        if self.straight_type == "x0_x1_interpolation":
            v_t_SC = ((a_t + s_t) * x_1_t - x_t) / a_t
        elif self.straight_type == "x1_scale_only":
            v_t_SC = x_1_t
        else:
            raise ValueError(f"Invalid straight_type='{self.straight_type}'!")
        # ------------------------------- #

        # Euler
        # ------------------------------- #
        x_tm1_SN = x_t_SN + (varphi_tm1 - varphi_t) * v_t_SC

        x_tm1 = x_tm1_SN * inv_k_tm1
        # ------------------------------- #

        results = {
            "x_t_SN": x_t_SN,
            "v_t_SC": v_t_SC,
            "a_t": a_t,
            "s_t": s_t,
            "inv_k_t": inv_k_t,
            "varphi_t": varphi_t,

            "x_tm1_SN": x_tm1_SN,
            "a_tm1": a_tm1,
            "s_tm1": s_tm1,
            "inv_k_tm1": inv_k_tm1,
            "varphi_tm1": varphi_tm1,
        }

        if return_intermediate_results:
            return x_tm1, results
        else:
            return x_tm1

    def step_Heun(
        self,
        model_fn: Callable,
        timestep: int,
        sample: FloatTensor,
        next_timestep: int,
        eps: float = 1e-3, model_kwargs: Dict = {},
        return_intermediate_results: bool = False,
    ):

        # Euler
        # ------------------------------- #
        x2, results = self.step_Euler(
            model_fn=model_fn, timestep=timestep,
            sample=sample, next_timestep=next_timestep,
            eps=eps, model_kwargs=model_kwargs,
            return_intermediate_results=True)
        # ------------------------------- #

        # Trapezoidal
        # ------------------------------- #
        x_t_SN = results['x_t_SN']
        v1_SC = results['v_t_SC']
        varphi_t = results['varphi_t']

        a_tm1 = results['a_tm1']
        s_tm1 = results['s_tm1']
        inv_k_tm1 = results['inv_k_tm1']
        varphi_tm1 = results['varphi_tm1']

        noise2 = model_fn(x2, next_timestep, **model_kwargs)

        if self.straight_type == "x0_x1_interpolation":
            v2_SC = ((a_tm1 + s_tm1) * noise2 - x2) / a_tm1
        elif self.straight_type == "x1_scale_only":
            v2_SC = noise2
        else:
            raise ValueError(f"Invalid straight_type='{self.straight_type}'!")
        # ------------------------------- #

        # Final
        # ------------------------------- #
        v_final = 0.5 * (v1_SC + v2_SC)

        x_tm1_SN = x_t_SN + (varphi_tm1 - varphi_t) * v_final

        x_tm1 = x_tm1_SN * inv_k_tm1
        # ------------------------------- #

        if return_intermediate_results:
            return x_tm1, results
        else:
            return x_tm1

    def step_ExplicitMidpoint(
        self,
        model_fn: Callable,
        timestep: int,
        sample: FloatTensor,
        next_timestep: int,
        eps: float = 1e-3, model_kwargs: Dict = {},
        return_intermediate_results: bool = False,
    ):

        assert (timestep + next_timestep) % 2 == 0, \
            f"timestep + next_timestep must be an even number. " \
            f"Found {timestep + next_timestep}!"

        # t_mid always exist in the list of training timesteps because
        # it is in [next_timestep, timestep]
        mid_timestep = (timestep + next_timestep) // 2

        # Euler (v1)
        # ------------------------------- #
        _, results = self.step_Euler(
            model_fn=model_fn, timestep=timestep,
            sample=sample, next_timestep=next_timestep,
            eps=eps, model_kwargs=model_kwargs,
            return_intermediate_results=True)
        # ------------------------------- #

        # v2
        # ------------------------------- #
        x_t_SN = results['x_t_SN']
        v1_SC = results['v_t_SC']
        varphi_t = results['varphi_t']

        s_t_mid = self.s_t(mid_timestep)
        a_t_mid = robust_clipping(self.a_t(mid_timestep), eps=eps)
        coeffs_t_mid = self.get_coefficients_SN(
            a_t_mid, s_t_mid,
            straight_type=self.straight_type)

        inv_k_t_mid = coeffs_t_mid['inv_k_t']
        varphi_t_mid = coeffs_t_mid['varphi_t']

        x2_SN = x_t_SN + (varphi_t_mid - varphi_t) * v1_SC
        x2 = x2_SN * inv_k_t_mid

        noise2 = model_fn(x2, mid_timestep, **model_kwargs)

        if self.straight_type == "x0_x1_interpolation":
            v2_SC = ((a_t_mid + s_t_mid) * noise2 - x2) / a_t_mid
        elif self.straight_type == "x1_scale_only":
            v2_SC = noise2
        else:
            raise ValueError(f"Invalid straight_type='{self.straight_type}'!")
        # ------------------------------- #

        # Final
        # ------------------------------- #
        inv_k_tm1 = results['inv_k_tm1']
        varphi_tm1 = results['varphi_tm1']

        v_final = v2_SC

        x_tm1_SN = x_t_SN + (varphi_tm1 - varphi_t) * v_final

        x_tm1 = x_tm1_SN * inv_k_tm1
        # ------------------------------- #

        if return_intermediate_results:
            return x_tm1, results
        else:
            return x_tm1

    def step_ClassicKutta3(
        self,
        model_fn: Callable,
        timestep: int,
        sample: FloatTensor,
        next_timestep: int,
        eps: float = 1e-3, model_kwargs: Dict = {},
        return_intermediate_results: bool = False,
    ):

        assert (timestep + next_timestep) % 2 == 0, \
            f"timestep + next_timestep must be an even number. " \
            f"Found {timestep + next_timestep}!"

        # t_mid always exist in the list of training timesteps because
        # it is in [next_timestep, timestep]
        mid_timestep = (timestep + next_timestep) // 2

        # Euler (v1)
        # ------------------------------- #
        _, results = self.step_Euler(
            model_fn=model_fn, timestep=timestep,
            sample=sample, next_timestep=next_timestep,
            eps=eps, model_kwargs=model_kwargs,
            return_intermediate_results=True)
        # ------------------------------- #

        # v2
        # ------------------------------- #
        x_t_SN = results['x_t_SN']
        v1_SC = results['v_t_SC']
        varphi_t = results['varphi_t']

        s_t_mid = self.s_t(mid_timestep)
        a_t_mid = robust_clipping(self.a_t(mid_timestep), eps=eps)
        coeffs_t_mid = self.get_coefficients_SN(
            a_t_mid, s_t_mid,
            straight_type=self.straight_type)

        inv_k_t_mid = coeffs_t_mid['inv_k_t']
        varphi_t_mid = coeffs_t_mid['varphi_t']

        x2_SN = x_t_SN + (varphi_t_mid - varphi_t) * v1_SC
        x2 = x2_SN * inv_k_t_mid

        noise2 = model_fn(x2, mid_timestep, **model_kwargs)

        if self.straight_type == "x0_x1_interpolation":
            v2_SC = ((a_t_mid + s_t_mid) * noise2 - x2) / a_t_mid
        elif self.straight_type == "x1_scale_only":
            v2_SC = noise2
        else:
            raise ValueError(f"Invalid straight_type='{self.straight_type}'!")
        # ------------------------------- #

        # v3
        # ------------------------------- #
        a_tm1 = results['a_tm1']
        s_tm1 = results['s_tm1']
        inv_k_tm1 = results['inv_k_tm1']
        varphi_tm1 = results['varphi_tm1']

        x3_SN = x_t_SN + (varphi_tm1 - varphi_t) * (2 * v2_SC - v1_SC)
        x3 = x3_SN * inv_k_tm1

        noise3 = model_fn(x3, next_timestep, **model_kwargs)

        if self.straight_type == "x0_x1_interpolation":
            v3_SC = ((a_tm1 + s_tm1) * noise3 - x3) / a_tm1
        elif self.straight_type == "x1_scale_only":
            v3_SC = noise3
        else:
            raise ValueError(f"Invalid straight_type='{self.straight_type}'!")
        # ------------------------------- #

        # Final result
        # ------------------------------- #
        x_tm1_SN = x_t_SN + (varphi_tm1 - varphi_t) * (v1_SC + 4 * v2_SC + v3_SC) / 6

        x_tm1 = x_tm1_SN * inv_k_tm1
        # ------------------------------- #

        if return_intermediate_results:
            return x_tm1, results
        else:
            return x_tm1

    def step_AB2(
        self,
        model_fn: Callable,
        timestep: int,
        sample: FloatTensor,
        next_timestep: int,
        old_dict: Dict,
        current_state_dict: Optional[Dict] = None,
        step1_solver: str = "euler",
        eps: float = 1e-3, model_kwargs: Dict = {},
        return_intermediate_results: bool = False,
    ):
        # The Adams-Bashforth solver of order 2

        if old_dict['v_m1'] is None:
            if step1_solver == "euler":
                x_tm1, results = self.step_Euler(
                    model_fn=model_fn, timestep=timestep,
                    sample=sample, next_timestep=next_timestep,
                    eps=eps, model_kwargs=model_kwargs,
                    return_intermediate_results=True)

            elif step1_solver == "heun":
                x_tm1, results = self.step_Heun(
                    model_fn=model_fn, timestep=timestep,
                    sample=sample, next_timestep=next_timestep,
                    eps=eps, model_kwargs=model_kwargs,
                    return_intermediate_results=True)

            elif step1_solver == "midpoint":
                x_tm1, results = self.step_ExplicitMidpoint(
                    model_fn=model_fn, timestep=timestep,
                    sample=sample, next_timestep=next_timestep,
                    eps=eps, model_kwargs=model_kwargs,
                    return_intermediate_results=True)

            else:
                raise ValueError(f"Invalid step1_solver='{step1_solver}'!")

            old_dict['v_m1'] = results["v_t_SC"]
            old_dict['t_m1'] = timestep

        else:
            assert old_dict['v_m1'] is not None, f"v_m1 must be not None!"
            assert old_dict['t_m1'] is not None, f"t_m1 must be not None!"

            a_t = robust_clipping(self.a_t(timestep), eps=eps)
            s_t = self.s_t(timestep)

            a_tm1 = robust_clipping(self.a_t(next_timestep), eps=eps)
            s_tm1 = self.s_t(next_timestep)

            coeffs_t = self.get_coefficients_SN(
                a_t, s_t, straight_type=self.straight_type)

            coeffs_tm1 = self.get_coefficients_SN(
                a_tm1, s_tm1, straight_type=self.straight_type)

            inv_k_t = coeffs_t['inv_k_t']
            varphi_t = coeffs_t['varphi_t']

            inv_k_tm1 = coeffs_tm1['inv_k_t']
            varphi_tm1 = coeffs_tm1['varphi_t']

            if (current_state_dict is None) or \
               (current_state_dict['v_0'] is None):
                x_t = sample
                x_t_SN = x_t / inv_k_t

                # Compute v_t_SC
                # ------------------------------- #
                x_1_t = model_fn(sample, timestep, **model_kwargs)

                if self.straight_type == "x0_x1_interpolation":
                    v_t_SC = ((a_t + s_t) * x_1_t - x_t) / a_t
                elif self.straight_type == "x1_scale_only":
                    v_t_SC = x_1_t
                else:
                    raise ValueError(f"Invalid straight_type='{self.straight_type}'!")
                # ------------------------------- #

            else:
                # print("[AB2] Current_state_dict is provided. Use current_state_dict!")
                x_t_SN = current_state_dict['x_0']
                v_t_SC = current_state_dict['v_0']

            v_m1 = old_dict['v_m1']
            t_m1 = old_dict['t_m1']

            # Compute x_tm1
            # ------------------------------- #
            l0 = (next_timestep + timestep - 2 * t_m1) * 0.5 / (timestep - t_m1)
            l1 = -(next_timestep - timestep) * 0.5 / (timestep - t_m1)

            v_final = l0 * v_t_SC + l1 * v_m1

            x_tm1_SN = x_t_SN + (varphi_tm1 - varphi_t) * v_final

            x_tm1 = x_tm1_SN * inv_k_tm1
            # ------------------------------- #

            # Update old_dict
            # ------------------------------- #
            old_dict['v_m1'] = v_t_SC
            old_dict['t_m1'] = timestep
            # ------------------------------- #

            results = {
                "x_t_SN": x_t_SN,
                "v_t_SC": v_t_SC,
                "a_t": a_t,
                "s_t": s_t,
                "inv_k_t": inv_k_t,
                "varphi_t": varphi_t,

                "x_tm1_SN": x_tm1_SN,
                "a_tm1": a_tm1,
                "s_tm1": s_tm1,
                "inv_k_tm1": inv_k_tm1,
                "varphi_tm1": varphi_tm1,
            }

        if return_intermediate_results:
            return x_tm1, results
        else:
            return x_tm1

    def step_AB3(
        self,
        model_fn: Callable,
        timestep: int,
        sample: FloatTensor,
        next_timestep: int,
        old_dict: Dict,
        step1_solver: str = "euler",
        step2_solver: str = "ab2",
        current_state_dict: Optional[Dict] = None,
        eps: float = 1e-3, model_kwargs: Dict = {},
        return_intermediate_results: bool = False,
    ):

        # The Adams-Bashforth solver of order 3

        if old_dict["v_m2"] is None:
            if step1_solver == "euler":
                step1_fn = self.step_Euler
            elif step1_solver == "heun":
                step1_fn = self.step_Heun
            elif step1_solver == "classic_kutta3":
                step1_fn = self.step_ClassicKutta3
            else:
                raise ValueError(f"Invalid step1_solver='{step1_solver}'!")

            x_tm1, results = step1_fn(
                model_fn=model_fn,
                timestep=timestep,
                sample=sample,
                next_timestep=next_timestep,
                eps=eps, model_kwargs=model_kwargs,
                return_intermediate_results=True)

            old_dict["v_m2"] = results["v_t_SC"]
            old_dict["t_m2"] = timestep

        elif old_dict["v_m1"] is None:
            assert old_dict["v_m2"] is not None, f"v_m2 must be not None!"
            assert old_dict["t_m2"] is not None, f"t_m2 must be not None!"

            if step2_solver == "heun":
                step2_fn = self.step_Heun
                add_kwargs = {}
            elif step2_solver == "classic_kutta3":
                step2_fn = self.step_ClassicKutta3
                add_kwargs = {}
            elif step2_solver == "ab2":
                step2_fn = self.step_AB2
                add_kwargs = {
                    'old_dict': {
                        'v_m1': old_dict['v_m2'],
                        't_m1': old_dict['t_m2'],
                    },
                    'current_state_dict': current_state_dict,
                }
            else:
                raise ValueError(f"Invalid step2_solver='{step2_solver}'!")

            x_tm1, results = step2_fn(
                model_fn=model_fn,
                timestep=timestep,
                sample=sample,
                next_timestep=next_timestep,
                eps=eps, model_kwargs=model_kwargs,
                return_intermediate_results=True,
                **add_kwargs
            )

            old_dict["v_m1"] = results["v_t_SC"]
            old_dict["t_m1"] = timestep

        else:
            assert old_dict["v_m1"] is not None, f"v_m1 must be not None!"
            assert old_dict["t_m1"] is not None, f"t_m1 must be not None!"

            a_t = robust_clipping(self.a_t(timestep), eps=eps)
            s_t = self.s_t(timestep)

            a_tm1 = robust_clipping(self.a_t(next_timestep), eps=eps)
            s_tm1 = self.s_t(next_timestep)

            coeffs_t = self.get_coefficients_SN(
                a_t, s_t, straight_type=self.straight_type)

            coeffs_tm1 = self.get_coefficients_SN(
                a_tm1, s_tm1, straight_type=self.straight_type)

            inv_k_t = coeffs_t['inv_k_t']
            varphi_t = coeffs_t['varphi_t']

            inv_k_tm1 = coeffs_tm1['inv_k_t']
            varphi_tm1 = coeffs_tm1['varphi_t']

            if (current_state_dict is None) or \
               (current_state_dict['v_0'] is None):
                x_t = sample
                x_t_SN = x_t / inv_k_t

                # Compute v_t_SC
                # ------------------------------- #
                x_1_t = model_fn(sample, timestep, **model_kwargs)

                if self.straight_type == "x0_x1_interpolation":
                    v_t_SC = ((a_t + s_t) * x_1_t - x_t) / a_t
                elif self.straight_type == "x1_scale_only":
                    v_t_SC = x_1_t
                else:
                    raise ValueError(f"Invalid straight_type='{self.straight_type}'!")
                # ------------------------------- #
            else:
                x_t_SN = current_state_dict['x_0']
                v_t_SC = current_state_dict['v_0']

            # Compute x_tm1
            # ------------------------------ #
            v_m2 = old_dict['v_m2']
            t_m2 = old_dict['t_m2']

            v_m1 = old_dict['v_m1']
            t_m1 = old_dict['t_m1']

            # print(f"timestep: {timestep}, t_m1: {t_m1}, t_m2: {t_m2}")

            l0 = (next_timestep**2 + next_timestep * timestep + timestep**2) / 3. \
                 - ((t_m1 + t_m2) * (next_timestep + timestep) / 2.) + t_m1 * t_m2
            l0 = l0 / ((timestep - t_m1) * (timestep - t_m2))

            l1 = (next_timestep - timestep) * (2 * next_timestep + timestep - 3 * t_m2) / 6.
            l1 = l1 / ((t_m1 - timestep) * (t_m1 - t_m2))

            l2 = (next_timestep - timestep) * (2 * next_timestep + timestep - 3 * t_m1) / 6.
            l2 = l2 / ((t_m2 - timestep) * (t_m2 - t_m1))

            # print(f"At step {timestep}, l0={l0}, l1={l1}, l2={l2}")

            v_final = l0 * v_t_SC + l1 * v_m1 + l2 * v_m2
            # v_final = 23./12 * v_t_SC - 16./12 * v_m1 + 5./12 * v_m2

            x_tm1_SN = x_t_SN + (varphi_tm1 - varphi_t) * v_final

            x_tm1 = x_tm1_SN * inv_k_tm1
            # ------------------------------ #

            # Update old_dict
            # ------------------------------ #
            old_dict["v_m2"] = old_dict["v_m1"]
            old_dict["t_m2"] = old_dict["t_m1"]

            old_dict["v_m1"] = v_t_SC
            old_dict["t_m1"] = timestep
            # ------------------------------ #

            results = {
                "x_t_SN": x_t_SN,
                "v_t_SC": v_t_SC,
                "a_t": a_t,
                "s_t": s_t,
                "inv_k_t": inv_k_t,
                "varphi_t": varphi_t,

                "x_tm1_SN": x_tm1_SN,
                "a_tm1": a_tm1,
                "s_tm1": s_tm1,
                "inv_k_tm1": inv_k_tm1,
                "varphi_tm1": varphi_tm1,
            }

        if return_intermediate_results:
            return x_tm1, results
        else:
            return x_tm1

    def step_AB1C(
        self,
        model_fn: Callable,
        timestep: int,
        sample: FloatTensor,
        next_timestep: int,

        current_state_dict: Dict,
        is_last_step: bool,
        correct_last_step: bool = False,

        eps: float = 1e-3, model_kwargs: Dict = {},
        return_intermediate_results: bool = False,
    ):
        # Multistep Heun method

        a_t = robust_clipping(self.a_t(timestep), eps=eps)
        s_t = self.s_t(timestep)

        a_tm1 = robust_clipping(self.a_t(next_timestep), eps=eps)
        s_tm1 = self.s_t(next_timestep)

        coeffs_t = self.get_coefficients_SN(
            a_t, s_t, straight_type=self.straight_type)

        coeffs_tm1 = self.get_coefficients_SN(
            a_tm1, s_tm1, straight_type=self.straight_type)

        inv_k_t = coeffs_t['inv_k_t']
        varphi_t = coeffs_t['varphi_t']

        inv_k_tm1 = coeffs_tm1['inv_k_t']
        varphi_tm1 = coeffs_tm1['varphi_t']

        if current_state_dict['v_0'] is None:
            # print("[AB1C]>>>Do not use current state dict!!!")
            x_t = sample
            x_t_SN = x_t / inv_k_t

            # Compute v_t_SC
            # ------------------------------- #
            x_1_t = model_fn(sample, timestep, **model_kwargs)

            if self.straight_type == "x0_x1_interpolation":
                v_t_SC = ((a_t + s_t) * x_1_t - x_t) / a_t
            elif self.straight_type == "x1_scale_only":
                v_t_SC = x_1_t
            else:
                raise ValueError(f"Invalid straight_type='{self.straight_type}'!")
            # ------------------------------- #
        else:
            # print("[AB1C]>>>Use current state dict!!!")
            x_t_SN = current_state_dict['x_0']
            v_t_SC = current_state_dict['v_0']

        x_tm1_SN = x_t_SN + (varphi_tm1 - varphi_t) * v_t_SC
        x_tm1 = x_tm1_SN * inv_k_tm1

        results = {
            "x_t_SN": x_t_SN,
            "v_t_SC": v_t_SC,
            "a_t": a_t,
            "s_t": s_t,
            "inv_k_t": inv_k_t,
            "varphi_t": varphi_t,

            "x_tm1_SN": x_tm1_SN,
            "a_tm1": a_tm1,
            "s_tm1": s_tm1,
            "inv_k_tm1": inv_k_tm1,
            "varphi_tm1": varphi_tm1,
        }

        if is_last_step and (not correct_last_step):
            x_tm1_c = x_tm1

        else:
            # Compute v_tm1_SC
            # ------------------------------- #
            x_1_tm1 = model_fn(x_tm1, next_timestep, **model_kwargs)

            if self.straight_type == "x0_x1_interpolation":
                v_tm1_SC = ((a_tm1 + s_tm1) * x_1_tm1 - x_tm1) / a_tm1
            elif self.straight_type == "x1_scale_only":
                v_tm1_SC = x_1_tm1
            else:
                raise ValueError(f"Invalid straight_type='{self.straight_type}'!")
            # ------------------------------- #

            # Correct x_tm1
            # ------------------------------- #
            v_final = 0.5 * (v_tm1_SC + v_t_SC)

            x_tm1_SN_c = x_t_SN + (varphi_tm1 - varphi_t) * v_final

            x_tm1_c = x_tm1_SN_c * inv_k_tm1

            current_state_dict['v_0'] = v_tm1_SC
            current_state_dict['x_0'] = x_tm1_SN_c

        if return_intermediate_results:
            return x_tm1_c, results
        else:
            return x_tm1_c

    def step_AB2C(
        self,
        model_fn: Callable,
        timestep: int,
        sample: FloatTensor,
        next_timestep: int,

        old_dict: Dict,
        current_state_dict: Dict,
        is_last_step: bool,

        step1_solver: str = "euler",
        corrector: str = "am2",
        correct_last_step: bool = False,

        eps: float = 1e-3, model_kwargs: Dict = {},
        return_intermediate_results: bool = False,
    ):
        # This solver is inspired by UniPC
        # We run a Predictor and Corrector in parallel.

        # Once we get the output from the Predictor,
        # we can use it in the next Corrector step with a higher order.

        # An important thing here is that we DON'T use the
        # output of the Corrector for the next Predictor step.
        # Thus, we can prevent computation from being doubled as in
        # AB2-AM2 methods.

        # If correct_last_step: Use corrector for the last step
        # This require one more call to the velocity network
        # We can avoid this by using only the predictor

        # print(f"\n[AB2C] At step {timestep}:")

        v_m1 = old_dict.get('v_m1')
        t_m1 = old_dict.get('t_m1')

        is_first_step = (v_m1 is None)
        if is_first_step:
            assert current_state_dict['v_0'] is None, \
                f"current_state_dict['v_0'] must be None " \
                f"if this is the first step!"

        # print(f"t_old1: {t_m1}, type(v_old1): {type(v_m1)}")

        # Predictor step
        x_tm1, results = self.step_AB2(
            model_fn=model_fn,
            timestep=timestep,
            sample=sample,
            next_timestep=next_timestep,
            old_dict=old_dict,
            current_state_dict=current_state_dict,
            step1_solver=step1_solver,
            eps=eps, model_kwargs=model_kwargs,
            return_intermediate_results=True,
        )

        if is_last_step and (not correct_last_step):
            # No correction for the last step
            x_tm1_c = x_tm1

            # print("Last step and do not correct last step!!!")

        else:
            a_tm1 = results['a_tm1']
            s_tm1 = results['s_tm1']

            # Compute v_tm1_SC
            # ------------------------------- #
            x_1_tm1 = model_fn(x_tm1, next_timestep, **model_kwargs)

            if self.straight_type == "x0_x1_interpolation":
                v_tm1_SC = ((a_tm1 + s_tm1) * x_1_tm1 - x_tm1) / a_tm1
            elif self.straight_type == "x1_scale_only":
                v_tm1_SC = x_1_tm1
            else:
                raise ValueError(f"Invalid straight_type='{self.straight_type}'!")
            # ------------------------------- #

            # Correct x_tm1
            # ------------------------------- #
            x_t_SN = results['x_t_SN']
            v_t_SC = results['v_t_SC']
            varphi_t = results['varphi_t']

            inv_k_tm1 = results['inv_k_tm1']
            varphi_tm1 = results['varphi_tm1']

            # At the first step, v_old1 is None
            # Therefore, we can only use the am2 for correction at this step
            if is_first_step:
                # At the first step, we can only use the AM2 corrector
                v_final = 0.5 * (v_tm1_SC + v_t_SC)

            else:
                if corrector == "am2":
                    v_final = 0.5 * (v_tm1_SC + v_t_SC)

                elif corrector == "am3":
                    l0 = (2 * next_timestep + timestep - 3 * t_m1) / (6. * (next_timestep - t_m1))
                    l1 = (next_timestep + 2 * timestep - 3 * t_m1) / (6. * (timestep - t_m1))
                    l2 = -((next_timestep - timestep)**2) / (6. * (t_m1 - next_timestep) * (t_m1 - timestep))

                    # print(f"l0={l0}, l1={l1}, l2={l2}")

                    v_final = l0 * v_tm1_SC + l1 * v_t_SC + l2 * v_m1
                    # v_final = 5./12 * v_tm1_SC + 8./12 * v_t_SC - 1./12 * v_old1

                else:
                    raise ValueError(f"Invalid corrector='{corrector}'!")

            x_tm1_SN_c = x_t_SN + (varphi_tm1 - varphi_t) * v_final

            x_tm1_c = x_tm1_SN_c * inv_k_tm1

            current_state_dict['v_0'] = v_tm1_SC
            current_state_dict['x_0'] = x_tm1_SN_c
            # ------------------------------- #

        if return_intermediate_results:
            return x_tm1_c, results
        else:
            return x_tm1_c

    def step_AB3C(
        self,

        model_fn: Callable,
        timestep: int,
        sample: FloatTensor,
        next_timestep: int,

        old_dict: Dict,
        current_state_dict: Dict,
        is_last_step: bool,

        step1_solver: str = "euler",
        corrector: str = "am3",
        correct_last_step: bool = False,

        eps: float = 1e-3, model_kwargs: Dict = {},
        return_intermediate_results: bool = False,
    ):

        # print(f"\n[AB3C] At step {timestep}:")

        v_m2 = old_dict['v_m2']
        t_m2 = old_dict['t_m2']

        v_m1 = old_dict['v_m1']
        t_m1 = old_dict['t_m1']

        is_first_step = (v_m2 is None)
        if is_first_step:
            assert current_state_dict['v_0'] is None, \
                f"current_state_dict['v_0'] must be None " \
                f"if this is the first step!"

        is_second_step = (v_m2 is not None) and (v_m1 is None)
        if is_second_step:
            assert current_state_dict['v_0'] is not None, \
                f"current_state_dict['v_0'] must be not None " \
                f"if this is the second step!"

        # print(f"t_m2: {t_m2}, type(v_m2): {type(v_m2)}")
        # print(f"t_m1: {t_m1}, type(v_m1): {type(v_m1)}")

        # Predictor step
        x_tm1, results = self.step_AB3(
            model_fn=model_fn,
            timestep=timestep,
            sample=sample,
            next_timestep=next_timestep,
            old_dict=old_dict,
            current_state_dict=current_state_dict,
            step1_solver=step1_solver,
            step2_solver="ab2",
            eps=eps, model_kwargs=model_kwargs,
            return_intermediate_results=True,
        )

        if is_last_step and (not correct_last_step):
            # No correction for the last step
            x_tm1_c = x_tm1

            # print("Last step and do not correct last step!!!")

        else:
            a_tm1 = results['a_tm1']
            s_tm1 = results['s_tm1']

            # Compute v_tm1_SC
            # ------------------------------- #
            x_1_tm1 = model_fn(x_tm1, next_timestep, **model_kwargs)

            if self.straight_type == "x0_x1_interpolation":
                v_tm1_SC = ((a_tm1 + s_tm1) * x_1_tm1 - x_tm1) / a_tm1
            elif self.straight_type == "x1_scale_only":
                v_tm1_SC = x_1_tm1
            else:
                raise ValueError(f"Invalid straight_type='{self.straight_type}'!")
            # ------------------------------- #

            # Correct x_tm1
            # ------------------------------- #
            x_t_SN = results['x_t_SN']
            v_t_SC = results['v_t_SC']
            varphi_t = results['varphi_t']

            inv_k_tm1 = results['inv_k_tm1']
            varphi_tm1 = results['varphi_tm1']

            # At the first step, v_old1 is None
            # Therefore, we can only use the am2 for correction at this step
            if is_first_step:
                # print("At first step!")

                # At the first step, we can only use the AM2 corrector
                v_final = 0.5 * (v_tm1_SC + v_t_SC)

            elif is_second_step:
                # At the second step, we can use the AM3 corrector
                # At this step, we haven't had t_m1 and v_m1,
                # so we need to use t_m2 and v_m2

                l0 = (2 * next_timestep + timestep - 3 * t_m2) / (6. * (next_timestep - t_m2))
                l1 = (next_timestep + 2 * timestep - 3 * t_m2) / (6. * (timestep - t_m2))
                l2 = -((next_timestep - timestep) ** 2) / (6. * (t_m2 - next_timestep) * (t_m2 - timestep))

                # print(f"At second step, l0={l0}, l1={l1}, l2={l2}")

                v_final = l0 * v_tm1_SC + l1 * v_t_SC + l2 * v_m2

                # v_final = 0.5 * (v_tm1_SC + v_t_SC)

            else:
                if corrector == "am3":
                    l0 = (2 * next_timestep + timestep - 3 * t_m1) / (6. * (next_timestep - t_m1))
                    l1 = (next_timestep + 2 * timestep - 3 * t_m1) / (6. * (timestep - t_m1))
                    l2 = -((next_timestep - timestep) ** 2) / (6. * (t_m1 - next_timestep) * (t_m1 - timestep))

                    # print(f"At step {timestep}, l0={l0}, l1={l1}, l2={l2}")

                    v_final = l0 * v_tm1_SC + l1 * v_t_SC + l2 * v_m1

                elif corrector == "am4":
                    raise NotImplementedError("The AM4 corrector has not been implemented yet!")

                else:
                    raise ValueError(f"Invalid corrector='{corrector}'!")

            x_tm1_SN_c = x_t_SN + (varphi_tm1 - varphi_t) * v_final

            x_tm1_c = x_tm1_SN_c * inv_k_tm1

            current_state_dict['v_0'] = v_tm1_SC
            current_state_dict['x_0'] = x_tm1_SN_c
            # ------------------------------- #

        if return_intermediate_results:
            return x_tm1_c, results
        else:
            return x_tm1_c

    def step(
        self,
        model_fn: Callable,
        timestep: int,
        sample: FloatTensor,
        next_timestep: int,
        eps=1e-3, model_kwargs={},
    ):

        assert self.num_inference_steps is not None, \
            "self.num_inference_steps is 'None', " \
            "you need to run 'set_timesteps' in advance!"

        assert self.config.prediction_type == "epsilon", \
            f"self.config.prediction_type must be epsilon. " \
            f"Found '{self.config.prediction_type}'!"

        x_tm1 = self.step_Euler(
            model_fn=model_fn, timestep=timestep,
            sample=sample, next_timestep=next_timestep,
            eps=eps, model_kwargs=model_kwargs)

        return x_tm1

    def __len__(self):
        return self.config.num_train_timesteps


class SCScheduler_Euler(SCBaseScheduler):
    order = 1


class SCScheduler_Heun(SCBaseScheduler):
    order = 2

    def set_timesteps(self, num_inference_steps: int,
                      is_nfe: bool = True,
                      device: Union[str, torch.device] = None):

        if not is_nfe:
            SCBaseScheduler.set_timesteps(
                self,
                num_inference_steps=num_inference_steps,
                device=device)
        else:
            # Recaculate the step because Heun require double the number of step
            num_inference_steps = (num_inference_steps // 2) + \
                                  (num_inference_steps % 2)
            SCBaseScheduler.set_timesteps(
                self,
                num_inference_steps=num_inference_steps,
                device=device)

    def step(
        self,
        model_fn: Callable,
        timestep: int,
        sample: FloatTensor,
        next_timestep: int,
        eps=1e-3, model_kwargs={},
    ):

        assert self.num_inference_steps is not None, \
            "self.num_inference_steps is 'None', " \
            "you need to run 'set_timesteps' in advance!"

        assert self.config.prediction_type == "epsilon", \
            f"self.config.prediction_type must be epsilon. " \
            f"Found '{self.config.prediction_type}'!"

        x_tm1 = self.step_Heun(
            model_fn=model_fn, timestep=timestep,
            sample=sample, next_timestep=next_timestep,
            eps=eps, model_kwargs=model_kwargs
        )

        return x_tm1


class SCScheduler_AB2(SCBaseScheduler):
    # Whether first step is Heun or Euler
    def set_timesteps(self, num_inference_steps: int,
                      step1_solver: str = "euler",
                      is_nfe: bool = True,
                      device: Union[str, torch.device] = None):

        if not is_nfe:
            SCBaseScheduler.set_timesteps(
                self,
                num_inference_steps=num_inference_steps,
                device=device)
        else:
            if step1_solver == "euler":
                SCBaseScheduler.set_timesteps(
                    self,
                    num_inference_steps=num_inference_steps,
                    device=device)

            elif step1_solver == "heun":
                assert num_inference_steps > 1, \
                    f"num_inference_steps must be > 1. " \
                    f"Found {num_inference_steps}!"

                SCBaseScheduler.set_timesteps(
                    self,
                    num_inference_steps=num_inference_steps-1,
                    device=device)

            elif step1_solver == "midpoint":
                assert num_inference_steps > 1, \
                    f"num_inference_steps must be > 1. " \
                    f"Found {num_inference_steps}!"

                SCBaseScheduler.set_timesteps(
                    self,
                    num_inference_steps=num_inference_steps-1,
                    device=device)

                step0 = self.timesteps[0]
                step1 = self.timesteps[1]

                if (step0 + step1) % 2 != 0:
                    self.timesteps[1] = step1 - 1
                    assert self.timesteps[1] > self.timesteps[2], \
                        f"self.timesteps[1] ({self.timesteps[1]}) must be greater than " \
                        f"self.timesteps[2] ({self.timesteps[2]})"

            else:
                raise ValueError(f"Invalid step1_solver='{step1_solver}'!")

    def step(
        self,
        model_fn: Callable,
        timestep: int,
        sample: FloatTensor,
        next_timestep: int,
        old_dict: Dict,
        step1_solver="euler",
        eps=1e-3, model_kwargs={},
    ):

        assert self.num_inference_steps is not None, \
            "self.num_inference_steps is 'None', " \
            "you need to run 'set_timesteps' in advance!"

        assert self.config.prediction_type == "epsilon", \
            f"self.config.prediction_type must be epsilon. " \
            f"Found '{self.config.prediction_type}'!"

        x_tm1 = self.step_AB2(
            model_fn=model_fn, timestep=timestep,
            sample=sample, next_timestep=next_timestep,
            old_dict=old_dict, step1_solver=step1_solver,
            eps=eps, model_kwargs=model_kwargs
        )

        return x_tm1


class SCScheduler_AB3(SCScheduler_AB2):
    def set_timesteps(self, num_inference_steps: int,
                      step1_solver: str = "euler",
                      step2_solver: str = "ab2",
                      is_nfe: bool = True,
                      device: Union[str, torch.device] = None):

        if not is_nfe:
            return SCBaseScheduler.set_timesteps(
                self,
                num_inference_steps=num_inference_steps,
                device=device)
        else:
            if step2_solver == "ab2":
                SCScheduler_AB2.set_timesteps(
                    self,
                    num_inference_steps=num_inference_steps,
                    step1_solver=step1_solver,
                    is_nfe=is_nfe, device=device)

            else:
                num_init_steps = 0
                if step2_solver == "heun":
                    num_init_steps += 1
                elif step2_solver == "classic_kutta3":
                    num_init_steps += 2
                else:
                    raise ValueError(f"Invalid step2_solver='{step2_solver}'!")

                if step1_solver == "heun":
                    num_init_steps += 1
                elif step1_solver == "classic_kutta3":
                    num_init_steps += 2
                else:
                    raise ValueError(f"Invalid step1_solver='{step1_solver}'!")

                assert num_inference_steps > num_init_steps, \
                    f"num_inference_steps ({num_inference_steps}) " \
                    f"must be > {num_init_steps} if " \
                    f"step1_solver='{step1_solver}' and " \
                    f"step2_solver='{step2_solver}'!"

                # We need to ensure the t_mid is support when
                # the step1_solver or step2_solver is "classic_kutta3"
                SCBaseScheduler.set_timesteps(
                    self,
                    num_inference_steps=num_inference_steps - num_init_steps,
                    device=device)

                if step1_solver == "classic_kutta3":
                    # Check to see whether step 0 and step 1
                    # has an even sum or not
                    step0 = self.timesteps[0]
                    step1 = self.timesteps[1]

                    if (step0 + step1) % 2 != 0:
                        self.timesteps[1] = step1 - 1
                        assert self.timesteps[1] > self.timesteps[2], \
                            f"self.timesteps[1] ({self.timesteps[1]}) must be greater than " \
                            f"self.timesteps[2] ({self.timesteps[2]})"

                if step2_solver == "classic_kutta3":
                    # Check to see whether step 0 and step 1
                    # has an even sum or not
                    step1 = self.timesteps[1]
                    step2 = self.timesteps[2]

                    if (step1 + step2) % 2 != 0:
                        self.timesteps[2] = step2 - 1
                        assert self.timesteps[2] > self.timesteps[3], \
                            f"self.timesteps[2] ({self.timesteps[2]}) must be greater than " \
                            f"self.timesteps[3] ({self.timesteps[3]})"

    def step(
        self,
        model_fn: Callable,
        timestep: int,
        sample: FloatTensor,
        next_timestep: int,
        old_dict: Dict,
        step1_solver="euler",
        step2_solver="ab2",
        eps=1e-3, model_kwargs={},
    ):

        assert self.num_inference_steps is not None, \
            "self.num_inference_steps is 'None', " \
            "you need to run 'set_timesteps' in advance!"

        assert self.config.prediction_type == "epsilon", \
            f"self.config.prediction_type must be epsilon. " \
            f"Found '{self.config.prediction_type}'!"

        x_tm1 = self.step_AB3(
            model_fn=model_fn, timestep=timestep,
            sample=sample, next_timestep=next_timestep,
            old_dict=old_dict,
            step1_solver=step1_solver,
            step2_solver=step2_solver,
            eps=eps, model_kwargs=model_kwargs
        )

        return x_tm1


class SCScheduler_AB1C(SCBaseScheduler):
    # Whether first step is Heun or Euler
    def set_timesteps(self, num_inference_steps: int,
                      correct_last_step: bool = False,
                      is_nfe: bool = True,
                      device: Union[str, torch.device] = None):

        if not is_nfe:
            SCBaseScheduler.set_timesteps(
                self,
                num_inference_steps=num_inference_steps,
                device=device)
        else:
            if correct_last_step:
                num_inference_steps = num_inference_steps - 1

            SCBaseScheduler.set_timesteps(
                self, num_inference_steps=num_inference_steps,
                device=device)

    def step(
        self,
        model_fn: Callable,
        timestep: int,
        sample: FloatTensor,
        next_timestep: int,

        current_state_dict: Dict,
        is_last_step: bool,
        correct_last_step: bool = False,

        eps: float = 1e-3, model_kwargs: Dict = {},
    ):

        assert self.num_inference_steps is not None, \
            "self.num_inference_steps is 'None', " \
            "you need to run 'set_timesteps' in advance!"

        assert self.config.prediction_type == "epsilon", \
            f"self.config.prediction_type must be epsilon. " \
            f"Found '{self.config.prediction_type}'!"

        x_tm1_c = self.step_AB1C(
            model_fn=model_fn, timestep=timestep,
            sample=sample, next_timestep=next_timestep,
            current_state_dict=current_state_dict,
            is_last_step=is_last_step,
            correct_last_step=correct_last_step,
            eps=eps, model_kwargs=model_kwargs
        )

        return x_tm1_c


class SCScheduler_AB2C(SCScheduler_AB2):
    # Whether first step is Heun or Euler
    def set_timesteps(self, num_inference_steps: int,
                      step1_solver: str = "euler",
                      correct_last_step: bool = False,
                      is_nfe: bool = True,
                      device: Union[str, torch.device] = None):

        if not is_nfe:
            SCBaseScheduler.set_timesteps(
                self,
                num_inference_steps=num_inference_steps,
                device=device)
        else:
            if correct_last_step:
                num_inference_steps = num_inference_steps - 1

            SCScheduler_AB2.set_timesteps(
                self, num_inference_steps=num_inference_steps,
                step1_solver=step1_solver, is_nfe=is_nfe, device=device)

    def step(
        self,
        model_fn: Callable,
        timestep: int,
        sample: FloatTensor,
        next_timestep: int,

        old_dict: Dict,
        current_state_dict: Dict,
        is_last_step: bool,

        step1_solver: str = "euler",
        corrector: str = "am2",
        correct_last_step: bool = False,

        eps: float = 1e-3, model_kwargs: Dict = {},
    ):

        assert self.num_inference_steps is not None, \
            "self.num_inference_steps is 'None', " \
            "you need to run 'set_timesteps' in advance!"

        assert self.config.prediction_type == "epsilon", \
            f"self.config.prediction_type must be epsilon. " \
            f"Found '{self.config.prediction_type}'!"

        x_tm1_c = self.step_AB2C(
            model_fn=model_fn, timestep=timestep,
            sample=sample, next_timestep=next_timestep,
            old_dict=old_dict, current_state_dict=current_state_dict,
            is_last_step=is_last_step,
            step1_solver=step1_solver, corrector=corrector,
            correct_last_step=correct_last_step,
            eps=eps, model_kwargs=model_kwargs
        )

        return x_tm1_c


class SCScheduler_AB3C(SCScheduler_AB3):
    # Whether first step is Heun or Euler
    def set_timesteps(self, num_inference_steps: int,
                      step1_solver: str = "euler",
                      correct_last_step: bool = False,
                      is_nfe: bool = True,
                      device: Union[str, torch.device] = None):

        if not is_nfe:
            SCBaseScheduler.set_timesteps(
                self,
                num_inference_steps=num_inference_steps,
                device=device)
        else:
            if correct_last_step:
                num_inference_steps = num_inference_steps - 1

            SCScheduler_AB3.set_timesteps(
                self, num_inference_steps=num_inference_steps,
                step1_solver=step1_solver, step2_solver="ab2",
                is_nfe=is_nfe, device=device)

    def step(
        self,
        model_fn: Callable,
        timestep: int,
        sample: FloatTensor,
        next_timestep: int,

        old_dict: Dict,
        current_state_dict: Dict,
        is_last_step: bool,

        step1_solver: str = "euler",
        corrector: str = "am3",
        correct_last_step: bool = False,

        eps: float = 1e-3, model_kwargs: Dict = {},
    ):

        assert self.num_inference_steps is not None, \
            "self.num_inference_steps is 'None', " \
            "you need to run 'set_timesteps' in advance!"

        assert self.config.prediction_type == "epsilon", \
            f"self.config.prediction_type must be epsilon. " \
            f"Found '{self.config.prediction_type}'!"

        x_tm1_c = self.step_AB3C(
            model_fn=model_fn, timestep=timestep,
            sample=sample, next_timestep=next_timestep,
            old_dict=old_dict, current_state_dict=current_state_dict,
            is_last_step=is_last_step,
            step1_solver=step1_solver,
            corrector=corrector, correct_last_step=correct_last_step,
            eps=eps, model_kwargs=model_kwargs
        )

        return x_tm1_c