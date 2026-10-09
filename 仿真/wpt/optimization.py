"""Local phase optimization for the configured square channel aperture.

The objective is signed power through the phone divided by total forward
radiation. The FFT adjoint differentiates both parts of that quotient.
"""

import numpy as np

from .config import MAX_OPTIMIZATION_ITERATIONS


class PhaseOptimizer:
    """Optimize continuous channel phases before final hardware quantization.

    The search is local. Reaching an iteration limit is not a convergence
    certificate, and rounding afterwards does not solve the discrete global
    optimization problem.
    """

    def __init__(self, grid, side, z):
        self.g, self.side, self.z = grid, side, z
        self.channel_count = grid.config["tx_channels_per_axis"]**2
        fill, labels, centers = grid.channel_layout(side)
        self.labels = np.fft.ifftshift(labels)
        self.fill = np.fft.ifftshift(fill)
        self.rx = np.fft.ifftshift(grid.rx)
        prop = np.exp(1j * grid.kz * z)
        self.lx = grid.factors[0] * prop
        self.ly = grid.factors[3] * prop
        x, y = np.meshgrid(centers, centers)
        self.initial = grid.focusing_phase(x, y, z).ravel()
        self.last_status = None

    def evaluate(self, phi, gradient=False):
        """Return the Rayleigh quotient and, optionally, its phase gradient."""
        grid = self.g
        field = self.fill * np.exp(1j * phi[self.labels])
        spectrum, norm = grid.raw_spectrum_and_power(field)
        ux = np.fft.ifft2(spectrum * self.lx)
        vy = np.fft.ifft2(spectrum * self.ly)
        cap = float(grid.dx**2 * np.sum(self.rx * (ux * vy.conj()).real))
        objective = cap / norm
        if not gradient:
            return objective
        grad_cap = (
            np.fft.ifft2(self.lx.conj() * np.fft.fft2(self.rx * vy))
            + np.fft.ifft2(self.ly.conj() * np.fft.fft2(self.rx * ux))
        )
        grad_norm = 2 * np.fft.ifft2(grid.sz * spectrum)
        pixel_gradient = grid.dx**2 / norm * np.imag(
            field.conj() * (grad_cap - objective * grad_norm)
        )
        grad = np.bincount(
            self.labels.ravel(), weights=pixel_gradient.ravel(),
            minlength=self.channel_count,
        )
        return objective, grad

    def spectrum(self, phi):
        """Return the candidate's normalized radiating spectrum."""
        field = self.fill * np.exp(1j * phi[self.labels])
        return self.g.normalized_spectrum(field)

    def optimize(
        self, initial=None, max_iterations=MAX_OPTIMIZATION_ITERATIONS
    ):
        """Run the original BB/backtracking search and return phases, history.

        ``last_status`` records why the search stopped without changing the
        return signature or the optimization steps. History records objectives
        evaluated at the beginning of iterations; final_objective also covers
        a final accepted step that was not followed by another iteration.
        """
        phi = self.initial.copy() if initial is None else initial.copy()
        history = []
        step = 100.0
        prev_phi = prev_grad = None
        reason = "max_iterations"
        accepted_steps = 0
        final_value = None
        last_gradient_inf_norm = None
        for iteration in range(max_iterations):
            value, grad = self.evaluate(phi, True)
            history.append(value)
            final_value = value
            last_gradient_inf_norm = float(np.max(np.abs(grad)))
            if last_gradient_inf_norm < 1e-8:
                reason = "gradient_tolerance"
                break
            if prev_phi is not None:
                delta_phi = phi - prev_phi
                delta_grad = grad - prev_grad
                curvature = -float(np.dot(delta_phi, delta_grad))
                if curvature > 1e-15:
                    step = np.clip(
                        float(np.dot(delta_phi, delta_phi)) / curvature,
                        0.05, 2e4,
                    )
            step = min(step, 0.5 / max(np.max(np.abs(grad)), 1e-12))
            accepted = False
            for _ in range(14):
                candidate = phi + step * grad
                new = self.evaluate(candidate)
                if new >= value + 1e-4 * step * float(np.dot(grad, grad)):
                    accepted = True
                    break
                step *= 0.5
            if not accepted:
                reason = "line_search_failed"
                break
            prev_phi, prev_grad = phi.copy(), grad.copy()
            phi = candidate
            accepted_steps += 1
            final_value = new
            if iteration > 12 and new - value < 1e-8:
                reason = "objective_tolerance"
                break
        self.last_status = {
            "stop_reason": reason,
            "iterations": len(history),
            "accepted_steps": accepted_steps,
            "final_objective": final_value,
            "last_evaluated_gradient_inf_norm": last_gradient_inf_norm,
        }
        return phi, history
