"""Forward vector angular-spectrum model without file-system side effects.

The time convention is exp(-i wt). Equivalent-aperture fields are normalized
to 1 W of forward radiation before propagation. Local flux remains signed.
This is not a hardware, room, or body-scattering full-wave model.
"""

from copy import deepcopy

import numpy as np

from .config import (
    C0,
    CFG,
    LONGITUDINAL_HALF_WIDTH_M,
    PUBLIC_PLANE_HALF_WIDTH_M,
    PUBLIC_RECTANGULAR_MARGIN_M,
    Z0,
)


def coverage(x, width, dx, center=0.0):
    """Return each cell's fractional overlap with a 1-D interval."""
    return np.maximum(
        0.0,
        np.minimum(x + dx / 2, center + width / 2)
        - np.maximum(x - dx / 2, center - width / 2),
    ) / dx


def quantize_phase(phase, bits):
    """Round phase to the bit resolution; None preserves continuous phase."""
    if bits is None:
        return phase
    step = 2 * np.pi / 2**bits
    return np.round(phase / step) * step


def field_metrics(fields):
    """Convert vector fields to signed flux and squared RMS fields."""
    ux, uy, uz, vy, vz = fields
    sz = (ux * vy.conj()).real
    e2 = Z0 * (np.abs(ux)**2 + np.abs(uy)**2 + np.abs(uz)**2)
    h2 = (np.abs(vy)**2 + np.abs(vz)**2) / Z0
    return {"sz": sz, "e2": e2, "h2": h2}


class Aperture:
    """A sampled equivalent aperture with a private configuration snapshot."""

    def __init__(self, n=None, dx_fraction=None, config=None):
        self.config = deepcopy(CFG)
        if config is not None:
            self.config.update(deepcopy(config))
        self.n = self.config["n"] if n is None else n
        dx_fraction = (
            self.config["dx_in_wavelengths"]
            if dx_fraction is None else dx_fraction
        )
        self.config["n"] = self.n
        self.config["dx_in_wavelengths"] = dx_fraction
        self.lam = C0 / (self.config["frequency_ghz"] * 1e9)
        self.dx = self.lam * dx_fraction
        self.x = (np.arange(self.n) - self.n // 2) * self.dx
        self.xx, self.yy = np.meshgrid(self.x, self.x)
        k = 2 * np.pi / self.lam
        q = 2 * np.pi * np.fft.fftfreq(self.n, d=self.dx) / k
        sx, sy = np.meshgrid(q, q)
        valid = sx * sx + sy * sy < 1.0 - 1e-10
        sz = np.sqrt(np.maximum(0.0, 1 - sx * sx - sy * sy))
        den = np.sqrt(np.maximum(1e-12, 1 - sx * sx))
        self.factors = [
            np.where(valid, (1 - sx * sx) / den, 0),
            np.where(valid, -sx * sy / den, 0),
            np.where(valid, -sx * sz / den, 0),
            np.where(valid, sz / den, 0),
            np.where(valid, -sy / den, 0),
        ]
        self.valid, self.sz, self.kz = valid, sz, k * sz
        cfg = self.config
        self.rx = self.mask(cfg["phone_width_m"], cfg["phone_height_m"])
        self.upper_rx = self.mask(
            cfg["phone_envelope_width_m"], cfg["phone_envelope_height_m"]
        )

    def mask(self, width, height, cx=0.0, cy=0.0):
        """Return a rectangular mask including fractional boundary cells."""
        return (
            coverage(self.x, width, self.dx, cx)[None, :]
            * coverage(self.x, height, self.dx, cy)[:, None]
        )

    def channel_layout(self, side):
        """Return centered-grid fill, labels, and 1-D channel centers."""
        count = self.config["tx_channels_per_axis"]
        tile = side / count
        ix = np.clip(
            np.floor((self.xx + side / 2) / tile), 0, count - 1
        ).astype(int)
        iy = np.clip(
            np.floor((self.yy + side / 2) / tile), 0, count - 1
        ).astype(int)
        labels = iy * count + ix
        centers = (np.arange(count) + 0.5) * tile - side / 2
        return self.mask(side, side), labels, centers

    def focusing_phase(self, x, y, focus):
        """Spherical conjugate phase for a target on the aperture normal."""
        return -2 * np.pi / self.lam * (
            np.sqrt(focus * focus + x * x + y * y) - focus
        )

    def raw_spectrum_and_power(self, field):
        """FFT a field and return its propagating spectrum and power.

        The returned spectrum is deliberately not normalized: the optimizer
        needs both the numerator and denominator of its Rayleigh quotient.
        """
        raw = np.fft.fft2(field)
        raw[~self.valid] = 0.0
        power = float(
            self.dx**2 / self.n**2 * np.sum(np.abs(raw)**2 * self.sz)
        )
        return raw, power

    def normalized_spectrum(self, field):
        """Normalize an unshifted field to 1 W of forward radiation."""
        raw, power = self.raw_spectrum_and_power(field)
        assert power > 0
        return raw / np.sqrt(power)

    def source(self, side, focus, architecture="144", bits="config"):
        """Construct a source; explicit bits=None disables phase quantization.

        The historical architecture name "144" means the configured square
        channel layout (12 by 12 in the default study).
        """
        if architecture == "144":
            fill, labels, centers = self.channel_layout(side)
            count = self.config["tx_channels_per_axis"]
            x, y = centers[labels % count], centers[labels // count]
        elif architecture == "continuous":
            fill = self.mask(side, side)
            x, y = self.xx, self.yy
        elif architecture == "unfocused":
            fill = self.mask(side, side)
            x, y = 0.0, 0.0
        else:
            raise ValueError(architecture)
        if bits == "config":
            bits = self.config["phase_bits"]
        phase = quantize_phase(self.focusing_phase(x, y, focus), bits)
        field = np.fft.ifftshift(fill * np.exp(1j * phase))
        return self.normalized_spectrum(field)

    def plane(self, spectrum, z):
        """Propagate a normalized spectrum onto a complete sampled plane."""
        prop = spectrum * np.exp(1j * self.kz * z)
        fields = [
            np.fft.fftshift(np.fft.ifft2(prop * factor))
            for factor in self.factors
        ]
        return field_metrics(fields)

    def summarize(self, plane):
        """Compute receiver budgets and specified-plane E/H screening."""
        cfg = self.config
        cap = float(np.sum(plane["sz"] * self.rx) * self.dx**2)
        cap_upper = float(np.sum(plane["sz"] * self.upper_rx) * self.dx**2)
        assert 0 < cap <= 1.00001 and cap <= cap_upper <= 1.00001
        total = float(plane["sz"].sum() * self.dx**2)
        rx_points = self.rx > 0
        outside = (
            (np.abs(self.xx) >= cfg["phone_width_m"] / 2
             + PUBLIC_RECTANGULAR_MARGIN_M)
            | (np.abs(self.yy) >= cfg["phone_height_m"] / 2
               + PUBLIC_RECTANGULAR_MARGIN_M)
        ) & (
            (np.abs(self.xx) <= PUBLIC_PLANE_HALF_WIDTH_M)
            & (np.abs(self.yy) <= PUBLIC_PLANE_HALF_WIDTH_M)
        )
        emax = float(plane["e2"][rx_points].max())
        hmax = float(plane["h2"][rx_points].max())
        cap_rad = min(
            cfg["public_e_rms_limit_v_m"]**2 / emax,
            cfg["public_h_rms_limit_a_m"]**2 / hmax,
        )
        eta = (
            cfg["tx_radiation_efficiency"] * cap
            * cfg["rx_collection_efficiency"]
            * cfg["rectifier_efficiency"] * cfg["pmic_efficiency"]
        )
        target_feed = cfg["target_dc_w"] / eta
        target_radiated = target_feed * cfg["tx_radiation_efficiency"]
        # Historical output keys are retained for existing reports and data.
        return {
            "capture_fraction": cap,
            "envelope_capture_fraction": cap_upper,
            "plane_power_w_per_radiated_w": total,
            "rf_feed_to_load_efficiency": eta,
            "rf_feed_for_5w_w": target_feed,
            "radiated_for_5w_w": target_radiated,
            "wall_for_5w_w": (
                target_feed / (cfg["psu_efficiency"] * cfg["pa_efficiency"])
                + cfg["aux_wall_w"]
            ),
            "rx_peak_e_rms_at_5w_v_m": np.sqrt(emax * target_radiated),
            "rx_peak_h_rms_at_5w_a_m": np.sqrt(hmax * target_radiated),
            "plane_outside_20cm_peak_e_rms_at_5w_v_m": np.sqrt(
                float(plane["e2"][outside].max()) * target_radiated
            ),
            "plane_outside_20cm_peak_h_rms_at_5w_a_m": np.sqrt(
                float(plane["h2"][outside].max()) * target_radiated
            ),
            "rx_plane_eh_constrained_rf_feed_w": (
                cap_rad / cfg["tx_radiation_efficiency"]
            ),
            "rx_plane_eh_constrained_load_w": (
                cap_rad * cap * cfg["rx_collection_efficiency"]
                * cfg["rectifier_efficiency"] * cfg["pmic_efficiency"]
            ),
            "minimum_local_sz_w_m2_per_radiated_w": float(plane["sz"].min()),
        }

    def longitudinal(self, spectrum, z_values):
        """Extract the y=0 slice; this is not a three-dimensional maximum."""
        selected = np.abs(self.x) <= LONGITUDINAL_HALF_WIDTH_M
        all_s, all_e, all_h = [], [], []
        for z in z_values:
            prop = spectrum * np.exp(1j * self.kz * z)
            fields = [
                np.fft.fftshift(
                    np.fft.ifft(np.sum(prop * factor, axis=0) / self.n)
                )[selected]
                for factor in self.factors
            ]
            metrics = field_metrics(fields)
            all_s.append(metrics["sz"])
            all_e.append(metrics["e2"])
            all_h.append(metrics["h2"])
        return (
            self.x[selected], np.array(all_s), np.array(all_e), np.array(all_h)
        )


def fresnel_capture(side, distance, a=None, b=None, config=None):
    """Independent scalar Fresnel capture check for a rectangular receiver."""
    cfg = CFG if config is None else config
    a = cfg["phone_width_m"] if a is None else a
    b = cfg["phone_height_m"] if b is None else b
    roots, weights = np.polynomial.legendre.leggauss(128)

    def fraction(width):
        q = side * width / (C0 / (cfg["frequency_ghz"] * 1e9) * distance)
        return float(q / 2 * np.sum(weights * np.sinc(q * roots / 2)**2))

    return fraction(a) * fraction(b)
