"""Shared parameters for the fixed-phone aperture study.

The default values reproduce the published simulation. Scan selections and
plotting/export extents are kept separate from the physical configuration.
"""

C0 = 299792458.0
Z0 = 376.730313668

CFG = {
    "frequency_ghz": 60.0,
    "phone_width_m": 0.05,
    "phone_height_m": 0.10,
    "phone_envelope_width_m": 0.07,
    "phone_envelope_height_m": 0.14,
    "rx_collection_efficiency": 0.60,
    "rectifier_efficiency": 0.50,
    "pmic_efficiency": 0.90,
    "tx_radiation_efficiency": 0.60,
    "psu_efficiency": 0.90,
    "pa_efficiency": 0.35,
    "aux_wall_w": 5.0,
    "reference_distance_m": 3.0,
    "tx_channels_per_axis": 12,
    "phase_bits": 6,
    "target_dc_w": 5.0,
    "public_e_rms_limit_v_m": 27.0,
    "public_h_rms_limit_a_m": 0.073,
    "n": 1024,
    "dx_in_wavelengths": 0.5,
    "tx_side_m": [
        0.03, 0.05, 0.075, 0.10, 0.125, 0.15, 0.175, 0.20, 0.225,
        0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60,
    ],
    "rf_feed_power_w": [1, 5, 10, 20, 30, 50, 75, 100, 150, 200, 300],
}

OPTIMIZED_TX_SIDES_M = (
    0.03, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60,
)
DISTANCE_SCAN_M = (1.0, 5.0)
DISTANCE_TX_SIDES_M = (0.10, 0.20, 0.30, 0.40, 0.60)
CONTROL_TX_SIDES_M = (0.20, 0.40, 0.60)
FIELD_REFERENCE_SIDE_M = 0.20
CHECK_GRIDS = (("resolution", 2048, 0.25), ("window", 2048, 0.5))

PUBLIC_RECTANGULAR_MARGIN_M = 0.20
PUBLIC_PLANE_HALF_WIDTH_M = 1.0
LONGITUDINAL_HALF_WIDTH_M = 0.75
TARGET_FIELD_HALF_WIDTH_M = 0.40
LONGITUDINAL_START_M = 0.5
LONGITUDINAL_STOP_M = 5.0
LONGITUDINAL_SAMPLES = 91

MAX_OPTIMIZATION_ITERATIONS = 70
GRADIENT_CHECK_PHASE_SEED = 20261008
GRADIENT_CHECK_DIRECTION_SEED = 4
