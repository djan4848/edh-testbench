"""
Phase 1 acceptance (calibration spec §7): the lambda-selection LOGIC is correct
on synthetic chi(N) curves. Does NOT run the simulation.

  * narrow unimodal peak centered at N=10 -> valid & centered
  * peak at the high edge (N=20)          -> interior False -> rejected, "alto"
  * two comparable peaks                   -> unimodal False -> rejected
  * flat / very wide                       -> sharp False    -> rejected
  * several valid rows                     -> smallest lambda chosen
"""
from __future__ import annotations

import numpy as np

from edh.finite_size import peak_report_from_chi
from exp_p1_calibrate_lambda import (
    CalibConfig,
    LambdaRow,
    classify_lambda,
    diagnose_failure,
    select_lambda,
)

N_GRID = np.arange(2, 21, dtype=float)
CFG = CalibConfig()


def _gauss(center, width, amp=1.0):
    return amp * np.exp(-0.5 * ((N_GRID - center) / width) ** 2)


def _row(lam, chi):
    peak = peak_report_from_chi(
        N_GRID, chi, accept_band=CFG.accept_band, target_band=CFG.target_band,
        fwhm_max=CFG.fwhm_max,
    )
    return classify_lambda(lam, N_GRID, chi, peak, converged_frac=1.0, cfg=CFG)


def test_narrow_centered_peak_is_valid():
    row = _row(1.0, _gauss(10, 1.2))
    assert row.interior and row.unimodal and row.sharp and row.centered
    assert row.valid
    assert abs(row.n_peak - 10.0) < 1.0


def test_edge_peak_rejected_high():
    row = _row(1.0, _gauss(21, 1.2))      # peak pushed past the high edge
    assert not row.interior
    assert not row.valid
    rows = [row]
    assert "HIGH edge" in diagnose_failure(rows)


def test_two_peaks_not_unimodal():
    chi = _gauss(7, 1.0) + _gauss(14, 0.98)
    row = _row(1.0, chi)
    assert not row.unimodal
    assert not row.valid


def test_flat_wide_not_sharp():
    row = _row(1.0, _gauss(10, 9.0))      # very broad peak, fwhm > fwhm_max
    assert not row.sharp
    assert not row.valid


def test_selects_smallest_valid_lambda():
    rows = [
        _row(0.3, _gauss(10, 1.2)),
        _row(1.0, _gauss(10, 1.1)),
        _row(3.0, _gauss(10, 1.0)),
    ]
    assert all(r.valid for r in rows)
    chosen, prov = select_lambda(rows)
    assert chosen == 0.3
    assert prov["chosen_lambda"] == 0.3


def test_no_valid_returns_diagnosis():
    rows = [_row(1.0, _gauss(21, 1.0)), _row(3.0, _gauss(22, 1.0))]
    chosen, prov = select_lambda(rows)
    assert chosen is None
    assert "reason" in prov and isinstance(prov["reason"], str)


def test_lowedge_diagnosis():
    rows = [LambdaRow(lam=0.1, n_peak=2.0, n_peak_int=2, chi_max=1.0, fwhm=1.0,
                      unimodal_ratio=5.0, interior=False, sharp=True,
                      centered=False, unimodal=True, converged=True, valid=False)]
    assert "LOW edge" in diagnose_failure(rows)
