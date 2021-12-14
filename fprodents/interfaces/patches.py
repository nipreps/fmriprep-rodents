# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Temporary patches
-----------------

"""

from random import randint
from time import sleep

from numpy.linalg.linalg import LinAlgError
from nipype.algorithms import confounds as nac
from nipype.interfaces.base import traits
from nipype.interfaces.fsl.preprocess import FAST, FASTInputSpec


class RobustACompCor(nac.ACompCor):
    """
    Runs aCompCor several times if it suddenly fails with
    https://github.com/nipreps/fmriprep/issues/776

    """

    def _run_interface(self, runtime):
        failures = 0
        while True:
            try:
                runtime = super(RobustACompCor, self)._run_interface(runtime)
                break
            except LinAlgError:
                failures += 1
                if failures > 10:
                    raise
                start = (failures - 1) * 10
                sleep(randint(start + 4, start + 10))

        return runtime


class RobustTCompCor(nac.TCompCor):
    """
    Runs tCompCor several times if it suddenly fails with
    https://github.com/nipreps/fmriprep/issues/940

    """

    def _run_interface(self, runtime):
        failures = 0
        while True:
            try:
                runtime = super(RobustTCompCor, self)._run_interface(runtime)
                break
            except LinAlgError:
                failures += 1
                if failures > 10:
                    raise
                start = (failures - 1) * 10
                sleep(randint(start + 4, start + 10))

        return runtime


class _FixTraitFASTInputSpec(FASTInputSpec):
    bias_iters = traits.Range(
        low=0,
        high=10,
        argstr="-I %d",
        desc="number of main-loop iterations during bias-field removal",
    )


class FixBiasItersFAST(FAST):
    """
    A replacement for nipype.interfaces.ants.resampling.ApplyTransforms that
    fixes the resampled image header to match the xform of the reference
    image
    """

    input_spec = _FixTraitFASTInputSpec

    def _run_interface(self, runtime, correct_return_codes=(0,)):
        # Run normally
        runtime = super(FixBiasItersFAST, self)._run_interface(
            runtime, correct_return_codes
        )

        return runtime
