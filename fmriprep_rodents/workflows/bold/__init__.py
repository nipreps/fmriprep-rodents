# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""

Pre-processing fMRI - BOLD signal workflows
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: fmriprep_rodents.workflows.bold.base
.. automodule:: fmriprep_rodents.workflows.bold.hmc
.. automodule:: fmriprep_rodents.workflows.bold.stc
.. automodule:: fmriprep_rodents.workflows.bold.t2s
.. automodule:: fmriprep_rodents.workflows.bold.registration
.. automodule:: fmriprep_rodents.workflows.bold.resampling
.. automodule:: fmriprep_rodents.workflows.bold.confounds


"""

from .base import init_func_preproc_wf
from .hmc import init_bold_hmc_wf
from .stc import init_bold_stc_wf
from .t2s import init_bold_t2s_wf
from .registration import (
    init_bold_t1_trans_wf,
    init_bold_reg_wf,
)
from .resampling import (
    init_bold_std_trans_wf,
    init_bold_surf_wf,
    init_bold_preproc_trans_wf,
)

from .confounds import (
    init_bold_confs_wf,
    init_ica_aroma_wf,
)

__all__ = [
    "init_bold_confs_wf",
    "init_bold_hmc_wf",
    "init_bold_std_trans_wf",
    "init_bold_preproc_trans_wf",
    "init_bold_reg_wf",
    "init_bold_stc_wf",
    "init_bold_surf_wf",
    "init_bold_t1_trans_wf",
    "init_bold_t2s_wf",
    "init_func_preproc_wf",
    "init_ica_aroma_wf",
]
