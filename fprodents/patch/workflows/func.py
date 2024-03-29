# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Patched functional workflows."""
from nipype.pipeline import engine as pe
from nipype.interfaces import utility as niu

from niworkflows.engine.workflows import LiterateWorkflow as Workflow
from niworkflows.interfaces.bold import NonsteadyStatesDetector
from niworkflows.interfaces.header import ValidateImage
from niworkflows.interfaces.images import RobustAverage
from niworkflows.interfaces.reportlets.masks import SimpleShowMaskRPT
from niworkflows.utils.connections import listify, pop_file
from niworkflows.utils.misc import pass_dummy_scans as _pass_dummy_scans

from ... import config

DEFAULT_MEMORY_MIN_GB = 0.01


def init_bold_reference_wf(
    omp_nthreads,
    bold_file=None,
    sbref_files=None,
    brainmask_thresh=0.85,
    pre_mask=False,
    multiecho=False,
    name="bold_reference_wf",
    gen_report=False,
):
    """
    Build a workflow that generates reference BOLD images for a series.
    The raw reference image is the target of :abbr:`HMC (head motion correction)`, and a
    contrast-enhanced reference is the subject of distortion correction, as well as
    boundary-based registration to T1w and template spaces.
    Workflow Graph
        .. workflow::
            :graph2use: orig
            :simple_form: yes
            from niworkflows.func.util import init_bold_reference_wf
            wf = init_bold_reference_wf(omp_nthreads=1)
    Parameters
    ----------
    omp_nthreads : :obj:`int`
        Maximum number of threads an individual process may use
    bold_file : :obj:`str`
        BOLD series NIfTI file
    sbref_files : :obj:`list` or :obj:`bool`
        Single band (as opposed to multi band) reference NIfTI file.
        If ``True`` is passed, the workflow is built to accommodate SBRefs,
        but the input is left undefined (i.e., it is left open for connection)
    brainmask_thresh: :obj:`float`
        Lower threshold for the probabilistic brainmask to obtain
        the final binary mask (default: 0.85).
    pre_mask : :obj:`bool`
        Indicates whether the ``pre_mask`` input will be set (and thus, step 1
        should be skipped).
    multiecho : :obj:`bool`
        If multiecho data was supplied, data from the first echo
        will be selected
    name : :obj:`str`
        Name of workflow (default: ``bold_reference_wf``)
    gen_report : :obj:`bool`
        Whether a mask report node should be appended in the end
    Inputs
    ------
    bold_file : str
        BOLD series NIfTI file
    bold_mask : bool
        A tentative brain mask to initialize the workflow (requires ``pre_mask``
        parameter set ``True``).
    dummy_scans : int or None
        Number of non-steady-state volumes specified by user at beginning of ``bold_file``
    sbref_file : str
        single band (as opposed to multi band) reference NIfTI file
    Outputs
    -------
    bold_file : str
        Validated BOLD series NIfTI file
    raw_ref_image : str
        Reference image to which BOLD series is motion corrected
    skip_vols : int
        Number of non-steady-state volumes selected at beginning of ``bold_file``
    algo_dummy_scans : int
        Number of non-steady-state volumes agorithmically detected at
        beginning of ``bold_file``
    ref_image : str
        Contrast-enhanced reference image
    ref_image_brain : str
        Skull-stripped reference image
    bold_mask : str
        Skull-stripping mask of reference image
    validation_report : str
        HTML reportlet indicating whether ``bold_file`` had a valid affine
    Subworkflows
        * :py:func:`~niworkflows.func.util.init_enhance_and_skullstrip_wf`
    """
    workflow = Workflow(name=name)
    workflow.__desc__ = f"""\
First, a reference volume and its skull-stripped version were generated
{'from the shortest echo of the BOLD run' * multiecho} using a custom
methodology of *fMRIPrep*.
"""

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=["bold_file", "bold_mask", "dummy_scans", "sbref_file"]
        ),
        name="inputnode",
    )
    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                "bold_file",
                "raw_ref_image",
                "skip_vols",
                "algo_dummy_scans",
                "ref_image",
                "ref_image_brain",
                "bold_mask",
                "validation_report",
                "mask_report",
            ]
        ),
        name="outputnode",
    )

    # Simplify manually setting input image
    if bold_file is not None:
        inputnode.inputs.bold_file = bold_file

    val_bold = pe.MapNode(
        ValidateImage(),
        name="val_bold",
        mem_gb=DEFAULT_MEMORY_MIN_GB,
        iterfield=["in_file"],
    )

    get_dummy = pe.Node(NonsteadyStatesDetector(), name="get_dummy")
    gen_avg = pe.Node(RobustAverage(), name="gen_avg", mem_gb=1)

    calc_dummy_scans = pe.Node(
        niu.Function(function=_pass_dummy_scans, output_names=["skip_vols_num"]),
        name="calc_dummy_scans",
        run_without_submitting=True,
        mem_gb=DEFAULT_MEMORY_MIN_GB,
    )
    bold_1st = pe.Node(
        niu.Select(index=[0]), name="bold_1st", run_without_submitting=True
    )
    validate_1st = pe.Node(
        niu.Select(index=[0]), name="validate_1st", run_without_submitting=True
    )

    # fmt: off
    workflow.connect([
        (inputnode, val_bold, [(("bold_file", listify), "in_file")]),
        (inputnode, get_dummy, [(("bold_file", pop_file), "in_file")]),
        (inputnode, calc_dummy_scans, [("dummy_scans", "dummy_scans")]),
        (val_bold, bold_1st, [(("out_file", listify), "inlist")]),
        (get_dummy, calc_dummy_scans, [("n_dummy", "algo_dummy_scans")]),
        (calc_dummy_scans, outputnode, [("skip_vols_num", "skip_vols")]),
        (gen_avg, outputnode, [("out_file", "raw_ref_image")]),
        (val_bold, validate_1st, [(("out_report", listify), "inlist")]),
        (bold_1st, outputnode, [("out", "bold_file")]),
        (validate_1st, outputnode, [("out", "validation_report")]),
        (get_dummy, outputnode, [("n_dummy", "algo_dummy_scans")]),
    ])
    # fmt: on

    if not pre_mask:
        from nirodents.workflows.brainextraction import init_rodent_brain_extraction_wf

        brain_extraction_wf = init_rodent_brain_extraction_wf(
            ants_affine_init=False,
            debug=config.execution.debug is True
        )
        # fmt: off
        workflow.connect([
            (gen_avg, brain_extraction_wf, [
                ("out_file", "inputnode.in_files"),
            ]),
            (brain_extraction_wf, outputnode, [
                ("outputnode.out_corrected", "ref_image"),
                ("outputnode.out_mask", "bold_mask"),
                ("outputnode.out_brain", "ref_image_brain"),
            ]),
        ])
        # fmt: on
    else:
        from niworkflows.interfaces.nibabel import ApplyMask
        mask_brain = pe.Node(ApplyMask(), name="mask_brain")

        # fmt: off
        workflow.connect([
            (inputnode, mask_brain, [("bold_mask", "in_mask")]),
            (inputnode, outputnode, [("bold_mask", "bold_mask")]),
            (gen_avg, outputnode, [("out_file", "ref_image")]),
            (gen_avg, mask_brain, [("out_file", "in_file")]),
            (mask_brain, outputnode, [("out_file", "ref_image_brain")]),
        ])
        # fmt: on

    if not sbref_files:
        # fmt: off
        workflow.connect([
            (val_bold, gen_avg, [(("out_file", pop_file), "in_file")]),  # pop first echo of ME-EPI
            (get_dummy, gen_avg, [("t_mask", "t_mask")]),
        ])
        # fmt: on
        return workflow

    from niworkflows.interfaces.nibabel import MergeSeries

    nsbrefs = 0
    if sbref_files is not True:
        # If not boolean, then it is a list-of or pathlike.
        inputnode.inputs.sbref_file = sbref_files
        nsbrefs = 1 if isinstance(sbref_files, str) else len(sbref_files)

    val_sbref = pe.MapNode(
        ValidateImage(),
        name="val_sbref",
        mem_gb=DEFAULT_MEMORY_MIN_GB,
        iterfield=["in_file"],
    )
    merge_sbrefs = pe.Node(MergeSeries(), name="merge_sbrefs")

    # fmt: off
    workflow.connect([
        (inputnode, val_sbref, [(("sbref_file", listify), "in_file")]),
        (val_sbref, merge_sbrefs, [("out_file", "in_files")]),
        (merge_sbrefs, gen_avg, [("out_file", "in_file")]),
    ])
    # fmt: on

    # Edit the boilerplate as the SBRef will be the reference
    workflow.__desc__ = f"""\
First, a reference volume and its skull-stripped version were generated
by aligning and averaging{' the first echo of' * multiecho}
{nsbrefs or ''} single-band references (SBRefs).
"""

    if gen_report:
        mask_reportlet = pe.Node(SimpleShowMaskRPT(), name="mask_reportlet")
        # fmt: off
        workflow.connect([
            (outputnode, mask_reportlet, [
                ("ref_image", "background_file"),
                ("bold_mask", "mask_file"),
            ]),
        ])
        # fmt: on

    return workflow
