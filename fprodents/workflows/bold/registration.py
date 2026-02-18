# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Registration workflows
++++++++++++++++++++++

.. autofunction:: init_bold_reg_wf
.. autofunction:: init_bold_t1_trans_wf

"""
from ... import config

from nipype.pipeline import engine as pe
from nipype.interfaces import utility as niu

from ...interfaces import DerivativesDataSink

DEFAULT_MEMORY_MIN_GB = config.DEFAULT_MEMORY_MIN_GB
LOGGER = config.loggers.workflow


def init_bold_reg_wf(
    bold2t1w_dof,
    bold2t1w_init,
    mem_gb,
    omp_nthreads,
    name="bold_reg_wf",
    use_compression=True,
    write_report=True,
):
    """
    Build a workflow to run same-subject, BOLD-to-T1w image-registration.

    Calculates the registration between a reference BOLD image and T1w-space.

    Workflow Graph
        .. workflow::
            :graph2use: orig
            :simple_form: yes

            from fprodents.workflows.bold.registration import init_bold_reg_wf
            wf = init_bold_reg_wf(mem_gb=3,
                                  omp_nthreads=1,
                                  bold2t1w_dof=9,
                                  bold2t1w_init='register')

    Parameters
    ----------
    bold2t1w_dof : 6, 9 or 12
        Degrees-of-freedom for BOLD-T1w registration
    bold2t1w_init : str, 'header' or 'register'
        If ``'header'``, use header information for initialization of BOLD and T1 images.
        If ``'register'``, align volumes by their centers.
    mem_gb : :obj:`float`
        Size of BOLD file in GB
    omp_nthreads : :obj:`int`
        Maximum number of threads an individual process may use
    name : :obj:`str`
        Name of workflow (default: ``bold_reg_wf``)
    use_compression : :obj:`bool`
        Save registered BOLD series as ``.nii.gz``
    use_fieldwarp : :obj:`bool`
        Include SDC warp in single-shot transform from BOLD to T1
    write_report : :obj:`bool`
        Whether a reportlet should be stored

    Inputs
    ------
    ref_bold_brain
        Reference image to which BOLD series is aligned
        If ``fieldwarp == True``, ``ref_bold_brain`` should be unwarped
    t1w_brain
        Skull-stripped ``t1w_preproc``

    Outputs
    -------
    bold2anat
        Affine transform from ``ref_bold_brain`` to T1 space (ITK format)
    anat2bold
        Affine transform from T1 space to BOLD space (ITK format)
    out_report
        Reportlet for assessing registration quality

    """
    from nipype.interfaces.ants.base import Info as ANTsInfo
    from niworkflows.engine.workflows import LiterateWorkflow as Workflow
    from niworkflows.interfaces.fixes import FixHeaderRegistration as Registration
    from niworkflows.interfaces.reportlets.registration import SimpleBeforeAfterRPT
    from pkg_resources import resource_filename as _pkg_fname

    workflow = Workflow(name=name)
    workflow.__desc__ = """\
The BOLD reference was then co-registered to the T2w reference using
`ANTs` [FSL {ants_ver}, @ants].
Co-registration was configured with six degrees of freedom.
""".format(
        ants_ver=ANTsInfo.version() or "(version unknown)"
    )
    inputnode = pe.Node(
        niu.IdentityInterface(fields=["ref_bold_brain", "t1w_brain"]), name="inputnode"
    )

    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=["bold2anat", "anat2bold", "out_report"]
        ),
        name="outputnode",
    )

    coreg = pe.Node(
        Registration(
            initial_moving_transform_com=1,
            output_warped_image=True,
            from_file=_pkg_fname("fprodents", "data/translation_rigid.json"),
        ),
        name="coreg",
        n_procs=omp_nthreads,
    )

    coreg_rpt = pe.Node(
        SimpleBeforeAfterRPT(before_label='anat', after_label='coreg'), name='coreg_rpt'
    )

    # fmt:off
    workflow.connect([
        (inputnode, coreg, [('ref_bold_brain', 'moving_image'),
                            ('t1w_brain', 'fixed_image')]),
        (inputnode, coreg_rpt, [('t1w_brain', 'before')]),
        (coreg, coreg_rpt, [("warped_image", "after")]),
        (coreg_rpt, outputnode, [('out_report', 'out_report')]),
        (coreg, outputnode, [('composite_transform', 'bold2anat')]),
        (coreg, outputnode, [('inverse_composite_transform', 'anat2bold')]),
    ])
    # fmt:on

    if write_report:
        ds_report_reg = pe.Node(
            DerivativesDataSink(datatype="figures", dismiss_entities=("echo",)),
            name="ds_report_reg",
            run_without_submitting=True,
            desc="coreg",
            mem_gb=DEFAULT_MEMORY_MIN_GB,
        )

        # fmt:off
        workflow.connect([
            (outputnode, ds_report_reg, [('out_report', 'in_file')]),
        ])
        # fmt:on

    return workflow


def init_bold_t1_trans_wf(
    mem_gb,
    omp_nthreads,
    multiecho=False,
    use_fieldwarp=False,
    use_compression=True,
    name="bold_t1_trans_wf",
):
    """
    Co-register the reference BOLD image to T1w-space.

    Workflow Graph
        .. workflow::
            :graph2use: orig
            :simple_form: yes

            from fprodents.workflows.bold.registration import init_bold_t1_trans_wf
            wf = init_bold_t1_trans_wf(mem_gb=3,
                                       omp_nthreads=1)

    Parameters
    ----------
    use_fieldwarp : :obj:`bool`
        Include SDC warp in single-shot transform from BOLD to T1
    multiecho : :obj:`bool`
        If multiecho data was supplied, HMC already performed
    mem_gb : :obj:`float`
        Size of BOLD file in GB
    omp_nthreads : :obj:`int`
        Maximum number of threads an individual process may use
    use_compression : :obj:`bool`
        Save registered BOLD series as ``.nii.gz``
    name : :obj:`str`
        Name of workflow (default: ``bold_reg_wf``)

    Inputs
    ------
    name_source
        BOLD series NIfTI file
        Used to recover original information lost during processing
    ref_bold_brain
        Reference image to which BOLD series is aligned
        If ``fieldwarp == True``, ``ref_bold_brain`` should be unwarped
    t1w_brain
        Skull-stripped bias-corrected structural template image
    t1w_mask
        Mask of the skull-stripped template image
    bold_split
        Individual 3D BOLD volumes, not motion corrected
    hmc_xforms
        List of affine transforms aligning each volume to ``ref_image`` in ITK format
    bold2anat
        Affine transform from ``ref_bold_brain`` to T1 space (ITK format)
    fieldwarp
        a :abbr:`DFM (displacements field map)` in ITK format

    Outputs
    -------
    bold_t1
        Motion-corrected BOLD series in T1 space
    bold_t1_ref
        Reference, contrast-enhanced summary of the motion-corrected BOLD series in T1w space


    """
    from niworkflows.engine.workflows import LiterateWorkflow as Workflow
    from niworkflows.interfaces.fixes import FixHeaderApplyTransforms as ApplyTransforms
    from niworkflows.interfaces.itk import MultiApplyTransforms
    from niworkflows.interfaces.nibabel import GenerateSamplingReference
    from niworkflows.interfaces.nilearn import Merge

    workflow = Workflow(name=name)
    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                "name_source",
                "ref_bold_brain",
                "t1w_brain",
                "t1w_mask",
                "bold_split",
                "fieldwarp",
                "hmc_xforms",
                "bold2anat",
            ]
        ),
        name="inputnode",
    )

    outputnode = pe.Node(
        niu.IdentityInterface(fields=["bold_t1", "bold_t1_ref", "bold_mask_t1"]),
        name="outputnode",
    )

    gen_ref = pe.Node(
        GenerateSamplingReference(), name="gen_ref", mem_gb=0.3
    )  # 256x256x256 * 64 / 8 ~ 150MB

    bold_ref_t1w_tfm = pe.Node(
        ApplyTransforms(interpolation="LanczosWindowedSinc"), name="bold_ref_t1w_tfm", mem_gb=0.1
    )

    # fmt:off
    workflow.connect([
        (inputnode, gen_ref, [('ref_bold_brain', 'moving_image'),
                              ('t1w_brain', 'fixed_image'),
                              ('t1w_mask', 'fov_mask')]),
        (inputnode, bold_ref_t1w_tfm, [('ref_bold_brain', 'input_image')]),
        (gen_ref, bold_ref_t1w_tfm, [('out_file', 'reference_image')]),
        (inputnode, bold_ref_t1w_tfm, [('bold2anat', 'transforms')]),
        (bold_ref_t1w_tfm, outputnode, [('output_image', 'bold_t1_ref')]),
    ])
    # fmt:on

    bold_to_t1w_transform = pe.Node(
        MultiApplyTransforms(
            interpolation="LanczosWindowedSinc", float=True, copy_dtype=True
        ),
        name="bold_to_t1w_transform",
        mem_gb=mem_gb * 3 * omp_nthreads,
        n_procs=omp_nthreads,
    )

    # merge 3D volumes into 4D timeseries
    merge = pe.Node(Merge(compress=use_compression), name="merge", mem_gb=mem_gb)

    if not multiecho:
        # Merge transforms placing the head motion correction last
        nforms = 2 + int(use_fieldwarp)
        merge_xforms = pe.Node(
            niu.Merge(nforms),
            name="merge_xforms",
            run_without_submitting=True,
            mem_gb=DEFAULT_MEMORY_MIN_GB,
        )
        if use_fieldwarp:
            # fmt:off
            workflow.connect([
                (inputnode, merge_xforms, [('fieldwarp', 'in2')])
            ])
            # fmt:on

        # fmt:off
        workflow.connect([
            # merge transforms
            (inputnode, merge_xforms, [
                ('hmc_xforms', 'in%d' % nforms),
                ('bold2anat', 'in1')]),
            (merge_xforms, bold_to_t1w_transform, [('out', 'transforms')]),
            (inputnode, bold_to_t1w_transform, [('bold_split', 'input_image')]),
        ])
        # fmt:on
    else:
        from nipype.interfaces.fsl import Split as FSLSplit

        bold_split = pe.Node(
            FSLSplit(dimension="t"), name="bold_split", mem_gb=DEFAULT_MEMORY_MIN_GB
        )

        # fmt:off
        workflow.connect([
            (inputnode, bold_split, [('bold_split', 'in_file')]),
            (bold_split, bold_to_t1w_transform, [('out_files', 'input_image')]),
            (inputnode, bold_to_t1w_transform, [('bold2anat', 'transforms')]),
        ])
        # fmt:on

    # fmt:off
    workflow.connect([
        (inputnode, merge, [('name_source', 'header_source')]),
        (gen_ref, bold_to_t1w_transform, [('out_file', 'reference_image')]),
        (bold_to_t1w_transform, merge, [('out_files', 'in_files')]),
        (merge, outputnode, [('out_file', 'bold_t1')]),
    ])
    # fmt:on

    return workflow
