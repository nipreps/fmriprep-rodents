# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Orchestrating the BOLD-preprocessing workflow
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: init_func_preproc_wf
.. autofunction:: init_func_derivatives_wf

"""
from ... import config

import os

import nibabel as nb
from nipype.interfaces.fsl import Split as FSLSplit
from nipype.pipeline import engine as pe
from nipype.interfaces import utility as niu

from niworkflows.utils.connections import pop_file, listify


from ...utils.meepi import combine_meepi_source

from ...interfaces import DerivativesDataSink
from ...interfaces.reports import FunctionalSummary

# BOLD workflows
from .confounds import init_bold_confs_wf, init_carpetplot_wf
from .hmc import init_bold_hmc_wf
from .stc import init_bold_stc_wf
from .t2s import init_bold_t2s_wf
from .registration import init_bold_t1_trans_wf, init_bold_reg_wf
from .resampling import (
    init_bold_std_trans_wf,
    init_bold_preproc_trans_wf,
)
from .outputs import init_func_derivatives_wf


def init_func_preproc_wf(bold_file, has_fieldmap=False):
    """
    This workflow controls the functional preprocessing stages of *fMRIPrep*.

    Workflow Graph
        .. workflow::
            :graph2use: orig
            :simple_form: yes

            from fprodents.workflows.tests import mock_config
            from fprodents import config
            from fprodents.workflows.bold.base import init_func_preproc_wf
            with mock_config():
                bold_file = config.execution.bids_dir / 'sub-01' / 'func' \
                    / 'sub-01_task-mixedgamblestask_run-01_bold.nii.gz'
                wf = init_func_preproc_wf(str(bold_file))

    Inputs
    ------
    bold_file
        BOLD series NIfTI file
    t1w_preproc
        Bias-corrected structural template image
    t1w_mask
        Mask of the skull-stripped template image
    anat_dseg
        Segmentation of preprocessed structural image, including
        gray-matter (GM), white-matter (WM) and cerebrospinal fluid (CSF)
    anat_tpms
        List of tissue probability maps in T1w space
    template
        List of templates to target
    anat2std_xfm
        List of transform files, collated with templates
    std2anat_xfm
        List of inverse transform files, collated with templates
    subjects_dir
        FreeSurfer SUBJECTS_DIR
    subject_id
        FreeSurfer subject ID

    Outputs
    -------
    bold_t1
        BOLD series, resampled to T1w space
    bold_mask_t1
        BOLD series mask in T1w space
    bold_std
        BOLD series, resampled to template space
    bold_mask_std
        BOLD series mask in template space
    confounds
        TSV of confounds
    melodic_mix
        FSL MELODIC mixing matrix

    See Also
    --------

    * :py:func:`~fprodents.workflows.bold.stc.init_bold_stc_wf`
    * :py:func:`~fprodents.workflows.bold.hmc.init_bold_hmc_wf`
    * :py:func:`~fprodents.workflows.bold.t2s.init_bold_t2s_wf`
    * :py:func:`~fprodents.workflows.bold.registration.init_bold_t1_trans_wf`
    * :py:func:`~fprodents.workflows.bold.registration.init_bold_reg_wf`
    * :py:func:`~fprodents.workflows.bold.confounds.init_bold_confounds_wf`
    * :py:func:`~fprodents.workflows.bold.confounds.init_ica_aroma_wf`
    * :py:func:`~fprodents.workflows.bold.resampling.init_bold_std_trans_wf`
    * :py:func:`~fprodents.workflows.bold.resampling.init_bold_preproc_trans_wf`
    * :py:func:`~fprodents.workflows.bold.resampling.init_bold_surf_wf`

    """
    from niworkflows.engine.workflows import LiterateWorkflow as Workflow
    from niworkflows.interfaces.fixes import FixHeaderApplyTransforms as ApplyTransforms
    from niworkflows.interfaces.nibabel import ApplyMask
    from niworkflows.interfaces.utility import KeySelect

    from ...patch.utils import extract_entities

    mem_gb = {"filesize": 1, "resampled": 1, "largemem": 1}
    bold_tlen = 10

    # Have some options handy
    omp_nthreads = config.nipype.omp_nthreads
    spaces = config.workflow.spaces
    output_dir = str(config.execution.output_dir)

    # Extract BIDS entities and metadata from BOLD file(s)
    entities = extract_entities(bold_file)
    layout = config.execution.layout

    all_metadata = [layout.get_metadata(fname) for fname in listify(bold_file)]
    # Take first file as reference
    ref_file = pop_file(bold_file)
    metadata = all_metadata[0]

    echo_idxs = listify(entities.get("echo", []))
    multiecho = len(echo_idxs) > 2
    if len(echo_idxs) == 1:
        config.loggers.warning(
            f"Running a single echo <{ref_file}> from a seemingly multi-echo dataset."
        )
        bold_file = ref_file  # Just in case - drop the list

    if len(echo_idxs) == 2:
        raise RuntimeError(
            "Multi-echo processing requires at least three different echos (found two)."
        )

    if multiecho:
        # Drop echo entity for future queries, have a boolean shorthand
        entities.pop("echo", None)
        # reorder echoes from shortest to largest
        tes, bold_file = zip(
            *sorted([(layout.get_metadata(bf)["EchoTime"], bf) for bf in bold_file])
        )
        ref_file = bold_file[0]  # Reset reference to be the shortest TE

    if os.path.isfile(ref_file):
        bold_tlen, mem_gb = _create_mem_gb(ref_file)

    wf_name = _get_wf_name(ref_file)
    config.loggers.workflow.debug(
        "Creating bold processing workflow for <%s> (%.2f GB / %d TRs). "
        "Memory resampled/largemem=%.2f/%.2f GB.",
        ref_file,
        mem_gb["filesize"],
        bold_tlen,
        mem_gb["resampled"],
        mem_gb["largemem"],
    )

    # Find associated sbref, if possible
    entities["suffix"] = "sbref"
    entities["extension"] = ["nii", "nii.gz"]  # Overwrite extensions
    sbref_files = layout.get(return_type="file", **entities)

    sbref_msg = f"No single-band-reference found for {os.path.basename(ref_file)}."
    if sbref_files and "sbref" in config.workflow.ignore:
        sbref_msg = "Single-band reference file(s) found and ignored."
    elif sbref_files:
        sbref_msg = "Using single-band reference file(s) {}.".format(
            ",".join([os.path.basename(sbf) for sbf in sbref_files])
        )
    config.loggers.workflow.info(sbref_msg)

    # Check whether STC must/can be run
    run_stc = (
        bool(metadata.get("SliceTiming"))
        and 'slicetiming' not in config.workflow.ignore
    )

    # Build workflow
    workflow = Workflow(name=wf_name)
    workflow.__postdesc__ = """\
All resamplings can be performed with *a single interpolation
step* by composing all the pertinent transformations (i.e. head-motion
transform matrices, susceptibility distortion correction when available,
and co-registrations to anatomical and output spaces).
Gridded (volumetric) resamplings were performed using `antsApplyTransforms` (ANTs),
configured with Lanczos interpolation to minimize the smoothing
effects of other kernels [@lanczos].
Non-gridded (surface) resamplings were performed using `mri_vol2surf`
(FreeSurfer).
"""

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                "bold_file",
                "ref_file",
                "bold_ref_xfm",
                "n_dummy_scans",
                "validation_report",
                "subjects_dir",
                "subject_id",
                "anat_preproc",
                "anat_mask",
                "anat_dseg",
                "anat_tpms",
                "anat2std_xfm",
                "std2anat_xfm",
                "template",
                "anat2fsnative_xfm",
                "fsnative2anat_xfm",
                "fmap",
                "fmap_ref",
                "fmap_coeff",
                "fmap_mask",
                "fmap_id",
                "sdc_method",
            ]
        ),
        name="inputnode",
    )
    inputnode.inputs.bold_file = bold_file

    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                "bold_mask",
                "bold_t1",
                "bold_t1_ref",
                "bold_mask_t1",
                "bold_std",
                "bold_std_ref",
                "bold_mask_std",
                "bold_native",
                "confounds",
                "melodic_mix",
                "nonaggr_denoised_file",
                "confounds_metadata",
            ]
        ),
        name="outputnode",
    )

    # Generate a brain-masked conversion of the t1w and bold reference images
    t1w_brain = pe.Node(ApplyMask(), name="t1w_brain")

    # BOLD buffer: an identity used as a pointer to either the original BOLD
    # or the STC'ed one for further use.
    boldbuffer = pe.Node(niu.IdentityInterface(fields=["bold_file"]), name="boldbuffer")

    summary = pe.Node(
        FunctionalSummary(
            slice_timing=run_stc,
            registration="FSL",
            registration_dof=config.workflow.bold2t1w_dof,
            registration_init=config.workflow.bold2t1w_init,
            pe_direction=metadata.get("PhaseEncodingDirection"),
            echo_idx=echo_idxs,
            tr=metadata.get("RepetitionTime"),
            distortion_correction="<not implemented>",
        ),
        name="summary",
        mem_gb=config.DEFAULT_MEMORY_MIN_GB,
        run_without_submitting=True,
    )
    summary.inputs.dummy_scans = config.workflow.dummy_scans

    func_derivatives_wf = init_func_derivatives_wf(
        bids_root=layout.root,
        metadata=metadata,
        output_dir=output_dir,
        spaces=spaces,
        use_aroma=config.workflow.use_aroma,
    )

    # fmt:off
    workflow.connect([
        (outputnode, func_derivatives_wf, [
            ('bold_t1', 'inputnode.bold_t1'),
            ('bold_t1_ref', 'inputnode.bold_t1_ref'),
            ('bold_mask_t1', 'inputnode.bold_mask_t1'),
            ('bold_native', 'inputnode.bold_native'),
            ('confounds', 'inputnode.confounds'),
            ('melodic_mix', 'inputnode.melodic_mix'),
            ('nonaggr_denoised_file', 'inputnode.nonaggr_denoised_file'),
            ('confounds_metadata', 'inputnode.confounds_metadata'),
        ]),
    ])
    # fmt:on

    # Top-level BOLD splitter
    bold_split = pe.Node(
        FSLSplit(dimension="t"), name="bold_split", mem_gb=mem_gb["filesize"] * 3
    )

    # HMC on the BOLD
    bold_hmc_wf = init_bold_hmc_wf(
        name="bold_hmc_wf", mem_gb=mem_gb["filesize"], omp_nthreads=omp_nthreads
    )

    # calculate BOLD registration to T1w
    bold_reg_wf = init_bold_reg_wf(
        bold2t1w_dof=config.workflow.bold2t1w_dof,
        bold2t1w_init=config.workflow.bold2t1w_init,
        mem_gb=mem_gb["resampled"],
        name="bold_reg_wf",
        omp_nthreads=omp_nthreads,
        use_compression=False,
        write_report=True,
    )

    # apply BOLD registration to T1w
    bold_t1_trans_wf = init_bold_t1_trans_wf(
        name="bold_t1_trans_wf",
        use_fieldwarp=False,
        multiecho=multiecho,
        mem_gb=mem_gb["resampled"],
        omp_nthreads=omp_nthreads,
        use_compression=False,
    )

    t1w_mask_bold_tfm = pe.Node(
        ApplyTransforms(interpolation="MultiLabel"), name="t1w_mask_bold_tfm", mem_gb=0.1
    )

    # get confounds
    bold_confounds_wf = init_bold_confs_wf(
        mem_gb=mem_gb["largemem"],
        metadata=metadata,
        regressors_all_comps=config.workflow.regressors_all_comps,
        regressors_fd_th=config.workflow.regressors_fd_th,
        regressors_dvars_th=config.workflow.regressors_dvars_th,
        name="bold_confounds_wf",
    )
    bold_confounds_wf.get_node("inputnode").inputs.t1_transform_flags = [False]

    # SLICE-TIME CORRECTION (or bypass) #############################################
    if run_stc:
        bold_stc_wf = init_bold_stc_wf(name="bold_stc_wf", metadata=metadata)
        # fmt:off
        workflow.connect([
            (inputnode, bold_stc_wf, [("n_dummy_scans", "inputnode.skip_vols")]),
            (bold_stc_wf, boldbuffer, [("outputnode.stc_file", "bold_file")]),
        ])
        # fmt:on
        if not multiecho:
            # fmt:off
            workflow.connect([
                (inputnode, bold_stc_wf, [('bold_file', 'inputnode.bold_file')])
            ])
            # fmt:on
        else:  # for meepi, iterate through stc_wf for all workflows
            meepi_echos = boldbuffer.clone(name="meepi_echos")
            meepi_echos.iterables = ("bold_file", bold_file)
            # fmt:off
            workflow.connect([
                (meepi_echos, bold_stc_wf, [('bold_file', 'inputnode.bold_file')])
            ])
            # fmt:on
    elif not multiecho:  # STC is too short or False
        # bypass STC from original BOLD to the splitter through boldbuffer
        # fmt:off
        workflow.connect([
            (inputnode, boldbuffer, [('bold_file', 'bold_file')])
        ])
        # fmt:on
    else:
        # for meepi, iterate over all meepi echos to boldbuffer
        boldbuffer.iterables = ("bold_file", bold_file)

    # MULTI-ECHO EPI DATA #############################################
    if multiecho:
        from niworkflows.func.util import init_skullstrip_bold_wf

        skullstrip_bold_wf = init_skullstrip_bold_wf(name="skullstrip_bold_wf")

        inputnode.inputs.bold_file = ref_file  # Replace reference w first echo

        join_echos = pe.JoinNode(
            niu.IdentityInterface(fields=["bold_files"]),
            joinsource=("meepi_echos" if run_stc is True else "boldbuffer"),
            joinfield=["bold_files"],
            name="join_echos",
        )

        # create optimal combination, adaptive T2* map
        bold_t2s_wf = init_bold_t2s_wf(
            echo_times=tes,
            mem_gb=mem_gb["resampled"],
            omp_nthreads=omp_nthreads,
            name="bold_t2smap_wf",
        )

        # fmt:off
        workflow.connect([
            (skullstrip_bold_wf, join_echos, [
                ('outputnode.skull_stripped_file', 'bold_files')]),
            (join_echos, bold_t2s_wf, [
                ('bold_files', 'inputnode.bold_file')]),
        ])
        # fmt:on

    # MAIN WORKFLOW STRUCTURE #######################################################
    # fmt:off
    workflow.connect([
        (inputnode, bold_reg_wf, [('anat_preproc', 'inputnode.t1w_brain'),
                                  ('ref_file', 'inputnode.ref_bold_brain')]),
        (inputnode, t1w_brain, [('anat_preproc', 'in_file'),
                                ('anat_mask', 'in_mask')]),
        # BOLD buffer has slice-time corrected if it was run, original otherwise
        (boldbuffer, bold_split, [('bold_file', 'in_file')]),
        (inputnode, summary, [('n_dummy_scans', 'algo_dummy_scans')]),
        # HMC
        (inputnode, bold_hmc_wf, [
            ("ref_file", "inputnode.raw_ref_image"),
            ("bold_file", "inputnode.bold_file"),
        ]),
        # EPI-T1 registration workflow
        (inputnode, bold_t1_trans_wf, [('bold_file', 'inputnode.name_source'),
                                       ('anat_mask', 'inputnode.t1w_mask')]),
        (t1w_brain, bold_t1_trans_wf, [('out_file', 'inputnode.t1w_brain')]),
        (bold_hmc_wf, bold_t1_trans_wf, [('outputnode.xforms', 'inputnode.hmc_xforms')]),
        (bold_reg_wf, bold_t1_trans_wf, [('outputnode.bold2anat', 'inputnode.bold2anat')]),
        (bold_t1_trans_wf, outputnode, [('outputnode.bold_t1', 'bold_t1'),
                                        ('outputnode.bold_t1_ref', 'bold_t1_ref')]),
        # transform T1 mask to BOLD
        (inputnode, t1w_mask_bold_tfm, [('anat_mask', 'input_image'),
                                        ('ref_file', 'reference_image')]),
        (bold_reg_wf, t1w_mask_bold_tfm, [('outputnode.anat2bold', 'transforms')]),
        # Connect bold_confounds_wf
        (inputnode, bold_confounds_wf, [('anat_tpms', 'inputnode.anat_tpms'),
                                        ('anat_mask', 'inputnode.t1w_mask')]),
        (bold_hmc_wf, bold_confounds_wf, [("outputnode.movpar_file", "inputnode.movpar_file")]),
        (bold_reg_wf, bold_confounds_wf, [('outputnode.anat2bold', 'inputnode.anat2bold')]),
        (inputnode, bold_confounds_wf, [('n_dummy_scans', 'inputnode.skip_vols')]),
        (bold_confounds_wf, outputnode, [('outputnode.confounds_file', 'confounds')]),
        (bold_confounds_wf, outputnode, [('outputnode.confounds_metadata', 'confounds_metadata')]),
        # Summary
        (outputnode, summary, [('confounds', 'confounds_file')]),
    ])
    # fmt:on

    # for standard EPI data, pass along correct file
    if not multiecho:
        # fmt:off
        workflow.connect([
            (inputnode, func_derivatives_wf, [('bold_file', 'inputnode.source_file')]),
            (bold_split, bold_t1_trans_wf, [('out_files', 'inputnode.bold_split')]),
        ])
        # fmt:on
    else:  # for meepi, create and use optimal combination
        # fmt:off
        workflow.connect([
            # update name source for optimal combination
            (inputnode, func_derivatives_wf, [
                (('bold_file', combine_meepi_source), 'inputnode.source_file')]),
            (bold_t2s_wf, bold_confounds_wf, [('outputnode.bold', 'inputnode.bold')]),
            (bold_t2s_wf, bold_t1_trans_wf, [('outputnode.bold', 'inputnode.bold_split')]),
        ])
        # fmt:on

    # Map final BOLD mask into T1w space (if required)
    nonstd_spaces = set(spaces.get_nonstandard())
    if nonstd_spaces.intersection(("T1w", "anat")):
        from niworkflows.interfaces.fixes import (
            FixHeaderApplyTransforms as ApplyTransforms,
        )

        boldmask_to_t1w = pe.Node(
            ApplyTransforms(interpolation="MultiLabel"),
            name="boldmask_to_t1w",
            mem_gb=0.1,
        )
        # fmt:off
        workflow.connect([
            (bold_reg_wf, boldmask_to_t1w, [('outputnode.bold2anat', 'transforms')]),
            (bold_t1_trans_wf, boldmask_to_t1w, [('outputnode.bold_mask_t1', 'reference_image')]),
            (t1w_mask_bold_tfm, boldmask_to_t1w, [('output_image', 'input_image')]),
            (boldmask_to_t1w, outputnode, [('output_image', 'bold_mask_t1')]),
        ])
        # fmt:on

    if spaces.get_spaces(nonstandard=False, dim=(3,)):
        # Apply transforms in 1 shot
        bold_std_trans_wf = init_bold_std_trans_wf(
            mem_gb=mem_gb["resampled"],
            omp_nthreads=omp_nthreads,
            spaces=spaces,
            name="bold_std_trans_wf",
            use_compression=not config.execution.low_mem,
            use_fieldwarp=has_fieldmap,
        )

        carpetplot_wf = init_carpetplot_wf(
            mem_gb=mem_gb["resampled"],
            metadata=metadata,
            name="carpetplot_wf",
        )
        # Xform to 'Fischer344' is always computed.
        carpetplot_select_std = pe.Node(
            KeySelect(fields=["std2anat_xfm"], key="Fischer344"),
            name="carpetplot_select_std",
            run_without_submitting=True,
        )

        # fmt:off
        workflow.connect([
            (inputnode, bold_std_trans_wf, [('template', 'inputnode.templates'),
                                            ('anat2std_xfm', 'inputnode.anat2std_xfm'),
                                            ('bold_file', 'inputnode.name_source')]),
            (t1w_mask_bold_tfm, bold_std_trans_wf, [('output_image', 'inputnode.bold_mask')]),
            (bold_hmc_wf, bold_std_trans_wf, [('outputnode.xforms', 'inputnode.hmc_xforms')]),
            (bold_reg_wf, bold_std_trans_wf, [('outputnode.bold2anat', 'inputnode.bold2anat')]),
            (bold_std_trans_wf, outputnode, [('outputnode.bold_std', 'bold_std'),
                                             ('outputnode.bold_std_ref', 'bold_std_ref'),
                                             ('outputnode.bold_mask_std', 'bold_mask_std')]),
            # func_derivatives_wf internally parametrizes over snapshotted spaces.
            (bold_std_trans_wf, func_derivatives_wf, [
                ('outputnode.template', 'inputnode.template'),
                ('outputnode.spatial_reference', 'inputnode.spatial_reference'),
                ('outputnode.bold_std_ref', 'inputnode.bold_std_ref'),
                ('outputnode.bold_std', 'inputnode.bold_std'),
                ('outputnode.bold_mask_std', 'inputnode.bold_mask_std'),
            ]),
            (inputnode, carpetplot_select_std, [('std2anat_xfm', 'std2anat_xfm'),
                                                ('template', 'keys')]),
            (inputnode, carpetplot_wf, [('n_dummy_scans', 'inputnode.dummy_scans')]),
            (carpetplot_select_std, carpetplot_wf, [('std2anat_xfm', 'inputnode.std2anat_xfm')]),
            (t1w_mask_bold_tfm, carpetplot_wf, [('output_image', 'inputnode.bold_mask')]),
            (bold_reg_wf, carpetplot_wf, [('outputnode.anat2bold', 'inputnode.anat2bold')]),
            (bold_confounds_wf, carpetplot_wf,
                [('outputnode.confounds_file', 'inputnode.confounds_file')]),
        ])
        # fmt:on

        if not multiecho:
            # fmt:off
            workflow.connect([
                (bold_split, bold_std_trans_wf, [("out_files", "inputnode.bold_split")])
            ])
            # fmt:on
        else:
            split_opt_comb = bold_split.clone(name="split_opt_comb")
            # fmt:off
            workflow.connect([
                (bold_t2s_wf, split_opt_comb, [('outputnode.bold', 'in_file')]),
                (split_opt_comb, bold_std_trans_wf, [('out_files', 'inputnode.bold_split')])
            ])
            # fmt:on

    # REPORTING ############################################################
    ds_report_summary = pe.Node(
        DerivativesDataSink(
            desc="summary", datatype="figures", dismiss_entities=("echo",)
        ),
        name="ds_report_summary",
        run_without_submitting=True,
        mem_gb=config.DEFAULT_MEMORY_MIN_GB,
    )

    ds_report_validation = pe.Node(
        DerivativesDataSink(
            base_directory=output_dir,
            desc="validation",
            datatype="figures",
            dismiss_entities=("echo",),
        ),
        name="ds_report_validation",
        run_without_submitting=True,
        mem_gb=config.DEFAULT_MEMORY_MIN_GB,
    )

    # fmt:off
    workflow.connect([
        (summary, ds_report_summary, [('out_report', 'in_file')]),
        (inputnode, ds_report_validation, [('validation_report', 'in_file')]),
    ])
    # fmt:on

    # Fill-in datasinks of reportlets seen so far
    for node in workflow.list_node_names():
        if node.split(".")[-1].startswith("ds_report"):
            workflow.get_node(node).inputs.base_directory = output_dir
            workflow.get_node(node).inputs.source_file = ref_file

    if not has_fieldmap:
        # Finalize workflow without SDC connections
        summary.inputs.distortion_correction = "None"

        # Resample in native space in just one shot
        bold_bold_trans_wf = init_bold_preproc_trans_wf(
            mem_gb=mem_gb["resampled"],
            omp_nthreads=omp_nthreads,
            use_compression=not config.execution.low_mem,
            use_fieldwarp=False,
            name="bold_bold_trans_wf",
        )
        bold_bold_trans_wf.inputs.inputnode.name_source = ref_file

        # fmt:off
        workflow.connect([
            # Connect bold_bold_trans_wf
            (inputnode, bold_t1_trans_wf, [('ref_file', 'inputnode.ref_bold_brain')]),
            (inputnode, bold_bold_trans_wf, [('ref_file', 'inputnode.bold_ref')]),
            (t1w_mask_bold_tfm, outputnode, [('output_image', 'bold_mask')]),
            (t1w_mask_bold_tfm, bold_confounds_wf, [('output_image', 'inputnode.bold_mask')]),
            (t1w_mask_bold_tfm, bold_bold_trans_wf, [('output_image', 'inputnode.bold_mask')]),
            (bold_split, bold_bold_trans_wf, [('out_files', 'inputnode.bold_file')]),
            (bold_hmc_wf, bold_bold_trans_wf, [('outputnode.xforms', 'inputnode.hmc_xforms')]),
        ])

        if not multiecho:
            workflow.connect([
                (bold_bold_trans_wf, bold_confounds_wf, [("outputnode.bold", "inputnode.bold")]),
                (bold_bold_trans_wf, carpetplot_wf, [('outputnode.bold', 'inputnode.bold')])
            ])
        else:
            workflow.connect([
                (bold_bold_trans_wf, skullstrip_bold_wf, [
                    ('outputnode.bold', 'inputnode.in_file')]),
                (bold_t2s_wf, carpetplot_wf, [('outputnode.bold', 'inputnode.bold')]),
            ])
        # fmt:on

        if nonstd_spaces.intersection(("func", "run", "bold", "boldref", "sbref")):
            # fmt:off
            workflow.connect([
                (bold_bold_trans_wf, outputnode, [('outputnode.bold', 'bold_native')]),
                (bold_bold_trans_wf, func_derivatives_wf, [
                    ('outputnode.bold_ref', 'inputnode.bold_native_ref'),
                    ('outputnode.bold_mask', 'inputnode.bold_mask_native')]),
            ])
            # fmt:on
        return workflow

    from niworkflows.interfaces.reportlets.registration import (
        SimpleBeforeAfterRPT as SimpleBeforeAfter,
    )
    from niworkflows.interfaces.utility import KeySelect
    from sdcflows.workflows.apply.registration import init_coeff2epi_wf
    from sdcflows.workflows.apply.correction import init_unwarp_wf
    from ...patch.workflows.func import init_bold_reference_wf

    coeff2epi_wf = init_coeff2epi_wf(
        debug=config.execution.debug,
        omp_nthreads=config.nipype.omp_nthreads,
        write_coeff=True,
    )

    unwarp_wf = init_unwarp_wf(
        debug=config.execution.debug,
        omp_nthreads=config.nipype.omp_nthreads,
    )
    unwarp_wf.inputs.inputnode.metadata = metadata

    final_boldref_wf = init_bold_reference_wf(
        omp_nthreads=config.nipype.omp_nthreads,
        pre_mask=True,
        multiecho=multiecho,
        name="bold_reference_wf",
        gen_report=True,
    )

    sdc_report = pe.Node(
        SimpleBeforeAfter(
            before_label="Distorted",
            after_label="Corrected",
            dismiss_affine=True,
        ),
        name="sdc_report",
        mem_gb=0.1,
    )

    ds_report_sdc = pe.Node(
        DerivativesDataSink(
            base_directory=output_dir,
            desc="sdc",
            suffix="bold",
            datatype="figures",
            dismiss_entities=("echo",),
        ),
        name="ds_report_sdc",
        run_without_submitting=True,
    )

    # fmt:off
    workflow.connect([
        (inputnode, summary, [(("sdc_method", pop_file), "distortion_correction")]),
        (inputnode, coeff2epi_wf, [
            (("fmap_ref", pop_file), "inputnode.fmap_ref"),
            ("fmap_coeff", "inputnode.fmap_coeff"),
            ("fmap_mask", "inputnode.fmap_mask"),
            ("ref_file", "inputnode.target_ref")]),
        (t1w_mask_bold_tfm, coeff2epi_wf, [('output_image', 'inputnode.target_mask')]),
        (t1w_mask_bold_tfm, final_boldref_wf, [('output_image', 'inputnode.bold_mask')]),
        (inputnode, unwarp_wf, [("ref_file", "inputnode.distorted_ref")]),
        (coeff2epi_wf, unwarp_wf, [("outputnode.fmap_coeff", "inputnode.fmap_coeff")]),
        (bold_hmc_wf, unwarp_wf, [("outputnode.xforms", "inputnode.hmc_xforms")]),
        (bold_split, unwarp_wf, [("out_files", "inputnode.distorted")]),
        (final_boldref_wf, bold_t1_trans_wf,
            [("outputnode.ref_image", "inputnode.ref_bold_brain")]),
        (inputnode, sdc_report, [("ref_file", "before")]),
        (final_boldref_wf, sdc_report, [("outputnode.ref_image", "after"),
                                        ("outputnode.bold_mask", "wm_seg")]),
        (inputnode, ds_report_sdc, [("bold_file", "source_file")]),
        (sdc_report, ds_report_sdc, [("out_report", "in_file")]),
        (final_boldref_wf, bold_confounds_wf, [("outputnode.bold_mask", 'inputnode.bold_mask')]),
    ])
    # fmt:on

    if not multiecho:
        # fmt:off
        workflow.connect([
            (unwarp_wf, outputnode, [("outputnode.corrected", "bold_native")]),
            # remaining workflow connections
            (unwarp_wf, final_boldref_wf, [("outputnode.corrected", "inputnode.bold_file")]),
            (unwarp_wf, bold_confounds_wf, [("outputnode.corrected", "inputnode.bold")]),
            (unwarp_wf, carpetplot_wf, [("outputnode.corrected", "inputnode.bold")]),
            (unwarp_wf, bold_t1_trans_wf, [
                # TEMPORARY: For the moment we can't use frame-wise fieldmaps
                (("outputnode.fieldwarp", pop_file), "inputnode.fieldwarp"),
            ]),
            (unwarp_wf, bold_std_trans_wf, [
                # TEMPORARY: For the moment we can't use frame-wise fieldmaps
                (("outputnode.fieldwarp", pop_file), "inputnode.fieldwarp"),
            ]),
        ])
        # fmt:on
        return workflow

    # Finalize connections if ME-EPI
    join_sdc_echos = pe.JoinNode(
        niu.IdentityInterface(
            fields=[
                "fieldmap",
                "fieldwarp",
                "corrected",
                "corrected_ref",
                "corrected_mask",
            ]
        ),
        joinsource="echo_index",
        joinfield=[
            "fieldmap",
            "fieldwarp",
            "corrected",
            "corrected_ref",
            "corrected_mask",
        ],
        name="join_sdc_echos",
    )

    def _dpop(list_of_lists):
        return list_of_lists[0][0]

    # fmt:off
    workflow.connect([
        (unwarp_wf, join_echos, [
            ("outputnode.corrected", "bold_files"),
        ]),
        (unwarp_wf, join_sdc_echos, [
            ("outputnode.fieldmap", "fieldmap"),
            ("outputnode.fieldwarp", "fieldwarp"),
            ("outputnode.corrected", "corrected"),
            ("outputnode.corrected_ref", "corrected_ref"),
            ("outputnode.corrected_mask", "corrected_mask"),
        ]),
        # remaining workflow connections
        (join_sdc_echos, final_boldref_wf, [
            ("corrected", "inputnode.bold_file"),
        ]),
        (join_sdc_echos, bold_t2s_wf, [
            (("corrected_mask", pop_file), "inputnode.bold_mask"),
        ]),
        (join_sdc_echos, bold_t1_trans_wf, [
            # TEMPORARY: For the moment we can't use frame-wise fieldmaps
            (("fieldwarp", _dpop), "inputnode.fieldwarp"),
        ]),
        (join_sdc_echos, bold_std_trans_wf, [
            # TEMPORARY: For the moment we can't use frame-wise fieldmaps
            (("fieldwarp", _dpop), "inputnode.fieldwarp"),
        ]),
    ])
    # fmt:on

    return workflow


def _create_mem_gb(bold_fname):
    bold_size_gb = os.path.getsize(bold_fname) / (1024 ** 3)
    bold_tlen = nb.load(bold_fname).shape[-1]
    mem_gb = {
        "filesize": bold_size_gb,
        "resampled": bold_size_gb * 4,
        "largemem": bold_size_gb * (max(bold_tlen / 100, 1.0) + 4),
    }

    return bold_tlen, mem_gb


def _get_wf_name(bold_fname):
    """
    Derive the workflow name for supplied BOLD file.

    >>> _get_wf_name('/completely/made/up/path/sub-01_task-nback_bold.nii.gz')
    'func_preproc_task_nback_wf'
    >>> _get_wf_name('/completely/made/up/path/sub-01_task-nback_run-01_echo-1_bold.nii.gz')
    'func_preproc_task_nback_run_01_echo_1_wf'

    """
    from nipype.utils.filemanip import split_filename

    fname = split_filename(bold_fname)[1]
    fname_nosub = "_".join(fname.split("_")[1:])
    # if 'echo' in fname_nosub:
    #     fname_nosub = '_'.join(fname_nosub.split("_echo-")[:1]) + "_bold"
    name = "func_preproc_" + fname_nosub.replace(".", "_").replace(" ", "").replace(
        "-", "_"
    ).replace("_bold", "_wf")

    return name


def _to_join(in_file, join_file):
    """Join two tsv files if the join_file is not ``None``."""
    from niworkflows.interfaces.utility import JoinTSVColumns

    if join_file is None:
        return in_file
    res = JoinTSVColumns(in_file=in_file, join_file=join_file).run()
    return res.outputs.out_file
