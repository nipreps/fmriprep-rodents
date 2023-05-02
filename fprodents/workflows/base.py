# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
fMRIPrep base processing workflows
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: init_fmriprep_wf
.. autofunction:: init_single_subject_wf

"""

import sys
from copy import deepcopy

from nipype.pipeline import engine as pe
from nipype.interfaces import utility as niu

from .. import config
from ..interfaces import SubjectSummary, AboutSummary, DerivativesDataSink
from .bold.base import init_func_preproc_wf


def init_fmriprep_wf():
    """
    Build *fMRIPrep*'s pipeline.

    This workflow organizes the execution of FMRIPREP, with a sub-workflow for
    each subject.

    If FreeSurfer's ``recon-all`` is to be run, a corresponding folder is created
    and populated with any needed template subjects under the derivatives folder.

    Workflow Graph
        .. workflow::
            :graph2use: orig
            :simple_form: yes

            from fprodents.workflows.tests import mock_config
            from fprodents.workflows.base import init_fmriprep_wf
            with mock_config():
                wf = init_fmriprep_wf()

    """
    from niworkflows.engine.workflows import LiterateWorkflow as Workflow

    fmriprep_wf = Workflow(name="fmriprep_wf")
    fmriprep_wf.base_dir = config.execution.work_dir

    for subject_id in config.execution.participant_label:
        single_subject_wf = init_single_subject_wf(subject_id)

        # Dump a copy of the config file into the log directory
        log_dir = (
            config.execution.output_dir
            / "fmriprep"
            / f"sub-{subject_id}"
            / "log"
            / config.execution.run_uuid
        )
        log_dir.mkdir(exist_ok=True, parents=True)
        config.to_filename(log_dir / "fmriprep.toml")

        single_subject_wf.config["execution"]["crashdump_dir"] = str(log_dir)
        for node in single_subject_wf._get_all_nodes():
            node.config = deepcopy(single_subject_wf.config)

        fmriprep_wf.add_nodes([single_subject_wf])

    return fmriprep_wf


def init_single_subject_wf(subject_id):
    """
    Organize the preprocessing pipeline for a single subject.

    It collects and reports information about the subject, and prepares
    sub-workflows to perform anatomical and functional preprocessing.
    Anatomical preprocessing is performed in a single workflow, regardless of
    the number of sessions.
    Functional preprocessing is performed using a separate workflow for each
    individual BOLD series.

    Workflow Graph
        .. workflow::
            :graph2use: orig
            :simple_form: yes

            from fprodents.workflows.tests import mock_config
            from fprodents.workflows.base import init_single_subject_wf
            with mock_config():
                wf = init_single_subject_wf('01')

    Parameters
    ----------
    subject_id : :obj:`str`
        Subject label for this single-subject workflow.

    Inputs
    ------
    subjects_dir : :obj:`str`
        FreeSurfer's ``$SUBJECTS_DIR``.

    """
    from nirodents.workflows.brainextraction import _bspline_grid
    from niworkflows.engine.workflows import LiterateWorkflow as Workflow
    from niworkflows.interfaces.bids import BIDSInfo
    from niworkflows.interfaces.nilearn import NILEARN_VERSION
    from niworkflows.utils.bids import collect_data
    from niworkflows.utils.spaces import Reference
    from ..patch.interfaces import BIDSDataGrabber
    from ..patch.utils import fix_multi_source_name
    from ..patch.workflows.anatomical import init_anat_preproc_wf

    subject_data = collect_data(
        bids_dir=config.execution.layout,
        participant_label=subject_id,
        task=config.execution.task_id,
        echo=config.execution.echo_idx,
        bids_filters=config.execution.bids_filters,
    )[0]

    anat_only = config.workflow.anat_only
    # Make sure we always go through these two checks
    if not anat_only and not subject_data["bold"]:
        task_id = config.execution.task_id
        raise RuntimeError(
            f"No BOLD images found for participant <{subject_id}> and "
            f"task <{task_id or 'all'}>. All workflows require BOLD images."
        )

    workflow = Workflow(name=f"single_subject_{subject_id}_wf")
    workflow.__desc__ = """
Results included in this manuscript come from preprocessing
performed using *fMRIPrep-rodents* {fmriprep_ver}
(@fmriprep1; @fmriprep2; RRID:SCR_016216),
which is based on *Nipype* {nipype_ver}
(@nipype1; @nipype2; RRID:SCR_002502).

""".format(
        fmriprep_ver=config.environment.version,
        nipype_ver=config.environment.nipype_version,
    )
    workflow.__postdesc__ = """

Many internal operations of *fMRIPrep* use
*Nilearn* {nilearn_ver} [@nilearn, RRID:SCR_001362],
mostly within the functional processing workflow.
For more details of the pipeline, see [the section corresponding
to workflows in *fMRIPrep*'s documentation]\
(https://fmriprep-rodents.readthedocs.io/en/latest/workflows.html \
"FMRIPrep's documentation").


### Copyright Waiver

The above boilerplate text was automatically generated by fMRIPrep
with the express intention that users should copy and paste this
text into their manuscripts *unchanged*.
It is released under the [CC0]\
(https://creativecommons.org/publicdomain/zero/1.0/) license.

### References

""".format(
        nilearn_ver=NILEARN_VERSION
    )

    spaces = config.workflow.spaces
    output_dir = str(config.execution.output_dir)

    inputnode = pe.Node(
        niu.IdentityInterface(fields=["subjects_dir"]), name="inputnode"
    )

    bidssrc = pe.Node(
        BIDSDataGrabber(
            subject_data=subject_data, anat_only=anat_only, subject_id=subject_id
        ),
        name="bidssrc",
    )

    bids_info = pe.Node(
        BIDSInfo(bids_dir=config.execution.bids_dir, bids_validate=False),
        name="bids_info",
    )

    summary = pe.Node(
        SubjectSummary(
            std_spaces=spaces.get_spaces(nonstandard=False),
            nstd_spaces=spaces.get_spaces(standard=False),
        ),
        name="summary",
        run_without_submitting=True,
    )

    about = pe.Node(
        AboutSummary(version=config.environment.version, command=" ".join(sys.argv)),
        name="about",
        run_without_submitting=True,
    )

    ds_report_summary = pe.Node(
        DerivativesDataSink(
            base_directory=output_dir,
            desc="summary",
            datatype="figures",
            dismiss_entities=("echo",),
        ),
        name="ds_report_summary",
        run_without_submitting=True,
    )

    ds_report_about = pe.Node(
        DerivativesDataSink(
            base_directory=output_dir,
            desc="about",
            datatype="figures",
            dismiss_entities=("echo",),
        ),
        name="ds_report_about",
        run_without_submitting=True,
    )

    anat_derivatives = config.execution.anat_derivatives
    if anat_derivatives:
        from smriprep.utils.bids import collect_derivatives

        std_spaces = spaces.get_spaces(nonstandard=False, dim=(3,))
        anat_derivatives = collect_derivatives(
            anat_derivatives.absolute(),
            subject_id,
            std_spaces,
            False,
        )
        if anat_derivatives is None:
            config.loggers.workflow.warning(
                f"""\
Attempted to access pre-existing anatomical derivatives at \
<{config.execution.anat_derivatives}>, however not all expectations of fMRIPrep \
were met (for participant <{subject_id}>, spaces <{', '.join(std_spaces)}>."""
            )

    # Preprocessing of T1w (includes registration to MNI)
    anat_preproc_wf = init_anat_preproc_wf(
        bids_root=str(config.execution.bids_dir),
        debug=config.execution.debug is True,
        existing_derivatives=anat_derivatives,
        longitudinal=config.workflow.longitudinal,
        omp_nthreads=config.nipype.omp_nthreads,
        output_dir=output_dir,
        skull_strip_fixed_seed=config.workflow.skull_strip_fixed_seed,
        skull_strip_mode=config.workflow.skull_strip_t1w,
        skull_strip_template=Reference.from_string(
            config.workflow.skull_strip_template
        )[0],
        spaces=spaces,
        t2w=subject_data["t2w"],
    )

    # fmt:off
    workflow.connect([
        (bidssrc, bids_info, [(('t2w', fix_multi_source_name), 'in_file')]),
        (inputnode, summary, [('subjects_dir', 'subjects_dir')]),
        (bidssrc, summary, [('t1w', 't1w'),
                            ('t2w', 't2w'),
                            ('bold', 'bold')]),
        (bids_info, summary, [('subject', 'subject_id')]),
        (bidssrc, anat_preproc_wf, [('t2w', 'inputnode.t2w'),
                                    ('roi', 'inputnode.roi')]),
        (bidssrc, ds_report_summary, [(('t2w', fix_multi_source_name), 'source_file')]),
        (summary, ds_report_summary, [('out_report', 'in_file')]),
        (bidssrc, ds_report_about, [(('t2w', fix_multi_source_name), 'source_file')]),
        (about, ds_report_about, [('out_report', 'in_file')]),
    ])
    # fmt:on

    # Overwrite ``out_path_base`` of smriprep's DataSinks
    for node in workflow.list_node_names():
        if node.split(".")[-1].startswith("ds_"):
            workflow.get_node(node).interface.out_path_base = "fmriprep"

    if anat_only:
        return workflow

    from sdcflows import fieldmaps as fm

    fmap_estimators = None

    if any(
        (
            "fieldmaps" not in config.workflow.ignore,
            config.workflow.use_syn_sdc,
            config.workflow.force_syn,
        )
    ):
        from sdcflows.utils.wrangler import find_estimators

        # SDC Step 1: Run basic heuristics to identify available data for fieldmap estimation
        # For now, no fmapless
        filters = None
        if config.execution.bids_filters is not None:
            filters = config.execution.bids_filters.get("fmap")
        fmap_estimators = find_estimators(
            layout=config.execution.layout,
            subject=subject_id,
            fmapless=bool(config.workflow.use_syn_sdc),
            force_fmapless=config.workflow.force_syn,
            bids_filters=filters,
            anat_suffix=["T2w", "T1w", "UNIT1"]
        )

        if config.workflow.use_syn_sdc and not fmap_estimators:
            message = (
                "Fieldmap-less (SyN) estimation was requested, but PhaseEncodingDirection "
                "information appears to be absent."
            )
            config.loggers.workflow.error(message)
            if config.workflow.use_syn_sdc == "error":
                raise ValueError(message)

        if "fieldmaps" in config.workflow.ignore and any(
            f.method == fm.EstimatorType.ANAT for f in fmap_estimators
        ):
            config.loggers.workflow.info(
                'Option "--ignore fieldmaps" was set, but either "--use-syn-sdc" '
                'or "--force-syn" were given, so fieldmap-less estimation will be executed.'
            )
            fmap_estimators = [f for f in fmap_estimators if f.method == fm.EstimatorType.ANAT]

        if fmap_estimators:
            config.loggers.workflow.info(
                "B0 field inhomogeneity map will be estimated with "
                f" the following {len(fmap_estimators)} estimators: "
                f"{[e.method for e in fmap_estimators]}."
            )

    # Append the functional section to the existing anatomical excerpt
    # That way we do not need to stream down the number of bold datasets
    func_pre_desc = """
Functional data preprocessing

: For each of the {num_bold} BOLD runs found per subject (across all
tasks and sessions), the following preprocessing was performed.
""".format(
        num_bold=len(subject_data['bold'])
    )

    func_preproc_wfs = []
    has_fieldmap = bool(fmap_estimators)
    for bold_file in subject_data['bold']:
        func_preproc_wf = init_func_preproc_wf(bold_file, has_fieldmap=has_fieldmap)
        if func_preproc_wf is None:
            continue

        func_preproc_wf.__desc__ = func_pre_desc + (func_preproc_wf.__desc__ or "")
        # fmt:off
        workflow.connect([
            (anat_preproc_wf, func_preproc_wf,
             [('outputnode.t2w_preproc', 'inputnode.anat_preproc'),
              ('outputnode.t2w_mask', 'inputnode.anat_mask'),
              ('outputnode.t2w_dseg', 'inputnode.anat_dseg'),
              ('outputnode.t2w_tpms', 'inputnode.anat_tpms'),
              ('outputnode.template', 'inputnode.template'),
              ('outputnode.anat2std_xfm', 'inputnode.anat2std_xfm'),
              ('outputnode.std2anat_xfm', 'inputnode.std2anat_xfm')]),
        ])
        # fmt:on
        func_preproc_wfs.append({'wf': func_preproc_wf, 'bold_file': bold_file})

    if not has_fieldmap:
        # do reference workflow since it won't otherwise be done
        for func_preproc_wf in func_preproc_wfs:
            ref_wf = init_reference_workflow(func_preproc_wf['bold_file'])
            workflow.connect([
                (ref_wf, func_preproc_wf['wf'], [
                    ("outputnode.ref_file", "inputnode.ref_file"),
                    ("outputnode.bold_ref_xfm", "inputnode.bold_ref_xfm"),
                    ("outputnode.validation_report", "inputnode.validation_report"),
                    ("outputnode.n_dummy_scans", "inputnode.n_dummy_scans")
                ]),
            ])
        return workflow

    from sdcflows.workflows.base import init_fmap_preproc_wf

    fmap_wf = init_fmap_preproc_wf(
        debug=config.execution.debug,
        estimators=fmap_estimators,
        omp_nthreads=config.nipype.omp_nthreads,
        output_dir=output_dir,
        subject=subject_id,
    )
    fmap_wf.__desc__ = f"""

Preprocessing of B<sub>0</sub> inhomogeneity mappings

: A total of {len(fmap_estimators)} fieldmaps were found available within the input
BIDS structure for this particular subject.
"""

    for func_preproc_wf in func_preproc_wfs:
        workflow.connect([
            (fmap_wf, func_preproc_wf['wf'], [
                ("outputnode.fmap", "inputnode.fmap"),
                ("outputnode.fmap_ref", "inputnode.fmap_ref"),
                ("outputnode.fmap_coeff", "inputnode.fmap_coeff"),
                ("outputnode.fmap_mask", "inputnode.fmap_mask"),
                ("outputnode.fmap_id", "inputnode.fmap_id"),
                ("outputnode.method", "inputnode.sdc_method"),
            ]),
        ])
    # fmt: on

    # Overwrite ``out_path_base`` of sdcflows's DataSinks
    for node in fmap_wf.list_node_names():
        if node.split(".")[-1].startswith("ds_"):
            fmap_wf.get_node(node).interface.out_path_base = "fmriprep"

    # Step 3: Manually connect PEPOLAR and ANAT workflows

    # Select "Fisher344" from standard references.
    # This node may be used by multiple ANAT estimators, so define outside loop.
    from niworkflows.interfaces.utility import KeySelect

    fmap_select_std = pe.Node(
        KeySelect(fields=["std2anat_xfm"], key="Fischer344"),
        name="fmap_select_std",
        run_without_submitting=True,
    )
    if any(estimator.method == fm.EstimatorType.ANAT for estimator in fmap_estimators):
        # fmt:off
        workflow.connect([
            (anat_preproc_wf, fmap_select_std, [
                ("outputnode.std2anat_xfm", "std2anat_xfm"),
                ("outputnode.template", "keys")]),
        ])
        # fmt:on
    else:
        # connect reference workflow since it won't be done otherwise
        for func_preproc_wf in func_preproc_wfs:
            ref_wf = init_reference_workflow(func_preproc_wf['bold_file'])
            # fmt:off
            workflow.connect([
                (ref_wf, func_preproc_wf['wf'], [
                    ("outputnode.ref_file", "inputnode.ref_file"),
                    ("outputnode.bold_ref_xfm", "inputnode.bold_ref_xfm"),
                    ("outputnode.validation_report", "inputnode.validation_report"),
                    ("outputnode.n_dummy_scans", "inputnode.n_dummy_scans")
                ]),
            ])
            # fmt:on

    for estimator in fmap_estimators:
        config.loggers.workflow.info(
            f"""\
Setting-up fieldmap "{estimator.bids_id}" ({estimator.method}) with \
<{', '.join(s.path.name for s in estimator.sources)}>"""
        )

        # Mapped and phasediff can be connected internally by SDCFlows
        if estimator.method in (fm.EstimatorType.MAPPED, fm.EstimatorType.PHASEDIFF):
            continue

        suffices = [s.suffix for s in estimator.sources]

        if estimator.method == fm.EstimatorType.PEPOLAR:
            if set(suffices) == {"epi"} or sorted(suffices) == ["bold", "epi"]:
                wf_inputs = getattr(fmap_wf.inputs, f"in_{estimator.bids_id}")
                wf_inputs.in_data = [str(s.path) for s in estimator.sources]
                wf_inputs.metadata = [s.metadata for s in estimator.sources]

                scale_topup_inputs = pe.Node(
                    niu.Function(function=_scale_header),
                    name="scale_topup_inputs"
                )
                fix_fieldcoeff_affine = pe.Node(
                    niu.Function(function=_fix_affine_order),
                    name="fix_fieldcoeff_affine"
                )
                reorient_fieldcoeff = pe.Node(
                    niu.Function(function=_reorient),
                    name="reorient_fieldcoeff"
                )
                pe_wf = fmap_wf.get_node(f"wf_{estimator.bids_id}")
                pe_wf.add_nodes([scale_topup_inputs, fix_fieldcoeff_affine, reorient_fieldcoeff])
                # if pe dir is IS/SI, need to reorient to LSA for topup
                # https://www.jiscmail.ac.uk/cgi-bin/wa-jisc.exe?A2=ind1504&L=FSL&D=0&P=277030
                pe_wf.inputs.to_las.target_orientation = "LSA"
                # two pass averaging is derailing averaging, so use single-pass instead
                pe_wf.inputs.setwise_avg.two_pass = False
                # adjust n4 parameters for surface coils
                pe_wf.inputs.brainextraction_wf.n4.args = _bspline_grid(estimator.sources[0].path)
                pe_wf.inputs.brainextraction_wf.n4.shrink_factor = 1
                pe_wf.inputs.brainextraction_wf.n4.n_iterations = [50] * 4
                # add scaling for sub-mm resolution compatibility with topup
                pe_wf.inputs.scale_topup_inputs.factor = 10.0
                to_las = pe_wf.get_node("to_las")
                topup = pe_wf.get_node("topup")
                fix_coeff = pe_wf.get_node("fix_coeff")
                setwise_avg = pe_wf.get_node("setwise_avg")
                outputnode = pe_wf.get_node("outputnode")
                # fmt: off
                pe_wf.disconnect(to_las, "out_file", topup, "in_file")
                pe_wf.disconnect(topup, "out_fieldcoef", fix_coeff, "in_coeff")
                pe_wf.disconnect(fix_coeff, "out_coeff", outputnode, "fmap_coeff")
                pe_wf.connect([
                    (to_las, scale_topup_inputs, [("out_file", "in_file")]),
                    (to_las, fix_fieldcoeff_affine, [("out_file", "reference_image")]),
                    (scale_topup_inputs, topup, [("out", "in_file")]),
                    (topup, fix_fieldcoeff_affine, [("out_fieldcoef", "moving_image")]),
                    (fix_fieldcoeff_affine, fix_coeff, [("out", "in_coeff")]),
                    (fix_coeff, reorient_fieldcoeff, [("out_coeff", "moving_image")]),
                    (setwise_avg, reorient_fieldcoeff, [("out_file", "reference_image")]),
                    (reorient_fieldcoeff, outputnode, [("out", "fmap_coeff")]),
                ])
                # fmt: on
                if config.execution.debug is False:
                    descale_topup_corrected = pe.Node(
                        niu.Function(function=_scale_header),
                        name="descale_topup_corrected"
                    )
                    descale_topup_field = pe.Node(
                        niu.Function(function=_scale_header),
                        name="descale_topup_field"
                    )
                    descale_topup_warp = pe.MapNode(
                        niu.Function(function=_scale_header),
                        name="descale_topup_warp",
                        iterfield=['in_file']
                    )
                    pe_wf.add_nodes([
                        descale_topup_corrected,
                        descale_topup_field,
                        descale_topup_warp
                    ])
                    from_las = pe_wf.get_node("from_las")
                    from_las_fmap = pe_wf.get_node("from_las_fmap")
                    # fmt: off
                    pe_wf.disconnect(topup, "out_corrected", from_las, "in_file")
                    pe_wf.disconnect(topup, "out_field", from_las_fmap, "in_file")
                    pe_wf.disconnect(topup, "out_warps", outputnode, "out_warps")
                    pe_wf.connect([
                        (topup, descale_topup_corrected, [("out_corrected", "in_file")]),
                        (descale_topup_corrected, from_las, [("out", "in_file")]),
                        (topup, descale_topup_field, [("out_field", "in_file")]),
                        (descale_topup_field, from_las_fmap, [("out", "in_file")]),
                        (topup, descale_topup_warp, [("out_warps", "in_file")]),
                        (descale_topup_warp, outputnode, [("out", "out_warps")]),
                    ])
                    # fmt: on
            else:
                raise NotImplementedError("Sophisticated PEPOLAR schemes are unsupported.")

        elif estimator.method == fm.EstimatorType.ANAT:
            from sdcflows.workflows.fit.syn import init_syn_preprocessing_wf

            sources = [str(s.path) for s in estimator.sources if s.suffix == "bold"]
            source_meta = [s.metadata for s in estimator.sources if s.suffix == "bold"]
            syn_preprocessing_wf = init_syn_preprocessing_wf(
                omp_nthreads=config.nipype.omp_nthreads,
                debug=config.execution.debug is True,
                auto_bold_nss=True,
                t1w_inversion=False,
                name=f"syn_preprocessing_{estimator.bids_id}",
            )
            syn_preprocessing_wf.inputs.inputnode.in_epis = sources
            syn_preprocessing_wf.inputs.inputnode.in_meta = source_meta

            #  The default N4 shrink factor (4) appears to artificially blur values across
            #  anisotropic voxels. Shrink factors are intended to speed up calculation
            #  but in most cases, the extra calculation time appears to be minimal.
            syn_preprocessing_wf.inputs.epi_reference_wf.n4_avgs.shrink_factor = 1
            #  Similarly, the use of an asymmetric bspline grid improves performance
            #  in anisotropic voxels, so set INU bspline grid based on voxel size
            syn_preprocessing_wf.inputs.epi_reference_wf.n4_avgs.args = _bspline_grid(sources[0])
            # The number of N4 iterations are also reduced.
            syn_preprocessing_wf.inputs.epi_reference_wf.n4_avgs.n_iterations = [50] * 4

            # Select "Fischer344" from standard references.
            create_prior = pe.Node(niu.Function(function=_create_atlas_prior), name="create_prior")
            create_prior.inputs.template = 'Fischer344'
            create_prior.inputs.suffix = 'T2w'

            # fmt:off
            workflow.connect([
                (anat_preproc_wf, syn_preprocessing_wf, [
                    ("outputnode.t2w_preproc", "inputnode.in_anat"),
                    ("outputnode.t2w_mask", "inputnode.mask_anat"),
                ]),
                (create_prior, syn_preprocessing_wf, [('out', 'prior2epi.input_image')]),
                (fmap_select_std, syn_preprocessing_wf, [
                    ("std2anat_xfm", "inputnode.std2anat_xfm"),
                ]),
                (syn_preprocessing_wf, fmap_wf, [
                    ("outputnode.epi_ref", f"in_{estimator.bids_id}.epi_ref"),
                    ("outputnode.epi_mask", f"in_{estimator.bids_id}.epi_mask"),
                    ("outputnode.anat_ref", f"in_{estimator.bids_id}.anat_ref"),
                    ("outputnode.anat_mask", f"in_{estimator.bids_id}.anat_mask"),
                    ("outputnode.sd_prior", f"in_{estimator.bids_id}.sd_prior"),
                ]),
            ])
            for func_preproc_wf in func_preproc_wfs:
                workflow.connect([
                    (syn_preprocessing_wf, func_preproc_wf['wf'], [
                        ("epi_reference_wf.outputnode.epi_ref_file", "inputnode.ref_file"),
                        ("epi_reference_wf.outputnode.xfm_files", "inputnode.bold_ref_xfm"),
                        ("epi_reference_wf.outputnode.validation_report",
                         "inputnode.validation_report"),
                        (("epi_reference_wf.outputnode.n_dummy", _pop), "inputnode.n_dummy_scans")
                    ]),
                ])
            # fmt:on
    return workflow


def _prefix(subid):
    return subid if subid.startswith("sub-") else f"sub-{subid}"


def _pop(inlist):
    if isinstance(inlist, (list, tuple)):
        return inlist[0]
    return inlist


def _create_atlas_prior(template=None, suffix=None):
    from templateflow.api import get
    import nibabel as nb
    import numpy as np
    from pathlib import Path
    from nipype.utils.filemanip import fname_presuffix

    if template is None:
        template = 'Fischer344'
    if suffix is None:
        suffix = 'T2w'

    tpl_path = get(template, suffix=suffix)
    tpl_img = nb.load(tpl_path)
    tpl_data = tpl_img.get_fdata()
    new_data = np.ones_like(tpl_data) * 4

    out = fname_presuffix(Path(tpl_path).name, suffix="_sdcprior", newpath=Path.cwd())
    hdr = tpl_img.header.copy()
    hdr.set_data_dtype("uint8")
    tpl_img.__class__(new_data.astype("uint8"), tpl_img.affine, hdr).to_filename(out)

    return out


def init_reference_workflow(bold_file):
    # run reference workflow if reference image is not passed
    from nipype.pipeline import engine as pe
    from nirodents.workflows.brainextraction import _bspline_grid
    from niworkflows.engine.workflows import LiterateWorkflow as Workflow
    from niworkflows.workflows.epi.refmap import init_epi_reference_wf
    from niworkflows.utils.connections import listify
    from pathlib import Path
    from ..patch.utils import extract_entities

    workflow = Workflow(name=f"{Path(bold_file).name.split('.')[0]}_ref_wf")

    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=["ref_file", "bold_ref_xfm", "validation_report", "n_dummy_scans"]),
        name="outputnode"
    )

    echoes = extract_entities(bold_file).get("echo", [])
    echo_idxs = listify(echoes)
    multiecho = len(echo_idxs) > 2

    bold_ref_wf = init_epi_reference_wf(
        auto_bold_nss=True,
        omp_nthreads=config.nipype.omp_nthreads
    )
    bold_ref_wf.inputs.inputnode.in_files = (bold_file if not multiecho else bold_file[0])
    # set INU bspline grid based on voxel size
    bspline_grid = _bspline_grid(
        bold_file if not multiecho else bold_file[0]
    )
    bold_ref_wf.inputs.n4_avgs.args = bspline_grid
    #  The default N4 shrink factor (4) appears to artificially blur values across
    #  anisotropic voxels. Shrink factors are intended to speed up calculation
    #  but in most cases, the extra calculation time appears to be minimal.
    #  Similarly, the use of an asymmetric bspline grid improves performance
    #  in anisotropic voxels. The number of N4 iterations are also reduced.
    bold_ref_wf.inputs.n4_avgs.shrink_factor = 1
    bold_ref_wf.inputs.n4_avgs.n_iterations = [50] * 4

    # fmt:off
    workflow.connect([
        (bold_ref_wf, outputnode,
            [('outputnode.epi_ref_file', 'ref_file'),
             ('outputnode.xfm_files', 'bold_ref_xfm'),
             ('outputnode.validation_report', 'validation_report'),
             (('outputnode.n_dummy', _pop), 'n_dummy_scans')])
    ])
    # fmt:on
    return workflow


def _scale_header(in_file, factor=0.1):
    from pathlib import Path
    import nibabel as nb
    from nipype.utils.filemanip import fname_presuffix

    fname = Path(in_file)
    img = nb.load(in_file)

    # first update header zooms
    new_hdr = img.header.copy()
    orig_zooms = img.header.get_zooms()
    zoom_lim = len(orig_zooms) if len(orig_zooms) <= 3 else 3
    new_zooms = [float(zoom * factor) for zoom in orig_zooms[0:zoom_lim]]

    if len(orig_zooms) > 3:
        new_zooms = [
            element for sublist in
            [new_zooms, list(new_hdr.get_zooms()[3:])]
            for element in sublist
        ]
    new_hdr.set_zooms(tuple(new_zooms))

    # next update affine
    new_affine = img.affine.copy()
    new_affine[:3] *= factor

    out = fname_presuffix(fname.name, suffix="_scaled", newpath=Path.cwd())
    img.__class__(img.dataobj, new_affine, new_hdr).to_filename(out)

    return out


def _fix_affine_order(reference_image, moving_image):
    from pathlib import Path
    import nibabel as nb
    from numpy import moveaxis
    from nipype.utils.filemanip import fname_presuffix

    ref = nb.load(reference_image)
    moving = nb.load(moving_image)

    ref_axis_order = nb.io_orientation(ref.affine)[:,0]
    moving_axis_order = nb.io_orientation(moving.affine)[:,0]

    # change the order of the rows based on the reference affine
    reindex = [list(ref_axis_order).index(val) for val in moving_axis_order]

    new_affine = moving.affine.copy()
    new_affine[:3,:] = moving.affine[reindex,:]

    # reorient to LAS as expected by sdcflows
    las_ornt = nb.orientations.ornt_transform(
        nb.io_orientation(new_affine),
        nb.orientations.axcodes2ornt("LAS")
    )
    moving_las = moving.as_reoriented(las_ornt)

    # reorder data matrix
    new_data = moveaxis(
        moving.get_fdata(),
        moving_axis_order.astype(int),
        ref_axis_order.astype(int)
    )
    out = fname_presuffix(moving_image, newpath=Path.cwd())

    moving_las.__class__(
        new_data,
        new_affine,
        moving.header
    ).to_filename(out)

    return out


def _reorient(reference_image, moving_image):
    from pathlib import Path
    import nibabel as nb
    from nipype.utils.filemanip import fname_presuffix

    ref = nb.load(reference_image)
    moving = nb.load(moving_image)

    new_affine = moving.affine.copy()

    # reorient to RAS
    reorient = nb.orientations.ornt_transform(
        nb.orientations.io_orientation(new_affine),
        nb.orientations.io_orientation(ref.affine)
    )

    out = fname_presuffix(moving_image, newpath=Path.cwd())
    moving.as_reoriented(reorient).to_filename(out)

    return out
