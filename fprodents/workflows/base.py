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
    from niworkflows.utils.connections import listify
    from niworkflows.utils.spaces import Reference
    from niworkflows.workflows.epi.refmap import init_epi_reference_wf
    from ..patch.interfaces import BIDSDataGrabber
    from ..patch.utils import extract_entities, fix_multi_source_name
    from ..patch.workflows.anatomical import init_anat_preproc_wf

    subject_data = collect_data(
        config.execution.layout,
        subject_id,
        config.execution.task_id,
        config.execution.echo_idx,
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
    from sdcflows.interfaces.brainmask import BrainExtraction
    fmap_estimators = None

    if "fieldmaps" not in config.workflow.ignore:
        # SDC Step 1: Run basic heuristics to identify available data for fieldmap estimation
        # For now, no fmapless
        base_entities = {
            "subject": subject_id,
            "extension": [".nii", ".nii.gz"],
            "scope": "raw",  # Ensure derivatives are not captured
        }

        # Find fieldmap-less schemes
        anat_file = config.execution.layout.get(suffix=["T2w", "T1w", "UNIT1"], **base_entities)
        if not anat_file:
            has_fieldmap = False
            fmap_estimators = None
        else:
            from contextlib import suppress
            from pathlib import Path
            from sdcflows.utils.epimanip import get_trt
            candidates = config.execution.layout.get(suffix="bold", **base_entities)
            # Filter out candidates without defined PE direction
            fmap_estimators = []
            epi_targets = []
            pe_dirs = []
            ro_totals = []

            for candidate in candidates:
                meta = candidate.get_metadata()
                pe_dir = meta.get("PhaseEncodingDirection")

                if not pe_dir:
                    continue

                pe_dirs.append(pe_dir)
                ro = 1.0
                with suppress(ValueError):
                    ro = get_trt(meta, candidate.path)
                ro_totals.append(ro)
                meta.update({"TotalReadoutTime": ro})
                epi_targets.append(fm.FieldmapFile(candidate.path, metadata=meta))

            for pe_dir in sorted(set(pe_dirs)):
                pe_ro = [ro for ro, pe in zip(ro_totals, pe_dirs) if pe == pe_dir]
                for ro_time in sorted(set(pe_ro)):
                    fmfiles, fmpaths = tuple(
                        zip(
                            *[
                                (target, str(Path(target.path)))
                                for i, target in enumerate(epi_targets)
                                if pe_dirs[i] == pe_dir and ro_totals[i] == ro_time
                            ]
                        )
                    )
                    fmap_estimators.append(
                        fm.FieldmapEstimation(
                            [
                                fm.FieldmapFile(
                                    anat_file[0], metadata={"IntendedFor": fmpaths}
                                ),
                                *fmfiles,
                            ]
                        )
                    )

            # fmap_estimators = [fm.FieldmapEstimation(
            #     [str(f) for f in config.execution.layout.get(
            #         return_type='file',
            #         subject=subject_id,
            #         suffix=['T2w', 'bold'],
            #         extension='nii.gz')]
            # )]

        if config.workflow.use_syn_sdc and not fmap_estimators:
            message = ("Fieldmap-less (SyN) estimation was requested, but "
                       "PhaseEncodingDirection information appears to be "
                       "absent.")
            config.loggers.workflow.error(message)
            if config.workflow.use_syn_sdc == "error":
                raise ValueError(message)

        if fmap_estimators:
            config.loggers.workflow.info(
                "B0 field inhomogeneity map will be estimated with "
                f" the following {len(fmap_estimators)} estimators: "
                f"{[e.method for e in fmap_estimators]}."
            )

    # Append the functional section to the existing anatomical exerpt
    # That way we do not need to stream down the number of bold datasets
    func_pre_desc = """

Functional data preprocessing

: For each of the {num_bold} BOLD runs found per subject (across all
tasks and sessions), the following preprocessing was performed.
""".format(num_bold=len(subject_data["bold"]))

    func_preproc_wfs = []
    has_fieldmap = bool(fmap_estimators)
    for bold_file in subject_data["bold"]:
        echoes = extract_entities(bold_file).get("echo", [])
        echo_idxs = listify(echoes)
        multiecho = len(echo_idxs) > 2

        bold_ref_wf = init_epi_reference_wf(
            auto_bold_nss=True,
            omp_nthreads=config.nipype.omp_nthreads,
        )
        bold_ref_wf.inputs.inputnode.in_files = (
            bold_file if not multiecho else bold_file[0]
        )
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

        skullstrip_ref = pe.Node(BrainExtraction(), name='skullstrip_ref')

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
            (bold_ref_wf, func_preproc_wf,
             [('outputnode.epi_ref_file', 'inputnode.ref_file'),
              ('outputnode.xfm_files', 'inputnode.bold_ref_xfm'),
              ('outputnode.validation_report', 'inputnode.validation_report'),
              (('outputnode.n_dummy', _pop), 'inputnode.n_dummy_scans')]),
            (bold_ref_wf, skullstrip_ref, [('outputnode.epi_ref_file', 'in_file')])
        ])
        # fmt:on
        func_preproc_wfs.append(func_preproc_wf)

    if not has_fieldmap:
        return workflow

    from sdcflows.workflows.base import init_fmap_preproc_wf

    # Select "Fischer344" from standard references.
    create_prior = pe.Node(niu.Function(function=_create_atlas_prior), name="create_prior")
    create_prior.inputs.template = 'Fischer344'
    create_prior.inputs.suffix = 'T2w'

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

    # Overwrite ``out_path_base`` of sdcflows's DataSinks
    for node in fmap_wf.list_node_names():
        if node.split(".")[-1].startswith("ds_"):
            fmap_wf.get_node(node).interface.out_path_base = ""

    refmap_metadata = pe.Node(niu.Function(function=_merge_meta), name="refmap_metadata")
    source_meta = [s.metadata for s in _pop(fmap_estimators).sources if s.suffix == 'bold']
    refmap_metadata.inputs.meta_list = source_meta

    # fmt: off
    workflow.connect([
        (anat_preproc_wf, fmap_wf, [
            ("outputnode.t2w_preproc", f"in_{_pop(fmap_estimators).bids_id}.anat_ref"),
            ("outputnode.t2w_mask", f"in_{_pop(fmap_estimators).bids_id}.anat_mask")]),
        (bold_ref_wf, refmap_metadata, [('outputnode.epi_ref_file', 'epi_ref')]),
        (refmap_metadata, fmap_wf, [
            ("out", f"in_{_pop(fmap_estimators).bids_id}.epi_ref")]),
        (skullstrip_ref, fmap_wf, [('out_mask', f"in_{_pop(fmap_estimators).bids_id}.epi_mask")]),
        (create_prior, fmap_wf, [("out", f"in_{_pop(fmap_estimators).bids_id}.sd_prior")])
    ])

    for func_preproc_wf in func_preproc_wfs:
        workflow.connect([
            (fmap_wf, func_preproc_wf, [
                ("outputnode.fmap", "inputnode.fmap"),
                ("outputnode.fmap_ref", "inputnode.fmap_ref"),
                ("outputnode.fmap_coeff", "inputnode.fmap_coeff"),
                ("outputnode.fmap_mask", "inputnode.fmap_mask"),
                ("outputnode.fmap_id", "inputnode.fmap_id"),
                ("outputnode.method", "inputnode.sdc_method"),
            ]),
        ])
    # fmt: on
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


def _merge_meta(epi_ref, meta_list):
    """Prepare a tuple of EPI reference and metadata."""
    return (epi_ref, meta_list[0])
