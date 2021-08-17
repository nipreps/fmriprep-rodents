# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Resampling workflows
++++++++++++++++++++

.. autofunction:: init_bold_surf_wf
.. autofunction:: init_bold_std_trans_wf
.. autofunction:: init_bold_preproc_trans_wf

"""
from ...config import DEFAULT_MEMORY_MIN_GB

from nipype.pipeline import engine as pe
from nipype.interfaces import utility as niu
from nipype.interfaces.fsl import Split as FSLSplit
import nipype.interfaces.workbench as wb


def init_bold_std_trans_wf(
    mem_gb,
    omp_nthreads,
    spaces,
    name="bold_std_trans_wf",
    use_compression=True,
    use_fieldwarp=False,
):
    """
    Sample fMRI into standard space with a single-step resampling of the original BOLD series.

    .. important::
        This workflow provides two outputnodes.
        One output node (with name ``poutputnode``) will be parameterized in a Nipype sense
        (see `Nipype iterables
        <https://miykael.github.io/nipype_tutorial/notebooks/basic_iteration.html>`__), and a
        second node (``outputnode``) will collapse the parameterized outputs into synchronous
        lists of the output fields listed below.

    Workflow Graph
        .. workflow::
            :graph2use: colored
            :simple_form: yes

            from niworkflows.utils.spaces import SpatialReferences
            from fprodents.workflows.bold.resampling import init_bold_std_trans_wf
            wf = init_bold_std_trans_wf(
                mem_gb=3,
                omp_nthreads=1,
                spaces=SpatialReferences(
                    spaces=['MNI152Lin',
                            ('MNIPediatricAsym', {'cohort': '6'})],
                    checkpoint=True),
            )

    Parameters
    ----------
    mem_gb : :obj:`float`
        Size of BOLD file in GB
    omp_nthreads : :obj:`int`
        Maximum number of threads an individual process may use
    spaces : :py:class:`~niworkflows.utils.spaces.SpatialReferences`
        A container for storing, organizing, and parsing spatial normalizations. Composed of
        :py:class:`~niworkflows.utils.spaces.Reference` objects representing spatial references.
        Each ``Reference`` contains a space, which is a string of either TemplateFlow template IDs
        (e.g., ``MNI152Lin``, ``MNI152NLin6Asym``, ``MNIPediatricAsym``), nonstandard references
        (e.g., ``T1w`` or ``anat``, ``sbref``, ``run``, etc.), or a custom template located in
        the TemplateFlow root directory. Each ``Reference`` may also contain a spec, which is a
        dictionary with template specifications (e.g., a specification of ``{'resolution': 2}``
        would lead to resampling on a 2mm resolution of the space).
    name : :obj:`str`
        Name of workflow (default: ``bold_std_trans_wf``)
    use_compression : :obj:`bool`
        Save registered BOLD series as ``.nii.gz``
    use_fieldwarp : :obj:`bool`
        Include SDC warp in single-shot transform from BOLD to MNI

    Inputs
    ------
    anat2std_xfm
        List of anatomical-to-standard space transforms generated during
        spatial normalization.
    bold_mask
        Skull-stripping mask of reference image
    bold_split
        Individual 3D volumes, not motion corrected
    fieldwarp
        a :abbr:`DFM (displacements field map)` in ITK format
    hmc_xforms
        List of affine transforms aligning each volume to ``ref_image`` in ITK format
    bold2anat
        Affine transform from ``ref_bold_brain`` to T1 space (ITK format)
    name_source
        BOLD series NIfTI file
        Used to recover original information lost during processing
    templates
        List of templates that were applied as targets during
        spatial normalization.

    Outputs
    -------
    bold_std
        BOLD series, resampled to template space
    bold_std_ref
        Reference, contrast-enhanced summary of the BOLD series, resampled to template space
    bold_mask_std
        BOLD series mask in template space
    template
        Template identifiers synchronized correspondingly to previously
        described outputs.

    """
    from niworkflows.engine.workflows import LiterateWorkflow as Workflow
    from niworkflows.interfaces.fixes import FixHeaderApplyTransforms as ApplyTransforms
    from niworkflows.interfaces.itk import MultiApplyTransforms
    from niworkflows.interfaces.utility import KeySelect
    from niworkflows.interfaces.nibabel import GenerateSamplingReference
    from niworkflows.interfaces.nilearn import Merge
    from niworkflows.utils.spaces import format_reference

    workflow = Workflow(name=name)
    output_references = spaces.cached.get_spaces(nonstandard=False, dim=(3,))
    std_vol_references = [
        (s.fullname, s.spec) for s in spaces.references if s.standard and s.dim == 3
    ]

    if len(output_references) == 1:
        workflow.__desc__ = """\
The BOLD time-series were resampled into standard space,
generating a *preprocessed BOLD run in {tpl} space*.
""".format(
            tpl=output_references[0]
        )
    elif len(output_references) > 1:
        workflow.__desc__ = """\
The BOLD time-series were resampled into several standard spaces,
correspondingly generating the following *spatially-normalized,
preprocessed BOLD runs*: {tpl}.
""".format(
            tpl=", ".join(output_references)
        )

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                "anat2std_xfm",
                "bold_mask",
                "bold_split",
                "fieldwarp",
                "hmc_xforms",
                "bold2anat",
                "name_source",
                "templates",
            ]
        ),
        name="inputnode",
    )

    iterablesource = pe.Node(
        niu.IdentityInterface(fields=["std_target"]), name="iterablesource"
    )
    # Generate conversions for every template+spec at the input
    iterablesource.iterables = [("std_target", std_vol_references)]

    split_target = pe.Node(
        niu.Function(
            function=_split_spec,
            input_names=["in_target"],
            output_names=["space", "template", "spec"],
        ),
        run_without_submitting=True,
        name="split_target",
    )

    select_std = pe.Node(
        KeySelect(fields=["anat2std_xfm"]),
        name="select_std",
        run_without_submitting=True,
    )

    select_tpl = pe.Node(
        niu.Function(function=_select_template),
        name="select_tpl",
        run_without_submitting=True,
    )

    gen_ref = pe.Node(
        GenerateSamplingReference(), name="gen_ref", mem_gb=0.3
    )  # 256x256x256 * 64 / 8 ~ 150MB)

    mask_std_tfm = pe.Node(
        ApplyTransforms(interpolation="MultiLabel"), name="mask_std_tfm", mem_gb=1
    )

    ref_std_tfm = pe.Node(
        ApplyTransforms(interpolation="LanczosWindowedSinc"), name="ref_std_tfm", mem_gb=1
    )

    # Write corrected file in the designated output dir
    mask_merge_tfms = pe.Node(
        niu.Merge(2),
        name="mask_merge_tfms",
        run_without_submitting=True,
        mem_gb=DEFAULT_MEMORY_MIN_GB,
    )

    nxforms = 3 + use_fieldwarp
    merge_xforms = pe.Node(
        niu.Merge(nxforms),
        name="merge_xforms",
        run_without_submitting=True,
        mem_gb=DEFAULT_MEMORY_MIN_GB,
    )
    workflow.connect([(inputnode, merge_xforms, [("hmc_xforms", "in%d" % nxforms)])])

    if use_fieldwarp:
        workflow.connect([(inputnode, merge_xforms, [("fieldwarp", "in3")])])

    bold_to_std_transform = pe.Node(
        MultiApplyTransforms(
            interpolation="LanczosWindowedSinc", float=True, copy_dtype=True
        ),
        name="bold_to_std_transform",
        mem_gb=mem_gb * 3 * omp_nthreads,
        n_procs=omp_nthreads,
    )

    merge = pe.Node(Merge(compress=use_compression), name="merge", mem_gb=mem_gb * 3)

    # Generate a reference on the target standard space
    # gen_final_ref = init_bold_reference_wf(omp_nthreads=omp_nthreads, pre_mask=True)

    # fmt:off
    workflow.connect([
        (iterablesource, split_target, [('std_target', 'in_target')]),
        (iterablesource, select_tpl, [('std_target', 'template')]),
        (inputnode, select_std, [('anat2std_xfm', 'anat2std_xfm'),
                                 ('templates', 'keys')]),
        (inputnode, mask_std_tfm, [('bold_mask', 'input_image')]),
        (inputnode, ref_std_tfm, [('bold_mask', 'input_image')]),
        (inputnode, gen_ref, [(('bold_split', _first), 'moving_image')]),
        (inputnode, merge_xforms, [
            (('bold2anat', _aslist), 'in2')]),
        (inputnode, merge, [('name_source', 'header_source')]),
        (inputnode, mask_merge_tfms, [(('bold2anat', _aslist), 'in2')]),
        (inputnode, bold_to_std_transform, [('bold_split', 'input_image')]),
        (split_target, select_std, [('space', 'key')]),
        (select_std, merge_xforms, [('anat2std_xfm', 'in1')]),
        (select_std, mask_merge_tfms, [('anat2std_xfm', 'in1')]),
        (split_target, gen_ref, [(('spec', _is_native), 'keep_native')]),
        (select_tpl, gen_ref, [('out', 'fixed_image')]),
        (merge_xforms, bold_to_std_transform, [('out', 'transforms')]),
        (gen_ref, bold_to_std_transform, [('out_file', 'reference_image')]),
        (gen_ref, mask_std_tfm, [('out_file', 'reference_image')]),
        (mask_merge_tfms, mask_std_tfm, [('out', 'transforms')]),
        (gen_ref, ref_std_tfm, [('out_file', 'reference_image')]),
        (mask_merge_tfms, ref_std_tfm, [('out', 'transforms')]),
        (bold_to_std_transform, merge, [('out_files', 'in_files')]),
    ])
    # fmt:on

    output_names = [
        "bold_mask_std",
        "bold_std",
        "bold_std_ref",
        "spatial_reference",
        "template",
    ]

    poutputnode = pe.Node(
        niu.IdentityInterface(fields=output_names), name="poutputnode"
    )
    # fmt:off
    workflow.connect([
        # Connecting outputnode
        (iterablesource, poutputnode, [
            (('std_target', format_reference), 'spatial_reference')]),
        (merge, poutputnode, [('out_file', 'bold_std')]),
        (ref_std_tfm, poutputnode, [('output_image', 'bold_std_ref')]),
        (mask_std_tfm, poutputnode, [('output_image', 'bold_mask_std')]),
        (select_std, poutputnode, [('key', 'template')]),
    ])
    # fmt:on

    # Connect parametric outputs to a Join outputnode
    outputnode = pe.JoinNode(
        niu.IdentityInterface(fields=output_names),
        name="outputnode",
        joinsource="iterablesource",
    )
    # fmt:off
    workflow.connect([
        (poutputnode, outputnode, [(f, f) for f in output_names]),
    ])
    # fmt:on
    return workflow


def init_bold_preproc_trans_wf(
    mem_gb,
    omp_nthreads,
    name="bold_preproc_trans_wf",
    use_compression=True,
    use_fieldwarp=False,
    split_file=False,
    interpolation="LanczosWindowedSinc",
):
    """
    Resample in native (original) space.

    This workflow resamples the input fMRI in its native (original)
    space in a "single shot" from the original BOLD series.

    Workflow Graph
        .. workflow::
            :graph2use: colored
            :simple_form: yes

            from fprodents.workflows.bold.resampling import init_bold_preproc_trans_wf
            wf = init_bold_preproc_trans_wf(mem_gb=3, omp_nthreads=1)

    Parameters
    ----------
    mem_gb : :obj:`float`
        Size of BOLD file in GB
    omp_nthreads : :obj:`int`
        Maximum number of threads an individual process may use
    name : :obj:`str`
        Name of workflow (default: ``bold_std_trans_wf``)
    use_compression : :obj:`bool`
        Save registered BOLD series as ``.nii.gz``
    use_fieldwarp : :obj:`bool`
        Include SDC warp in single-shot transform from BOLD to MNI
    split_file : :obj:`bool`
        Whether the input file should be splitted (it is a 4D file)
        or it is a list of 3D files (default ``False``, do not split)
    interpolation : :obj:`str`
        Interpolation type to be used by ANTs' ``applyTransforms``
        (default ``'LanczosWindowedSinc'``)

    Inputs
    ------
    bold_file
        Individual 3D volumes, not motion corrected
    bold_mask
        Skull-stripping mask of reference image
    bold_ref
        BOLD reference image: an average-like 3D image of the time-series
    name_source
        BOLD series NIfTI file
        Used to recover original information lost during processing
    hmc_xforms
        List of affine transforms aligning each volume to ``ref_image`` in ITK format
    fieldwarp
        a :abbr:`DFM (displacements field map)` in ITK format

    Outputs
    -------
    bold
        BOLD series, resampled in native space, including all preprocessing

    """
    from bids.utils import listify
    from niworkflows.engine.workflows import LiterateWorkflow as Workflow
    from niworkflows.interfaces.itk import MultiApplyTransforms
    from niworkflows.interfaces.nilearn import Merge

    workflow = Workflow(name=name)
    workflow.__desc__ = """\
The BOLD time-series (including slice-timing correction when applied)
were resampled onto their original, native space by applying
{transforms}.
These resampled BOLD time-series will be referred to as *preprocessed
BOLD in original space*, or just *preprocessed BOLD*.
""".format(
        transforms="""\
a single, composite transform to correct for head-motion and
susceptibility distortions"""
        if use_fieldwarp
        else """\
the transforms to correct for head-motion"""
    )

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=["name_source", "bold_file", "bold_mask", "bold_ref", "hmc_xforms", "fieldwarp"]
        ),
        name="inputnode",
    )

    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=["bold", "bold_mask", "bold_ref", "bold_ref_brain"]
        ),
        name="outputnode",
    )

    bold_transform = pe.Node(
        MultiApplyTransforms(interpolation=interpolation, float=True, copy_dtype=True),
        name="bold_transform",
        mem_gb=mem_gb * 3 * omp_nthreads,
        n_procs=omp_nthreads,
    )

    merge = pe.Node(Merge(compress=use_compression), name="merge", mem_gb=mem_gb * 3)

    # fmt:off
    workflow.connect([
        (inputnode, merge, [('name_source', 'header_source')]),
        (bold_transform, merge, [('out_files', 'in_files')]),
        (inputnode, bold_transform, [(('hmc_xforms', listify), 'transforms')]),
        (merge, outputnode, [('out_file', 'bold')]),
    ])
    # fmt:on

    # Input file is not splitted
    if split_file:
        bold_split = pe.Node(
            FSLSplit(dimension="t"), name="bold_split", mem_gb=mem_gb * 3
        )
        # fmt:off
        workflow.connect([
            (inputnode, bold_split, [('bold_file', 'in_file')]),
            (bold_split, bold_transform, [
                ('out_files', 'input_image'),
                (('out_files', _first), 'reference_image'),
            ])
        ])
        # fmt:on
    else:
        # fmt:off
        workflow.connect([
            (inputnode, bold_transform, [('bold_file', 'input_image'),
                                         (('bold_file', _first), 'reference_image')]),
        ])
        # fmt:on

    return workflow


def init_bold_grayords_wf(
    grayord_density, mem_gb, repetition_time, name="bold_grayords_wf"
):
    """
    Sample Grayordinates files onto the fsLR atlas.

    Outputs are in CIFTI2 format.

    Workflow Graph
        .. workflow::
            :graph2use: colored
            :simple_form: yes

            from fprodents.workflows.bold.resampling import init_bold_grayords_wf
            wf = init_bold_grayords_wf(mem_gb=0.1, grayord_density='91k')

    Parameters
    ----------
    grayord_density : :obj:`str`
        Either `91k` or `170k`, representing the total of vertices or *grayordinates*.
    mem_gb : :obj:`float`
        Size of BOLD file in GB
    name : :obj:`str`
        Unique name for the subworkflow (default: ``'bold_grayords_wf'``)

    Inputs
    ------
    bold_std : :obj:`str`
        List of BOLD conversions to standard spaces.
    spatial_reference :obj:`str`
        List of unique identifiers corresponding to the BOLD standard-conversions.
    subjects_dir : :obj:`str`
        FreeSurfer's subjects directory.
    surf_files : :obj:`str`
        List of BOLD files resampled on the fsaverage (ico7) surfaces.
    surf_refs :
        List of unique identifiers corresponding to the BOLD surface-conversions.

    Outputs
    -------
    cifti_bold : :obj:`str`
        List of BOLD grayordinates files - (L)eft and (R)ight.
    cifti_variant : :obj:`str`
        Only ``'HCP Grayordinates'`` is currently supported.
    cifti_metadata : :obj:`str`
        Path of metadata files corresponding to ``cifti_bold``.
    cifti_density : :obj:`str`
        Density (i.e., either `91k` or `170k`) of ``cifti_bold``.

    """
    import templateflow.api as tf
    from niworkflows.engine.workflows import LiterateWorkflow as Workflow
    from niworkflows.interfaces.cifti import GenerateCifti
    from niworkflows.interfaces.utility import KeySelect

    workflow = Workflow(name=name)
    workflow.__desc__ = """\
*Grayordinates* files [@hcppipelines] containing {density} samples were also
generated using the highest-resolution ``fsaverage`` as intermediate standardized
surface space.
""".format(
        density=grayord_density
    )

    fslr_density, mni_density = (
        ("32k", "2") if grayord_density == "91k" else ("59k", "1")
    )

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                "bold_std",
                "spatial_reference",
                "subjects_dir",
                "surf_files",
                "surf_refs",
            ]
        ),
        name="inputnode",
    )

    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=["cifti_bold", "cifti_variant", "cifti_metadata", "cifti_density"]
        ),
        name="outputnode",
    )

    # extract out to BOLD base
    select_std = pe.Node(
        KeySelect(fields=["bold_std"]),
        name="select_std",
        run_without_submitting=True,
        nohash=True,
    )
    select_std.inputs.key = "MNI152NLin6Asym_res-%s" % mni_density

    select_fs_surf = pe.Node(
        KeySelect(fields=["surf_files"]),
        name="select_fs_surf",
        run_without_submitting=True,
        mem_gb=DEFAULT_MEMORY_MIN_GB,
    )
    select_fs_surf.inputs.key = "fsaverage"

    # Setup Workbench command. LR ordering for hemi can be assumed, as it is imposed
    # by the iterfield of the MapNode in the surface sampling workflow above.
    resample = pe.MapNode(
        wb.MetricResample(method="ADAP_BARY_AREA", area_metrics=True),
        name="resample",
        iterfield=[
            "in_file",
            "out_file",
            "new_sphere",
            "new_area",
            "current_sphere",
            "current_area",
        ],
    )
    resample.inputs.current_sphere = [
        str(tf.get("fsaverage", hemi=hemi, density="164k", desc="std", suffix="sphere"))
        for hemi in "LR"
    ]
    resample.inputs.current_area = [
        str(
            tf.get(
                "fsaverage",
                hemi=hemi,
                density="164k",
                desc="vaavg",
                suffix="midthickness",
            )
        )
        for hemi in "LR"
    ]
    resample.inputs.new_sphere = [
        str(
            tf.get(
                "fsLR",
                space="fsaverage",
                hemi=hemi,
                density=fslr_density,
                suffix="sphere",
            )
        )
        for hemi in "LR"
    ]
    resample.inputs.new_area = [
        str(
            tf.get(
                "fsLR",
                hemi=hemi,
                density=fslr_density,
                desc="vaavg",
                suffix="midthickness",
            )
        )
        for hemi in "LR"
    ]
    resample.inputs.out_file = [
        "space-fsLR_hemi-%s_den-%s_bold.gii" % (h, grayord_density) for h in "LR"
    ]

    gen_cifti = pe.Node(
        GenerateCifti(
            volume_target="MNI152NLin6Asym",
            surface_target="fsLR",
            TR=repetition_time,
            surface_density=fslr_density,
        ),
        name="gen_cifti",
    )

    # fmt:off
    workflow.connect([
        (inputnode, gen_cifti, [('subjects_dir', 'subjects_dir')]),
        (inputnode, select_std, [('bold_std', 'bold_std'),
                                 ('spatial_reference', 'keys')]),
        (inputnode, select_fs_surf, [('surf_files', 'surf_files'),
                                     ('surf_refs', 'keys')]),
        (select_fs_surf, resample, [('surf_files', 'in_file')]),
        (select_std, gen_cifti, [('bold_std', 'bold_file')]),
        (resample, gen_cifti, [('out_file', 'surface_bolds')]),
        (gen_cifti, outputnode, [('out_file', 'cifti_bold'),
                                 ('variant', 'cifti_variant'),
                                 ('out_metadata', 'cifti_metadata'),
                                 ('density', 'cifti_density')]),
    ])
    # fmt:on
    return workflow


def _split_spec(in_target):
    space, spec = in_target
    template = space.split(":")[0]
    return space, template, spec


def _select_template(template):
    from fprodents.patch.utils import get_template_specs

    template, specs = template
    template = template.split(":")[0]  # Drop any cohort modifier if present
    specs = specs.copy()
    specs["suffix"] = specs.get("suffix", "T2w")

    # Sanitize resolution
    res = specs.pop("res", None) or specs.pop("resolution", None) or "native"
    if res != "native":
        specs["resolution"] = res
        return get_template_specs(template, template_spec=specs)[0]

    # Map nonstandard resolutions to existing resolutions
    if template == "Fischer344":
        default_res = None
    else:
        default_res = 2

    out = get_template_specs(
        template, template_spec=specs, default_resolution=default_res
    )

    return out[0]


def _first(inlist):
    return inlist[0]


def _aslist(in_value):
    if isinstance(in_value, list):
        return in_value
    return [in_value]


def _is_native(in_value):
    return in_value.get("resolution") == "native" or in_value.get("res") == "native"


def _itk2lta(in_file, src_file, dst_file):
    import nitransforms as nt
    from pathlib import Path

    out_file = Path("out.lta").absolute()
    nt.linear.load(
        in_file, fmt="fs" if in_file.endswith(".lta") else "itk", reference=src_file
    ).to_filename(out_file, moving=dst_file, fmt="fs")
    return str(out_file)
