"""Patched workflows for compatibility"""

from nipype import logging
from nipype.pipeline import engine as pe
from nipype.interfaces import fsl, utility as niu
from nipype.interfaces.ants.base import Info as ANTsInfo
from nirodents.workflows.brainextraction import init_rodent_brain_extraction_wf
from niworkflows.interfaces.images import ValidateImage
from niworkflows.engine.workflows import LiterateWorkflow as Workflow
from niworkflows.interfaces.utility import KeySelect
from smriprep.utils.misc import apply_lut as _apply_bids_lut
from smriprep.workflows.anatomical import init_anat_template_wf
from smriprep.workflows.outputs import init_anat_derivatives_wf
from templateflow.api import get_metadata

from ..interfaces import TemplateFlowSelect
from ..utils import fix_multi_source_name

LOGGER = logging.getLogger('nipype.workflow')


def init_anat_preproc_wf(
        *,
        bids_root,
        longitudinal,
        t2w,
        omp_nthreads,
        output_dir,
        skull_strip_mode,
        skull_strip_template,
        spaces,
        debug=False,
        existing_derivatives=None,
        freesurfer=False,
        name='anat_preproc_wf',
        skull_strip_fixed_seed=False,
):
    """
    Stage the anatomical preprocessing steps of *sMRIPrep*.
    This includes:
      - T1w reference: realigning and then averaging T1w images.
      - Brain extraction and INU (bias field) correction.
      - Brain tissue segmentation.
      - Spatial normalization to standard spaces.
    .. include:: ../links.rst
    Workflow Graph
        .. workflow::
            :graph2use: orig
            :simple_form: yes
            from niworkflows.utils.spaces import SpatialReferences, Reference
            from smriprep.workflows.anatomical import init_anat_preproc_wf
            wf = init_anat_preproc_wf(
                bids_root='.',
                longitudinal=False,
                t2w=['t2w.nii.gz'],
                omp_nthreads=1,
                output_dir='.',
                skull_strip_mode='force',
                skull_strip_template=Reference('OASIS30ANTs'),
                spaces=SpatialReferences(spaces=['WHS']),
            )
    Parameters
    ----------
    bids_root : :obj:`str`
        Path of the input BIDS dataset root
    existing_derivatives : :obj:`dict` or None
        Dictionary mapping output specification attribute names and
        paths to corresponding derivatives.
    longitudinal : :obj:`bool`
        Create unbiased structural template, regardless of number of inputs
        (may increase runtime)
    t1w : :obj:`list`
        List of T1-weighted structural images.
    omp_nthreads : :obj:`int`
        Maximum number of threads an individual process may use
    output_dir : :obj:`str`
        Directory in which to save derivatives
    skull_strip_template : :py:class:`~niworkflows.utils.spaces.Reference`
        Spatial reference to use in atlas-based brain extraction.
    spaces : :py:class:`~niworkflows.utils.spaces.SpatialReferences`
        Object containing standard and nonstandard space specifications.
    debug : :obj:`bool`
        Enable debugging outputs
    name : :obj:`str`, optional
        Workflow name (default: anat_preproc_wf)
    skull_strip_mode : :obj:`str`
        Determiner for T1-weighted skull stripping (`force` ensures skull stripping,
        `skip` ignores skull stripping, and `auto` automatically ignores skull stripping
        if pre-stripped brains are detected).
    skull_strip_fixed_seed : :obj:`bool`
        Do not use a random seed for skull-stripping - will ensure
        run-to-run replicability when used with --omp-nthreads 1
        (default: ``False``).
    Inputs
    ------
    t1w
        List of T1-weighted structural images
    t2w
        List of T2-weighted structural images
    roi
        A mask to exclude regions during standardization
    flair
        List of FLAIR images
    subjects_dir
        FreeSurfer SUBJECTS_DIR
    subject_id
        FreeSurfer subject ID
    Outputs
    -------
    t1w_preproc
        The T1w reference map, which is calculated as the average of bias-corrected
        and preprocessed T1w images, defining the anatomical space.
    t1w_brain
        Skull-stripped ``t1w_preproc``
    t1w_mask
        Brain (binary) mask estimated by brain extraction.
    t1w_dseg
        Brain tissue segmentation of the preprocessed structural image, including
        gray-matter (GM), white-matter (WM) and cerebrospinal fluid (CSF).
    t1w_tpms
        List of tissue probability maps corresponding to ``t1w_dseg``.
    std_preproc
        T1w reference resampled in one or more standard spaces.
    std_mask
        Mask of skull-stripped template, in MNI space
    std_dseg
        Segmentation, resampled into MNI space
    std_tpms
        List of tissue probability maps in MNI space
    subjects_dir
        FreeSurfer SUBJECTS_DIR
    anat2std_xfm
        Nonlinear spatial transform to resample imaging data given in anatomical space
        into standard space.
    std2anat_xfm
        Inverse transform of the above.
    See Also
    --------
    * :py:func:`~niworkflows.anat.ants.init_brain_extraction_wf`
    * :py:func:`~smriprep.workflows.surfaces.init_surface_recon_wf`
    """
    workflow = Workflow(name=name)
    num_t2w = len(t2w)
    desc = """Anatomical data preprocessing
: """
    desc += """\
A total of {num_t2w} T2-weighted (T2w) images were found within the input
BIDS dataset."""

    inputnode = pe.Node(
        niu.IdentityInterface(fields=['t2w', 'roi']),
        name='inputnode')

    outputnode = pe.Node(niu.IdentityInterface(
        fields=['t2w_preproc', 't2w_mask', 't2w_dseg', 't2w_tpms',
                'std_preproc', 'std_mask', 'std_dseg', 'std_tpms',
                'anat2std_xfm', 'std2anat_xfm', 'template']),
        name='outputnode')

    # Connect reportlets workflows
    anat_reports_wf = init_anat_reports_wf(
        freesurfer=False,
        output_dir=output_dir,
    )
    workflow.connect([
        (outputnode, anat_reports_wf, [
            ('t2w_preproc', 'inputnode.t1w_preproc'),
            ('t2w_mask', 'inputnode.t1w_mask'),
            ('t2w_dseg', 'inputnode.t1w_dseg')]),
    ])

    if existing_derivatives is not None:
        LOGGER.log(25, "Anatomical workflow will reuse prior derivatives found in the "
                   "output folder (%s).", output_dir)
        desc += """
Anatomical preprocessing was reused from previously existing derivative objects.\n"""
        workflow.__desc__ = desc

        templates = existing_derivatives.pop('template')
        templatesource = pe.Node(niu.IdentityInterface(
            fields=['template']), name='templatesource')
        templatesource.iterables = [('template', templates)]
        outputnode.inputs.template = templates

        for field, value in existing_derivatives.items():
            setattr(outputnode.inputs, field, value)

        anat_reports_wf.inputs.inputnode.source_file = fix_multi_source_name(
            [existing_derivatives['t2w_preproc']], modality='T2w')

        stdselect = pe.Node(KeySelect(
            fields=['std_preproc', 'std_mask'], keys=templates),
            name='stdselect', run_without_submitting=True)
        workflow.connect([
            (inputnode, outputnode, [('subjects_dir', 'subjects_dir'),
                                     ('subject_id', 'subject_id')]),
            (inputnode, anat_reports_wf, [
                ('subjects_dir', 'inputnode.subjects_dir'),
                ('subject_id', 'inputnode.subject_id')]),
            (templatesource, stdselect, [('template', 'key')]),
            (outputnode, stdselect, [('std_preproc', 'std_preproc'),
                                     ('std_mask', 'std_mask')]),
            (stdselect, anat_reports_wf, [
                ('key', 'inputnode.template'),
                ('std_preproc', 'inputnode.std_t1w'),
                ('std_mask', 'inputnode.std_mask'),
            ]),
        ])
        return workflow

    # The workflow is not cached.
    desc += """
All of them were corrected for intensity non-uniformity (INU)
""" if num_t2w > 1 else """\
The T2-weighted (T2w) image was corrected for intensity non-uniformity (INU)
"""
    desc += """\
with `N4BiasFieldCorrection` [@n4], distributed with ANTs {ants_ver} \
[@ants, RRID:SCR_004757]"""
    desc += '.\n' if num_t2w > 1 else ", and used as T2w-reference throughout the workflow.\n"

    desc += """\
The T2w-reference was then skull-stripped with a *Nipype* implementation of
the `antsBrainExtraction.sh` workflow (from ANTs), using {skullstrip_tpl}
as target template.
Brain tissue segmentation of cerebrospinal fluid (CSF),
white-matter (WM) and gray-matter (GM) was performed on
the brain-extracted T1w using `fast` [FSL {fsl_ver}, RRID:SCR_002823,
@fsl_fast].
"""

    workflow.__desc__ = desc.format(
        ants_ver=ANTsInfo.version() or '(version unknown)',
        fsl_ver=fsl.FAST().version or '(version unknown)',
        num_t2w=num_t2w,
        skullstrip_tpl=skull_strip_template.fullname,
    )

    buffernode = pe.Node(niu.IdentityInterface(
        fields=['t2w_brain', 't2w_mask']), name='buffernode')

    # 1. Anatomical reference generation - average input T1w images.
    anat_template_wf = init_anat_template_wf(longitudinal=longitudinal, omp_nthreads=omp_nthreads,
                                             num_t1w=num_t2w)

    anat_validate = pe.Node(ValidateImage(), name='anat_validate',
                            run_without_submitting=True)

    # 2. Brain-extraction and INU (bias field) correction.
    if skull_strip_mode == 'auto':
        import numpy as np
        import nibabel as nb

        def _is_skull_stripped(imgs):
            """Check if T1w images are skull-stripped."""
            def _check_img(img):
                data = np.abs(nb.load(img).get_fdata(dtype=np.float32))
                sidevals = data[0, :, :].sum() + data[-1, :, :].sum() + \
                    data[:, 0, :].sum() + data[:, -1, :].sum() + \
                    data[:, :, 0].sum() + data[:, :, -1].sum()
                return sidevals < 10

            return all(_check_img(img) for img in imgs)

        skull_strip_mode = _is_skull_stripped(t2w)

    if skull_strip_mode in (True, 'skip'):
        raise NotImplementedError("Cannot run on already skull-stripped images.")
    else:
        # ants_affine_init?
        brain_extraction_wf = init_rodent_brain_extraction_wf(
            in_template=skull_strip_template.space,
            omp_nthreads=omp_nthreads,
        )

    # 4. Spatial normalization
    anat_norm_wf = init_anat_norm_wf(
        debug=debug,
        omp_nthreads=omp_nthreads,
        templates=spaces.get_spaces(nonstandard=False, dim=(3,)),
    )

    workflow.connect([
        # Step 1.
        (inputnode, anat_template_wf, [('t2w', 'inputnode.t1w')]),
        (anat_template_wf, anat_validate, [
            ('outputnode.t1w_ref', 'in_file')]),
        (anat_validate, brain_extraction_wf, [
            ('out_file', 'inputnode.in_files')]),
        (brain_extraction_wf, outputnode, [
            (('outputnode.out_corrected', _pop), 't2w_preproc')]),
        (anat_template_wf, outputnode, [
            ('outputnode.t1w_realign_xfm', 't2w_ref_xfms')]),
        (buffernode, outputnode, [('t2w_brain', 't2w_brain'),
                                  ('t2w_mask', 't2w_mask')]),
        # Steps 2, 3 and 4
        (inputnode, anat_norm_wf, [
            (('t2w', fix_multi_source_name), 'inputnode.orig_t1w'),
            ('roi', 'inputnode.lesion_mask')]),
        (brain_extraction_wf, anat_norm_wf, [
            (('outputnode.out_corrected', _pop), 'inputnode.moving_image')]),
        (buffernode, anat_norm_wf, [('t2w_mask', 'inputnode.moving_mask')]),
        (anat_norm_wf, outputnode, [
            ('poutputnode.standardized', 'std_preproc'),
            ('poutputnode.std_mask', 'std_mask'),
            ('poutputnode.std_dseg', 'std_dseg'),
            ('poutputnode.std_tpms', 'std_tpms'),
            ('outputnode.template', 'template'),
            ('outputnode.anat2std_xfm', 'anat2std_xfm'),
            ('outputnode.std2anat_xfm', 'std2anat_xfm'),
        ]),
    ])

    # Change LookUp Table - BIDS wants: 0 (bg), 1 (gm), 2 (wm), 3 (csf)
    lut_t1w_dseg = pe.Node(niu.Function(function=_apply_bids_lut),
                           name='lut_t1w_dseg')

    workflow.connect([
        (lut_t1w_dseg, anat_norm_wf, [
            ('out', 'inputnode.moving_segmentation')]),
        (lut_t1w_dseg, outputnode, [('out', 't2w_dseg')]),
    ])

    # Connect reportlets
    workflow.connect([
        (inputnode, anat_reports_wf, [
            (('t2w', fix_multi_source_name), 'inputnode.source_file')]),
        (outputnode, anat_reports_wf, [
            ('std_preproc', 'inputnode.std_t1w'),
            ('std_mask', 'inputnode.std_mask'),
        ]),
        (anat_template_wf, anat_reports_wf, [
            ('outputnode.out_report', 'inputnode.t1w_conform_report')]),
        (anat_norm_wf, anat_reports_wf, [
            ('poutputnode.template', 'inputnode.template')]),
    ])

    # Write outputs ############################################3
    anat_derivatives_wf = init_anat_derivatives_wf(
        bids_root=bids_root,
        freesurfer=freesurfer,
        num_t1w=num_t2w,
        output_dir=output_dir,
    )

    workflow.connect([
        # Connect derivatives
        (anat_template_wf, anat_derivatives_wf, [
            ('outputnode.t1w_valid_list', 'inputnode.source_files')]),
        (anat_norm_wf, anat_derivatives_wf, [
            ('poutputnode.template', 'inputnode.template'),
            ('poutputnode.anat2std_xfm', 'inputnode.anat2std_xfm'),
            ('poutputnode.std2anat_xfm', 'inputnode.std2anat_xfm')
        ]),
        (outputnode, anat_derivatives_wf, [
            ('std_preproc', 'inputnode.std_t1w'),
            ('t2w_ref_xfms', 'inputnode.t1w_ref_xfms'),
            ('t2w_preproc', 'inputnode.t1w_preproc'),
            ('t2w_mask', 'inputnode.t1w_mask'),
            ('t2w_dseg', 'inputnode.t1w_dseg'),
            ('t2w_tpms', 'inputnode.t1w_tpms'),
            ('std_mask', 'inputnode.std_mask'),
            ('std_dseg', 'inputnode.std_dseg'),
            ('std_tpms', 'inputnode.std_tpms'),
        ]),
    ])

    # Brain tissue segmentation - FAST produces: 0 (bg), 1 (wm), 2 (csf), 3 (gm)
    t1w_dseg = pe.Node(fsl.FAST(segments=True, no_bias=True, probability_maps=True),
                       name='t1w_dseg', mem_gb=3)
    lut_t1w_dseg.inputs.lut = (0, 3, 1, 2)  # Maps: 0 -> 0, 3 -> 1, 1 -> 2, 2 -> 3.
    fast2bids = pe.Node(niu.Function(function=_probseg_fast2bids), name="fast2bids",
                        run_without_submitting=True)

    workflow.connect([
        (brain_extraction_wf, buffernode, [
            (('outputnode.out_brain', _pop), 't2w_brain'),
            ('outputnode.out_mask', 't2w_mask')]),
        (buffernode, t1w_dseg, [('t2w_brain', 'in_files')]),
        (t1w_dseg, lut_t1w_dseg, [('partial_volume_map', 'in_dseg')]),
        (t1w_dseg, fast2bids, [('partial_volume_files', 'inlist')]),
        (fast2bids, anat_norm_wf, [('out', 'inputnode.moving_tpms')]),
        (fast2bids, outputnode, [('out', 't2w_tpms')]),
    ])
    return workflow


def init_anat_norm_wf(
    *,
    debug,
    omp_nthreads,
    templates,
    name="anat_norm_wf",
):
    """
    Build an individual spatial normalization workflow using ``antsRegistration``.

    Workflow Graph
        .. workflow ::
            :graph2use: orig
            :simple_form: yes

            from fmriprep_rodents.patch.workflows.anatomical import init_anat_norm_wf
            wf = init_anat_norm_wf(
                debug=False,
                omp_nthreads=1,
                templates=['WHS'],
            )

    .. important::
        This workflow defines an iterable input over the input parameter ``templates``,
        so Nipype will produce one copy of the downstream workflows which connect
        ``poutputnode.template`` or ``poutputnode.template_spec`` to their inputs
        (``poutputnode`` stands for *parametric output node*).
        Nipype refers to this expansion of the graph as *parameterized execution*.
        If a joint list of values is required (and thus cutting off parameterization),
        please use the equivalent outputs of ``outputnode`` (which *joins* all the
        parameterized execution paths).

    Parameters
    ----------
    debug : :obj:`bool`
        Apply sloppy arguments to speed up processing. Use with caution,
        registration processes will be very inaccurate.
    omp_nthreads : :obj:`int`
        Maximum number of threads an individual process may use.
    templates : :obj:`list` of :obj:`str`
        List of standard space fullnames (e.g., ``MNI152NLin6Asym``
        or ``MNIPediatricAsym:cohort-4``) which are targets for spatial
        normalization.

    Inputs
    ------
    moving_image
        The input image that will be normalized to standard space.
    moving_mask
        A precise brain mask separating skull/skin/fat from brain
        structures.
    moving_segmentation
        A brain tissue segmentation of the ``moving_image``.
    moving_tpms
        tissue probability maps (TPMs) corresponding to the
        ``moving_segmentation``.
    lesion_mask
        (optional) A mask to exclude regions from the cost-function
        input domain to enable standardization of lesioned brains.
    orig_t1w
        The original T1w image from the BIDS structure.
    template
        Template name and specification

    Outputs
    -------
    standardized
        The T1w after spatial normalization, in template space.
    anat2std_xfm
        The T1w-to-template transform.
    std2anat_xfm
        The template-to-T1w transform.
    std_mask
        The ``moving_mask`` in template space (matches ``standardized`` output).
    std_dseg
        The ``moving_segmentation`` in template space (matches ``standardized``
        output).
    std_tpms
        The ``moving_tpms`` in template space (matches ``standardized`` output).
    template
        Template name extracted from the input parameter ``template``, for further
        use in downstream nodes.
    template_spec
        Template specifications extracted from the input parameter ``template``, for
        further use in downstream nodes.

    """
    from collections import defaultdict
    from niworkflows.interfaces.ants import ImageMath
    from niworkflows.interfaces.fixes import FixHeaderApplyTransforms as ApplyTransforms
    from smriprep.interfaces.templateflow import TemplateDesc
    from ..interfaces import RobustMNINormalization

    ntpls = len(templates)
    workflow = Workflow(name=name)

    if templates:
        workflow.__desc__ = """\
Volume-based spatial normalization to {targets} ({targets_id}) was performed through
nonlinear registration with `antsRegistration` (ANTs {ants_ver}),
using brain-extracted versions of both T1w reference and the T1w template.
The following template{tpls} selected for spatial normalization:
""".format(
            ants_ver=ANTsInfo.version() or '(version unknown)',
            targets='%s standard space%s' % (defaultdict(
                'several'.format, {1: 'one', 2: 'two', 3: 'three', 4: 'four'})[ntpls],
                's' * (ntpls != 1)),
            targets_id=', '.join(templates),
            tpls=(' was', 's were')[ntpls != 1]
        )

        # Append template citations to description
        for template in templates:
            template_meta = get_metadata(template.split(':')[0])
            template_refs = ['@%s' % template.split(':')[0].lower()]

            if template_meta.get('RRID', None):
                template_refs += ['RRID:%s' % template_meta['RRID']]

            workflow.__desc__ += """\
*{template_name}* [{template_refs}; TemplateFlow ID: {template}]""".format(
                template=template,
                template_name=template_meta['Name'],
                template_refs=', '.join(template_refs)
            )
            workflow.__desc__ += (', ', '.')[template == templates[-1][0]]

    inputnode = pe.Node(niu.IdentityInterface(fields=[
        'lesion_mask',
        'moving_image',
        'moving_mask',
        'moving_segmentation',
        'moving_tpms',
        'orig_t1w',
        'template',
    ]), name='inputnode')
    inputnode.iterables = [('template', templates)]

    out_fields = [
        'anat2std_xfm',
        'standardized',
        'std2anat_xfm',
        'std_dseg',
        'std_mask',
        'std_tpms',
        'template',
        'template_spec',
    ]
    poutputnode = pe.Node(niu.IdentityInterface(fields=out_fields), name='poutputnode')

    split_desc = pe.Node(TemplateDesc(), run_without_submitting=True, name='split_desc')

    tf_select = pe.Node(TemplateFlowSelect(resolution=1 + debug),
                        name='tf_select', run_without_submitting=True)

    # With the improvements from nipreps/niworkflows#342 this truncation is now necessary
    trunc_mov = pe.Node(ImageMath(operation='TruncateImageIntensity', op2='0.01 0.999 256'),
                        name='trunc_mov')

    registration = pe.Node(RobustMNINormalization(
        float=True, flavor=['precise', 'testing'][debug],
    ), name='registration', n_procs=omp_nthreads, mem_gb=2)

    # Resample T1w-space inputs
    tpl_moving = pe.Node(ApplyTransforms(
        dimension=3, default_value=0, float=True,
        interpolation='LanczosWindowedSinc'), name='tpl_moving')

    std_mask = pe.Node(ApplyTransforms(interpolation='MultiLabel'), name='std_mask')
    std_dseg = pe.Node(ApplyTransforms(interpolation='MultiLabel'), name='std_dseg')

    std_tpms = pe.MapNode(ApplyTransforms(dimension=3, default_value=0, float=True,
                                          interpolation='Gaussian'),
                          iterfield=['input_image'], name='std_tpms')

    workflow.connect([
        (inputnode, split_desc, [('template', 'template')]),
        (inputnode, poutputnode, [('template', 'template')]),
        (inputnode, trunc_mov, [('moving_image', 'op1')]),
        (inputnode, registration, [
            ('moving_mask', 'moving_mask'),
            ('lesion_mask', 'lesion_mask')]),
        (inputnode, tpl_moving, [('moving_image', 'input_image')]),
        (inputnode, std_mask, [('moving_mask', 'input_image')]),
        (split_desc, tf_select, [('name', 'template'),
                                 ('spec', 'template_spec')]),
        (split_desc, registration, [('name', 'template'),
                                    (('spec', _no_atlas), 'template_spec')]),
        (tf_select, tpl_moving, [('t2star_file', 'reference_image')]),
        (tf_select, std_mask, [('t2star_file', 'reference_image')]),
        (tf_select, std_dseg, [('t2star_file', 'reference_image')]),
        (tf_select, std_tpms, [('t2star_file', 'reference_image')]),
        (trunc_mov, registration, [
            ('output_image', 'moving_image')]),
        (registration, tpl_moving, [('composite_transform', 'transforms')]),
        (registration, std_mask, [('composite_transform', 'transforms')]),
        (inputnode, std_dseg, [('moving_segmentation', 'input_image')]),
        (registration, std_dseg, [('composite_transform', 'transforms')]),
        (inputnode, std_tpms, [('moving_tpms', 'input_image')]),
        (registration, std_tpms, [('composite_transform', 'transforms')]),
        (registration, poutputnode, [
            ('composite_transform', 'anat2std_xfm'),
            ('inverse_composite_transform', 'std2anat_xfm')]),
        (tpl_moving, poutputnode, [('output_image', 'standardized')]),
        (std_mask, poutputnode, [('output_image', 'std_mask')]),
        (std_dseg, poutputnode, [('output_image', 'std_dseg')]),
        (std_tpms, poutputnode, [('output_image', 'std_tpms')]),
        (split_desc, poutputnode, [('spec', 'template_spec')]),
    ])

    # Provide synchronized output
    outputnode = pe.JoinNode(niu.IdentityInterface(fields=out_fields),
                             name='outputnode', joinsource='inputnode')
    workflow.connect([
        (poutputnode, outputnode, [(f, f) for f in out_fields]),
    ])

    return workflow


def init_anat_reports_wf(*, freesurfer, output_dir,
                         name='anat_reports_wf'):
    """
    Set up a battery of datasinks to store reports in the right location.
    Parameters
    ----------
    freesurfer : :obj:`bool`
        FreeSurfer was enabled
    output_dir : :obj:`str`
        Directory in which to save derivatives
    name : :obj:`str`
        Workflow name (default: anat_reports_wf)
    Inputs
    ------
    source_file
        Input T1w image
    std_t1w
        T1w image resampled to standard space
    std_mask
        Mask of skull-stripped template
    subject_dir
        FreeSurfer SUBJECTS_DIR
    subject_id
        FreeSurfer subject ID
    t1w_conform_report
        Conformation report
    t1w_preproc
        The T1w reference map, which is calculated as the average of bias-corrected
        and preprocessed T1w images, defining the anatomical space.
    t1w_dseg
        Segmentation in T1w space
    t1w_mask
        Brain (binary) mask estimated by brain extraction.
    template
        Template space and specifications
    """
    from niworkflows.interfaces import SimpleBeforeAfter
    from niworkflows.interfaces.masks import ROIsPlot
    from smriprep.interfaces import DerivativesDataSink

    workflow = Workflow(name=name)

    inputfields = ['source_file', 't1w_conform_report',
                   't1w_preproc', 't1w_dseg', 't1w_mask',
                   'template', 'std_t1w', 'std_mask',
                   'subject_id', 'subjects_dir']
    inputnode = pe.Node(niu.IdentityInterface(fields=inputfields),
                        name='inputnode')

    seg_rpt = pe.Node(ROIsPlot(colors=['b', 'magenta'], levels=[1.5, 2.5]),
                      name='seg_rpt')

    t1w_conform_check = pe.Node(niu.Function(
        function=_empty_report),
        name='t1w_conform_check', run_without_submitting=True)

    ds_t1w_conform_report = pe.Node(
        DerivativesDataSink(base_directory=output_dir, desc='conform', datatype="figures",
                            dismiss_entities=("session",)),
        name='ds_t1w_conform_report', run_without_submitting=True)

    ds_t1w_dseg_mask_report = pe.Node(
        DerivativesDataSink(base_directory=output_dir, suffix='dseg', datatype="figures",
                            dismiss_entities=("session",)),
        name='ds_t1w_dseg_mask_report', run_without_submitting=True)

    workflow.connect([
        (inputnode, t1w_conform_check, [('t1w_conform_report', 'in_file')]),
        (t1w_conform_check, ds_t1w_conform_report, [('out', 'in_file')]),
        (inputnode, ds_t1w_conform_report, [('source_file', 'source_file')]),
        (inputnode, ds_t1w_dseg_mask_report, [('source_file', 'source_file')]),
        (inputnode, seg_rpt, [('t1w_preproc', 'in_file'),
                              ('t1w_mask', 'in_mask'),
                              ('t1w_dseg', 'in_rois')]),
        (seg_rpt, ds_t1w_dseg_mask_report, [('out_report', 'in_file')]),
    ])

    # Generate reportlets showing spatial normalization
    tf_select = pe.Node(TemplateFlowSelect(resolution=1),
                        name='tf_select', run_without_submitting=True)
    norm_msk = pe.Node(niu.Function(
        function=_rpt_masks, output_names=['before', 'after'],
        input_names=['mask_file', 'before', 'after', 'after_mask']),
        name='norm_msk')
    norm_rpt = pe.Node(SimpleBeforeAfter(), name='norm_rpt', mem_gb=0.1)
    norm_rpt.inputs.after_label = 'Participant'  # after

    ds_std_t1w_report = pe.Node(
        DerivativesDataSink(base_directory=output_dir, suffix='T1w', datatype="figures",
                            dismiss_entities=("session",)),
        name='ds_std_t1w_report', run_without_submitting=True)

    workflow.connect([
        (inputnode, tf_select, [('template', 'template')]),
        (inputnode, norm_rpt, [('template', 'before_label')]),
        (inputnode, norm_msk, [('std_t1w', 'after'),
                               ('std_mask', 'after_mask')]),
        (tf_select, norm_msk, [('t2star_file', 'before'),
                               ('brain_mask', 'mask_file')]),
        (norm_msk, norm_rpt, [('before', 'before'),
                              ('after', 'after')]),
        (inputnode, ds_std_t1w_report, [
            (('template', _fmt_cohort), 'space'),
            ('source_file', 'source_file')]),
        (norm_rpt, ds_std_t1w_report, [('out_report', 'in_file')]),
    ])

    if freesurfer:
        from ..interfaces.reports import FSSurfaceReport
        recon_report = pe.Node(FSSurfaceReport(), name='recon_report')
        recon_report.interface._always_run = True

        ds_recon_report = pe.Node(
            DerivativesDataSink(
                base_directory=output_dir, desc='reconall', datatype="figures",
                dismiss_entities=("session",)),
            name='ds_recon_report', run_without_submitting=True)
        workflow.connect([
            (inputnode, recon_report, [('subjects_dir', 'subjects_dir'),
                                       ('subject_id', 'subject_id')]),
            (recon_report, ds_recon_report, [('out_report', 'in_file')]),
            (inputnode, ds_recon_report, [('source_file', 'source_file')])
        ])

    return workflow


def _pop(inlist):
    if isinstance(inlist, (list, tuple)):
        return inlist[0]
    return inlist


def _probseg_fast2bids(inlist):
    """Reorder a list of probseg maps from FAST (CSF, WM, GM) to BIDS (GM, WM, CSF)."""
    return (inlist[1], inlist[2], inlist[0])


def _empty_report(in_file=None):
    from pathlib import Path
    from nipype.interfaces.base import isdefined
    if in_file is not None and isdefined(in_file):
        return in_file

    out_file = Path('tmp-report.html').absolute()
    out_file.write_text("""\
                <h4 class="elem-title">A previously computed T1w template was provided.</h4>
""")
    return str(out_file)


def _fmt_cohort(in_template):
    return in_template.replace(':', '_')


def _rpt_masks(mask_file, before, after, after_mask=None):
    from os.path import abspath
    import nibabel as nb
    msk = nb.load(mask_file).get_fdata() > 0
    bnii = nb.load(before)
    nb.Nifti1Image(bnii.get_fdata() * msk,
                   bnii.affine, bnii.header).to_filename('before.nii.gz')
    if after_mask is not None:
        msk = nb.load(after_mask).get_fdata() > 0

    anii = nb.load(after)
    nb.Nifti1Image(anii.get_fdata() * msk,
                   anii.affine, anii.header).to_filename('after.nii.gz')
    return abspath('before.nii.gz'), abspath('after.nii.gz')


def _no_atlas(spec):
    spec['atlas'] = None
    return spec
