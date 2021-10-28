def fix_multi_source_name(in_files, modality="T2w"):
    """
    Make up a generic source name when there are multiple
    >>> fix_multi_source_name([
    ...     '/path/to/sub-045_ses-test_T2w.nii.gz',
    ...     '/path/to/sub-045_ses-retest_T2w.nii.gz'])
    '/path/to/sub-045_T2w.nii.gz'
    """
    import os
    from nipype.utils.filemanip import filename_to_list

    base, in_file = os.path.split(filename_to_list(in_files)[0])
    subject_label = in_file.split("_", 1)[0].split("-")[1]
    return os.path.join(base, f"sub-{subject_label}_{modality}.nii.gz")


def get_template_specs(in_template, template_spec=None, default_resolution=None):
    """
    Parse template specifications
    >>> get_template_specs('MNI152NLin2009cAsym', {'suffix': 'T1w'})[1]
    {'resolution': 1}
    >>> get_template_specs('MNI152NLin2009cAsym', {'res': '2', 'suffix': 'T1w'})[1]
    {'resolution': '2'}
    >>> specs = get_template_specs('MNIInfant', {'res': '2', 'cohort': '10', 'suffix': 'T1w'})[1]
    >>> sorted(specs.items())
    [('cohort', '10'), ('resolution', '2')]
    >>> get_template_specs('MNI152NLin2009cAsym',
    ...                    {'suffix': 'T1w', 'cohort': 1})[1] # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    RuntimeError:
    ...
    >>> get_template_specs('MNI152NLin2009cAsym',
    ...                    {'suffix': 'T1w', 'res': '1|2'})[1] # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    RuntimeError:
    ...
    """
    from templateflow.api import get as get_template

    # Massage spec (start creating if None)
    template_spec = template_spec or {}
    template_spec["desc"] = template_spec.get("desc", None)
    template_spec["atlas"] = template_spec.get("atlas", None)
    template_spec["resolution"] = template_spec.pop(
        "res", template_spec.get("resolution", default_resolution)
    )

    common_spec = {"resolution": template_spec["resolution"]}
    if "cohort" in template_spec:
        common_spec["cohort"] = template_spec["cohort"]

    tpl_target_path = get_template(in_template, raise_empty=True, **template_spec)

    if isinstance(tpl_target_path, list):
        raise RuntimeError(
            """\
The available template modifiers ({0}) did not select a unique template \
(got "{1}"). Please revise your template argument.""".format(
                template_spec, ", ".join([str(p) for p in tpl_target_path])
            )
        )

    return str(tpl_target_path), common_spec


def extract_entities(file_list):
    """
    Return a dictionary of common entities given a list of files.

    Examples
    --------
    >>> extract_entities('sub-01/anat/sub-01_T1w.nii.gz')
    {'subject': '01', 'suffix': 'T1w', 'datatype': 'anat', 'extension': 'nii.gz'}
    >>> extract_entities(['sub-01/anat/sub-01_T1w.nii.gz'] * 2)
    {'subject': '01', 'suffix': 'T1w', 'datatype': 'anat', 'extension': 'nii.gz'}
    >>> extract_entities(['sub-01/anat/sub-01_run-1_T1w.nii.gz',
    ...                   'sub-01/anat/sub-01_run-2_T1w.nii.gz'])
    {'subject': '01', 'run': [1, 2], 'suffix': 'T1w', 'datatype': 'anat',
     'extension': 'nii.gz'}

    """
    from collections import defaultdict
    from bids.layout import parse_file_entities
    from niworkflows.utils.connections import listify

    entities = defaultdict(list)
    for e, v in [
        ev_pair
        for f in listify(file_list)
        for ev_pair in parse_file_entities(f).items()
    ]:
        entities[e].append(v)

    def _unique(inlist):
        inlist = sorted(set(inlist))
        if len(inlist) == 1:
            return inlist[0]
        return inlist

    return {k: _unique(v) for k, v in entities.items()}
