def fix_multi_source_name(in_files, modality='T2w'):
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
