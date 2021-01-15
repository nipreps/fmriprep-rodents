*fMRIPrep/rodents*: Analysis-grade BOLD fMRI data of rats
=========================================================
This tool adapts *fMRIPrep* to the use-case of rodent preclinical imaging -- 
because MRI is some times done on species other than the *homo sapiens*.
At the moment, only rats are fully-supported.
However, the vision is to generalize the preprocessing to other rodents,
starting with mice.

.. image:: https://circleci.com/gh/poldracklab/fmriprep-rodents/tree/master.svg?style=shield
  :target: https://circleci.com/gh/poldracklab/fmriprep-rodents/tree/master

.. image:: https://github.com/poldracklab/fmriprep-rodents/workflows/Python%20package/badge.svg
  :target: https://github.com/poldracklab/fmriprep-rodents/actions

About
-----
Development was initiated by (soon-to-be) Dr. E. MacNicol in a research visit to the 
`Poldrack lab at Stanford University <https://poldracklab.stanford.edu/>`__.
Currently, it is maintained by the NiPreps community.

.. image:: https://github.com/oesteban/fmriprep/raw/38a63e9504ab67812b63813c5fe9af882109408e/docs/_static/fmriprep-workflow-all.png

*fMRIPrep* is a functional magnetic resonance imaging (fMRI) data
preprocessing pipeline.
*fMRIPrep/rodents* adapts the original pipeline to work on rodents.
The software is designed to provide an easily accessible interface,
and the pipeline is robust to variations in scan acquisition
protocols.
This is possible with the adoption of `BIDS (Brain Imaging Data Structure)
<https://bids-specification.readthedocs.io/>`__, which allows the tool to
implement such a design.
In practice, *fMRIPrep* (and *fMRIPrep/rodents*) require minimal user input and
provide interpretable, comprehensive visual reports.
*fMRIPrep/rodents* performs basic processing steps (coregistration, normalization, 
unwarping, noise component extraction, segmentation, skullstripping etc.) providing
outputs that can be easily submitted to a variety of group level analyses,
including task-based or resting-state fMRI, graph theory measures, surface or
volume-based statistics, etc.

.. note::

   *fMRIPrep* performs minimal preprocessing.
   Here we define 'minimal preprocessing'  as motion correction, field
   unwarping, normalization, bias field correction, and brain extraction.
   See the `workflows section of our documentation
   <https://fmriprep.readthedocs.io/en/latest/workflows.html>`__ for more details.

The *fMRIPrep/rodents* pipeline uses a combination of tools from well-known neuroimaging
packages, including FSL_, ANTs_, and AFNI_.

Principles
----------
*fMRIPrep* is built around three principles:

1. **Robustness** - The pipeline adapts the preprocessing steps depending on
   the input dataset and should provide results as good as possible
   independently of scanner make, scanning parameters or presence of additional
   correction scans (such as fieldmaps).
2. **Ease of use** - Thanks to dependence on the BIDS standard, manual
   parameter input is reduced to a minimum, allowing the pipeline to run in an
   automatic fashion.
3. **"Glass box"** philosophy - Automation should not mean that one should not
   visually inspect the results or understand the methods.
   Thus, *fMRIPrep* provides visual reports for each subject, detailing the
   accuracy of the most important processing steps.
   This, combined with the documentation, can help researchers to understand
   the process and decide which subjects should be kept for the group level
   analysis.

Limitations
-----------
We count as limitations `those inherited from the upstream project, *fMRIPrep*
<https://fmriprep.org/en/stable/#limitations-and-reasons-not-to-use-fmriprep>`__,
in addition to:

1. Mice are not yet supported, although the infrastructure is all set for quickly
   extending the support.
   In particular, the processing of mice imaging will require the inclusion of a
   suitable mice template on the `TemplateFlow <https://www.templateflow.org>`__ Archive.

Acknowledgements
----------------
Please acknowledge this work using the citation boilerplate that *fMRIPrep* includes
in the visual report generated for every subject processed.
