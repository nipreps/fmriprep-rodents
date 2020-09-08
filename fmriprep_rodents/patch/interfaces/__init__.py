import logging
import os.path as op
from niworkflows.interfaces.bids import BIDSDataGrabber as _BIDSDataGrabber
from niworkflows.interfaces.mni import (
    _RobustMNINormalizationInputSpec as _NormInputSpec,
    RobustMNINormalization as _Norm,
)
from smriprep.interfaces.templateflow import (
    TemplateFlowSelect as _TFSelect,
    _TemplateFlowSelectOutputSpec as TFSelectOutputSpec,
    traits,
    File,
    isdefined,
)
from templateflow import api as tf

LOGGER = logging.getLogger("nipype.interface")


class BIDSDataGrabber(_BIDSDataGrabber):
    def _run_interface(self, runtime):
        bids_dict = self.inputs.subject_data

        self._results["out_dict"] = bids_dict
        self._results.update(bids_dict)
        if not bids_dict['t2w']:
            raise FileNotFoundError(
                "No T2w images found for subject sub-{}".format(self.inputs.subject_id)
            )
        if self._require_funcs and not bids_dict["bold"]:
            raise FileNotFoundError(
                "No functional images found for subject sub-{}".format(
                    self.inputs.subject_id
                )
            )
        for imtype in ["bold", "t1w", "flair", "fmap", "sbref", "roi"]:
            if not bids_dict[imtype]:
                LOGGER.info(
                    'No "%s" images found for sub-%s', imtype, self.inputs.subject_id
                )
        return runtime


class _TemplateFlowSelectOutputSpec(TFSelectOutputSpec):
    brain_mask = File(desc="Template's brain mask")
    t2w_file = File(desc="Template's T2w image")


class TemplateFlowSelect(_TFSelect):
    output_spec = _TemplateFlowSelectOutputSpec

    def _run_interface(self, runtime):
        specs = self.inputs.template_spec
        if isdefined(self.inputs.resolution):
            specs['resolution'] = self.inputs.resolution
        if isdefined(self.inputs.atlas):
            specs['atlas'] = self.inputs.atlas
        else:
            specs['atlas'] = None

        name = self.inputs.template.strip(":").split(":", 1)
        if len(name) > 1:
            specs.update({
                k: v for modifier in name[1].split(":")
                for k, v in [tuple(modifier.split("-"))]
                if k not in specs
            })

        self._results['brain_mask'] = tf.get(
            name[0], desc='brain', suffix='mask', **specs
        )
        self._results['t2w_file'] = tf.get(
            name[0], suffix='T2w', **specs
        )
        return runtime


class _RobustMNINormalizationInputSpec(_NormInputSpec):
    reference = traits.Enum(
        "T2w",
        "T1w",
        "boldref",
        "PDw",
        mandatory=True,
        usedefault=True,
        desc="set the reference modality for registration",
    )


class RobustMNINormalization(_Norm):
    input_spec = _RobustMNINormalizationInputSpec

    def _get_ants_args(self):
        from niworkflows.interfaces.mni import mask, create_cfm
        args = {
            "moving_image": self.inputs.moving_image,
            "num_threads": self.inputs.num_threads,
            "float": self.inputs.float,
            "terminal_output": "file",
            "write_composite_transform": True,
            "initial_moving_transform": self.inputs.initial_moving_transform,
        }

        """
        Moving image handling - The following truth table maps out the intended action
        sequence. Future refactoring may more directly encode this.
        moving_mask and lesion_mask are files
        True = file
        False = None
        | moving_mask | explicit_masking | lesion_mask | action
        |-------------|------------------|-------------|-------------------------------------------
        | True        | True             | True        | Update `moving_image` after applying
        |             |                  |             | mask.
        |             |                  |             | Set `moving_image_masks` applying
        |             |                  |             | `create_cfm` with `global_mask=True`.
        |-------------|------------------|-------------|-------------------------------------------
        | True        | True             | False       | Update `moving_image` after applying
        |             |                  |             | mask.
        |-------------|------------------|-------------|-------------------------------------------
        | True        | False            | True        | Set `moving_image_masks` applying
        |             |                  |             | `create_cfm` with `global_mask=False`
        |-------------|------------------|-------------|-------------------------------------------
        | True        | False            | False       | args['moving_image_masks'] = moving_mask
        |-------------|------------------|-------------|-------------------------------------------
        | False       | *                | True        | Set `moving_image_masks` applying
        |             |                  |             | `create_cfm` with `global_mask=True`
        |-------------|------------------|-------------|-------------------------------------------
        | False       | *                | False       | No action
        """
        # If a moving mask is provided...
        if isdefined(self.inputs.moving_mask):
            # If explicit masking is enabled...
            if self.inputs.explicit_masking:
                # Mask the moving image.
                # Do not use a moving mask during registration.
                args["moving_image"] = mask(
                    self.inputs.moving_image,
                    self.inputs.moving_mask,
                    "moving_masked.nii.gz",
                )

            # If explicit masking is disabled...
            else:
                # Use the moving mask during registration.
                # Do not mask the moving image.
                args["moving_image_masks"] = self.inputs.moving_mask

            # If a lesion mask is also provided...
            if isdefined(self.inputs.lesion_mask):
                # Create a cost function mask with the form:
                # [global mask - lesion mask] (if explicit masking is enabled)
                # [moving mask - lesion mask] (if explicit masking is disabled)
                # Use this as the moving mask.
                args["moving_image_masks"] = create_cfm(
                    self.inputs.moving_mask,
                    lesion_mask=self.inputs.lesion_mask,
                    global_mask=self.inputs.explicit_masking,
                )

        # If no moving mask is provided...
        # But a lesion mask *IS* provided...
        elif isdefined(self.inputs.lesion_mask):
            # Create a cost function mask with the form: [global mask - lesion mask]
            # Use this as the moving mask.
            args["moving_image_masks"] = create_cfm(
                self.inputs.moving_image,
                lesion_mask=self.inputs.lesion_mask,
                global_mask=True,
            )

        """
        Reference image handling - The following truth table maps out the intended action
        sequence. Future refactoring may more directly encode this.
        reference_mask and lesion_mask are files
        True = file
        False = None
        | reference_mask | explicit_masking | lesion_mask | action
        |----------------|------------------|-------------|----------------------------------------
        | True           | True             | True        | Update `fixed_image` after applying
        |                |                  |             | mask.
        |                |                  |             | Set `fixed_image_masks` applying
        |                |                  |             | `create_cfm` with `global_mask=True`.
        |----------------|------------------|-------------|----------------------------------------
        | True           | True             | False       | Update `fixed_image` after applying
        |                |                  |             | mask.
        |----------------|------------------|-------------|----------------------------------------
        | True           | False            | True        | Set `fixed_image_masks` applying
        |                |                  |             | `create_cfm` with `global_mask=False`
        |----------------|------------------|-------------|----------------------------------------
        | True           | False            | False       | args['fixed_image_masks'] = fixed_mask
        |----------------|------------------|-------------|----------------------------------------
        | False          | *                | True        | Set `fixed_image_masks` applying
        |                |                  |             | `create_cfm` with `global_mask=True`
        |----------------|------------------|-------------|----------------------------------------
        | False          | *                | False       | No action
        """
        # If a reference image is provided...
        if isdefined(self.inputs.reference_image):
            # Use the reference image as the fixed image.
            args["fixed_image"] = self.inputs.reference_image
            self._reference_image = self.inputs.reference_image

            # If a reference mask is provided...
            if isdefined(self.inputs.reference_mask):
                # If explicit masking is enabled...
                if self.inputs.explicit_masking:
                    # Mask the reference image.
                    # Do not use a fixed mask during registration.
                    args["fixed_image"] = mask(
                        self.inputs.reference_image,
                        self.inputs.reference_mask,
                        "fixed_masked.nii.gz",
                    )

                    # If a lesion mask is also provided...
                    if isdefined(self.inputs.lesion_mask):
                        # Create a cost function mask with the form: [global mask]
                        # Use this as the fixed mask.
                        args["fixed_image_masks"] = create_cfm(
                            self.inputs.reference_mask,
                            lesion_mask=None,
                            global_mask=True,
                        )

                # If a reference mask is provided...
                # But explicit masking is disabled...
                else:
                    # Use the reference mask as the fixed mask during registration.
                    # Do not mask the fixed image.
                    args["fixed_image_masks"] = self.inputs.reference_mask

            # If no reference mask is provided...
            # But a lesion mask *IS* provided ...
            elif isdefined(self.inputs.lesion_mask):
                # Create a cost function mask with the form: [global mask]
                # Use this as the fixed mask
                args["fixed_image_masks"] = create_cfm(
                    self.inputs.reference_image, lesion_mask=None, global_mask=True
                )

        # If no reference image is provided, fall back to the default template.
        else:
            from niworkflows.utils.misc import get_template_specs

            # Raise an error if the user specifies an unsupported image orientation.
            if self.inputs.orientation == "LAS":
                raise NotImplementedError

            template_spec = (
                self.inputs.template_spec
                if isdefined(self.inputs.template_spec)
                else {}
            )

            default_resolution = {"precise": 1, "fast": 2, "testing": 2}[
                self.inputs.flavor
            ]

            # Set the template resolution.
            if isdefined(self.inputs.template_resolution):
                template_spec["res"] = self.inputs.template_resolution

            template_spec["suffix"] = self.inputs.reference
            template_spec["desc"] = None

            # HACK: since Fischer has no resolutions
            if self.inputs.template == 'Fischer344':
                default_resolution = None

            ref_template, template_spec = get_template_specs(
                self.inputs.template,
                template_spec=template_spec,
                default_resolution=default_resolution,
            )

            template_spec['atlas'] = None
            template_spec['hemi'] = None

            # Set reference image
            self._reference_image = ref_template
            if not op.isfile(self._reference_image):
                raise ValueError(
                    """\
The registration reference must be an existing file, but path "%s" \
cannot be found."""
                    % ref_template
                )

            # Get the template specified by the user.
            ref_mask = tf.get(
                self.inputs.template, desc="brain", suffix="mask", **template_spec
            )

            # Default is explicit masking disabled
            args["fixed_image"] = ref_template
            # Use the template mask as the fixed mask.
            args["fixed_image_masks"] = str(ref_mask)

            # Overwrite defaults if explicit masking
            if self.inputs.explicit_masking:
                # Mask the template image with the template mask.
                args["fixed_image"] = mask(
                    ref_template, str(ref_mask), "fixed_masked.nii.gz"
                )
                # Do not use a fixed mask during registration.
                args.pop("fixed_image_masks", None)

                # If a lesion mask is provided...
                if isdefined(self.inputs.lesion_mask):
                    # Create a cost function mask with the form: [global mask]
                    # Use this as the fixed mask.
                    args["fixed_image_masks"] = create_cfm(
                        str(ref_mask), lesion_mask=None, global_mask=True
                    )

        return args
