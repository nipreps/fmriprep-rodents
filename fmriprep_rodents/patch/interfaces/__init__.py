import logging
from niworkflows.interfaces.bids import BIDSDataGrabber as _BIDSDataGrabber

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
