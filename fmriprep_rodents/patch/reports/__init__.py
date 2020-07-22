from pathlib import Path
from niworkflows.reports.core import Report as _Report


class Report(_Report):
    def _load_config(self, config):
        from yaml import safe_load as load

        settings = load(config.read_text())
        self.packagename = self.packagename or settings.get("package", None)

        if self.packagename is not None:
            self.root = self.root / self.packagename.replace('_', '-')
            self.out_dir = self.out_dir / self.packagename.replace('_', '-')

        if self.subject_id is not None:
            self.root = self.root / "sub-{}".format(self.subject_id)

        if "template_path" in settings:
            self.template_path = config.parent / settings["template_path"]

        self.index(settings["sections"])


def generate_reports(
    subject_list, output_dir, run_uuid, config=None, work_dir=None, packagename=None
):
    """Execute run_reports on a list of subjects."""
    reportlets_dir = None
    if work_dir is not None:
        reportlets_dir = Path(work_dir) / "reportlets"
    report_errors = [
        Report(
            output_dir,
            subject_label,
            run_uuid,
            config=config,
            packagename=packagename,
            reportlets_dir=reportlets_dir,
        ).generate_report()
        for subject_label in subject_list
    ]

    errno = sum(report_errors)
    if errno:
        import logging

        logger = logging.getLogger("cli")
        error_list = ", ".join(
            "%s (%d)" % (subid, err)
            for subid, err in zip(subject_list, report_errors)
            if err
        )
        logger.error(
            "Preprocessing did not finish successfully. Errors occurred while processing "
            "data from participants: %s. Check the HTML reports for details.",
            error_list,
        )
    return errno
