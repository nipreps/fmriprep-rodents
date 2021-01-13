"""Check the configuration module and file."""
from pathlib import Path
from pkg_resources import resource_filename as pkgrf
from toml import loads
from niworkflows.utils.spaces import format_reference
import pytest

from .. import config


@pytest.mark.skip(reason="Need to update config.toml")
def test_config_spaces():
    """Check that all necessary spaces are recorded in the config."""
    filename = Path(pkgrf("fmriprep_rodents", "data/tests/config.toml"))
    settings = loads(filename.read_text())
    for sectionname, configs in settings.items():
        if sectionname != "environment":
            section = getattr(config, sectionname)
            section.load(configs, init=False)
    config.nipype.init()
    config.loggers.init()
    config.init_spaces()

    spaces = config.workflow.spaces
    assert "Fischer344:res-native" not in [
        str(s) for s in spaces.get_standard(full_spec=True)
    ]

    config.init_spaces()
    spaces = config.workflow.spaces

    assert "Fischer344:res-native" in [
        str(s) for s in spaces.get_standard(full_spec=True)
    ]

    config.execution.output_spaces = None
    config.workflow.use_aroma = False
    config.init_spaces()
    spaces = config.workflow.spaces

    assert not [str(s) for s in spaces.get_standard(full_spec=True)]

    assert [
        format_reference((s.fullname, s.spec))
        for s in spaces.references
        if s.standard and s.dim == 3
    ] == ["Fischer344"]
