from importlib.metadata import version

import kaprag


def test_package_version_matches_metadata():
    assert kaprag.__version__ == version("kaprag")
