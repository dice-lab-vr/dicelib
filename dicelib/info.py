# -*- coding: UTF-8 -*-

# Format version as expected by setup.py (string of form "X.Y.Z")
_version_major = 1
_version_minor = 2
_version_micro = 2
_version_extra = '' #'.dev'
__version__    = "%s.%s.%s%s" % (_version_major,_version_minor,_version_micro,_version_extra)

NAME                = 'dicelib'
DESCRIPTION         = 'Software library of the Diffusion Imaging and Connectivity Estimation (DICE) lab'
LONG_DESCRIPTION    = """
=========
 DICElib
=========

Software library of the Diffusion Imaging and Connectivity Estimation (DICE) lab.
"""
URL                 = "N/A"
DOWNLOAD_URL        = "N/A"
LICENSE             = "N/A"
AUTHOR              = "DICE lab"
AUTHOR_EMAIL        = "alessandro.daducci@univr.it"
PLATFORMS           = "OS independent"
MAJOR               = _version_major
MINOR               = _version_minor
MICRO               = _version_micro
VERSION             = __version__