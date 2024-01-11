#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import git


"""utils.py: Utility functions."""

__author__  = "Vratislav Besta"
__group__   = "50"
__version__ = "1.0.1"
__email__   = "bestavra@fel.cvut.cz"
__date__    = "2024/01/10" 


def find_git_root_folder():
    """
    Finds the root folder of the git repository to automatically set the paths to the data folders.
    """
    try:
        repo = git.Repo(search_parent_directories=True)
        git_root = repo.git.rev_parse("--show-toplevel")
        return git_root
    except git.InvalidGitRepositoryError:
        return None