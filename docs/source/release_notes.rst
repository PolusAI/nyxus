.. _release-notes:

Release notes
=============

Each Nyxus release is tagged with its version, and its notes are published on the
`GitHub releases page <https://github.com/PolusAI/nyxus/releases>`__. These pages describe the
development head of the repository; facts about published packages (wheels, conda-forge builds
and Docker images) describe what is published now.

The repository is set up to make releases with release-please from the commit history; no release
has been made that way yet, and the notes of 0.11.0 and earlier are GitHub's generated lists.
Commit subjects follow the Conventional Commits format, and release-please groups them by type:
features, bug fixes, performance improvements, reverts, documentation, refactoring and build
changes. Chores, style, test and CI commits are left out of the notes. The commit conventions are
described in ``CONTRIBUTING.md`` and in the :ref:`developer guide <devguide-release>`.
