"""Journal exception taxonomy."""

from __future__ import annotations


class JournalError(Exception):
    """Base class for journal failures."""


class JournalCorruptError(JournalError):
    """A segment file contains a record that could not be parsed.

    Per STRATEGY R3 the kernel must not mutate corrupt journal segments on
    its own. Reporting and quarantine are reserved to explicit recovery
    commands.
    """


class JournalReplayError(JournalError):
    """Replay could not reconstruct a coherent intent record from the
    journal — e.g. the seq counter goes backwards across segments."""


class JournalLocked(JournalError):
    """Another writer already holds the journal's exclusive flock.

    Two writers on the same journal would each maintain an independent seq
    counter and produce duplicate seqs — guaranteed corruption — so the
    second opener fails fast.
    """
