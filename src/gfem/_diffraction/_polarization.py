"""The polarization names and the Jones ordering shared by the diffraction readers."""

#: Names a reader accepts for the incident polarization (besides a Jones pair).
POLARIZATIONS = ("TE", "TM", "RCP", "LCP", "unpolarized")
#: The order of the two linear polarizations along a Jones axis: TM, then TE.
JONES_ORDER = ("TM", "TE")
