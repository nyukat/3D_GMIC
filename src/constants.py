"""
Defines constants used in src.
"""

class VIEWS:
    L_CC = "L-CC"
    R_CC = "R-CC"
    L_MLO = "L-MLO"
    R_MLO = "R-MLO"

    LIST = [L_CC, R_CC, L_MLO, R_MLO]

    @classmethod
    def is_cc(cls, view):
        return view in (cls.L_CC, cls.R_CC)

    @classmethod
    def is_mlo(cls, view):
        return view in (cls.L_MLO, cls.R_MLO)

    @classmethod
    def is_left(cls, view):
        return view in (cls.L_CC, cls.L_MLO)

    @classmethod
    def is_right(cls, view):
        return view in (cls.R_CC, cls.R_MLO)


INPUT_SIZE_DICT = {
    VIEWS.L_CC: (2116, 1339),
    VIEWS.R_CC: (2116, 1339),
    VIEWS.L_MLO: (2116, 1339),
    VIEWS.R_MLO: (2116, 1339),
}

PERCENT_T_DICT = {
    "1": 1.5108578607685,
    "2": 2.0584660301930686,
    "3": 1.1336909878403076,
    "4": 1.5651680987233705,
    "5": 2.293890202354881
}

TOP_K_DICT = {
    "1": 12,
    "2": 12,
    "3": 8,
    "4": 16,
    "5": 8
}