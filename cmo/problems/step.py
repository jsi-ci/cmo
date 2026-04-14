import os

from cmo.problems.cdtlz import C2DTLZ2, C3DTLZ4, C1DTLZ1
from cmo.problems.cf import CF4, CF8
from cmo.problems.dascmop import DASCMOP9, DASCMOP8
from cmo.problems.dcdtlz import DC2DTLZ1, DC2DTLZ3, DC1DTLZ3
from cmo.problems.mw import MW8, MW11, MW7
from cmo.problems.rcm import RCM2
from cmo.problems.utils import load_pareto_front_from_file
from cmo.problems.zxhcf import ZXHCF16


class STEP1(DC2DTLZ3):
    def __init__(self):
        super(STEP1, self).__init__(n_var=2, n_obj=2)
        self.name = self.__class__.__name__

    def _calc_pareto_front(self, *args, **kwargs):
        fname = f"{self.name.lower()}_M{self.n_obj}_D{self.n_var}.pf"
        return load_pareto_front_from_file(os.path.join("STEP", fname))


class STEP2(MW7):
    def __init__(self):
        super(STEP2, self).__init__(n_var=2, n_obj=2)
        self.name = self.__class__.__name__

    def _calc_pareto_front(self, *args, **kwargs):
        fname = f"{self.name.lower()}_M{self.n_obj}_D{self.n_var}.pf"
        return load_pareto_front_from_file(os.path.join("STEP", fname))


class STEP3(MW11):
    def __init__(self):
        super(STEP3, self).__init__(n_var=2, n_obj=2)
        self.name = self.__class__.__name__

    def _calc_pareto_front(self, *args, **kwargs):
        fname = f"{self.name.lower()}_M{self.n_obj}_D{self.n_var}.pf"
        return load_pareto_front_from_file(os.path.join("STEP", fname))


class STEP4(MW11):
    def __init__(self):
        super(STEP4, self).__init__(n_var=3, n_obj=2)
        self.name = self.__class__.__name__

    def _calc_pareto_front(self, *args, **kwargs):
        fname = f"{self.name.lower()}_M{self.n_obj}_D{self.n_var}.pf"
        return load_pareto_front_from_file(os.path.join("STEP", fname))


class STEP5(DC2DTLZ1):
    def __init__(self):
        super(STEP5, self).__init__(n_var=3, n_obj=2)
        self.name = self.__class__.__name__

    def _calc_pareto_front(self, *args, **kwargs):
        fname = f"{self.name.lower()}_M{self.n_obj}_D{self.n_var}.pf"
        return load_pareto_front_from_file(os.path.join("STEP", fname))


class STEP6(CF4):
    def __init__(self):
        super(STEP6, self).__init__(n_var=3, n_obj=2)
        self.name = self.__class__.__name__

    def _calc_pareto_front(self, *args, **kwargs):
        fname = f"{self.name.lower()}_M{self.n_obj}_D{self.n_var}.pf"
        return load_pareto_front_from_file(os.path.join("STEP", fname))


class STEP7(C2DTLZ2):
    def __init__(self):
        super(STEP7, self).__init__(n_var=5, n_obj=2)
        self.name = self.__class__.__name__

    def _calc_pareto_front(self, *args, **kwargs):
        fname = f"{self.name.lower()}_M{self.n_obj}_D{self.n_var}.pf"
        return load_pareto_front_from_file(os.path.join("STEP", fname))


class STEP8(RCM2):
    def __init__(self):
        super(STEP8, self).__init__(n_var=5, n_obj=2)
        self.name = self.__class__.__name__

    def _calc_pareto_front(self, *args, **kwargs):
        fname = f"{self.name.lower()}_M{self.n_obj}_D{self.n_var}.pf"
        return load_pareto_front_from_file(os.path.join("STEP", fname))


class STEP9(DC2DTLZ1):
    def __init__(self):
        super(STEP9, self).__init__(n_var=5, n_obj=2)
        self.name = self.__class__.__name__

    def _calc_pareto_front(self, *args, **kwargs):
        fname = f"{self.name.lower()}_M{self.n_obj}_D{self.n_var}.pf"
        return load_pareto_front_from_file(os.path.join("STEP", fname))


class STEP10(C2DTLZ2):
    def __init__(self):
        super(STEP10, self).__init__(n_var=10, n_obj=2)
        self.name = self.__class__.__name__

    def _calc_pareto_front(self, *args, **kwargs):
        fname = f"{self.name.lower()}_M{self.n_obj}_D{self.n_var}.pf"
        return load_pareto_front_from_file(os.path.join("STEP", fname))


class STEP11(CF4):
    def __init__(self):
        super(STEP11, self).__init__(n_var=10, n_obj=2)
        self.name = self.__class__.__name__

    def _calc_pareto_front(self, *args, **kwargs):
        fname = f"{self.name.lower()}_M{self.n_obj}_D{self.n_var}.pf"
        return load_pareto_front_from_file(os.path.join("STEP", fname))


class STEP12(MW8):
    def __init__(self):
        super(STEP12, self).__init__(n_var=10, n_obj=2)
        self.name = self.__class__.__name__

    def _calc_pareto_front(self, *args, **kwargs):
        fname = f"{self.name.lower()}_M{self.n_obj}_D{self.n_var}.pf"
        return load_pareto_front_from_file(os.path.join("STEP", fname))


class STEP13(DC2DTLZ1):
    def __init__(self):
        super(STEP13, self).__init__(n_var=30, n_obj=2)
        self.name = self.__class__.__name__

    def _calc_pareto_front(self, *args, **kwargs):
        fname = f"{self.name.lower()}_M{self.n_obj}_D{self.n_var}.pf"
        return load_pareto_front_from_file(os.path.join("STEP", fname))


class STEP14(ZXHCF16):
    def __init__(self):
        super(STEP14, self).__init__(n_var=30, n_obj=2)
        self.name = self.__class__.__name__

    def _calc_pareto_front(self, *args, **kwargs):
        fname = f"{self.name.lower()}_M{self.n_obj}_D{self.n_var}.pf"
        return load_pareto_front_from_file(os.path.join("STEP", fname))


class STEP15(MW11):
    def __init__(self):
        super(STEP15, self).__init__(n_var=30, n_obj=2)
        self.name = self.__class__.__name__

    def _calc_pareto_front(self, *args, **kwargs):
        fname = f"{self.name.lower()}_M{self.n_obj}_D{self.n_var}.pf"
        return load_pareto_front_from_file(os.path.join("STEP", fname))


class STEP16(DC2DTLZ3):
    def __init__(self):
        super(STEP16, self).__init__(n_var=2, n_obj=3)
        self.name = self.__class__.__name__

    def _calc_pareto_front(self, *args, **kwargs):
        fname = f"{self.name.lower()}_M{self.n_obj}_D{self.n_var}.pf"
        return load_pareto_front_from_file(os.path.join("STEP", fname))


class STEP17(DASCMOP8):
    def __init__(self):
        super(STEP17, self).__init__(n_var=2, n_obj=3)
        self.name = self.__class__.__name__

    def _calc_pareto_front(self, *args, **kwargs):
        fname = f"{self.name.lower()}_M{self.n_obj}_D{self.n_var}.pf"
        return load_pareto_front_from_file(os.path.join("STEP", fname))


class STEP18(DC1DTLZ3):
    def __init__(self):
        super(STEP18, self).__init__(n_var=2, n_obj=3)
        self.name = self.__class__.__name__

    def _calc_pareto_front(self, *args, **kwargs):
        fname = f"{self.name.lower()}_M{self.n_obj}_D{self.n_var}.pf"
        return load_pareto_front_from_file(os.path.join("STEP", fname))


class STEP19(DC2DTLZ3):
    def __init__(self):
        super(STEP19, self).__init__(n_var=3, n_obj=3)
        self.name = self.__class__.__name__

    def _calc_pareto_front(self, *args, **kwargs):
        fname = f"{self.name.lower()}_M{self.n_obj}_D{self.n_var}.pf"
        return load_pareto_front_from_file(os.path.join("STEP", fname))


class STEP20(C3DTLZ4):
    def __init__(self):
        super(STEP20, self).__init__(n_var=3, n_obj=3)
        self.name = self.__class__.__name__

    def _calc_pareto_front(self, *args, **kwargs):
        fname = f"{self.name.lower()}_M{self.n_obj}_D{self.n_var}.pf"
        return load_pareto_front_from_file(os.path.join("STEP", fname))


class STEP21(DASCMOP8):
    def __init__(self):
        super(STEP21, self).__init__(n_var=3, n_obj=3)
        self.name = self.__class__.__name__

    def _calc_pareto_front(self, *args, **kwargs):
        fname = f"{self.name.lower()}_M{self.n_obj}_D{self.n_var}.pf"
        return load_pareto_front_from_file(os.path.join("STEP", fname))


class STEP22(DC2DTLZ3):
    def __init__(self):
        super(STEP22, self).__init__(n_var=5, n_obj=3)
        self.name = self.__class__.__name__

    def _calc_pareto_front(self, *args, **kwargs):
        fname = f"{self.name.lower()}_M{self.n_obj}_D{self.n_var}.pf"
        return load_pareto_front_from_file(os.path.join("STEP", fname))


class STEP23(C3DTLZ4):
    def __init__(self):
        super(STEP23, self).__init__(n_var=5, n_obj=3)
        self.name = self.__class__.__name__

    def _calc_pareto_front(self, *args, **kwargs):
        fname = f"{self.name.lower()}_M{self.n_obj}_D{self.n_var}.pf"
        return load_pareto_front_from_file(os.path.join("STEP", fname))


class STEP24(C1DTLZ1):
    def __init__(self):
        super(STEP24, self).__init__(n_var=5, n_obj=3)
        self.name = self.__class__.__name__

    def _calc_pareto_front(self, *args, **kwargs):
        fname = f"{self.name.lower()}_M{self.n_obj}_D{self.n_var}.pf"
        return load_pareto_front_from_file(os.path.join("STEP", fname))


class STEP25(DC2DTLZ1):
    def __init__(self):
        super(STEP25, self).__init__(n_var=10, n_obj=3)
        self.name = self.__class__.__name__

    def _calc_pareto_front(self, *args, **kwargs):
        fname = f"{self.name.lower()}_M{self.n_obj}_D{self.n_var}.pf"
        return load_pareto_front_from_file(os.path.join("STEP", fname))


class STEP26(MW8):
    def __init__(self):
        super(STEP26, self).__init__(n_var=10, n_obj=3)
        self.name = self.__class__.__name__

    def _calc_pareto_front(self, *args, **kwargs):
        fname = f"{self.name.lower()}_M{self.n_obj}_D{self.n_var}.pf"
        return load_pareto_front_from_file(os.path.join("STEP", fname))


class STEP27(C3DTLZ4):
    def __init__(self):
        super(STEP27, self).__init__(n_var=10, n_obj=3)
        self.name = self.__class__.__name__

    def _calc_pareto_front(self, *args, **kwargs):
        fname = f"{self.name.lower()}_M{self.n_obj}_D{self.n_var}.pf"
        return load_pareto_front_from_file(os.path.join("STEP", fname))


class STEP28(DC2DTLZ3):
    def __init__(self):
        super(STEP28, self).__init__(n_var=30, n_obj=3)
        self.name = self.__class__.__name__

    def _calc_pareto_front(self, *args, **kwargs):
        fname = f"{self.name.lower()}_M{self.n_obj}_D{self.n_var}.pf"
        return load_pareto_front_from_file(os.path.join("STEP", fname))


class STEP29(DASCMOP9):
    def __init__(self):
        super(STEP29, self).__init__(n_var=30, n_obj=3)
        self.name = self.__class__.__name__

    def _calc_pareto_front(self, *args, **kwargs):
        fname = f"{self.name.lower()}_M{self.n_obj}_D{self.n_var}.pf"
        return load_pareto_front_from_file(os.path.join("STEP", fname))


class STEP30(CF8):
    def __init__(self):
        super(STEP30, self).__init__(n_var=30, n_obj=3)
        self.name = self.__class__.__name__

    def _calc_pareto_front(self, *args, **kwargs):
        fname = f"{self.name.lower()}_M{self.n_obj}_D{self.n_var}.pf"
        return load_pareto_front_from_file(os.path.join("STEP", fname))
