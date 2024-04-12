import os

from cmo.problems.cdtlz import C1DTLZ3, C2DTLZ2, C3DTLZ4, C3DTLZ1
from cmo.problems.ctp import CTP1
from cmo.problems.dascmop import DASCMOP9, DASCMOP7, DASCMOP6
from cmo.problems.dcdtlz import DC3DTLZ3, DC2DTLZ1, DC2DTLZ3
from cmo.problems.mw import MW10, MW8
from cmo.problems.nctp import NCTP14, NCTP4
from cmo.problems.rcm import RCM16
from cmo.problems.zxhcf import ZXHCF10, ZXHCF3
from cmo.problems.utils import load_pareto_front_from_file


class STEP1(RCM16):
    def __init__(self):
        super(STEP1, self).__init__(n_var=2, n_obj=2)
        self.name = self.__class__.__name__

    def _calc_pareto_front(self, *args, **kwargs):
        fname = f"{self.name.lower()}_M{self.n_obj}_D{self.n_var}.pf"
        return load_pareto_front_from_file(os.path.join("STEP", fname))


class STEP2(C1DTLZ3):
    def __init__(self):
        super(STEP2, self).__init__(n_var=2, n_obj=2)
        self.name = self.__class__.__name__

    def _calc_pareto_front(self, *args, **kwargs):
        fname = f"{self.name.lower()}_M{self.n_obj}_D{self.n_var}.pf"
        return load_pareto_front_from_file(os.path.join("STEP", fname))


class STEP3(DC3DTLZ3):
    def __init__(self):
        super(STEP3, self).__init__(n_var=2, n_obj=2)
        self.name = self.__class__.__name__

    def _calc_pareto_front(self, *args, **kwargs):
        fname = f"{self.name.lower()}_M{self.n_obj}_D{self.n_var}.pf"
        return load_pareto_front_from_file(os.path.join("STEP", fname))


class STEP4(C3DTLZ4):
    def __init__(self):
        super(STEP4, self).__init__(n_var=3, n_obj=2)
        self.name = self.__class__.__name__

    def _calc_pareto_front(self, *args, **kwargs):
        fname = f"{self.name.lower()}_M{self.n_obj}_D{self.n_var}.pf"
        return load_pareto_front_from_file(os.path.join("STEP", fname))


class STEP5(C3DTLZ1):
    def __init__(self):
        super(STEP5, self).__init__(n_var=3, n_obj=2)
        self.name = self.__class__.__name__

    def _calc_pareto_front(self, *args, **kwargs):
        fname = f"{self.name.lower()}_M{self.n_obj}_D{self.n_var}.pf"
        return load_pareto_front_from_file(os.path.join("STEP", fname))


class STEP6(DC2DTLZ3):
    def __init__(self):
        super(STEP6, self).__init__(n_var=3, n_obj=2)
        self.name = self.__class__.__name__

    def _calc_pareto_front(self, *args, **kwargs):
        fname = f"{self.name.lower()}_M{self.n_obj}_D{self.n_var}.pf"
        return load_pareto_front_from_file(os.path.join("STEP", fname))


class STEP7(NCTP14):
    def __init__(self):
        super(STEP7, self).__init__(n_var=5, n_obj=2)
        self.name = self.__class__.__name__

    def _calc_pareto_front(self, *args, **kwargs):
        fname = f"{self.name.lower()}_M{self.n_obj}_D{self.n_var}.pf"
        return load_pareto_front_from_file(os.path.join("STEP", fname))


class STEP8(DC2DTLZ3):
    def __init__(self):
        super(STEP8, self).__init__(n_var=5, n_obj=2)
        self.name = self.__class__.__name__

    def _calc_pareto_front(self, *args, **kwargs):
        fname = f"{self.name.lower()}_M{self.n_obj}_D{self.n_var}.pf"
        return load_pareto_front_from_file(os.path.join("STEP", fname))


class STEP9(CTP1):
    def __init__(self):
        super(STEP9, self).__init__(n_var=5, n_obj=2)
        self.name = self.__class__.__name__

    def _calc_pareto_front(self, *args, **kwargs):
        fname = f"{self.name.lower()}_M{self.n_obj}_D{self.n_var}.pf"
        return load_pareto_front_from_file(os.path.join("STEP", fname))


class STEP10(DASCMOP6):
    def __init__(self):
        super(STEP10, self).__init__(n_var=10, n_obj=2)
        self.name = self.__class__.__name__

    def _calc_pareto_front(self, *args, **kwargs):
        fname = f"{self.name.lower()}_M{self.n_obj}_D{self.n_var}.pf"
        return load_pareto_front_from_file(os.path.join("STEP", fname))


class STEP11(DC2DTLZ3):
    def __init__(self):
        super(STEP11, self).__init__(n_var=10, n_obj=2)
        self.name = self.__class__.__name__

    def _calc_pareto_front(self, *args, **kwargs):
        fname = f"{self.name.lower()}_M{self.n_obj}_D{self.n_var}.pf"
        return load_pareto_front_from_file(os.path.join("STEP", fname))


class STEP12(CTP1):
    def __init__(self):
        super(STEP12, self).__init__(n_var=10, n_obj=2)
        self.name = self.__class__.__name__

    def _calc_pareto_front(self, *args, **kwargs):
        fname = f"{self.name.lower()}_M{self.n_obj}_D{self.n_var}.pf"
        return load_pareto_front_from_file(os.path.join("STEP", fname))


class STEP13(DC2DTLZ3):
    def __init__(self):
        super(STEP13, self).__init__(n_var=30, n_obj=2)
        self.name = self.__class__.__name__

    def _calc_pareto_front(self, *args, **kwargs):
        fname = f"{self.name.lower()}_M{self.n_obj}_D{self.n_var}.pf"
        return load_pareto_front_from_file(os.path.join("STEP", fname))


class STEP14(MW10):
    def __init__(self):
        super(STEP14, self).__init__(n_var=30, n_obj=2)
        self.name = self.__class__.__name__

    def _calc_pareto_front(self, *args, **kwargs):
        fname = f"{self.name.lower()}_M{self.n_obj}_D{self.n_var}.pf"
        return load_pareto_front_from_file(os.path.join("STEP", fname))


class STEP15(NCTP4):
    def __init__(self):
        super(STEP15, self).__init__(n_var=30, n_obj=2)
        self.name = self.__class__.__name__

    def _calc_pareto_front(self, *args, **kwargs):
        fname = f"{self.name.lower()}_M{self.n_obj}_D{self.n_var}.pf"
        return load_pareto_front_from_file(os.path.join("STEP", fname))


class STEP16(DASCMOP9):
    def __init__(self):
        super(STEP16, self).__init__(n_var=2, n_obj=3)
        self.name = self.__class__.__name__

    def _calc_pareto_front(self, *args, **kwargs):
        fname = f"{self.name.lower()}_M{self.n_obj}_D{self.n_var}.pf"
        return load_pareto_front_from_file(os.path.join("STEP", fname))


class STEP17(DC2DTLZ3):
    def __init__(self):
        super(STEP17, self).__init__(n_var=2, n_obj=3)
        self.name = self.__class__.__name__

    def _calc_pareto_front(self, *args, **kwargs):
        fname = f"{self.name.lower()}_M{self.n_obj}_D{self.n_var}.pf"
        return load_pareto_front_from_file(os.path.join("STEP", fname))


class STEP18(MW8):
    def __init__(self):
        super(STEP18, self).__init__(n_var=2, n_obj=3)
        self.name = self.__class__.__name__

    def _calc_pareto_front(self, *args, **kwargs):
        fname = f"{self.name.lower()}_M{self.n_obj}_D{self.n_var}.pf"
        return load_pareto_front_from_file(os.path.join("STEP", fname))


class STEP19(C2DTLZ2):
    def __init__(self):
        super(STEP19, self).__init__(n_var=3, n_obj=3)
        self.name = self.__class__.__name__

    def _calc_pareto_front(self, *args, **kwargs):
        fname = f"{self.name.lower()}_M{self.n_obj}_D{self.n_var}.pf"
        return load_pareto_front_from_file(os.path.join("STEP", fname))


class STEP20(C1DTLZ3):
    def __init__(self):
        super(STEP20, self).__init__(n_var=3, n_obj=3)
        self.name = self.__class__.__name__

    def _calc_pareto_front(self, *args, **kwargs):
        fname = f"{self.name.lower()}_M{self.n_obj}_D{self.n_var}.pf"
        return load_pareto_front_from_file(os.path.join("STEP", fname))


class STEP21(C3DTLZ4):
    def __init__(self):
        super(STEP21, self).__init__(n_var=3, n_obj=3)
        self.name = self.__class__.__name__

    def _calc_pareto_front(self, *args, **kwargs):
        fname = f"{self.name.lower()}_M{self.n_obj}_D{self.n_var}.pf"
        return load_pareto_front_from_file(os.path.join("STEP", fname))


class STEP22(ZXHCF10):
    def __init__(self):
        super(STEP22, self).__init__(n_var=5, n_obj=3)
        self.name = self.__class__.__name__

    def _calc_pareto_front(self, *args, **kwargs):
        fname = f"{self.name.lower()}_M{self.n_obj}_D{self.n_var}.pf"
        return load_pareto_front_from_file(os.path.join("STEP", fname))


class STEP23(C3DTLZ1):
    def __init__(self):
        super(STEP23, self).__init__(n_var=5, n_obj=3)
        self.name = self.__class__.__name__

    def _calc_pareto_front(self, *args, **kwargs):
        fname = f"{self.name.lower()}_M{self.n_obj}_D{self.n_var}.pf"
        return load_pareto_front_from_file(os.path.join("STEP", fname))


class STEP24(DC2DTLZ1):
    def __init__(self):
        super(STEP24, self).__init__(n_var=5, n_obj=3)
        self.name = self.__class__.__name__

    def _calc_pareto_front(self, *args, **kwargs):
        fname = f"{self.name.lower()}_M{self.n_obj}_D{self.n_var}.pf"
        return load_pareto_front_from_file(os.path.join("STEP", fname))


class STEP25(DC2DTLZ3):
    def __init__(self):
        super(STEP25, self).__init__(n_var=10, n_obj=3)
        self.name = self.__class__.__name__

    def _calc_pareto_front(self, *args, **kwargs):
        fname = f"{self.name.lower()}_M{self.n_obj}_D{self.n_var}.pf"
        return load_pareto_front_from_file(os.path.join("STEP", fname))


class STEP26(DASCMOP7):
    def __init__(self):
        super(STEP26, self).__init__(n_var=10, n_obj=3)
        self.name = self.__class__.__name__

    def _calc_pareto_front(self, *args, **kwargs):
        fname = f"{self.name.lower()}_M{self.n_obj}_D{self.n_var}.pf"
        return load_pareto_front_from_file(os.path.join("STEP", fname))


class STEP27(ZXHCF3):
    def __init__(self):
        super(STEP27, self).__init__(n_var=10, n_obj=3)
        self.name = self.__class__.__name__

    def _calc_pareto_front(self, *args, **kwargs):
        fname = f"{self.name.lower()}_M{self.n_obj}_D{self.n_var}.pf"
        return load_pareto_front_from_file(os.path.join("STEP", fname))


class STEP28(DC2DTLZ1):
    def __init__(self):
        super(STEP28, self).__init__(n_var=30, n_obj=3)
        self.name = self.__class__.__name__

    def _calc_pareto_front(self, *args, **kwargs):
        fname = f"{self.name.lower()}_M{self.n_obj}_D{self.n_var}.pf"
        return load_pareto_front_from_file(os.path.join("STEP", fname))


class STEP29(DASCMOP7):
    def __init__(self):
        super(STEP29, self).__init__(n_var=30, n_obj=3)
        self.name = self.__class__.__name__

    def _calc_pareto_front(self, *args, **kwargs):
        fname = f"{self.name.lower()}_M{self.n_obj}_D{self.n_var}.pf"
        return load_pareto_front_from_file(os.path.join("STEP", fname))


class STEP30(C3DTLZ1):
    def __init__(self):
        super(STEP30, self).__init__(n_var=30, n_obj=3)
        self.name = self.__class__.__name__

    def _calc_pareto_front(self, *args, **kwargs):
        fname = f"{self.name.lower()}_M{self.n_obj}_D{self.n_var}.pf"
        return load_pareto_front_from_file(os.path.join("STEP", fname))
