import stats_lib.utils as utils
import stats_lib.calculator as clc
from stats_lib.calculator import StatsCalculator as Calc
import numpy as np

DEFAULT_DET_POOL = 0.95


class BaseClass(Calc):
    """
    An object oriented wrapper over the StatsCalculator.

    Attributes:
    -----------
        - proficiency
        - determination
        - brutality
        - fortune
        - stats_sum

    Methods:
    --------
    Overridden StatsCalculator methods:
        - dominance_mult
        - dominance_mult_diff
        - crit_chance_calculate

    Other methods:
        - get_default_lb - default lower bounds for stats
        - crit_randomize - returns True with the crit probability
        - noncrit_dmg - total damage multiplier ignoring fortune
        - dmg - mean value of total damage multiplier with fortune
          (see StatsCalculator.dmg_calculate)
        - stats_optimize - optimal stats distribution
        - stats_round - rounds stats to integers
    """

    class _StatsDescriptor:
        def __init__(self, index):
            self._index = index

        def __get__(self, instance, owner):
            return instance._stats[self._index]

        def __set__(self, instance, value):
            if value < 0.0:
                print('Negative stat value, '
                      'stat_index = ' + str(self._index) +
                      'stat_value = ' + str(value))
            instance._stats[self._index] = value

    proficiency = _StatsDescriptor(0)
    determination = _StatsDescriptor(1)
    brutality = _StatsDescriptor(2)
    fortune = _StatsDescriptor(3)
    dominance = _StatsDescriptor(4)

    _dom_cff = 0.0
    _default_crit_chance = 0.0

    def __init__(self, stats_sum=0.0):
        n = self.stats_number()
        self._stats = stats_sum / n * np.ones(n)

    @staticmethod
    def get_default_lb():
        """
        Returns default lower bounds for stats.

        Returns
        -------
        np.ndarray

        Notes
        -----
        This stat values characters have by default, without equip,
        buffs and freely distributable stats (treatise on perfection.
        In 'BaseClass' stats from holy weapons, spark milestones,
        guild milestones and ornaments are considered.
        In derived classes also considers the class milestones.
        """
        return np.array([48, 35, 35, 88, 40])

    @classmethod
    def dominance_mult(cls, dom):
        """
        Calculates the dominance multiplier.

        Parameters
        ----------
        dom : int, float
            Amount of dominance.

        Returns
        -------
        float
        """
        return Calc._special_stat_multiplier(dom, cls._dom_cff)

    @classmethod
    def dominance_mult_diff(cls, dom):
        """
        Calculates the derivative of the dominance multiplier.

        Parameters
        ----------
        dom : int, float
            Amount of dominance.

        Returns
        -------
        float
        """
        return Calc._special_stat_multiplier_diff(dom, cls._dom_cff)

    @classmethod
    @utils.documentation_inheritance(Calc.crit_chance_calculate)
    def crit_chance_calculate(cls, fort):
        return Calc.crit_chance_calculate(fort) + cls._default_crit_chance

    @property
    def stats_sum(self):
        """
        Returns the sum of stats.

        Returns
        -------
        float
        """
        return sum(self._stats)

    def crit_randomize(self):
        """
        Returns True with probability equals to crit chance and False otherwise.

        Returns
        -------
        bool
        """
        crit_chance = self.crit_chance_calculate(self.fortune)
        return utils.random_event_generate(crit_chance)

    def noncrit_dmg(self, *, det_pool=DEFAULT_DET_POOL, lost_hp=None):
        """
        Calculates the total noncrit damage multiplier (ignoring fortune).

        Parameters
        ----------
        det_pool : float, optional
            Normalized value of determination pool from [0, 1] interval.
        lost_hp : float, optional
            Normalized value of lost hp from [0, 1] interval, if lost_hp is None,
            the brutality multiplier will be calculated with lost_hp = gamma(brut).

        Returns
        -------
        float

        See also
        --------
        StatsCalculator.noncrit_dmg_calculate
        """
        return self.noncrit_dmg_calculate(self._stats, det_pool=det_pool, lost_hp=lost_hp)

    def dmg(self, *, det_pool=DEFAULT_DET_POOL, lost_hp=None, inst=0.0):
        """
        Calculates the total damage multiplier
        (crits are considered in terms of mean value).

        Parameters
        ----------
        det_pool : float
            Normalized value of determination pool from [0, 1] interval.
        lost_hp : float, optional
            Normalized value of lost hp from [0, 1] interval, if lost_hp is None,
            the brutality multiplier will be calculated with lost_hp = gamma(brut).
        inst : int, float, optional
            Amount of target's instinct.

        Returns
        -------
        float

        See also
        --------
        StatsCalculator.dmg_calculate
        """
        return self.dmg_calculate(self._stats, det_pool=det_pool, lost_hp=lost_hp, inst=inst)

    def stats_optimize(self, *, stats_sum=None, det_pool=DEFAULT_DET_POOL,
                       lost_hp=None, inst=0.0, lb=None, ub=None):
        """
        Sets stats according to the optimal distribution with fixed sum and
        lower and upper bounds (crits are considered in terms of mean value).

        Parameters
        ----------
        stats_sum : int, float, optional
            Sum of stats. If None then current stats sum will be used.
        det_pool : float, optional
            Normalized value of determination pool from [0, 1] interval.
        lost_hp : float, optional
            Normalized value of lost hp from [0, 1] interval, if lost_hp is None,
            the brutality multiplier will be calculated with lost_hp = gamma(brut).
        inst : int, float, optional
            Amount of target's instinct.
        lb : list, tuple, ndarray, optional
            List of lower bounds for stats. By default is None, in this case
            lower bounds will be 0 for all stats.
        ub : list, tuple, ndarray, optional
            List of upper bounds for stats. By default is None, in this case
            upper bounds will be stats_sum for all stats.

        Notes
        -----
        The algorithm will try to find solution analytically.
        If the analytical solution is inappropriate
        (it does not fit lower or upper boundaries,
        some of special stats reaches diminishing threshold
        or the 'lost_hp' argument is passed), then solution will
        be found numerically, using analytical result as initial
        precision.

        See also
        --------
        StatsCalculator.optimization
        StatsCalculator.multi_approximate_optimization
        StatsCalculator.analytical_optimization
        """
        if stats_sum is None:
            stats_sum = self.stats_sum
        self._stats = self.analytical_optimization(stats_sum, det_pool=det_pool, inst=inst)
        if lost_hp is None and self._analytical_solution_check(lb, ub):
            return
        self._stats = self.optimization(stats_sum, det_pool=det_pool, lost_hp=lost_hp,
                                        inst=inst, lb=lb, ub=ub, stats0=self._stats)

    def stats_round(self):
        """
        Rounding the stats to integers.

        Notes
        -----
        Doesn't work like rounding up, down or to nearest integer.
        Finds the closest integers, having sum which differs from
        current sum no more than by 0.5.

        For example for class with stats [1.3, 1.4, 1.2, 1.8, 1.3]
        the result will be [1, 2, 1, 2, 1].
        """
        rounded_stats = np.round(self._stats)
        rounding_err = self._stats - rounded_stats
        sum_discrepancy = round(sum(rounding_err))
        sign = int(np.sign(sum_discrepancy))
        sum_discrepancy *= sign
        rounding_err *= sign
        rounding_err, indices = utils.indexed_sort(rounding_err, reverse=True)
        for index in indices:
            if sum_discrepancy > 0:
                rounded_stats[index] += sign
                sum_discrepancy -= 1
        self._stats = rounded_stats

    def _analytical_solution_check(self, lb, ub):
        n = self.stats_number()
        th = clc.SpecialSpline.get_diminishing_threshold()
        if lb is not None:
            for i in range(n):
                if self._stats[i] < lb[i] - clc.EPSILON:
                    return False
        if ub is not None:
            for i in range(n):
                if self._stats[i] > ub[i] + clc.EPSILON:
                    return False
        if self.fortune > th + clc.EPSILON or self.dominance > th + clc.EPSILON:
            return False
        return True


@utils.documentation_inheritance(BaseClass)
class Bard(BaseClass):
    _dom_cff = 2e-4
    _default_crit_chance = 0.08

    @staticmethod
    @utils.documentation_inheritance(BaseClass.get_default_lb)
    def get_default_lb():
        lb = BaseClass.get_default_lb()
        lb[2] += 90
        lb[4] += 120
        return lb


@utils.documentation_inheritance(BaseClass)
class Druid(BaseClass):
    _dom_cff = 4e-4
    _default_crit_chance = 0.1

    @staticmethod
    @utils.documentation_inheritance(BaseClass.get_default_lb)
    def get_default_lb():
        lb = BaseClass.get_default_lb()
        lb[1] += 90
        lb[2] += 90
        lb[4] += 120
        return lb


@utils.documentation_inheritance(BaseClass)
class Engineer(BaseClass):
    _dom_cff = 3e-4

    @staticmethod
    @utils.documentation_inheritance(BaseClass.get_default_lb)
    def get_default_lb():
        lb = BaseClass.get_default_lb()
        lb[0] += 90
        lb[4] += 120
        return lb


@utils.documentation_inheritance(BaseClass)
class Mage(BaseClass):
    _dom_cff = 5e-4

    @staticmethod
    @utils.documentation_inheritance(BaseClass.get_default_lb)
    def get_default_lb():
        lb = BaseClass.get_default_lb()
        lb[2] += 90
        lb[4] += 120
        return lb


@utils.documentation_inheritance(BaseClass)
class Necromancer(BaseClass):
    _dom_cff = 6e-4

    @staticmethod
    @utils.documentation_inheritance(BaseClass.get_default_lb)
    def get_default_lb():
        lb = BaseClass.get_default_lb()
        lb[1] += 90
        lb[2] += 90
        lb[4] += 120
        return lb


@utils.documentation_inheritance(BaseClass)
class Paladin(BaseClass):
    _dom_cff = 2.5e-4

    @staticmethod
    @utils.documentation_inheritance(BaseClass.get_default_lb)
    def get_default_lb():
        lb = BaseClass.get_default_lb()
        lb[0] += 90
        lb[4] += 120
        return lb


@utils.documentation_inheritance(BaseClass)
class Priest(BaseClass):
    _dom_cff = 5e-4

    @staticmethod
    @utils.documentation_inheritance(BaseClass.get_default_lb)
    def get_default_lb():
        lb = BaseClass.get_default_lb()
        lb[1] += 90
        lb[2] += 90
        lb[4] += 120
        return lb


@utils.documentation_inheritance(BaseClass)
class Psionic(BaseClass):
    _dom_cff = 4.0e-4
    _default_crit_chance = 0.06

    @staticmethod
    @utils.documentation_inheritance(BaseClass.get_default_lb)
    def get_default_lb():
        lb = BaseClass.get_default_lb()
        lb[2] += 90
        lb[4] += 120
        return lb


@utils.documentation_inheritance(BaseClass)
class Stalker(BaseClass):
    _dom_cff = 6e-4

    @staticmethod
    @utils.documentation_inheritance(BaseClass.get_default_lb)
    def get_default_lb():
        lb = BaseClass.get_default_lb()
        lb[2] += 120
        lb[4] += 120
        return lb


@utils.documentation_inheritance(BaseClass)
class Warrior(BaseClass):
    _dom_cff = 4.5e-4
    _default_crit_chance = 0.01

    @staticmethod
    @utils.documentation_inheritance(BaseClass.get_default_lb)
    def get_default_lb():
        lb = BaseClass.get_default_lb()
        lb[2] += 90
        lb[4] += 120
        return lb


@utils.documentation_inheritance(BaseClass)
class Warlock(BaseClass):
    _dom_cff = 2.5e-4
    _default_crit_chance = 0.1

    @staticmethod
    @utils.documentation_inheritance(BaseClass.get_default_lb)
    def get_default_lb():
        lb = BaseClass.get_default_lb()
        lb[2] += 90
        lb[4] += 120
        return lb


