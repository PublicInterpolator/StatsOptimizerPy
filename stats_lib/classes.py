import numpy as np

import stats_lib.utils as utils
from stats_lib.stats_approximations import EPSILON, DEFAULT_DET_POOL, Brutality, SpecialStats
from stats_lib.calculator import StatsCalculator


class BaseClass(StatsCalculator):
    """
    An object oriented wrapper over the StatsCalculator.

    Attributes:
    -----------
        - proficiency
        - determination
        - brutality
        - fortune
        - stats
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

        def __set_name__(self, owner, name):
            self._name = name

        def __get__(self, instance, owner):
            return instance.stats[self._index]

        def __set__(self, instance, value):
            if value < 0.0:
                print('Negative stat {} (val = {})'.format(self._name, value))
            instance.stats[self._index] = value

    dominance_coefficient = 0.0
    default_crit_chance = 0.0

    proficiency = _StatsDescriptor(0)
    determination = _StatsDescriptor(1)
    brutality = _StatsDescriptor(2)
    fortune = _StatsDescriptor(3)
    dominance = _StatsDescriptor(4)

    @classmethod
    def dominance_mult(cls, dom):
        """
        Parameters
        ----------
        dom : int, float
            Amount of dominance.

        Returns
        -------
        float
        """
        return SpecialStats.special_stat_multiplier(dom, cls.dominance_coefficient)

    @classmethod
    def dominance_mult_diff(cls, dom):
        """
        Parameters
        ----------
        dom : int, float
            Amount of dominance.

        Returns
        -------
        float
        """
        return SpecialStats.special_stat_multiplier_diff(dom, cls.dominance_coefficient)

    @classmethod
    @utils.documentation_inheritance(StatsCalculator.crit_chance_calculate)
    def crit_chance_calculate(cls, fort):
        return StatsCalculator.crit_chance_calculate(fort) + cls.default_crit_chance

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

    def __init__(self, stats_sum=0.0):
        n = self.stats_number()
        self._stats = stats_sum / n * np.ones(n)

    @property
    def stats(self):
        """
        Returns
        -------
        np.ndarray
        """
        return self._stats

    @stats.setter
    def stats(self, stats):
        """
        Parameters
        ----------
        stats : list, np.ndarray
        """
        for i, stat in enumerate(stats):
            self._stats[i] = stat

    @property
    def stats_sum(self):
        """
        Returns the sum of stats.

        Returns
        -------
        float
        """
        return sum(self.stats)

    def crit_randomize(self, crit_chance=None):
        """
        Returns True with probability equals to crit_chance and False otherwise.

        Parameters
        ----------
        crit_chance : float, optional
            Probability (from [0, 1]) of critical damage.
            If None (default), then self.crit_chance_calculate will be used.

        Returns
        -------
        bool
        """
        if crit_chance is None:
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
        return self.noncrit_dmg_calculate(self.stats, det_pool=det_pool, lost_hp=lost_hp)

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
        return self.dmg_calculate(self.stats, det_pool=det_pool, lost_hp=lost_hp, inst=inst)

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
        self.stats = self.analytical_optimization(stats_sum, det_pool=det_pool, inst=inst)
        if lost_hp is None and self._analytical_solution_check(lb, ub):
            return
        self.stats = self.optimization(stats_sum, det_pool=det_pool, lost_hp=lost_hp,
                                       inst=inst, lb=lb, ub=ub, stats0=self.stats)

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
        rounded_stats = np.round(self.stats)
        rounding_err = self.stats - rounded_stats
        sum_discrepancy = round(sum(rounding_err))
        sign = int(np.sign(sum_discrepancy))
        sum_discrepancy *= sign
        rounding_err *= sign
        rounding_err, indices = utils.indexed_sort(rounding_err, reverse=True)
        for index in indices:
            if sum_discrepancy > 0:
                rounded_stats[index] += sign
                sum_discrepancy -= 1
        self.stats = rounded_stats

    def _analytical_solution_check(self, lb, ub):
        n = self.stats_number()
        th = SpecialStats.get_diminishing_threshold()
        if lb is not None:
            for i in range(n):
                if self.stats[i] < lb[i] - EPSILON:
                    return False
        if ub is not None:
            for i in range(n):
                if self.stats[i] > ub[i] + EPSILON:
                    return False
        if self.fortune > th + EPSILON or self.dominance > th + EPSILON:
            return False
        return True


@utils.documentation_inheritance(BaseClass)
class Bard(BaseClass):
    dominance_coefficient = 2e-4
    default_crit_chance = 0.08

    @staticmethod
    @utils.documentation_inheritance(BaseClass.get_default_lb)
    def get_default_lb():
        lb = BaseClass.get_default_lb()
        lb[2] += 90
        lb[4] += 120
        return lb


@utils.documentation_inheritance(BaseClass)
class Druid(BaseClass):
    dominance_coefficient = 4e-4
    default_crit_chance = 0.1

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
    dominance_coefficient = 3e-4

    @staticmethod
    @utils.documentation_inheritance(BaseClass.get_default_lb)
    def get_default_lb():
        lb = BaseClass.get_default_lb()
        lb[0] += 90
        lb[4] += 120
        return lb


@utils.documentation_inheritance(BaseClass)
class Mage(BaseClass):
    dominance_coefficient = 5e-4

    @staticmethod
    @utils.documentation_inheritance(BaseClass.get_default_lb)
    def get_default_lb():
        lb = BaseClass.get_default_lb()
        lb[2] += 90
        lb[4] += 120
        return lb


@utils.documentation_inheritance(BaseClass)
class Necromancer(BaseClass):
    dominance_coefficient = 6e-4

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
    dominance_coefficient = 2.5e-4

    @staticmethod
    @utils.documentation_inheritance(BaseClass.get_default_lb)
    def get_default_lb():
        lb = BaseClass.get_default_lb()
        lb[0] += 90
        lb[4] += 120
        return lb


@utils.documentation_inheritance(BaseClass)
class Priest(BaseClass):
    dominance_coefficient = 5e-4

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
    dominance_coefficient = 4.0e-4
    default_crit_chance = 0.06

    @staticmethod
    @utils.documentation_inheritance(BaseClass.get_default_lb)
    def get_default_lb():
        lb = BaseClass.get_default_lb()
        lb[2] += 90
        lb[4] += 120
        return lb


@utils.documentation_inheritance(BaseClass)
class Stalker(BaseClass):
    dominance_coefficient = 6e-4

    @staticmethod
    @utils.documentation_inheritance(BaseClass.get_default_lb)
    def get_default_lb():
        lb = BaseClass.get_default_lb()
        lb[2] += 120
        lb[4] += 120
        return lb


@utils.documentation_inheritance(BaseClass)
class Warrior(BaseClass):
    dominance_coefficient = 4.5e-4
    default_crit_chance = 0.01

    @staticmethod
    @utils.documentation_inheritance(BaseClass.get_default_lb)
    def get_default_lb():
        lb = BaseClass.get_default_lb()
        lb[2] += 90
        lb[4] += 120
        return lb


@utils.documentation_inheritance(BaseClass)
class Warlock(BaseClass):
    dominance_coefficient = 2.5e-4
    default_crit_chance = 0.1

    @staticmethod
    @utils.documentation_inheritance(BaseClass.get_default_lb)
    def get_default_lb():
        lb = BaseClass.get_default_lb()
        lb[2] += 90
        lb[4] += 120
        return lb
