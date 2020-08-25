import functools

import numpy as np
import scipy.optimize as opt

import stats_lib.utils as utils
from stats_lib.stats_approximations import EPSILON, DEFAULT_DET_POOL, Brutality, SpecialStats


class StatsCalculator:
    """
    Provides methods of calculation the optimal stats distribution.
    Also includes methods for stats multipliers and their derivatives.

    Does not imply instantiation.

    Methods:
    --------
    Stats multipliers and their derivatives:
        - proficiency_mult
        - proficiency_mult_diff
        - determination_mult
        - determination_mult_diff
        - brutality_mult
        - brutality_mult_diff
        - fortune_mult
        - fortune_mult_diff
        - dominance_mult
        - dominance_mult_diff
        - vitality_mult
        - survivability_mult
        - caution_mult
        - instinct_mult

    Optimization methods:
        - analytical_optimization
        - optimization
        - multi_approximate_optimization

    Other methods:
        - stats_number - number of supported offensive stats
        - crit_chance_calculate - calculates the crit probability
        - crit_chance_diff - calculates the derivative of the crit probability by fortune
        - noncrit_dmg_calculate - calculates the total damage multiplier ignoring fortune
        - dmg_calculate - calculates the mean value of total damage multiplier with fortune
        - dmg_grad - calculates gradient vector of 'dmg_calculate' by all stats
    """
    _n = 5

    # //////////////////// Public section ////////////////////
    @staticmethod
    def proficiency_mult(prof):
        """
        Parameters
        ----------
        prof : int, float
            Amount of proficiency.

        Returns
        -------
        float
        """
        return 1.0 + 5e-4 * prof

    @staticmethod
    def proficiency_mult_diff(prof=0):
        """
        Parameters
        ----------
        prof : int, float
            Amount of proficiency
            (since the derivative is constant, this argument is ignored).

        Returns
        -------
        float
        """
        return 5e-4

    @staticmethod
    def determination_mult(det, det_pool=DEFAULT_DET_POOL):
        """
        Parameters
        ----------
        det : int, float
            Amount of determination.
        det_pool : float
            Normalized value of determination pool from [0, 1] interval.

        Returns
        -------
        float
        """
        return 1.0 + 7.5e-4 * det_pool * det

    @staticmethod
    def determination_mult_diff(det=0, det_pool=DEFAULT_DET_POOL):
        """
        Parameters
        ----------
        det : int, float
            Amount of determination
            (since the derivative is constant, this argument is ignored).
        det_pool : float
            Normalized value of determination pool from [0, 1] interval.

        Returns
        -------
        float
        """
        return 7.5e-4 * det_pool

    @staticmethod
    def brutality_mult(brut, lost_hp=None):
        """
        Parameters
        ----------
        brut : int, float
        lost_hp : float, optional

        Returns
        -------
        float

        See also
        --------
        Brutality.brutality_multiplier
        Brutality.gamma
        """
        return Brutality.brutality_multiplier(brut, lost_hp)

    @staticmethod
    def brutality_mult_diff(brut, lost_hp=None):
        """
        Parameters
        ----------
        brut : int, float
            Amount of brutality.
        lost_hp : float, optional
            Normalized value of lost hp from [0, 1] interval, if lost_hp is None,
            the derivative is taken from multiplier with lost_hp = gamma(brut).

        Returns
        -------
        float

        See also
        --------
        Brutality.brutality_multiplier
        Brutality.gamma
        """
        return Brutality.brutality_multiplier_diff(brut, lost_hp)

    @classmethod
    def fortune_mult(cls, fort, inst=0.0):
        """
        Parameters
        ----------
        fort : int, float
            Amount of fortune.
        inst : int, float, optional
            Amount of target's instinct.

        Returns
        -------
        float

        Notes
        -----
        The multiplier isn't random and is calculated in terms of mean value of damage.
        Also may consider the target's instinct.
        """
        cff = cls._fortune_cff_calculate(inst)
        return 1.0 + cff * cls.crit_chance_calculate(fort)

    @classmethod
    def fortune_mult_diff(cls, fort, inst=0.0):
        """
        Parameters
        ----------
        fort : int, float
            Amount of fortune.
        inst : int, float, optional
            Amount of target's instinct.

        Returns
        -------
        float

        Notes
        -----
        The multiplier isn't random and is calculated in terms of mean value of damage.
        Also may consider the target's instinct.
        """
        cff = cls._fortune_cff_calculate(inst)
        return cff * cls.crit_chance_diff(fort)

    @staticmethod
    def dominance_mult(dom):
        """
        Parameters
        ----------
        dom : int, float
            Amount of dominance
            (since the dominance isn't supported, this argument is ignored).

        Returns
        -------
        float

        Notes
        -----
        Always returns 1 since dominance isn't supported by default,
        but could be implemented in derived classes for every
        class-specific dominance mechanics. Must be overridden only in pair
        with dominance_mult_diff.
        """
        return 1.0

    @staticmethod
    def dominance_mult_diff(dom):
        """
        Parameters
        ----------
        dom : int, float
            Amount of dominance
            (since the dominance isn't supported, this argument is ignored).

        Returns
        -------
        float

        Notes
        -----
        Always returns 0 since dominance isn't supported by default,
        but could be implemented in derived classes for every
        class-specific dominance mechanics. Must be overridden only in pair
        with dominance_mult.
        """
        return 0.0

    @staticmethod
    def vitality_mult(vit):
        """
        Parameters
        ----------
        vit : int, float
            Amount of vitality.

        Returns
        -------
        float
        """
        return 1.0 + 5e-4 * vit

    @staticmethod
    def survivability_mult(surv):
        """
        Parameters
        ----------
        surv : int, float
            Amount of survivability.

        Returns
        -------
        float
        """
        return SpecialStats.special_stat_multiplier(surv, 8e-4)

    @staticmethod
    def caution_mult(caut):
        """
        The multiplier represents damage reduction factor at 40% and less hp.

        Parameters
        ----------
        caut : int, float
            Amount of caution.

        Returns
        -------
        float
        """
        return SpecialStats.special_stat_multiplier(caut, -4e-4)

    @staticmethod
    def instinct_mult(inst):
        """
        The multiplier represents the critical damage damage reduction factor.

        Parameters
        ----------
        inst : int, float
            Amount of instinct.

        Returns
        -------
        float
        """
        return SpecialStats.special_stat_multiplier(inst, -4e-4)

    @staticmethod
    def crit_chance_calculate(fort):
        """
        Calculates the probability (from [0, 1]) of critical damage.

        Parameters
        ----------
        fort : int, float
            Amount of fortune.

        Returns
        -------
        float
        """
        return 8e-4 * SpecialStats.spline(fort) * fort

    @staticmethod
    def crit_chance_diff(fort):
        """
        Calculates the derivative of the crit chance.

        Parameters
        ----------
        fort : int, float
            Amount of fortune.

        Returns
        -------
        float
        """
        return SpecialStats.special_stat_multiplier_diff(fort, 8e-4)

    @staticmethod
    def stats_number():
        """
        Returns the number of supported offensive stats.

        Returns
        -------
        int
        """
        return StatsCalculator._n

    @classmethod
    def noncrit_dmg_calculate(cls, stats, *, det_pool=DEFAULT_DET_POOL, lost_hp=None):
        """
        Calculates the total noncrit damage multiplier (ignoring fortune).

        Parameters
        ----------
        stats :  list, tuple, ndarray
            List of stat values.
        det_pool : float
            Normalized value of determination pool from [0, 1] interval.
        lost_hp : float, optional
            Normalized value of lost hp from [0, 1] interval, if lost_hp is None,
            the brutality multiplier will be calculated with lost_hp = gamma(brut).

        Returns
        -------
        float

        See also
        --------
        Brutality.l_calculate
        Brutality.gamma
        """
        dmg = 1.0
        dmg *= cls.proficiency_mult(stats[0])
        dmg *= cls.determination_mult(stats[1], det_pool)
        dmg *= cls.brutality_mult(stats[2], lost_hp)
        dmg *= cls.dominance_mult(stats[4])
        return dmg

    @classmethod
    def dmg_calculate(cls, stats, *, det_pool=DEFAULT_DET_POOL, lost_hp=None, inst=0.0):
        """
        Calculates the total damage multiplier
        (crits are considered in terms of mean value).

        Parameters
        ----------
        stats : list, tuple, ndarray
            List of stat values.
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
        Brutality.l_calculate
        Brutality.gamma
        """
        noncrit_dmg = cls.noncrit_dmg_calculate(stats, det_pool=det_pool, lost_hp=lost_hp)
        return noncrit_dmg * cls.fortune_mult(stats[3], inst)

    @classmethod
    def dmg_grad(cls, stats, *, det_pool=DEFAULT_DET_POOL, lost_hp=None, inst=0.0):
        """
        Calculates the gradient vector of the total damage multiplier
        (crits are considered in terms of mean value).

        Parameters
        ----------
        stats : list, tuple, ndarray
            List of stat values.
        det_pool : float
            Normalized value of determination pool from [0, 1] interval.
        lost_hp : float, optional
            Normalized value of lost hp from [0, 1] interval, if lost_hp is None,
            the brutality multiplier will be calculated with lost_hp = gamma(brut).
        inst : int, float, optional
            Amount of target's instinct.

        Returns
        -------
        np.ndarray

        See also
        --------
        Brutality.l_calculate
        Brutality.gamma
        """
        dmg = cls.dmg_calculate(stats, det_pool=det_pool, lost_hp=lost_hp, inst=inst)
        prof_diff = dmg * cls.proficiency_mult_diff(stats[0]) / cls.proficiency_mult(stats[0])
        det_diff = dmg * cls.determination_mult_diff(stats[1], det_pool)
        det_diff /= cls.determination_mult(stats[1], det_pool)
        brut_diff = dmg * cls.brutality_mult_diff(stats[2], lost_hp)
        brut_diff /= cls.brutality_mult(stats[2], lost_hp)
        fort_diff = dmg * cls.fortune_mult_diff(stats[3], inst) / cls.fortune_mult(stats[3], inst)
        dom_diff = dmg * cls.dominance_mult_diff(stats[4]) / cls.dominance_mult(stats[4])
        return np.array([prof_diff, det_diff, brut_diff, fort_diff, dom_diff], copy=False).T

    @classmethod
    def multi_approximate_optimization(cls, stats_sum, *, det_pool=DEFAULT_DET_POOL,
                                       lost_hp=None, inst=0.0, lb=None, ub=None, stats0=None):
        """
        Calculates the optimal stats distribution with fixed sum and
        lower and upper bounds (crits are considered in terms of mean value).

        Parameters
        ----------
        stats_sum : int, float
            Sum of stats.
        det_pool : float
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
        stats0 : list, tuple, ndarray, optional
            List of stat values that will be used as one of the the initial precisions for
            numerical optimization algorithm in addition to default list of precisions.

        Returns
        -------
        np.ndarray

        Notes
        -----
        Optimization algorithm will be run from several different initial precisions.
        From all results will be returned the one with the greatest value of goal function
        (dmg_calculate), regardless of whether the algorithm's convergence from
        corresponding initial precision was successful.

        In most cases is better to call "optimization" method, than this method directly.
        The only reason to directly call "multi_approximate_optimization" instead of "optimization" is that
        objective function is expected to have several local maximum.

        See also
        --------
        optimization
        analytical_optimization
        """
        n = cls.stats_number()
        if np.abs(stats_sum) < EPSILON:
            return np.zeros(n)
        if lb is None:
            lb = np.zeros(n)
        if ub is None:
            ub = stats_sum * np.ones(n)

        approximations = []
        if stats0 is not None:
            approximations.append(stats0)
        approximations.append(cls._get_initial_approximation(stats_sum, lb=lb, ub=ub))
        for index in range(n):
            approximations.append(cls._get_initial_approximation(stats_sum, lb=lb,
                                                                 ub=ub, index=index))

        results = []
        for approximation in approximations:
            results.append(cls._numerical_optimize(stats_sum, det_pool=det_pool, lost_hp=lost_hp,
                                                   inst=inst, lb=lb, ub=ub, stats0=approximation))
        results.sort(key=lambda result: -result.fun)
        return np.array(results[-1].x, copy=False)

    @classmethod
    def optimization(cls, stats_sum, *, det_pool=DEFAULT_DET_POOL,
                     lost_hp=None, inst=0.0, lb=None, ub=None, stats0=None):
        """
        Calculates the optimal stats distribution with fixed sum and
        lower and upper bounds (crits are considered in terms of mean value).

        Parameters
        ----------
        stats_sum : int, float
            Sum of stats.
        det_pool : float
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
        stats0 : list, tuple, ndarray, optional
            List of stat values that will be used as the initial precision for
            numerical optimization algorithm. If stats0 is None, the default precision
            well be used.

        Returns
        -------
        np.ndarray

        Notes
        -----
        Optimization algorithm will be run from single initial precision.
        If the algorithm doesn't converge successfully, then "multi_approximate_optimization"
        method will be called in order to achieve convergence from another initial precisions.

        In most cases is better to call this method, than directly the "multi_approximate_optimization" method.
        The only reason to directly call "multi_approximate_optimization" instead of "optimization" is that
        the objective function is expected to have several local maximum.

        See also
        --------
        multi_approximate_optimization
        analytical_optimization
        """
        n = cls.stats_number()
        if np.abs(stats_sum) < EPSILON:
            return np.zeros(n)
        if lb is None:
            lb = np.zeros(n)
        if ub is None:
            ub = stats_sum * np.ones(n)
        if stats0 is None:
            approximation = cls._get_initial_approximation(stats_sum, lb=lb, ub=ub)
        else:
            approximation = stats0

        result = cls._numerical_optimize(stats_sum, det_pool=det_pool, lost_hp=lost_hp,
                                         inst=inst, lb=lb, ub=ub, stats0=approximation)
        if result.success:
            return np.array(result.x, copy=False)
        else:
            return cls.multi_approximate_optimization(stats_sum, det_pool=det_pool, lost_hp=lost_hp,
                                                      inst=inst, lb=lb, ub=ub, stats0=stats0)

    @classmethod
    def analytical_optimization(cls, stats_sum, *, det_pool=DEFAULT_DET_POOL, inst=0.0):
        """
        Calculates the optimal stats distribution with fixed sum
        (crits are considered in terms of mean value).

        Parameters
        ----------
        stats_sum : int, float
            Sum of stats.
        det_pool : float
            Normalized value of determination pool from [0, 1] interval.
        inst : int, float, optional
            Amount of target's instinct.

        Returns
        -------
        np.ndarray

        Notes
        -----
        Analytically finds optimal stats distribution. Works much faster than
        numerical methods like "optimization" or "multi_approximate_optimization",
        but does not supports lower or upper bounds and fixed lost hp.

        Since all the stats multipliers are considered linear, algorithm works correctly
        only before any stat reaches diminishing threshold.

        See also
        --------
        optimization
        multi_approximate_optimization
        """
        thresholds, indices = cls._threshold_matrix_calculate(det_pool, inst)
        n = cls.stats_number()

        def stats_calculate(th_number):
            stats = np.zeros(n)
            for j in range(th_number + 1):
                index = indices[j]
                stats[index] = stats_sum - thresholds[th_number][j]
            return stats / (th_number + 1)

        for i in range(n - 1):
            if thresholds[i][i] <= stats_sum < thresholds[i + 1][i + 1]:
                return stats_calculate(i)
        if thresholds[n - 1][n - 1] <= stats_sum:
            return stats_calculate(n - 1)
        return np.zeros(n)

    # //////////////////// End public section ////////////////////

    # //////////////////// Private section ////////////////////
    @classmethod
    @functools.lru_cache(maxsize=128)
    def _fortune_cff_calculate(cls, inst):
        inst_mult = 1.0
        if np.abs(inst) >= EPSILON:
            inst_mult = cls.instinct_mult(inst)
        return 3.0 / 2.0 * inst_mult - 1.0

    @classmethod
    def _get_initial_approximation(cls, stats_sum, *, lb, ub, index=None):
        n = cls.stats_number()
        c = np.zeros(n)
        if index is not None:
            if not 0 <= index < n:
                index = 0
            c[index] = 1.0
        solution = opt.linprog(c, A_eq=np.ones((1, n)), b_eq=stats_sum, bounds=list(zip(lb, ub)))
        return np.array(solution.x)

    @classmethod
    def _numerical_optimize(cls, stats_sum, *, det_pool, lost_hp, inst, lb, ub, stats0):
        def objective(s):
            return -cls.dmg_calculate(s, det_pool=det_pool, lost_hp=lost_hp, inst=inst)

        def grad(s):
            return -cls.dmg_grad(s, det_pool=det_pool, lost_hp=lost_hp, inst=inst)

        n = cls.stats_number()
        bounds = opt.Bounds(lb, ub)
        constraint = opt.LinearConstraint(np.ones(n), stats_sum, stats_sum)
        return opt.minimize(objective, stats0, bounds=bounds, constraints=constraint,
                            jac=grad, method='trust-constr', options={'maxiter': n * 5})

    @classmethod
    @functools.lru_cache(maxsize=16)
    def _threshold_matrix_calculate(cls, det_pool, inst):
        """
        Do not modify the result!!!!
        """
        n = cls.stats_number()
        coefficients = cls.dmg_grad(np.zeros(n), det_pool=det_pool, inst=inst)
        coefficients[np.abs(coefficients) < EPSILON] = EPSILON
        coefficients, indices = utils.indexed_sort(coefficients, reverse=True)

        def threshold_calculate(stats_number, stat_index):
            inverse_sum = 0.0
            for k in range(stats_number):
                inverse_sum += 1.0 / coefficients[k]
            return stats_number / coefficients[stat_index] - inverse_sum

        thresholds = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1):
                thresholds[i][j] = threshold_calculate(i + 1, j)
        return thresholds, indices
    # //////////////////// End private section ////////////////////
