import random
import functools
import numpy as np
import scipy.optimize as opt
import scipy.interpolate as itp
from stats_lib.utils import indexed_sort

EPSILON = 1e-8
""" Numbers with absolute value less than EPSILON will be considered as zeros. """


class Brutality:
    """
    Provides methods which are required to calculate brutality multiplier and its' derivative.
    Nonlinear dependencies are approximated by parametric regression.

    Does not imply instantiation.
    """

    _params = [2.031e-3, 6.575, 5.315e-4]

    _x = (0.0, 25.0, 50.0, 75.0, 100.0, 125.0, 150.0, 175.0, 200.0, 225.0,
          250.0, 275.0, 300.0, 325.0, 350.0, 375.0, 400.0, 425.0, 450.0, 475.0,
          500.0, 525.0, 550.0, 575.0, 600.0, 625.0, 650.0, 675.0, 700.0, 725.0,
          750.0, 775.0, 800.0, 825.0, 850.0, 875.0, 900.0, 925.0, 950.0, 975.0,
          1000.0, 1025.0, 1050.0, 1075.0, 1100.0, 1125.0, 1150.0, 1175.0, 1200.0, 1225.0,
          1250.0, 1275.0, 1300.0, 1325.0, 1350.0, 1375.0, 1400.0, 1425.0, 1450.0, 1475.0,
          1500.0, 1525.0, 1550.0, 1575.0, 1600.0, 1625.0, 1650.0, 1675.0, 1700.0, 1725.0,
          1750.0, 1775.0, 1800.0, 1825.0, 1850.0, 1875.0, 1900.0, 1925.0, 1950.0, 1975.0, 2000.0)

    _y = (0.0, 0.025, 0.05, 0.076, 0.102, 0.128, 0.154, 0.18, 0.206, 0.233,
          0.26, 0.287, 0.314, 0.342, 0.369, 0.397, 0.425, 0.453, 0.482, 0.51,
          0.539, 0.567, 0.596, 0.626, 0.655, 0.684, 0.714, 0.744, 0.774, 0.804,
          0.834, 0.864, 0.895, 0.925, 0.956, 0.987, 1.02, 1.05, 1.08, 1.11,
          1.14, 1.18, 1.21, 1.24, 1.27, 1.3, 1.34, 1.37, 1.4, 1.44,
          1.47, 1.5, 1.53, 1.57, 1.6, 1.63, 1.67, 1.7, 1.74, 1.77,
          1.81, 1.84, 1.87, 1.91, 1.94, 1.98, 2.01, 2.05, 2.08, 2.12,
          2.15, 2.19, 2.23, 2.26, 2.3, 2.33, 2.37, 2.4, 2.44, 2.48, 2.51)

    @classmethod
    def l_calculate(cls, brut, a=_params[0], b=_params[1], c=_params[2]):
        """
        Parameters
        ----------
        brut : int, float
            Amount of brutality.
        a, b, c : float, optional
            Parameters of l(brut)
            (The default values is a good approximation,
            use this arguments only if you know, what you are doing).

        Returns
        -------
        float

        Notes
        -----
        l(brut) is such function that brutality multiplier can be calculated
        as 1 + l(brut) * HP_lost/HP_max. The explicit form of l(brut)
        is c * ln(a*brut + b) * brut.
        """
        return c * np.log(a * brut + b) * brut

    @classmethod
    def l_diff(cls, brut, a=_params[0], b=_params[1], c=_params[2]):
        """
        Calculates the derivative of l(brut) by brutality.

        Parameters
        ----------
        brut : int, float
            Amount of brutality.
        a, b, c : float, optional
            Parameters of l(brut)
            (The default values is a good approximation,
            use this arguments only if you know, what you are doing).

        Returns
        -------
        float

        See also
        --------
        l_calculate
        """
        ex = a * brut + b
        return c * (np.log(ex) + a * brut / ex)

    @classmethod
    def l_grad(cls, brut, a=_params[0], b=_params[1], c=_params[2]):
        """
        Calculates the gradient of l(brut) by parameters.

        Parameters
        ----------
        brut : int, float
           Amount of brutality.
        a, b, c : float, optional
           Parameters of l(brut).

        Returns
        -------
        np.ndarray

        See also
        --------
        l_calculate
        """
        ex = a * brut + b
        diff_c = np.log(ex) * brut
        diff_b = c * brut / ex
        diff_a = diff_b * brut
        return np.array([diff_a, diff_b, diff_c], copy=False).T

    @classmethod
    def gamma(cls, brut, a=_params[0], b=_params[1], c=_params[2]):
        """
        Parameters
        ----------
        brut : int, float
            Amount of brutality.
        a, b, c : float, optional
            Parameters of l(brut)
            (The default values is a good approximation,
            use this arguments only if you know, what you are doing).

        Returns
        -------
        float

        Notes
        -----
        gamma(brut) is such function that brutality multiplier equals
        1 + l(brut) * gamma(brut) following a full solo boss fight.

        See also
        --------
        l_calculate
        """
        if np.abs(brut) < EPSILON:
            return 0.5
        l = cls.l_calculate(brut, a, b, c)
        return 1.0 / np.log(1.0 + l) - 1.0 / l

    @classmethod
    def regression_initialize(cls):
        """
        Calculates the parameters p1, p2, p3 of l(brut) using nonlinear regression.
        If this function isn't called, the initial approximations will be used
        for those parameters.

        See also
        --------
        l_calculate
        """
        params, _ = opt.curve_fit(cls.l_calculate, cls._x, cls._y,
                                  jac=cls.l_grad, p0=cls._params, maxfev=5000)
        cls._params = list(params)


class SpecialSpline:
    """
    Provides the calculation of the special stats diminishing coefficient and its' derivative
    using spline interpolation.

    Does not imply instantiation.

    Notes
    -----
    The detailed spline structure:
        - Before reaching the diminishing threshold the coefficient is constant.
        - From threshold to the end of known table values coefficient is interpolated with B-spline.
        - On the interval between first and second table values B-spline is replaced with a cubic spline
          in order smooth the joint between constant and B-spline.
        - From the last table value spline is extrapolated with "tail". Different types of "tails"
          is implemented, the hyperbolic one seems to be the most appropriate.

    The resulting spline is continuous and has continuous first derivative.
    """

    _x = (750.0, 757.0, 770.0, 778.0, 783.0, 800.0, 808.0, 813.0, 818.0, 831.0,
          841.0, 844.0, 855.0, 857.0, 865.0, 916.0, 921.0, 955.0, 963.0, 965.0,
          979.0, 991.0, 995.0, 1024.0, 1067.0, 1218.0, 1370.0, 1521.0)

    _y = (1.0, 0.9997, 0.9985, 0.9962, 0.9951, 0.9898, 0.9861, 0.9843, 0.9823, 0.9760,
          0.9710, 0.9692, 0.9636, 0.9623, 0.9579, 0.9287, 0.9260, 0.9063, 0.9017, 0.9004,
          0.8924, 0.8854, 0.8832, 0.8674, 0.8445, 0.7739, 0.7160, 0.6672)

    # The several last table values are ignored in order to stabilize and smooth the spline
    _last_values_cutoff = 2
    _x = _x[:-_last_values_cutoff]
    _y = _y[:-_last_values_cutoff]

    class _BSpline:
        def __init__(self, x, y, degree=5, smoothness=1.0):
            self._spline = itp.splrep(x, y, k=degree, s=smoothness)

        def __call__(self, x):
            return itp.splev(x, self._spline)

        def diff(self, x):
            return itp.splev(x, self._spline, der=1)

    class _HyperbolicTail:
        """ The quasi - hyperbolic extrapolation, y = 1 / (a*x + b*ln(x) + c). """

        def __init__(self, x, y, y_diff, inf_lim=8e-4):
            # tail(x) * x -> inf_lim, x -> inf
            # the default value is chosen so that crit chance -> 1 as fortune -> inf
            self._a = inf_lim
            self._b = -x * (y_diff / y ** 2.0 + self._a)
            self._c = 1 / y - self._a * x - self._b * np.log(x)

        def __call__(self, x):
            return 1.0 / (self._a * x + self._b * np.log(x) + self._c)

        def diff(self, x):
            return -(self._a + self._b / x) * self(x) ** 2.0

    class _SqrtTail:
        """ The reciprocal square root extrapolation, y = a / sqrt(x - b). """

        def __init__(self, x, y, y_diff):
            self._a = np.sqrt(-0.5 * y ** 3.0 / y_diff)
            self._b = x - self._a ** 2.0 / y ** 2.0

        def __call__(self, x):
            return self._a / np.sqrt(x - self._b)

        def diff(self, x):
            return -0.5 / (x - self._b) * self(x)

    class _ExponentialTail:
        """ The exponential extrapolation, y = a^(x - b). """

        def __init__(self, x, y, y_diff):
            self._a = np.exp(y_diff / y)
            self._b = x - np.log(y) / np.log(self._a)

        def __call__(self, x):
            return np.power(self._a, x - self._b)

        def diff(self, x):
            return np.log(self._a) * self(x)

    class _LinearTail:
        """ The linear extrapolation, y = a*x + b. """

        def __init__(self, x, y, y_diff):
            self._a = y_diff
            self._b = y - self._a * x

        def __call__(self, x):
            return self._a * x + self._b

        def diff(self, x):
            return self._a

    _b_spline = _BSpline(_x, _y)
    _tail = _HyperbolicTail(_x[-1], _b_spline(_x[-1]), _b_spline.diff(_x[-1]))
    _cubic_spline = itp.CubicSpline(_x[0:2], [_y[0], _b_spline(_x[1])],
                                    bc_type=((1, 0.0), (1, _b_spline.diff(_x[1]))))
    _cubic_spline.diff = functools.partial(_cubic_spline.__call__, nu=1)

    def __new__(cls, x):
        """
        Calculates the value of spline at x.

        Parameters
        ----------
        x : float

        Returns
        -------
        float

        Notes
        -----
        The way to evaluate spline like 'y = SpecialSpline(x)'
        is to override SpecialSpline instantiation.
        """
        if x < cls._x[0]:
            return cls._y[0]
        if cls._x[0] <= x < cls._x[1]:
            return cls._cubic_spline(x)
        if x >= cls._x[-1]:
            return cls._tail(x)
        return cls._b_spline(x)

    @classmethod
    def diff(cls, x):
        """
        Calculates the derivative of spline at x.

        Parameters
        ----------
        x : float

        Returns
        -------
        float
        """
        if x < cls._x[0]:
            return 0.0
        if cls._x[0] <= x < cls._x[1]:
            return cls._cubic_spline.diff(x)
        if x >= cls._x[-1]:
            return cls._tail.diff(x)
        return cls._b_spline.diff(x)

    @classmethod
    def get_diminishing_threshold(cls):
        """
        Returns the special stat amount starting from which
        the spline ceases to be linear and special stat efficiency decreases.

        Returns
        -------
        float
        """
        return cls._x[0]


class StatsCalculator:
    """
    Provides methods of calculation the optimal stats distribution.
    Also includes methods for stats multipliers and their derivatives.

    Does not imply instantiation.

    Methods:
    --------
    Stats multipliers and their derivatives:
        proficiency_mult
        proficiency_mult_diff
        determination_mult
        determination_mult_diff
        brutality_mult
        brutality_mult_diff
        fortune_mult
        fortune_mult_diff
        dominance_mult
        dominance_mult_diff
        vitality_mult
        survivability_mult
        caution_mult
        instinct_mult

    Optimization methods:
        analytical_optimization
        optimization
        multi_approximate_optimization

    Other methods:
        stats_number - number of supported offensive stats
        crit_randomize - returns True with the given probability
        crit_chance_calculate - calculates crit probability
        noncrit_dmg_calculate - calculates the total damage multiplier ignoring fortune
        dmg_calculate - calculates the mean value of total damage multiplier with fortune
        dmg_grad - calculates gradient vector of "dmg_calculate" by all stats
    """
    Brutality.regression_initialize()
    _n = 5

    # //////////////////// Public section ////////////////////
    @staticmethod
    def proficiency_mult(prof):
        """
        Calculates the proficiency multiplier.

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
    def proficiency_mult_diff(prof):
        """
        Calculates the derivative of the proficiency multiplier.

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
    def determination_mult(det, det_pool):
        """
        Calculates the determination multiplier.

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
    def determination_mult_diff(det, det_pool):
        """ Calculates the derivative of the determination multiplier.

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
        Calculates the brutality multiplier.

        Parameters
        ----------
        brut : int, float
            Amount of brutality.
        lost_hp : float, optional
            Normalized value of lost hp from [0, 1] interval, if lost_hp is None,
            the multiplier is calculated for lost_hp = gamma(brut).

        Returns
        -------
        float

        See also
        --------
        Brutality.l_calculate
        Brutality.gamma
        """
        if np.abs(brut) < EPSILON:
            return 1.0
        l = Brutality.l_calculate(brut)
        if lost_hp is None:
            return l / np.log(1.0 + l)
        else:
            return 1.0 + l * lost_hp

    @staticmethod
    def brutality_mult_diff(brut, lost_hp=None):
        """
        Calculates the derivative of the brutality multiplier.

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
        Brutality.l_calculate
        Brutality.gamma
        """
        l_diff = Brutality.l_diff(brut)
        if lost_hp is None:
            if np.abs(brut) < EPSILON:
                return l_diff / 2.0
            l = Brutality.l_calculate(brut)
            lg = np.log(l + 1.0)
            return (1.0 / lg - l / (l + 1.0) / lg ** 2.0) * l_diff
        else:
            return l_diff * lost_hp

    @staticmethod
    def fortune_mult(fort, inst=0.0):
        """
        Calculates the fortune multiplier.
        May consider the target's instinct.

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
        """
        cff = StatsCalculator._fortune_cff_calculate(inst)
        return StatsCalculator._special_stat_multiplier(fort, cff)

    @staticmethod
    def fortune_mult_diff(fort, inst=0.0):
        """
        Calculates the derivative of the fortune multiplier.
        May consider the target's instinct.

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
        """
        cff = StatsCalculator._fortune_cff_calculate(inst)
        return StatsCalculator._special_stat_multiplier_diff(fort, cff)

    @staticmethod
    def dominance_mult(dom):
        """
        Calculates the dominance multiplier.

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
        Calculates the derivative of the dominance multiplier.

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
        Calculates the vitality multiplier.

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
        Calculates the survivability multiplier.

        Parameters
        ----------
        surv : int, float
            Amount of survivability.

        Returns
        -------
        float
        """
        return StatsCalculator._special_stat_multiplier(surv, 8e-4)

    @staticmethod
    def caution_mult(caut):
        """
        Calculates the caution multiplier.
        The multiplier represents damage reduction factor at 40% and less hp.

        Parameters
        ----------
        caut : int, float
            Amount of caution.

        Returns
        -------
        float
        """
        return StatsCalculator._special_stat_multiplier(caut, -4e-4)

    @staticmethod
    def instinct_mult(inst):
        """
        Calculates the instinct multiplier.
        The multiplier represents the critical damage damage reduction factor.

        Parameters
        ----------
        inst : int, float
            Amount of instinct.

        Returns
        -------
        float
        """
        return StatsCalculator._special_stat_multiplier(inst, -4e-4)

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
        return 8e-4 * SpecialSpline(fort) * fort

    @staticmethod
    def crit_randomize(crit_chance):
        """
        Returns True with probability equals to crit_chance and False otherwise.

        Parameters
        ----------
        crit_chance : float
            Probability (from [0, 1]) of critical damage.

        Returns
        -------
        bool

        See also
        --------
        crit_chance_calculate
        """
        rand_max = 10000
        if random.randrange(0, rand_max) < crit_chance * rand_max:
            return True
        return False

    @classmethod
    def stats_number(cls):
        """
        Returns the number of supported offensive stats.

        Returns
        -------
        int
        """
        return cls._n

    @classmethod
    def noncrit_dmg_calculate(cls, stats, *, det_pool, lost_hp=None):
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
    def dmg_calculate(cls, stats, *, det_pool, lost_hp=None, inst=0.0):
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
    def dmg_grad(cls, stats, *, det_pool, lost_hp=None, inst=0.0):
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
    def multi_approximate_optimization(cls, stats_sum, *, det_pool, lost_hp=None,
                                       inst=0.0, lb=None, ub=None, stats0=None):
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
    def optimization(cls, stats_sum, *, det_pool, lost_hp=None,
                     inst=0.0, lb=None, ub=None, stats0=None):
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
    def analytical_optimization(cls, stats_sum, *, det_pool, inst):
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
    @staticmethod
    @functools.lru_cache(maxsize=128)
    def _fortune_cff_calculate(inst):
        inst_mult = 1.0
        if np.abs(inst) >= EPSILON:
            inst_mult = StatsCalculator.instinct_mult(inst)
        return 8e-4 * (3.0 / 2.0 * inst_mult - 1.0)

    @staticmethod
    def _special_stat_multiplier(stat, cff):
        return 1.0 + cff * SpecialSpline(stat) * stat

    @staticmethod
    def _special_stat_multiplier_diff(stat, cff):
        return cff * (SpecialSpline.diff(stat) * stat + SpecialSpline(stat))

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
        n = cls.stats_number()
        coefficients = cls.dmg_grad(np.zeros(n), det_pool=det_pool, inst=inst)
        coefficients[np.abs(coefficients) < EPSILON] = EPSILON
        coefficients, indices = indexed_sort(coefficients, reverse=True)

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
