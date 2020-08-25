import functools

import numpy as np
import scipy.optimize as opt
import scipy.interpolate as itp

EPSILON = 1e-8
""" Numbers with absolute value less than EPSILON will be considered as zeros. """

DEFAULT_DET_POOL = 0.95
""" The default pool of determination. """


class Brutality:
    """
    Provides methods which are required to calculate brutality multiplier and its derivative.
    Nonlinear dependencies are approximated by parametric regression.

    Does not imply instantiation.
    """

    class L:
        """
        Approximating 'l(brut)' with nonlinear regression.
        The parametrized explicit form of 'l(brut)' is chosen as
        'c * ln(a*brut + b) * brut'.
        """

        def __init__(self, x, y, p0=None):
            p_opt, p_cov = opt.curve_fit(self.call_p, x, y, p0=p0, jac=self.grad_p, maxfev=5000)
            self.params = dict(zip(('a', 'b', 'c'), p_opt))

        def __call__(self, brut):
            return self.call_p(brut, **self.params)

        def diff(self, brut):
            return self.diff_p(brut, **self.params)

        @staticmethod
        def call_p(brut, a, b, c):
            """
            Calculates the value of l(brut).
            """
            return c * np.log(a * brut + b) * brut

        @staticmethod
        def diff_p(brut, a, b, c):
            """
            Calculates the derivative of l(brut) by brutality.
            """
            lin_exp = a * brut + b
            return c * (np.log(lin_exp) + a * brut / lin_exp)

        @staticmethod
        def grad_p(brut, a, b, c):
            """
            Calculates the gradient of l(brut) by parameters.
            """
            ex = a * brut + b
            diff_c = np.log(ex) * brut
            diff_b = c * brut / ex
            diff_a = diff_b * brut
            return np.array([diff_a, diff_b, diff_c], copy=False).T

    _x = [0.0, 25.0, 50.0, 75.0, 100.0, 125.0, 150.0, 175.0, 200.0, 225.0,
          250.0, 275.0, 300.0, 325.0, 350.0, 375.0, 400.0, 425.0, 450.0, 475.0,
          500.0, 525.0, 550.0, 575.0, 600.0, 625.0, 650.0, 675.0, 700.0, 725.0,
          750.0, 775.0, 800.0, 825.0, 850.0, 875.0, 900.0, 925.0, 950.0, 975.0,
          1000.0, 1025.0, 1050.0, 1075.0, 1100.0, 1125.0, 1150.0, 1175.0, 1200.0, 1225.0,
          1250.0, 1275.0, 1300.0, 1325.0, 1350.0, 1375.0, 1400.0, 1425.0, 1450.0, 1475.0,
          1500.0, 1525.0, 1550.0, 1575.0, 1600.0, 1625.0, 1650.0, 1675.0, 1700.0, 1725.0,
          1750.0, 1775.0, 1800.0, 1825.0, 1850.0, 1875.0, 1900.0, 1925.0, 1950.0, 1975.0, 2000.0]

    _y = [0.0, 0.025, 0.05, 0.076, 0.102, 0.128, 0.154, 0.18, 0.206, 0.233,
          0.26, 0.287, 0.314, 0.342, 0.369, 0.397, 0.425, 0.453, 0.482, 0.51,
          0.539, 0.567, 0.596, 0.626, 0.655, 0.684, 0.714, 0.744, 0.774, 0.804,
          0.834, 0.864, 0.895, 0.925, 0.956, 0.987, 1.02, 1.05, 1.08, 1.11,
          1.14, 1.18, 1.21, 1.24, 1.27, 1.3, 1.34, 1.37, 1.4, 1.44,
          1.47, 1.5, 1.53, 1.57, 1.6, 1.63, 1.67, 1.7, 1.74, 1.77,
          1.81, 1.84, 1.87, 1.91, 1.94, 1.98, 2.01, 2.05, 2.08, 2.12,
          2.15, 2.19, 2.23, 2.26, 2.3, 2.33, 2.37, 2.4, 2.44, 2.48, 2.51]

    _p0 = [2.031e-3, 6.575, 5.315e-4]

    _l = L(_x, _y, _p0)

    @staticmethod
    def l_calculate(brut):
        """
        Calculates 'l(brut)'.

        Parameters
        ----------
        brut : int, float
            Amount of brutality.

        Returns
        -------
        float

        Notes
        -----
        'l(brut)' is such function that brutality multiplier can be calculated
        as 1 + l(brut) * HP_lost/HP_max.
        """
        return Brutality._l(brut)

    @staticmethod
    def l_diff(brut):
        """
        Calculates the derivative of 'l(brut)'.

        Parameters
        ----------
        brut : int, float
            Amount of brutality.

        Returns
        -------
        float

        See also
        --------
        l_calculate
        """
        return Brutality._l.diff(brut)

    @classmethod
    def brutality_multiplier(cls, brut, lost_hp=None):
        """
        Calculates the brutality multiplier.

        Parameters
        ----------
        brut : int, float
            Amount of brutality.
        lost_hp : float, optional
            Normalized value of lost hp from [0, 1] interval, if 'lost_hp' is None,
            the multiplier is calculated for 'lost_hp = gamma(brut)'.

        Returns
        -------
        float

        Notes
        -----
        Brutality multiplier means that if without brutality you wil deal 'D' damage
        then having 'brut' brutality you will deal 'D * brutality_multiplier(brut, lost_hp)'
        damage to target with '(1-lost_hp)*100%' percents health.

        See also
        --------
        gamma
        """
        if np.abs(brut) < EPSILON:
            return 1.0
        l_val = cls.l_calculate(brut)
        if lost_hp is None:
            return l_val / np.log(1.0 + l_val)
        else:
            return 1.0 + l_val * lost_hp

    @classmethod
    def brutality_multiplier_diff(cls, brut, lost_hp=None):
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
        gamma
        brutality_multiplier
        """
        l_diff = cls.l_diff(brut)
        if lost_hp is None:
            if np.abs(brut) < EPSILON:
                return l_diff / 2.0
            l_val = cls.l_calculate(brut)
            lg = np.log(l_val + 1.0)
            return (1.0 / lg - l_val / (l_val + 1.0) / lg ** 2.0) * l_diff
        else:
            return l_diff * lost_hp

    @classmethod
    def gamma(cls, brut):
        """
        Calculates the value of 'gamma(brut)'.

        Parameters
        ----------
        brut : int, float
            Amount of brutality.

        Returns
        -------
        float

        Notes
        -----
        gamma(brut) is such function that brutality multiplier equals
        1 + l(brut) * gamma(brut) following a full solo boss fight.
        It may be considered as 'mean value' of target's lost hp.

        See also
        --------
        brutality_multiplier
        l_calculate
        """
        if np.abs(brut) < EPSILON:
            return 0.5
        l_val = cls.l_calculate(brut)
        return 1.0 / np.log(1.0 + l_val) - 1.0 / l_val


class SpecialStats:
    """
    Provides methods which are required to calculate the special stats multipliers
    and their derivatives.
    Nonlinear dependencies are approximated with spline interpolation.

    Does not imply instantiation.
    """

    class Spline:
        """
        Spline approximating the special stats diminishing coefficient.

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

        class BSpline:
            def __init__(self, x, y, degree=5, smoothness=1.0):
                self._spline = itp.splrep(x, y, k=degree, s=smoothness)

            def __call__(self, x):
                return itp.splev(x, self._spline)

            def diff(self, x):
                return itp.splev(x, self._spline, der=1)

        class HyperbolicTail:
            """ The quasi - hyperbolic extrapolation, y = 1 / (a*x + b*ln(x) + c). """

            def __init__(self, x, y, y_diff, inf_lim=8e-4):
                """
                Tail(x) * x -> 'inf_lim', x -> inf.
                The default value is chosen so that crit chance -> 1 as fortune -> inf.
                """
                self._a = inf_lim
                self._b = -x * (y_diff / y ** 2.0 + self._a)
                self._c = 1 / y - self._a * x - self._b * np.log(x)

            def __call__(self, x):
                return 1.0 / (self._a * x + self._b * np.log(x) + self._c)

            def diff(self, x):
                return -(self._a + self._b / x) * self(x) ** 2.0

        class SqrtTail:
            """ The reciprocal square root extrapolation, y = a / sqrt(x - b). """

            def __init__(self, x, y, y_diff):
                self._a = np.sqrt(-0.5 * y ** 3.0 / y_diff)
                self._b = x - self._a ** 2.0 / y ** 2.0

            def __call__(self, x):
                return self._a / np.sqrt(x - self._b)

            def diff(self, x):
                return -0.5 / (x - self._b) * self(x)

        class ExponentialTail:
            """ The exponential extrapolation, y = a^(x - b). """

            def __init__(self, x, y, y_diff):
                self._a = np.exp(y_diff / y)
                self._b = x - np.log(y) / np.log(self._a)

            def __call__(self, x):
                return np.power(self._a, x - self._b)

            def diff(self, x):
                return np.log(self._a) * self(x)

        class LinearTail:
            """ The linear extrapolation, y = a*x + b. """

            def __init__(self, x, y, y_diff):
                self._a = y_diff
                self._b = y - self._a * x

            def __call__(self, x):
                return self._a * x + self._b

            def diff(self, x=0):
                return self._a

        # The several last table values are ignored in order to stabilize and smooth the spline
        last_values_cutoff = 2

        def __init__(self, x, y, tail_class=HyperbolicTail):
            self._x = x[:-self.last_values_cutoff]
            self._y = y[:-self.last_values_cutoff]
            self._b_spline = self.BSpline(x, y)

            x = self._x[0:2]
            y = [self._y[0], self._b_spline(self._x[1])]
            bc_type = ((1, 0.0), (1, self._b_spline.diff(self._x[1])))
            self._cubic_spline = itp.CubicSpline(x, y, bc_type=bc_type)
            self._cubic_spline.diff = functools.partial(self._cubic_spline.__call__, nu=1)

            x_last = self._x[-1]
            self._tail = tail_class(x_last, self._b_spline(x_last), self._b_spline.diff(x_last))

        @property
        def linearity_end(self):
            """
            The point of joint between constant and nonlinear part of spline.

            Returns
            -------
            float
            """
            return self._x[0]

        def __call__(self, x):
            """
            Calculates the value of spline at x.

            Parameters
            ----------
            x : float

            Returns
            -------
            float
            """
            if x < self.linearity_end:
                return self._y[0]
            if self.linearity_end <= x < self._x[1]:
                return self._cubic_spline(x)
            if x >= self._x[-1]:
                return self._tail(x)
            return self._b_spline(x)

        def diff(self, x):
            """
            Calculates the derivative of spline at x.

            Parameters
            ----------
            x : float

            Returns
            -------
            float
            """
            if x < self.linearity_end:
                return 0.0
            if self.linearity_end <= x < self._x[1]:
                return self._cubic_spline.diff(x)
            if x >= self._x[-1]:
                return self._tail.diff(x)
            return self._b_spline.diff(x)

    _x = (750.0, 757.0, 770.0, 778.0, 783.0, 800.0, 808.0, 813.0, 818.0, 831.0,
          841.0, 844.0, 855.0, 857.0, 865.0, 916.0, 921.0, 955.0, 963.0, 965.0,
          979.0, 991.0, 995.0, 1024.0, 1067.0, 1218.0, 1370.0, 1521.0)

    _y = (1.0, 0.9997, 0.9985, 0.9962, 0.9951, 0.9898, 0.9861, 0.9843, 0.9823, 0.9760,
          0.9710, 0.9692, 0.9636, 0.9623, 0.9579, 0.9287, 0.9260, 0.9063, 0.9017, 0.9004,
          0.8924, 0.8854, 0.8832, 0.8674, 0.8445, 0.7739, 0.7160, 0.6672)

    _spline = Spline(_x, _y, Spline.HyperbolicTail)

    @staticmethod
    def get_diminishing_threshold():
        """
        Returns the special stat amount starting from which
        the spline ceases to be linear and special stat efficiency decreases.

        Returns
        -------
        float
        """
        return SpecialStats._spline.linearity_end

    @staticmethod
    def spline(x):
        """
        Returns the value of spline at x.

        Parameters
        ----------
        x : int, float

        Returns
        -------
        float
        """
        return SpecialStats._spline(x)

    @staticmethod
    def spline_diff(x):
        """
        Returns the derivative of spline at x.

        Parameters
        ----------
        x : int, float

        Returns
        -------
        float

        See Also
        --------
        spline
        """
        return SpecialStats._spline.diff(x)

    @classmethod
    def special_stat_multiplier(cls, stat, cff):
        """
        Calculates the abstract special stat multiplier.
        The explicit form of multiplier is '1 + cff * spline(stat) * stat'.

        Parameters
        ----------
        stat : int, float
            Amount of special stat.
        cff :
            Coefficient of special stat.

        Returns
        -------
        float
        """
        if stat <= cls.get_diminishing_threshold():
            return 1.0 + cff * stat
        return 1.0 + cff * cls.spline(stat) * stat

    @classmethod
    def special_stat_multiplier_diff(cls, stat, cff):
        """
        Calculates the derivative of the abstract special stat multiplier.

        Parameters
        ----------
        stat : int, float
            Amount of special stat.
        cff :
            Coefficient of special stat.

        Returns
        -------
        float

        See Also
        --------
        special_stat_multiplier
        """
        if stat <= cls.get_diminishing_threshold():
            return cff
        return cff * (cls.spline_diff(stat) * stat + cls.spline(stat))
