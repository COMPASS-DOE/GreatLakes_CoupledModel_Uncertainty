import chaospy
import pint
from abc import ABC
from enum import Enum
from typing import Union
from pint import Quantity

units = pint.UnitRegistry()


class VariableDistribution(Enum):
    GAUSSIAN = chaospy.Normal
    UNIFORM = chaospy.Uniform
    DISCRETEUNIFORM = chaospy.DiscreteUniform


class Variable(ABC):
    name: str

    def __init__(
        self, unit: pint.Unit = None,
    ):
        self.unit = unit

    @property
    def unit(self) -> pint.Unit:
        return self._unit

    @unit.setter
    def unit(self, unit: Union[str, pint.Unit]):
        if not isinstance(unit, pint.Unit):
            if unit is None:
                unit = ""
            unit = units.Unit(unit)
        self.__unit = unit


class PerturbedVariable(Variable, ABC):
    variable_distribution: VariableDistribution
    lower_bound: Quantity
    upper_bound: Quantity
    mean: Quantity
    standard_deviaton: Quantity

    def __init__(
        self, unit: pint.Unit = None,
    ):
        super().__init__(unit=unit)

    @classmethod
    def chaospy_distribution(self, normalize: bool = False) -> chaospy.Distribution:
        if self.variable_distribution == VariableDistribution.GAUSSIAN:
            if normalize:
                distribution = chaospy.Normal(mu=0.0, sigma=1.0)
            else:
                distribution = chaospy.Normal(mu=self.mean, sigma=self.standard_deviation)
        elif self.variable_distribution == VariableDistribution.UNIFORM:
            if normalize:
                distribution = chaospy.Uniform(lower=0.0, upper=1.0)
            else:
                distribution = chaospy.Uniform(lower=self.lower_bound, upper=self.upper_bound)
        elif self.variable_distribution == VariableDistribution.DISCRETEUNIFORM:
            if normalize:
                distribution = chaospy.DiscreteUniform(lower=0, upper=1)
            else:
                distribution = chaospy.DiscreteUniform(
                    lower=self.lower_bound, upper=self.upper_bound
                )
        else:
            raise ValueError(f'perturbation type {self.variable_distribution} not recognized')

        return distribution

    @classmethod
    def class_from_variable_name(self, name):
        variable_classes = [
            WRF_PBL_SFCLAY,
            WRF_WaterZ0,
            WRF_MP,
            WRF_RA,
            WRF_LM,
            FVCOM_WindStress,
            FVCOM_VerticalMixing,
            FVCOM_Prandtl,
            FVCOM_SWRadiationAbsorption,
        ]
        for variable_class in variable_classes:
            if variable_class.name == name:
                return variable_class
        raise ValueError(f'{name} not recognized')

    @classmethod
    def class_from_scheme_name(self, name):
        variable_classes = [
            WRF_PBL_SFCLAY,
            WRF_WaterZ0,
            WRF_MP,
            WRF_RA,
            WRF_LM,
            FVCOM_WindStress,
            FVCOM_VerticalMixing,
            FVCOM_Prandtl,
            FVCOM_SWRadiationAbsorption,
        ]
        for variable_class in variable_classes:
            if variable_class.variable_distribution == VariableDistribution.DISCRETEUNIFORM:
                for value in range(variable_class.lower_bound, variable_class.upper_bound + 1):
                    scheme_name = variable_class.return_scheme_name(value)
                    if scheme_name == name:
                        return variable_class
            else:
                if variable_class.name == name:
                    return variable_class

        raise ValueError(f'{name} not recognized')


class WRF_PBL_SFCLAY(PerturbedVariable):
    """
    ``WRF_PBL_SFCLAY`` (``WRF planetary bounday layer and surface layer scheme``)
    Discrete uniform distribution on [1,3].
        1: YSU PBL (=1) and revised MM5 SFCLAY (=1)
        2: MYJ PBL (=2) and MOJ SFCLAY (=2)
        3: MYNN2.5 PBL (=5) and MYNN SFCLAY (=5)
    """

    name = 'WRF planetary bounday layer and surface layer scheme'
    variable_distribution = VariableDistribution.DISCRETEUNIFORM
    lower_bound = 1
    upper_bound = 3
    mean = None
    standard_deviation = None

    def __init__(self):
        super().__init__(unit=None)

    @classmethod
    def return_scheme_name(self, value) -> str:
        if value == 1:
            name = 'YSU PBL (=1) and revised MM5 SFCLAY (=1)'
        elif value == 2:
            name = 'MYJ PBL (=2) and MOJ SFCLAY (=2)'
        elif value == 3:
            name = 'MYNN2.5 PBL (=5) and MYNN SFCLAY (=5)'
        return name


class WRF_WaterZ0(PerturbedVariable):
    """
    ``WRF_WaterZ0`` (``WRF surface roughness (z0) scheme over water``)
    Discrete uniform distribution on [1,3].
        1: COARE 3.5 (Edson et al., 2013)
        2: Constant Charnock = 0.0185
        3: Depth Dependent (Jiménez & Dudhia, 2018)
    """

    name = 'WRF surface roughness (z0) scheme over water'
    variable_distribution = VariableDistribution.DISCRETEUNIFORM
    lower_bound = 1
    upper_bound = 3
    mean = None
    standard_deviation = None

    def __init__(self):
        super().__init__(unit=None)

    @classmethod
    def return_scheme_name(self, value) -> str:
        if value == 1:
            name = 'COARE 3.5 (Edson et al., 2013)'
        elif value == 2:
            name = 'Constant Charnock = 0.0185'
        elif value == 3:
            name = 'Depth Dependent (Jiménez & Dudhia, 2018)'
        return name


class WRF_MP(PerturbedVariable):
    """
    ``WRF_MP`` (``WRF microphysics scheme``)
    Discrete uniform distribution on [1,3].
        1: Morrison 2-moment 6-class (=10)
        2: Thompson 2-moment 6-class (=8)
        3: WSM 6-class graupel scheme (=6)
    """

    name = 'WRF microphysics scheme'
    variable_distribution = VariableDistribution.DISCRETEUNIFORM
    lower_bound = 1
    upper_bound = 3
    mean = None
    standard_deviation = None

    def __init__(self):
        super().__init__(unit=None)

    @classmethod
    def return_scheme_name(self, value) -> str:
        if value == 1:
            name = 'Morrison 2-moment 6-class (=10)'
        elif value == 2:
            name = 'Thompson 2-moment 6-class (=8)'
        elif value == 3:
            name = 'WSM 6-class graupel scheme (=6)'
        return name


class WRF_RA(PerturbedVariable):
    """
    ``WRF_RA`` (``WRF radiation scheme``)
    Discrete uniform distribution on [1,3].
        1: RRTM longwave and CAM shortwave (=1,3)
        2: RRTMG longwave and shortwave (=4,4)
        3: Goddard longwave and shortwave (=5,5)
    """

    name = 'WRF radiation scheme'
    variable_distribution = VariableDistribution.DISCRETEUNIFORM
    lower_bound = 1
    upper_bound = 3
    mean = None
    standard_deviation = None

    def __init__(self):
        super().__init__(unit=None)

    @classmethod
    def return_scheme_name(self, value) -> str:
        if value == 1:
            name = 'RRTM longwave and CAM shortwave (=1,3)'
        elif value == 2:
            name = 'RRTMG longwave and shortwave (=4,4)'
        elif value == 3:
            name = 'Goddard longwave and shortwave (=5,5)'
        return name


class WRF_LM(PerturbedVariable):
    """
    ``WRF_LM`` (``WRF land model (surface physics)``)
    Discrete uniform distribution on [1,2].
        1: Noah (=2)
        2: Noah MP (=4)
    """

    name = 'WRF land model (surface physics)'
    variable_distribution = VariableDistribution.DISCRETEUNIFORM
    lower_bound = 1
    upper_bound = 2
    mean = None
    standard_deviation = None

    def __init__(self):
        super().__init__(unit=None)

    @classmethod
    def return_scheme_name(self, value) -> str:
        if value == 1:
            name = 'Noah (=2)'
        elif value == 2:
            name = 'Noah MP (=4)'
        return name


class FVCOM_VerticalMixing(PerturbedVariable):
    """
    ``FVCOM_VerticalMixing`` (``FVCOM vertical mixing scheme``)
    Discrete uniform distribution on [1,2].
        1: FVCOM MY-2.5 for Great Lakes (=1)
        2: GOTM k-epsilon (=2)
    """

    name = 'FVCOM vertical mixing scheme'
    variable_distribution = VariableDistribution.DISCRETEUNIFORM
    lower_bound = 1
    upper_bound = 2
    mean = None
    standard_deviation = None

    def __init__(self):
        super().__init__(unit=None)

    @classmethod
    def return_scheme_name(self, value) -> str:
        if value == 1:
            name = 'FVCOM MY-2.5 for Great Lakes'
        elif value == 2:
            name = 'GOTM k-epsilon'
        return name


class FVCOM_WindStress(PerturbedVariable):
    """
    ``FVCOM_WindStress`` (``FVCOM bulk wind stress formulation``)
    Discrete uniform distribution on [1,2].
        1: Large and Pond (1981)
        2: Andreas et al. (2012)
    """

    name = 'FVCOM bulk wind stress formulation'
    variable_distribution = VariableDistribution.DISCRETEUNIFORM
    lower_bound = 1
    upper_bound = 2
    mean = None
    standard_deviation = None

    def __init__(self):
        super().__init__(unit=None,)

    @classmethod
    def return_scheme_name(self, value) -> str:
        if value == 1:
            name = 'Large and Pond (1981)'
        elif value == 2:
            name = 'Andreas et al. (2012)'
        return name


class FVCOM_SWRadiationAbsorption(PerturbedVariable):
    """
    ``FVCOM_SWRadiationAbsorption`` (``FVCOM shortwave radiation absorption: R``)
    Uniform distribution on R = [0.74,0.78], then with alpha = (0.74-R)/0.04 -> [0,-1]:
    Z1 = 1.7 + 0.3*alpha -> [1.7,1.4] [m]
    Z2 = 16 + 9.7*alpha -> [16,6.3] [m]

    Upper bound is the values given in FVCOM User Manual which is low water quality type III,
    and lower bound is from Paulson & Simpson (1977). https://doi.org/10.1175/1520-0485(1977)007<0952:IMITUO>2.0.CO;2
    which corresponds to type IA (pretty clear water).

    Definitions:
    R: HEATING_LONGWAVE_PERCTAGE, The fraction of the total shortwave flux associated with the longer wavelength irradiance
    Z1: HEATING_LONGWAVE_LENGTHSCALE, The attenuation depth for the longer wavelength component of shortwave irradiance
    Z2: HEATING_SHORTWAVE_LENGTHSCALE, The attenuation depth for shorter wavelength component of shortwave irradiance
    """

    name = 'FVCOM shortwave radiation absorption: R'
    variable_distribution = VariableDistribution.UNIFORM
    lower_bound = 0.74
    upper_bound = 0.78
    mean = None
    standard_deviation = None

    def __init__(self):
        super().__init__(unit=None)

    @classmethod
    def calc_Z1(self, R) -> float:
        alpha = (0.74 - R) / 0.04
        Z1 = 1.7 + 0.3 * alpha
        return Z1  # [m]

    @classmethod
    def calc_Z2(self, R) -> float:
        alpha = (0.74 - R) / 0.04
        Z2 = 16 + 9.7 * alpha
        return Z2  # [m]


class FVCOM_Prandtl(PerturbedVariable):
    """
    ``FVCOM_Prandtl`` (``FVCOM Prandlt Number``)
    Uniform distribution on [1,10] of the horizontal=vertical Prandtl number.

    IMPORTANT NOTE: Traditionally Pr = momentum/thermal diffusion,
    which we use here. However FVCOM implements the inverse (1/Pr),
    so user will need to enter 1/Pr in the FVCOM namelist file.
    e.g., Pr = 10, enter:
    HORIZONTAL_PRANDTL_NUMBER =  0.1,
    ...
    VERTICAL_PRANDTL_NUMBER =  0.1,

    """

    name = 'FVCOM Prandtl'
    variable_distribution = VariableDistribution.UNIFORM
    lower_bound = 1.0
    upper_bound = 10.0
    mean = None
    standard_deviation = None

    def __init__(self):
        super().__init__(unit=None)
