from itertools import chain

import metpy.calc as mpcalc
import numpy as np
import pint.quantity
from metpy.units import units


def showalter_index(pressure: pint.Quantity, temperature: pint.Quantity,
                    dewpt: pint.Quantity) -> pint.Quantity:
    """Calculate Showalter Index from pressure temperature and 850 hPa lcl.
    Showalter Index derived from [Galway1956]_:

    shox = T500 - Tp500
    where:
    - T500 is the measured temperature at 500 hPa
    - Tp500 is the temperature of the lifted parcel at 500 hPa

    Parameters
    ----------
    pressure : :class:`pint.Quantity`
        Atmospheric pressure level(s) of interest, in order from highest
        to lowest pressure

    temperature : :class:`pint.Quantity`
        Parcel temperature for corresponding pressure

    dewpt : :class:`pint.Quantity`
        Parcel dew point temperatures for corresponding pressure

    Returns
    -------
    shox : :class:`pint.Quantity`
       Showalter index in delta degrees celsius
    """
    shox = mpcalc.showalter_index(pressure, temperature, dewpt)
    return shox


def get_skewt_vars(p: pint.Quantity, tc: pint.Quantity, tdc: pint.Quantity,
                   pro: pint.Quantity) -> str:
    """This function processes the dataset values and returns a string element
    which can be used as a subtitle to replicate the styles of NCL Skew-T
    Diagrams.

    Parameters
    ----------
    p : :class:`pint.Quantity`
        Pressure level input from dataset

    tc : :class:`pint.Quantity`
        Temperature for parcel from dataset

    tdc : :class:`pint.Quantity`
        Dew point temperature for parcel from dataset

    pro : :class:`pint.Quantity`
        Parcel profile temperature converted to degC

    Returns
    -------
    joined : :class:`str`
        A string element with the format "Plcl=<value> Tlcl[C]=<value> Shox=<value> Pwat[cm]=<value> Cape[J]=<value>" where:

        - Cape  -  Convective Available Potential Energy [J]
        - Pwat  -  Precipitable Water [cm]
        - Shox  -  Showalter Index (stability)
        - Plcl  -  Pressure of the lifting condensation level [hPa]
        - Tlcl  -  Temperature at the lifting condensation level [C]

    See Also
    --------
    Related NCL Functions:
    `skewT_PlotData <https://www.ncl.ucar.edu/Document/Functions/Skewt_func/skewT_PlotData.shtml>`_,
    `skewt_BackGround <https://www.ncl.ucar.edu/Document/Functions/Skewt_func/skewT_BackGround.shtml>_
    """

    # CAPE
    cape = mpcalc.cape_cin(p, tc, tdc, pro)
    cape = cape[0].magnitude

    # Precipitable Water
    pwat = mpcalc.precipitable_water(p, tdc)
    pwat = (pwat.magnitude / 10) * units.cm  # Convert mm to cm
    pwat = pwat.magnitude

    # Pressure and temperature of lcl
    lcl = mpcalc.lcl(p[0], tc[0], tdc[0])
    plcl = lcl[0].magnitude
    tlcl = lcl[1].magnitude

    # Showalter index
    shox = showalter_index(p, tc, tdc)
    shox = shox[0].magnitude

    # Place calculated values in iterable list
    vals = [plcl, tlcl, shox, pwat, cape]
    vals = np.round(vals).astype(int)

    # Define variable names for calculated values
    names = ['Plcl=', 'Tlcl[C]=', 'Shox=', 'Pwat[cm]=', 'Cape[J]=']

    # Combine the list of values with their corresponding labels
    lst = list(chain.from_iterable(zip(names, vals)))
    lst = map(str, lst)

    # Create one large string for later plotting use
    joined = ' '.join(lst)

    return joined
