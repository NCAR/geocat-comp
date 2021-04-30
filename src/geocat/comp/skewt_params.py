from itertools import chain

import metpy.calc as mpcalc
import numpy as np
from metpy.units import units


def showalter_index(pressure, temperature, dewpt):
    """Calculate Showalter Index from pressure temperature and 850 hPa lcl.

    Showalter Index derived from [Galway1956]_:
    SI = T500 - Tp500

    where:
    T500 is the measured temperature at 500 hPa
    Tp500 is the temperature of the lifted parcel at 500 hPa

    Parameters
    ----------

        pressure : `pint.Quantity`
            Atmospheric pressure level(s) of interest, in order from highest
            to lowest pressure

        temperature : `pint.Quantity`
            Parcel temperature for corresponding pressure

        dewpt (:class: `pint.Quantity`):
            Parcel dew point temperatures for corresponding pressure


     Returns
     -------

     `pint.Quantity`
        Showalter index in delta degrees celsius
    """

    # find the measured temperature and dew point temperature at 850 hPa.
    idx850 = np.where(pressure == 850 * units.hPa)
    T850 = temperature[idx850]
    Td850 = dewpt[idx850]

    # find the parcel profile temperature at 500 hPa.
    idx500 = np.where(pressure == 500 * units.hPa)
    Tp500 = temperature[idx500]

    # Calculate lcl at the 850 hPa level
    lcl_calc = mpcalc.lcl(850 * units.hPa, T850[0], Td850[0])
    lcl_calc = lcl_calc[0]

    # Define start and end heights for dry and moist lapse rate calculations
    p_strt = 1000 * units.hPa
    p_end = 500 * units.hPa

    # Calculate parcel temp when raised dry adiabatically from surface to lcl
    dl = mpcalc.dry_lapse(lcl_calc, temperature[0], p_strt)
    dl = (dl.magnitude - 273.15) * units.degC  # Change units to C

    # Calculate parcel temp when raised moist adiabatically from lcl to 500mb
    ml = mpcalc.moist_lapse(p_end, dl, lcl_calc)

    # Calculate the Showalter index
    shox = Tp500 - ml
    return shox


def get_skewt_vars(p, tc, tdc, pro):
    """This function processes the dataset values and returns a string element
    which can be used as a subtitle to replicate the styles of NCL Skew-T
    Diagrams.

    Args:

        p (:class: `pint.quantity.build_quantity_class.<locals>.Quantity`):
            Pressure level input from dataset

        tc (:class: `pint.quantity.build_quantity_class.<locals>.Quantity`):
            Temperature for parcel from dataset

        tdc (:class: `pint.quantity.build_quantity_class.<locals>.Quantity`):
            Dew point temperature for parcel from dataset

        pro (:class: `pint.quantity.build_quantity_class.<locals>.Quantity`):
            Parcel profile temperature converted to degC


    Returns:
        :class: 'str'
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
