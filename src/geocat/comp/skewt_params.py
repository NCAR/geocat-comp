from itertools import chain

import metpy.calc as mpcalc
import numpy as np
from metpy.units import units


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
    shox = mpcalc.showalter_index(p, tc, tdc)
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
