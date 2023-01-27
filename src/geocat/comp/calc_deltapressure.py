import numpy as np

def _calc_deltapressure_1D(pressure_lev, surface_pressure):
    """Calculates the pressure layer thickness (delta pressure) of a one-dimensional pressure level array.

    Arguments:
    pressure_lev -- The pressure level array. May be in ascending or descending order. Must have the same units as `surface_pressure`.
    surface_pressure -- The scalar surface pressure. Must have the same units as `pressure_lev`.
    
    Returns:
    delta_pressure -- The pressure layer thickness array. Shares dimensions and units of `pressure_lev`.
    """
    pressure_top = min(pressure_lev)

    # safety checks
    try:
        surface_pressure == True
    except AttributeError:
        print("'surface_pressure1 can't equal a missing value.")

    try:
        pressure_top >= 0
    except AttributeError:
        print("'pressure_lev` values must all be positive.")
        
    try:
        pressure_top < surface_pressure
    except AttributeError:
        print("`surface_pressure` must be greater than minimul `pressure_lev` value.")

    # resort so pressure increases (array goes from top of atmosphere to bottom)
    is_pressuredecreasing = pressure_lev[1] < pressure_lev[0]
    if is_pressuredecreasing:
        pressure_lev = np.flip(pressure_lev)

    # calculate delta pressure
    delta_pressure = np.empty_like(pressure_lev)

    delta_pressure[0] = (pressure_lev[0]+pressure_lev[1]) / 2 - pressure_top
    for i in (np.arange(1, len(pressure_lev) - 1)):
        delta_pressure[i] = (pressure_lev[i + 1] - pressure_lev[i - 1]) / 2
        i += 1     
    delta_pressure[-1] = surface_pressure - (pressure_lev[-1] + pressure_lev[-2]) / 2

    # delta pressure sanity check
    try:
        sum(delta_pressure) == surface_pressure - pressure_top
    except ValueError:
        print("The total pressure layer thickens `sum(delta_pressure)` must be equal to the different in surface and top pressures (`surface_pressure - pressure-top`).")

    # return delta_pressure to original order
    if is_pressuredecreasing:
        delta_pressure = np.flip(delta_pressure)

    return delta_pressure


def calc_deltapressure(pressure_lev, surface_pressure):
    """Calculates the pressure layer thickness (delta pressure) of a constant pressure level coordinate system.

    Arguments:
    pressure_lev: The pressure level array. May be in ascending or descending order. Must have the same units as `surface_pressure`.
    surface_pressure -- The scalar or N-dimensional surface pressure array. Must have the same units as `pressure_lev`. Cannot exceed 3 dimensions.
    
    Returns:
    delta_pressure -- The pressure layer thickness array. Shares units with `pressure_lev`. If `surface_pressure` is scalar, shares dimensions with `pressure_level`. If `surface_pressure` is an array than the returned array will have an additional dimension [e.g. (lat, lon, time) becomes (lat, lon, time, lev)].
    """
    # Get dimensions of `surface_pressure`
    if (type(surface_pressure) == np.ndarray):
        dims = len(surface_pressure)
    else:
        dims = 0

    # Attribute check
    try:
        dims <= 3
    except AttributeError:
        print("`surface_pressure` cannot have more than 3 dimensions.")

    # Create array to hold delta pressure values
    if dims == 0:
        delta_pressure_shape = len(pressure_lev)
    else:
        shape = surface_pressure.shape
        delta_pressure_shape = shape + (len(pressure_lev),)
        
    delta_pressure = np.empty(delta_pressure_shape)
    
    # Depending on dimension size, loop through coordinates and calculate delta pressure
    if dims == 0:
        delta_pressure = _calc_deltapressure_1D(pressure_lev, surface_pressure)
    elif dims == 1:
        for i in range(shape[0]):
            delta_pressure_1D = _calc_deltapressure_1D(pressure_lev, surface_pressure[i])
            delta_pressure[i] = delta_pressure_1D
    elif dims == 2:
        for i in range(shape[0]):
            for j in range(shape[1]):
                delta_pressure_1D = _calc_deltapressure_1D(pressure_lev, surface_pressure[i][j])
                delta_pressure[i][j] = delta_pressure_1D
    elif dims == 3: 
        for i in range(shape[0]):
            for j in range(shape[1]):
                for k in range(shape[2]):
                    delta_pressure_1D = _calc_deltapressure_1D(pressure_lev, surface_pressure[i][j][k])
                    delta_pressure[i][j][k] = delta_pressure_1D
    
    return delta_pressure
