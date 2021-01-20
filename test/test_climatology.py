import pytest
import xarray as xr
import geocat.comp

dset_a = xr.tutorial.open_dataset("rasm")
dset_b = xr.tutorial.open_dataset("air_temperature")
dset_c = dset_a.copy().rename({"time": "Times"})
dset_encoded = xr.tutorial.open_dataset("rasm", decode_cf=False)


def test_climatology_invalid_freq():
    with pytest.raises(ValueError):
        geocat.comp.climatology(dset_a, "hourly")


def test_climatology_encoded_time():
    with pytest.raises(ValueError):
        geocat.comp.climatology(dset_encoded, "monthly")


@pytest.mark.parametrize("dataset", [dset_a, dset_b, dset_c["Tair"]])
@pytest.mark.parametrize("freq", ["day", "month", "year", "season"])
def test_climatology_setup(dataset, freq):
    computed_dset = geocat.comp.climatology(dataset, freq)
    assert type(dataset) == type(computed_dset)


@pytest.mark.parametrize("dataset", [dset_a, dset_b, dset_c["Tair"]])
@pytest.mark.parametrize("freq", ["day", "month", "year", "season"])
def test_anomaly_setup(dataset, freq):
    computed_dset = geocat.comp.anomaly(dataset, freq)
    assert type(dataset) == type(computed_dset)