import geocat.datafiles as gdf
import numpy as np
import numpy.testing as nt
import xarray as xr
import pytest
import pint

from geocat.comp import (
    interp_multidim,
    interp_hybrid_to_pressure,
    interp_sigma_to_hybrid,
    pressure_at_hybrid_levels,
    delta_pressure_hybrid,
)

# Global input data

# Open the netCDF data file "atmos.nc" and read in common variables
try:
    ds_atmos = xr.open_dataset(gdf.get("netcdf_files/atmos.nc"), decode_times=False)
except Exception:
    ds_atmos = xr.open_dataset("test/atmos.nc", decode_times=False)

_hyam = ds_atmos.hyam
_hybm = ds_atmos.hybm
_p0 = 1000.0 * 100  # Pa


@pytest.fixture(scope="module")
def ds_ccsm():
    # Open the netCDF data file with the input data
    try:
        return xr.open_dataset(
            gdf.get("netcdf_files/ccsm35.h0.0021-01.demo.nc"), decode_times=False
        )
    except Exception:
        return xr.open_dataset("test/ccsm35.h0.0021-01.demo.nc", decode_times=False)


class Test_delta_pressure_hybrid:
    @pytest.fixture(scope="class")
    def dph_out(self):
        # output from test_dpres_hybrid_ccm.ncl from atmos.nc
        try:
            return xr.open_dataarray("dpres_hybrid_ccm_output.nc")
        except Exception:
            return xr.open_dataarray("test/dpres_hybrid_ccm_output.nc")

    def test_delta_pressure_correctness(self, dph_out):
        ps = ds_atmos.PS[0, :, :].drop('time')
        dph = delta_pressure_hybrid(ps, _hyam, _hybm, _p0)
        dph_np = delta_pressure_hybrid(ps.values, _hyam.values, _hybm.values, _p0)

        xr.testing.assert_allclose(dph, dph_out, rtol=1e-5)
        nt.assert_allclose(dph.values, dph_np)

    def test_delta_pressure_hybrid_input_validation(self):
        ps = ds_atmos.PS[0, :, :].drop('time')

        # non dataarray or np ndarray input
        with pytest.raises(TypeError):
            delta_pressure_hybrid(ps.to_dataset(), _hyam, _hybm, _p0)

        # non scalar p0
        with pytest.raises(TypeError):
            delta_pressure_hybrid(ps, _hyam, _hybm, [_p0, _p0])

        # type(hya) != type(hyb)
        with pytest.raises(TypeError):
            delta_pressure_hybrid(ps, _hyam, _hybm.values, _p0)

        # hya.shape != hyb.shape
        with pytest.raises(ValueError):
            delta_pressure_hybrid(ps, _hyam[:-1], _hybm, _p0)

        # hya not 1D
        with pytest.raises(ValueError):
            delta_pressure_hybrid(ps, np.array([_hyam, _hyam]), _hybm.values, _p0)

        psnp = delta_pressure_hybrid(ps.values, _hyam, _hybm, _p0)
        assert isinstance(psnp, np.ndarray)

        psxr = delta_pressure_hybrid(ps, _hyam.values, _hybm.values, _p0)
        assert isinstance(psxr, xr.DataArray)
        assert psxr.attrs["long_name"] == "pressure layer thickness"

    def test_delta_pressure_bad_dims(self, p_out) -> None:
        dph = delta_pressure_hybrid(p_out.ps, p_out.hyam, p_out.hybm)

        # check to make sure the dimensions are what we expect
        assert list(dph.shape) == [len(p_out.hyam) - 1] + list(p_out.ps.shape)

    def test_delta_pressure_time_dim(self):
        ps = (
            ds_atmos.PS[0, :, :].drop('time').isel(lat=slice(20, 23), lon=slice(60, 63))
        )
        t = [1, 2]
        ps = ps.expand_dims({'time': t})

        hya = _hyam.isel(lev=slice(20, 25))
        hyb = _hybm.isel(lev=slice(20, 25))

        dph = delta_pressure_hybrid(ps, hya, hyb)

        # check to make sure the dimensions are what we expect
        assert list(dph.shape) == [len(hya) - 1] + list(ps.shape)


class Test_interp_hybrid_to_pressure:
    # Sample input data
    data = ds_atmos.U[0, :, :, :]
    ps = ds_atmos.PS
    pres3d = np.asarray([1000, 950, 800, 700, 600, 500, 400, 300, 200])  # mb
    pres3d = pres3d * 100  # mb to Pa

    def test_pressure_at_hybrid_levels(self, p_out) -> None:
        # np input
        pm = pressure_at_hybrid_levels(
            self.ps[0, :7, :7].values, p_out.hyam.values, p_out.hybm.values
        )
        nt.assert_allclose(pm, p_out.pm.values, rtol=1e-6)

        # xr input
        pm = pressure_at_hybrid_levels(self.ps[0, :7, :7], p_out.hyam, p_out.hybm)
        nt.assert_allclose(pm, p_out.pm, rtol=1e-6)

    def test_pressure_at_hybrid_levels_validation(self, p_out) -> None:
        # mismatched input types
        with pytest.raises(TypeError):
            pressure_at_hybrid_levels(self.ps[0, :7, :7].values, p_out.hyam, p_out.hybm)

        # mismatched dim names
        hya = p_out.hyam.rename({p_out.hyam.dims[0]: '1234'})  # explicit dim rename
        with pytest.warns(UserWarning):
            pm_mismatch = pressure_at_hybrid_levels(self.ps[0, :7, :7], hya, p_out.hybm)
        nt.assert_allclose(pm_mismatch, p_out.pm, rtol=1e-6)

    def test_pressure_at_hybrid_levels_2d(self, p_out) -> None:
        # xr w/ time dim on hya/hyb w/ badly named vert dim
        nt = 2
        hya_t = p_out.hyam.expand_dims(dim={"time": nt})
        hyb_t = p_out.hybm.expand_dims(dim={"time": nt})
        ps_t = self.ps[0, :7, :7].drop_vars('time').expand_dims(dim={"time": nt})
        out = pressure_at_hybrid_levels(ps_t, hya_t, hyb_t)
        assert out.shape == (nt, len(p_out.hyam), len(ps_t.lat), len(ps_t.lon))

        # numpy version of above, require xr inputs for > 1D hya/hyb
        with pytest.raises(ValueError):
            pressure_at_hybrid_levels(ps_t.values, hya_t.values, hyb_t.values)

    def test_interp_hybrid_to_pressure_atmos(self, vinth2p_output) -> None:
        u_int = interp_hybrid_to_pressure(
            self.data,
            self.ps[0, :, :],
            _hyam,
            _hybm,
            p0=_p0,
            new_levels=self.pres3d,
            method="log",
        )

        uzon = u_int.mean(dim='lon')

        nt.assert_array_almost_equal(vinth2p_output.uzon, uzon, 5)

    def test_interp_hybrid_to_pressure_multidim(self) -> None:
        nt = 2
        data = self.data.drop_vars('time').expand_dims(dim={"time": nt})
        ps = self.ps[0, :, :].drop_vars('time').expand_dims(dim={"time": nt})
        hya = _hyam.expand_dims(dim={"time": nt})
        hyb = _hybm.expand_dims(dim={"time": nt})
        out = interp_hybrid_to_pressure(
            data, ps, hya, hyb, p0=_p0, new_levels=self.pres3d, method="log"
        )
        assert out.shape == (
            nt,
            len(self.pres3d),
            len(self.data.lat),
            len(self.data.lon),
        )

    def test_interp_hybrid_to_pressure_atmos_pint(self, vinth2p_output) -> None:
        unit = pint.UnitRegistry()
        u_int = interp_hybrid_to_pressure(
            self.data * unit.meter / unit.second,
            self.ps[0, :, :] * unit.pascal,
            _hyam,
            _hybm,
            p0=_p0 * unit.pascal,
            new_levels=self.pres3d * unit.pascal,
            method="log",
        )

        uzon = u_int.mean(dim='lon')

        assert isinstance(uzon.data, pint.Quantity)
        nt.assert_array_almost_equal(vinth2p_output.uzon, uzon, 5)

    def test_interp_hybrid_to_pressure_atmos_4d(self, vinth2p_output) -> None:
        data_t = self.data.expand_dims("time")

        u_int = interp_hybrid_to_pressure(
            data_t, self.ps, _hyam, _hybm, p0=_p0, new_levels=self.pres3d, method="log"
        )

        uzon = u_int.mean(dim='lon')

        uzon_expected_t = vinth2p_output.uzon.expand_dims("time")
        nt.assert_array_almost_equal(uzon_expected_t, uzon, 5)

    def test_interp_hybrid_to_pressure_atmos_wrong_method(self) -> None:
        with pytest.raises(ValueError):
            interp_hybrid_to_pressure(
                self.data,
                self.ps[0, :, :],
                _hyam,
                _hybm,
                p0=_p0,
                new_levels=self.pres3d,
                method="wrong_method",
            )


class Test_interp_hybrid_to_pressure_extrapolate:
    @pytest.fixture(scope="class")
    def ds_out(self):
        # Open the netCDF file with the output data from running vinth2p_ecmwf.ncl
        try:
            return xr.open_dataset("test/vinth2p_ecmwf_output.nc", decode_times=False)
        except Exception:
            return xr.open_dataset("vinth2p_ecmwf_output.nc", decode_times=False)

    @pytest.fixture(scope="class")
    def _hyam(self, ds_ccsm):
        return ds_ccsm.hyam

    @pytest.fixture(scope="class")
    def _hybm(self, ds_ccsm):
        return ds_ccsm.hybm

    @pytest.fixture(scope="class")
    def temp_in(self, ds_ccsm):
        return ds_ccsm.T[:, :, :3, :2]

    @pytest.fixture(scope="class")
    def t_bot(self, ds_ccsm):
        return ds_ccsm.TS[:, :3, :2]

    @pytest.fixture(scope="class")
    def geopotential_in(self, ds_ccsm):
        return ds_ccsm.Z3[:, :, :3, :2]

    @pytest.fixture(scope="class")
    def humidity_in(self, ds_ccsm):
        return ds_ccsm.Q[:, :, :3, :2] * 1000  # g/kg

    @pytest.fixture(scope="class")
    def press_in(self, ds_ccsm):
        return ds_ccsm.PS[:, :3, :2]

    @pytest.fixture(scope="class")
    def phis(self, ds_ccsm):
        return ds_ccsm.PHIS[:, :3, :2]

    new_levels = np.asarray([500, 925, 950, 1000])
    new_levels *= 100  # new levels in Pa
    _p0 = 1000 * 100  # reference pressure in Pa

    def test_interp_hybrid_to_pressure_interp_temp(
        self, temp_in, press_in, _hyam, _hybm, ds_out
    ) -> None:
        result = interp_hybrid_to_pressure(
            temp_in,
            press_in,
            _hyam,
            _hybm,
            p0=self._p0,
            new_levels=self.new_levels,
            method="linear",
        )
        result = result.transpose('time', 'plev', 'lat', 'lon')
        result = result.assign_coords(dict(plev=self.new_levels / 100))
        temp_interp_expected = ds_out.Tp.rename(lev_p='plev')
        xr.testing.assert_allclose(temp_interp_expected, result)

    def test_interp_hybrid_to_pressure_extrap_temp(
        self, temp_in, press_in, _hyam, _hybm, t_bot, phis, ds_out
    ) -> None:
        result = interp_hybrid_to_pressure(
            temp_in,
            press_in,
            _hyam,
            _hybm,
            p0=self._p0,
            new_levels=self.new_levels,
            method="linear",
            extrapolate=True,
            variable='temperature',
            t_bot=temp_in.isel(lev=-1, drop=True),
            phi_sfc=phis,
        )
        result = result.transpose('time', 'plev', 'lat', 'lon')
        result = result.assign_coords(dict(plev=self.new_levels / 100))
        temp_extrap_expected = ds_out.Tpx.rename(lev_p='plev')
        xr.testing.assert_allclose(temp_extrap_expected, result)

    def test_interp_hybrid_to_pressure_extrap_geopotential(
        self, geopotential_in, press_in, _hyam, _hybm, t_bot, phis, ds_out
    ) -> None:
        result = interp_hybrid_to_pressure(
            geopotential_in,
            press_in,
            _hyam,
            _hybm,
            p0=self._p0,
            new_levels=self.new_levels,
            method="linear",
            extrapolate=True,
            variable='geopotential',
            t_bot=t_bot,
            phi_sfc=phis,
        )
        result = result.transpose('time', 'plev', 'lat', 'lon')
        result = result.assign_coords(dict(plev=self.new_levels / 100))
        geopotential_extrap_expected = ds_out.Zpx.rename(lev_p='plev')
        xr.testing.assert_allclose(geopotential_extrap_expected, result)

    def test_interp_hybrid_to_pressure_extrap_other(
        self, humidity_in, press_in, _hyam, _hybm, t_bot, phis, ds_out
    ) -> None:
        result = interp_hybrid_to_pressure(
            humidity_in,
            press_in,
            _hyam,
            _hybm,
            p0=self._p0,
            new_levels=self.new_levels,
            method="linear",
            extrapolate=True,
            variable='other',
            t_bot=t_bot,
            phi_sfc=phis,
        )
        result = result.transpose('time', 'plev', 'lat', 'lon')
        result = result.assign_coords(dict(plev=self.new_levels / 100))
        humidity_extrap_expected = ds_out.Qpx.rename(lev_p='plev')
        xr.testing.assert_allclose(humidity_extrap_expected, result)

    def test_interp_hybrid_to_pressure_extrap_kwargs(
        self, humidity_in, press_in, _hyam, _hybm
    ) -> None:
        with pytest.raises(ValueError):
            interp_hybrid_to_pressure(
                humidity_in,
                press_in,
                _hyam,
                _hybm,
                p0=self._p0,
                new_levels=self.new_levels,
                method="linear",
                extrapolate=True,
            )

    def test_interp_hybrid_to_pressure_extrap_invalid_var(
        self, humidity_in, press_in, _hyam, _hybm, t_bot, phis
    ) -> None:
        with pytest.raises(ValueError):
            interp_hybrid_to_pressure(
                humidity_in,
                press_in,
                _hyam,
                _hybm,
                p0=self._p0,
                new_levels=self.new_levels,
                method="linear",
                extrapolate=True,
                variable=' ',
                t_bot=t_bot,
                phi_sfc=phis,
            )


class Test_interp_sigma_to_hybrid:
    @pytest.fixture(scope="class")
    def ds_u(self):
        # Open the netCDF data file "u.89335.1.nc" and read in input data
        try:
            return xr.open_dataset(
                gdf.get("netcdf_files/u.89335.1_subset_time361.nc"), decode_times=False
            )
        except Exception:
            return xr.open_dataset(
                "test/u.89335.1_subset_time361.nc", decode_times=False
            )

    @pytest.fixture(scope="class")
    def ds_ps(self):
        # Open the netCDF data file "ps.89335.1.nc" and read in additional input
        # data
        try:
            return xr.open_dataset(
                gdf.get("netcdf_files/ps.89335.1.nc"), decode_times=False
            )
        except Exception:
            return xr.open_dataset("test/ps.89335.1.nc", decode_times=False)

    @pytest.fixture(scope="class")
    def ds_out(self):
        # Expected output from above sample input
        try:
            return xr.open_dataset(
                "sigma2hybrid_output.nc"
            )  # Generated by running ncl_tests/test_sigma2hybrid.ncl
        except Exception:
            return xr.open_dataset("test/sigma2hybrid_output.nc")

    hyam = xr.DataArray([0.0108093, 0.0130731, 0.03255911, 0.0639471])
    hybm = xr.DataArray([0.0108093, 0.0173664, 0.06069280, 0.1158237])

    @pytest.fixture(scope="class")
    def u(self, ds_u):
        return ds_u.u[:, 0:3, 0:2]

    @pytest.fixture(scope="class")
    def ps(self, ds_ps):
        return ds_ps.ps[361, 0:3, 0:2] * 100  # Pa

    @pytest.fixture(scope="class")
    def sigma(self, ds_ps):
        return ds_ps.sigma

    @pytest.fixture(scope="class")
    def xh_expected(self, ds_out):
        return ds_out.xh.transpose("ncl3", "ncl1", "ncl2")  # Expected output

    def test_interp_sigma_to_hybrid_1d(self, u, sigma, ps, xh_expected) -> None:
        xh = interp_sigma_to_hybrid(
            u[:, 0, 0], sigma, ps[0, 0], self.hyam, self.hybm, p0=_p0, method="linear"
        )
        nt.assert_array_almost_equal(xh_expected[:, 0, 0], xh, 5)

    def test_interp_sigma_to_hybrid_3d(self, u, sigma, ps, xh_expected) -> None:
        xh = interp_sigma_to_hybrid(
            u, sigma, ps, self.hyam, self.hybm, p0=_p0, method="linear"
        )
        nt.assert_array_almost_equal(xh_expected, xh, 5)

    def test_interp_sigma_to_hybrid_3d_transposed(
        self, u, sigma, ps, xh_expected
    ) -> None:
        xh = interp_sigma_to_hybrid(
            u.transpose('ycoord', 'sigma', 'xcoord'),
            sigma,
            ps.transpose('ycoord', 'xcoord'),
            self.hyam,
            self.hybm,
            p0=_p0,
            method="linear",
        )
        nt.assert_array_almost_equal(
            xh_expected.transpose('ncl2', 'ncl3', 'ncl1'), xh, 5
        )

    def test_interp_sigma_to_hybrid_wrong_method(self, u, sigma, ps) -> None:
        with pytest.raises(ValueError):
            interp_sigma_to_hybrid(
                u, sigma, ps, self.hyam, self.hybm, p0=_p0, method="wrong_method"
            )


class Test_interp_manually_calc:
    @pytest.fixture(scope="class")
    def test_input(self):
        return xr.load_dataset(gdf.get("netcdf_files/interpolation_test_input_data.nc"))

    @pytest.fixture(scope="class")
    def test_output(self):
        return xr.load_dataset(
            gdf.get("netcdf_files/interpolation_test_output_data.nc")
        )

    def test_float32(self, test_input, test_output) -> None:
        np.testing.assert_almost_equal(
            test_output['normal'].values.astype(np.float32),
            interp_multidim(
                xr.DataArray(
                    test_input['normal'].values.astype(np.float32),
                    dims=['lat', 'lon'],
                    coords={
                        'lat': test_input['normal']['lat'].values,
                        'lon': test_input['normal']['lon'].values,
                    },
                ),
                test_output['normal']['lat'].values,
                test_output['normal']['lon'].values,
                cyclic=True,
            ).values,
            decimal=7,
        )

    def test_float64(self, test_input, test_output) -> None:
        np.testing.assert_almost_equal(
            test_output['normal'].values.astype(np.float64),
            interp_multidim(
                xr.DataArray(
                    test_input['normal'].values.astype(np.float64),
                    dims=['lat', 'lon'],
                    coords={
                        'lat': test_input['normal']['lat'].values,
                        'lon': test_input['normal']['lon'].values,
                    },
                ),
                test_output['normal']['lat'].values,
                test_output['normal']['lon'].values,
                cyclic=True,
            ).values,
            decimal=8,
        )

    def test_missing(self, test_input, test_output) -> None:
        np.testing.assert_almost_equal(
            test_output['missing'],
            interp_multidim(
                test_input['missing'],
                test_output['normal']['lat'].values,
                test_output['normal']['lon'].values,
                cyclic=True,
            ).values,
            decimal=8,
        )

    def test_nan(self, test_input, test_output) -> None:
        np.testing.assert_almost_equal(
            test_output['nan'],
            interp_multidim(
                test_input['nan'],
                test_output['normal']['lat'].values,
                test_output['normal']['lon'].values,
                cyclic=True,
            ).values,
            decimal=8,
        )

    def test_mask(self, test_input, test_output) -> None:
        np.testing.assert_almost_equal(
            test_output['mask'],
            interp_multidim(
                test_input['mask'],
                test_output['normal']['lat'].values,
                test_output['normal']['lon'].values,
                cyclic=True,
            ).values,
            decimal=8,
        )

    def test_2_nans(self, test_input, test_output) -> None:
        np.testing.assert_almost_equal(
            test_output['nan_2'],
            interp_multidim(
                test_input['nan_2'],
                test_output['normal']['lat'].values,
                test_output['normal']['lon'].values,
                cyclic=True,
            ).values,
            decimal=8,
        )

    def test_numpy(self, test_input, test_output) -> None:
        np.testing.assert_almost_equal(
            test_output['normal'].values,
            interp_multidim(
                test_input['normal'].values,
                test_output['normal']['lat'].values,
                test_output['normal']['lon'].values,
                lat_in=test_input['normal']['lat'].values,
                lon_in=test_input['normal']['lon'].values,
                cyclic=True,
            ),
            decimal=8,
        )

    def test_extrapolate(self, test_input, test_output) -> None:
        np.testing.assert_almost_equal(
            test_output['normal'].values,
            interp_multidim(
                test_input['normal'],
                test_output['normal']['lat'].values,
                test_output['normal']['lon'].values,
                cyclic=True,
                fill_value='extrapolate',
            ),
            decimal=8,
        )


class Test_interp_larger_dataset:
    @pytest.fixture(scope="class")
    def test_input(self):
        return xr.load_dataset(gdf.get("netcdf_files/spherical_noise_input.nc"))[
            'spherical_noise'
        ]

    @pytest.fixture(scope="class")
    def test_output(self):
        return xr.load_dataset(gdf.get("netcdf_files/spherical_noise_output.nc"))[
            'spherical_noise'
        ]

    def test_10x(self, test_input, test_output) -> None:
        data_xr = interp_multidim(
            test_input, test_output.coords['lat'], test_output.coords['lon']
        )
        np.testing.assert_almost_equal(
            test_output,
            data_xr.values,
            decimal=8,
        )

    def test_chunked(self, test_input, test_output) -> None:
        data_xr = interp_multidim(
            test_input.chunk(2), test_output.coords['lat'], test_output.coords['lon']
        )

        np.testing.assert_almost_equal(test_output, data_xr.values, decimal=8)
