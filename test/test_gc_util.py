import pytest
import xarray as xr
import numpy as np
from geocat.comp.gc_util import _find_var, _find_optional_var


class TestFindVar:
    """Tests for _find_var function with case-insensitive matching."""

    @pytest.fixture
    def sample_dataset(self):
        """Create a sample dataset with various case conventions."""
        ds = xr.Dataset(
            {
                'TEMP': xr.DataArray(
                    np.random.rand(10),
                    attrs={
                        'standard_name': 'air_temperature',
                        'long_name': 'Air Temperature',
                    },
                ),
                'pressure': xr.DataArray(
                    np.random.rand(10),
                    attrs={
                        'standard_name': 'SURFACE_AIR_PRESSURE',
                        'long_name': 'surface pressure',
                    },
                ),
                'Meridional_Wind': xr.DataArray(
                    np.random.rand(10),
                    attrs={
                        'long_name': 'Meridional_Wind_Component',
                    },
                ),
                'v_component': xr.DataArray(np.random.rand(10), attrs={}),
            }
        )
        return ds

    def test_standard_name_exact_match(self, sample_dataset):
        """Test exact match on standard_name attribute."""
        result = _find_var(sample_dataset, standard_name='air_temperature')
        assert result == 'TEMP'

    def test_standard_name_case_insensitive_attr(self, sample_dataset):
        """Test case-insensitive match on standard_name attribute."""
        result = _find_var(sample_dataset, standard_name='surface_air_pressure')
        assert result == 'pressure'

    def test_standard_name_matches_variable_name_lowercase(self, sample_dataset):
        """Test when standard_name matches variable name (lowercase)."""
        result = _find_var(sample_dataset, standard_name='pressure')
        assert result == 'pressure'

    def test_standard_name_matches_variable_name_uppercase(self, sample_dataset):
        """Test when standard_name matches variable name (uppercase)."""
        result = _find_var(sample_dataset, standard_name='temp')
        assert result == 'TEMP'

    def test_standard_name_title_case(self, sample_dataset):
        """Test title case variation matching."""
        sample_dataset['wind'] = xr.DataArray(
            np.random.rand(10), attrs={'standard_name': 'Northward_Wind'}
        )
        result = _find_var(sample_dataset, standard_name='northward_wind')
        assert result == 'wind'

    def test_long_name_exact_match(self, sample_dataset):
        """Test exact match on long_name attribute."""
        result = _find_var(sample_dataset, long_name='Air Temperature')
        assert result == 'TEMP'

    def test_long_name_case_insensitive(self, sample_dataset):
        """Test case-insensitive match on long_name attribute."""
        result = _find_var(sample_dataset, long_name='Surface Pressure')
        assert result == 'pressure'

    def test_long_name_with_underscores(self, sample_dataset):
        """Test long_name with underscores."""
        result = _find_var(sample_dataset, long_name='meridional_wind_component')
        assert result == 'Meridional_Wind'

    def test_long_name_matches_variable_name(self, sample_dataset):
        """Test when long_name matches the variable name itself."""
        result = _find_var(sample_dataset, long_name='Meridional_Wind')
        assert result == 'Meridional_Wind'

    def test_possible_names_exact_match(self, sample_dataset):
        """Test exact match from possible_names list."""
        result = _find_var(
            sample_dataset, possible_names=['V', 'v_component', 'meridional']
        )
        assert result == 'v_component'

    def test_possible_names_case_insensitive_lowercase(self, sample_dataset):
        """Test case-insensitive matching with lowercase."""
        result = _find_var(sample_dataset, possible_names=['temp', 'temperature'])
        assert result == 'TEMP'

    def test_possible_names_case_insensitive_uppercase(self, sample_dataset):
        """Test case-insensitive matching with uppercase."""
        result = _find_var(sample_dataset, possible_names=['PRESSURE', 'PRESS'])
        assert result == 'pressure'

    def test_possible_names_title_case(self, sample_dataset):
        """Test title case matching."""
        result = _find_var(sample_dataset, possible_names=['meridional_wind', 'v_wind'])
        assert result == 'Meridional_Wind'

    def test_possible_names_mixed_case_in_list(self, sample_dataset):
        """Test with mixed case variations in possible_names list."""
        result = _find_var(
            sample_dataset, possible_names=['V_COMPONENT', 'v_comp', 'V_Comp']
        )
        assert result == 'v_component'

    def test_priority_standard_name_over_long_name(self, sample_dataset):
        """Test that standard_name has priority over long_name."""
        result = _find_var(
            sample_dataset,
            standard_name='air_temperature',
            long_name='Meridional_Wind_Component',
        )
        assert result == 'TEMP'

    def test_priority_standard_name_over_possible_names(self, sample_dataset):
        """Test that standard_name has priority over possible_names."""
        result = _find_var(
            sample_dataset,
            standard_name='air_temperature',
            possible_names=['pressure', 'TEMP'],
        )
        assert result == 'TEMP'

    def test_priority_long_name_over_possible_names(self, sample_dataset):
        """Test that long_name has priority over possible_names."""
        result = _find_var(
            sample_dataset,
            long_name='Air Temperature',
            possible_names=['pressure', 'TEMP'],
        )
        assert result == 'TEMP'

    def test_fallback_to_possible_names(self, sample_dataset):
        """Test fallback to possible_names when others fail."""
        result = _find_var(
            sample_dataset,
            standard_name='nonexistent_standard',
            long_name='Nonexistent Long Name',
            possible_names=['v_component'],
        )
        assert result == 'v_component'

    def test_not_found_raises_keyerror(self, sample_dataset):
        """Test that KeyError is raised when variable not found."""
        with pytest.raises(KeyError) as excinfo:
            _find_var(
                sample_dataset,
                standard_name='nonexistent',
                long_name='Does not exist',
                possible_names=['missing1', 'missing2'],
            )

        error_msg = str(excinfo.value)
        assert 'Could not find variable in dataset' in error_msg
        assert 'Tried standard_name: nonexistent' in error_msg
        assert 'Tried long_name: Does not exist' in error_msg  # Changed this line
        assert "Tried names: ['missing1', 'missing2']" in error_msg

    def test_custom_description_in_error(self, sample_dataset):
        """Test that custom description appears in error message."""
        with pytest.raises(KeyError) as excinfo:
            _find_var(
                sample_dataset,
                standard_name='nonexistent',
                description='meridional wind component',
            )

        assert 'Could not find meridional wind component in dataset' in str(
            excinfo.value
        )

    def test_error_message_includes_all_attempts(self, sample_dataset):
        """Test that error message includes all search attempts."""
        with pytest.raises(KeyError) as excinfo:
            _find_var(
                sample_dataset,
                standard_name='not_here',
                long_name='also_not_here',
                possible_names=['nope'],
            )

        error_msg = str(excinfo.value)
        assert 'Tried standard_name:' in error_msg
        assert 'Tried long_name:' in error_msg
        assert 'Tried names:' in error_msg

    def test_empty_attributes_no_crash(self, sample_dataset):
        """Test that empty/missing attributes don't cause crashes."""
        result = _find_var(sample_dataset, possible_names=['v_component'])
        assert result == 'v_component'

    def test_attribute_with_spaces(self, sample_dataset):
        """Test matching with spaces in attribute values."""
        result = _find_var(sample_dataset, long_name='Air Temperature')
        assert result == 'TEMP'

    def test_underscores_in_names(self, sample_dataset):
        """Test proper handling of underscores in capitalization."""
        result = _find_var(sample_dataset, possible_names=['meridional_wind'])
        assert result == 'Meridional_Wind'


class TestFindOptionalVar:
    """Tests for _find_optional_var function."""

    @pytest.fixture
    def sample_dataset(self):
        """Create a sample dataset."""
        ds = xr.Dataset(
            {
                'temp': xr.DataArray(
                    np.random.rand(10), attrs={'standard_name': 'air_temperature'}
                ),
                'PRESSURE': xr.DataArray(
                    np.random.rand(10), attrs={'long_name': 'Surface Pressure'}
                ),
            }
        )
        return ds

    def test_find_existing_variable_returns_name(self, sample_dataset):
        """Test finding an existing variable returns its name."""
        result = _find_optional_var(sample_dataset, standard_name='air_temperature')
        assert result == 'temp'

    def test_find_nonexistent_returns_none(self, sample_dataset):
        """Test that nonexistent variable returns None (no error raised)."""
        result = _find_optional_var(
            sample_dataset,
            standard_name='nonexistent',
            long_name='does_not_exist',
            possible_names=['nope', 'also_nope'],
        )
        assert result is None

    def test_none_not_exception(self, sample_dataset):
        """Test that None is returned, not an exception object."""
        result = _find_optional_var(sample_dataset, standard_name='nonexistent')
        assert result is None
        assert not isinstance(result, Exception)
