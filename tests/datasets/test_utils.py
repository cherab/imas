import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from cherab.imas.datasets import clear_cache, iter_jintrac, iter_solps
from cherab.imas.datasets._registry import method_files_map, registry
from cherab.imas.datasets._utils import _clear_cache


class TestClearCache:
    """Test suite for the clear_cache function."""

    def test_clear_cache_no_args_with_mock(self):
        """Test clear_cache with no arguments (should clear all cache)."""
        with patch("cherab.imas.datasets._utils._clear_cache") as mock_clear:
            clear_cache()
            mock_clear.assert_called_once_with(None)

    def test_clear_cache_single_dataset(self):
        """Test clear_cache with a single dataset function."""
        with patch("cherab.imas.datasets._utils._clear_cache") as mock_clear:
            clear_cache(iter_jintrac)
            mock_clear.assert_called_once_with(iter_jintrac)

    def test_clear_cache_multiple_datasets(self):
        """Test clear_cache with multiple dataset functions."""
        datasets = [iter_jintrac, iter_solps]
        with patch("cherab.imas.datasets._utils._clear_cache") as mock_clear:
            clear_cache(datasets)
            mock_clear.assert_called_once_with(datasets)

    def test_clear_cache_with_temp_directory(self):
        """Test _clear_cache with a temporary directory setup."""
        # Create a temporary directory structure
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir) / "test_cache"
            cache_dir.mkdir(parents=True)

            # Create test files matching the method_files_map
            test_files = list(registry.keys())

            for file in test_files:
                (cache_dir / file).touch()

            # Verify files exist before clearing
            assert all((cache_dir / file).exists() for file in test_files)

            # Clear all cache
            _clear_cache(None, cache_dir)

            # Verify cache directory is removed
            assert not cache_dir.exists()

    def test_clear_cache_single_dataset_with_temp_directory(self, capsys):
        """Test _clear_cache with a single dataset and temporary directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir) / "test_cache"
            cache_dir.mkdir(parents=True)

            # Create test files
            jintrac_file = method_files_map["iter_jintrac"][0]
            solps_file = method_files_map["iter_solps"][0]

            (cache_dir / jintrac_file).touch()
            (cache_dir / solps_file).touch()

            # Clear only jintrac dataset
            _clear_cache(iter_jintrac, cache_dir)

            # Verify only jintrac file is removed
            assert not (cache_dir / jintrac_file).exists()
            assert (cache_dir / solps_file).exists()

            # Check console output
            captured = capsys.readouterr()
            assert "iter_jintrac" in captured.out
            assert jintrac_file in captured.out

    def test_clear_cache_multiple_datasets_with_temp_directory(self, capsys):
        """Test _clear_cache with multiple datasets and temporary directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir) / "test_cache"
            cache_dir.mkdir(parents=True)

            # Create test files
            jintrac_file = method_files_map["iter_jintrac"][0]
            solps_file = method_files_map["iter_solps"][0]
            jorek_file = method_files_map["iter_jorek"][0]

            (cache_dir / jintrac_file).touch()
            (cache_dir / solps_file).touch()
            (cache_dir / jorek_file).touch()

            # Clear jintrac and solps datasets
            _clear_cache([iter_jintrac, iter_solps], cache_dir)

            # Verify jintrac and solps files are removed, jorek remains
            assert not (cache_dir / jintrac_file).exists()
            assert not (cache_dir / solps_file).exists()
            assert (cache_dir / jorek_file).exists()

    def test_clear_cache_nonexistent_cache_directory(self, capsys):
        """Test _clear_cache with non-existent cache directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            nonexistent_dir = Path(temp_dir) / "nonexistent"

            _clear_cache(None, nonexistent_dir)

            captured = capsys.readouterr()
            assert "doesn't exist" in captured.out
            assert "Nothing to clear" in captured.out

    def test_clear_cache_nonexistent_file(self, capsys):
        """Test _clear_cache when specific dataset files don't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir) / "test_cache"
            cache_dir.mkdir(parents=True)

            # Don't create any files, just try to clear jintrac
            _clear_cache(iter_jintrac, cache_dir)

            captured = capsys.readouterr()
            assert "doesn't exist" in captured.out
            assert "Nothing to clear" in captured.out

    def test_clear_cache_invalid_dataset_type(self):
        """Test _clear_cache with invalid dataset type."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir) / "test_cache"
            cache_dir.mkdir(parents=True)

            with pytest.raises(TypeError, match="Expected a callable dataset method"):
                _clear_cache("invalid_dataset", cache_dir)

    def test_clear_cache_unknown_dataset_method(self):
        """Test _clear_cache with unknown dataset method."""

        def unknown_dataset():
            pass

        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir) / "test_cache"
            cache_dir.mkdir(parents=True)

            with pytest.raises(ValueError, match="Dataset method unknown_dataset doesn't exist"):
                _clear_cache(unknown_dataset, cache_dir)

    def test_clear_cache_missing_platformdirs(self):
        """Test clear_cache behavior when platformdirs is not available."""
        with patch("cherab.imas.datasets._utils.platformdirs", None):
            with pytest.raises(ImportError, match="Missing optional dependency 'pooch'"):
                _clear_cache(None, None)

    @patch("cherab.imas.datasets._utils.platformdirs")
    def test_clear_cache_default_cache_dir(self, mock_platformdirs):
        """Test _clear_cache uses default cache directory when none provided."""
        mock_cache_dir = "/mock/cache/dir"
        mock_platformdirs.user_cache_dir.return_value = mock_cache_dir

        with patch("cherab.imas.datasets._utils.Path") as mock_path:
            mock_path_instance = MagicMock()
            mock_path.return_value = mock_path_instance
            mock_path_instance.exists.return_value = False

            _clear_cache(None, None)

            mock_platformdirs.user_cache_dir.assert_called_once_with("cherab/imas")
            mock_path.assert_called_once_with(mock_cache_dir)

    def test_clear_cache_custom_method_map(self, capsys):
        """Test _clear_cache with custom method map."""

        def custom_dataset():
            pass

        custom_method_map = {"custom_dataset": ["custom_file.nc"]}

        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir) / "test_cache"
            cache_dir.mkdir(parents=True)

            custom_file = cache_dir / "custom_file.nc"
            custom_file.touch()

            _clear_cache(custom_dataset, cache_dir, custom_method_map)

            assert not custom_file.exists()
            captured = capsys.readouterr()
            assert "custom_dataset" in captured.out
            assert "custom_file.nc" in captured.out
