import os
import pytest

from src.tools import general_tools


class TestGeneralTools:
    """Class testing the general_tools module."""

    @staticmethod
    def test_get_folder_path(monkeypatch: pytest.fixture) -> None:
        """Test the get_folder_path method.

        Args:
            monkeypatch (pytest.fixture): A convenient fixture for monkey-patching.

        Returns:
            None
        """
        # mock value of os method
        monkeypatch.setattr(os.path, 'realpath', lambda x: 'real/path/')
        assert general_tools.get_folder_path('src/') == 'real/path/src/'
        assert general_tools.get_folder_path('src/some_other_dir/') == 'real/path/src/some_other_dir/'

        # mock value of os method
        monkeypatch.setattr(os.path, 'realpath', lambda x: 'real/path/src/')
        assert general_tools.get_folder_path('dir/') == 'real/path/dir/'

    @staticmethod
    def test_get_filepath(monkeypatch: pytest.fixture) -> None:
        """Test the get_filepath method.

        Args:
            monkeypatch (pytest.fixture): A convenient fixture for monkey-patching.

        Returns:
            None
        """
        # mock value of os method
        monkeypatch.setattr(os.path, 'realpath', lambda x: 'real/path/')
        assert general_tools.get_filepath('my/favorite/folder', filename='my_file') \
               == 'real/path/my/favorite/folder/my_file'

        # mock value of os method
        monkeypatch.setattr(os.path, 'realpath', lambda x: 'real/path/src/')
        assert general_tools.get_filepath('my/favorite/folder', filename='my_file') \
               == 'real/path/my/favorite/folder/my_file'

        # mock value of os method
        monkeypatch.setattr(os.path, 'realpath', lambda x: 'real/path/src/')
        assert general_tools.get_filepath('my/favorite/folder', filename='my_file_with/slash') \
               == 'real/path/my/favorite/folder/my_file_with:slash'

    @staticmethod
    def test_is_number() -> None:
        """Test the is_number method."""
        assert general_tools.is_number(1) is True
        assert general_tools.is_number('1') is True
        assert general_tools.is_number('0') is True

    @staticmethod
    def test_is_not_number() -> None:
        """Test the is_number method."""
        assert general_tools.is_number('a') is False
        assert general_tools.is_number('five') is False
        assert general_tools.is_number(None) is False
        assert general_tools.is_number('') is False

    @staticmethod
    def test_is_empty() -> None:
        """Tests if a collection is empty."""
        assert general_tools.is_empty([]) is True
        assert general_tools.is_empty({}) is True
        assert general_tools.is_empty(()) is True
        assert general_tools.is_empty('') is True
        assert general_tools.is_empty(None) is True

    @staticmethod
    def test_is_not_empty() -> None:
        """Tests if a collection is empty."""
        assert general_tools.is_empty(['value_1', 'value_2']) is False
        assert general_tools.is_empty((1, 2, 3)) is False
        assert general_tools.is_empty({1, 2, 3}) is False
        assert general_tools.is_empty('value_1') is False

    @staticmethod
    def test_extract_info_from_url() -> None:
        """Test extract_info_from_url method"""
        components = {'protocol': 'https', 'hostname': 'app.foodakai.com', 'path': '',
                      'params': '', 'query': '', 'username': None, 'password': None, 'port': None}
        assert general_tools.extract_info_from_url(url='https://app.foodakai.com') == components

        components = {'protocol': 'https', 'hostname': 'app.foodakai.com', 'path': '/content/4/ad5f5c40.PNG',
                      'params': '', 'query': '', 'username': None, 'password': None, 'port': None}
        assert general_tools.extract_info_from_url(url='https://app.foodakai.com/content/4/ad5f5c40.PNG') == components
