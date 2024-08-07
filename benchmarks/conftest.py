from _pytest.config import Config


def pytest_configure(config: Config) -> None:
    # cspell:ignore addinivalue
    config.addinivalue_line("python_files", "*.py")
