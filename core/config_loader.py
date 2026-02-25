"""
Configuration Loader

Centralized configuration management for OpenAD.
Loads settings from YAML files and provides easy access.
"""

import yaml
from pathlib import Path
from typing import Any, Dict, Optional

# Singleton pattern for config
_config_cache: Optional[Dict[str, Any]] = None


def load_config(reload: bool = False) -> Dict[str, Any]:
    """
    Load all configuration files.

    Args:
        reload: Force reload configuration from disk

    Returns:
        Dictionary containing all configuration
    """
    global _config_cache

    if _config_cache is not None and not reload:
        return _config_cache

    config_dir = Path(__file__).parent.parent / "config"

    config = {}

    # Load settings.yaml
    settings_path = config_dir / "settings.yaml"
    if settings_path.exists():
        with open(settings_path) as f:
            config['settings'] = yaml.safe_load(f)
    else:
        print(f"Warning: {settings_path} not found, using defaults")
        config['settings'] = _get_default_settings()

    # Load packages.yaml
    packages_path = config_dir / "packages.yaml"
    if packages_path.exists():
        with open(packages_path) as f:
            config['packages'] = yaml.safe_load(f)
    else:
        print(f"Warning: {packages_path} not found, using defaults")
        config['packages'] = _get_default_packages()

    _config_cache = config
    return config


def get_setting(path: str, default: Any = None) -> Any:
    """
    Get a setting value by dot-notation path.

    Example:
        get_setting('system.max_retries')  # Returns 2
        get_setting('llm.bedrock.model_id')  # Returns model ID

    Args:
        path: Dot-separated path to setting (e.g., 'system.max_retries')
        default: Default value if setting not found

    Returns:
        The setting value or default
    """
    config = load_config()

    keys = path.split('.')
    value = config.get('settings', {})

    for key in keys:
        if isinstance(value, dict):
            value = value.get(key)
        else:
            return default

        if value is None:
            return default

    return value


def get_package_config(package_name: str) -> Optional[Dict[str, Any]]:
    """
    Get configuration for a specific package.

    Args:
        package_name: Name of the package (e.g., 'pyod', 'pygod')

    Returns:
        Package configuration dictionary or None
    """
    config = load_config()
    packages = config.get('packages', {}).get('packages', {})
    return packages.get(package_name)


def get_all_packages() -> Dict[str, Any]:
    """
    Get all package configurations.

    Returns:
        Dictionary of all package configurations
    """
    config = load_config()
    return config.get('packages', {}).get('packages', {})


def get_algorithms(package_name: str) -> list:
    """
    Get list of algorithms for a package.

    Args:
        package_name: Name of the package

    Returns:
        List of algorithm names
    """
    pkg_config = get_package_config(package_name)
    if pkg_config:
        return pkg_config.get('algorithms', [])
    return []


def _get_default_settings() -> Dict[str, Any]:
    """Default settings if settings.yaml not found."""
    return {
        'system': {
            'sandbox_mode': 'modal',
            'timeout_seconds': 60,
            'max_retries': 2,
        },
        'llm': {
            'default_provider': 'bedrock',
            'bedrock': {
                'model_id': 'meta.llama3-1-70b-instruct-v1:0',
                'region': 'us-west-2',
            },
            'openai': {
                'model': 'gpt-4o',
            },
        },
    }


def _get_default_packages() -> Dict[str, Any]:
    """Default package config if packages.yaml not found."""
    return {
        'packages': {
            'pyod': {
                'name': 'PyOD',
                'algorithms': ['IForest', 'ABOD', 'LOF'],
            }
        }
    }
