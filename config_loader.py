"""
Configuration loader for CAPTCHA Solver Bot
Handles loading and validation of configuration files
"""

import yaml
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class SolverConfig:
    """Configuration for individual solvers"""
    enabled: bool = True
    confidence_threshold: float = 0.7
    timeout: int = 30
    max_retries: int = 3
    custom_params: Dict[str, Any] = None

    def __post_init__(self):
        if self.custom_params is None:
            self.custom_params = {}


@dataclass
class BrowserConfig:
    """Browser configuration"""
    headless: bool = True
    user_agent: str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    window_size: tuple = (1920, 1080)
    implicit_wait: int = 10
    page_load_timeout: int = 30
    disable_images: bool = False
    disable_javascript: bool = False
    proxy: Optional[str] = None
    extensions: list = None

    def __post_init__(self):
        if self.extensions is None:
            self.extensions = []


@dataclass
class ModelConfig:
    """AI model configuration"""
    model_path: str = ""
    device: str = "auto"  # auto, cpu, cuda, mps
    batch_size: int = 1
    confidence_threshold: float = 0.8
    preprocessing_enabled: bool = True
    cache_predictions: bool = True


class ConfigLoader:
    """Loads and manages configuration files"""

    def __init__(self, base_path: Optional[Path] = None):
        self.base_path = base_path or Path(__file__).parent.parent.parent
        self.config_cache = {}

    def load_config(self, config_path: Union[str, Path]) -> Dict[str, Any]:
        """Load configuration from file"""

        config_path = Path(config_path)

        if not config_path.is_absolute():
            config_path = self.base_path / config_path

        cache_key = str(config_path)
        if cache_key in self.config_cache:
            modification_time = config_path.stat().st_mtime
            if self.config_cache[cache_key]['modified'] >= modification_time:
                return self.config_cache[cache_key]['config']

        try:
            logger.info(f"Loading configuration from: {config_path}")

            with open(config_path, 'r', encoding='utf-8') as file:
                if config_path.suffix.lower() in ['.yaml', '.yml']:
                    config = yaml.safe_load(file)
                elif config_path.suffix.lower() == '.json':
                    config = json.load(file)
                else:
                    raise ValueError(f"Unsupported config format: {config_path.suffix}")

            config = self._substitute_env_vars(config)
            config = self._process_includes(config, config_path.parent)
            config = self._apply_defaults(config)

            self.config_cache[cache_key] = {
                'config': config,
                'modified': config_path.stat().st_mtime
            }

            return config

        except Exception as e:
            logger.error(f"Failed to load config {config_path}: {e}")
            raise

    def _substitute_env_vars(self, config: Any) -> Any:
        if isinstance(config, dict):
            return {k: self._substitute_env_vars(v) for k, v in config.items()}
        elif isinstance(config, list):
            return [self._substitute_env_vars(i) for i in config]
        elif isinstance(config, str):
            return os.path.expandvars(config)
        else:
            return config

    def _process_includes(self, config: Dict[str, Any], base_dir: Path) -> Dict[str, Any]:
        includes = config.pop('includes', [])
        if not isinstance(includes, list):
            includes = [includes]

        merged_config = {}
        for include in includes:
            include_path = base_dir / include
            included_config = self.load_config(include_path)
            merged_config.update(included_config)

        merged_config.update(config)
        return merged_config

    def _apply_defaults(self, config: Dict[str, Any]) -> Dict[str, Any]:
        config.setdefault('solver', {})
        config.setdefault('browser', {})
        config.setdefault('model', {})

        config['solver'] = SolverConfig(**config['solver']).__dict__
        config['browser'] = BrowserConfig(**config['browser']).__dict__
        config['model'] = ModelConfig(**config['model']).__dict__

        return config
