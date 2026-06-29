#!/usr/bin/env python3
"""
ECOS2026 - Main Entry Point

This is the primary CLI interface for the ECOS2026 pipeline.
All configuration is read from CONFIG.yml - no hardcoded parameters.

Usage:
    python main.py                                    # Run with default CONFIG.yml
    python main.py --config custom_config.yml        # Run with custom config
    python main.py --dataset MyDataset                # Process specific dataset
    python main.py --debug                            # Verbose logging
    python main.py --slurm                            # Submit to SLURM cluster
    python main.py --dry-run                          # Validate config only
"""

import argparse
import sys
import logging
import warnings
from pathlib import Path
from typing import Optional, Dict, Any

import yaml

# Suppress warnings if not debugging
warnings.filterwarnings("ignore")


def setup_logging(
    log_level: str = "INFO",
    save_logs: bool = True,
    log_dir: Optional[Path] = None,
) -> logging.Logger:
    """
    Configure logging before any module imports.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        save_logs: Save logs to file
        log_dir: Directory for log files
    
    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger("ecos2026")
    logger.setLevel(log_level)
    
    # Remove existing handlers
    logger.handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_format = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler
    if save_logs and log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"ecos2026_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_format = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)
        logger.info(f"Logging to {log_file}")
    
    return logger


def load_config(config_path: Path, logger: logging.Logger) -> Dict[str, Any]:
    """
    Load and validate configuration from YAML file.
    
    Args:
        config_path: Path to CONFIG.yml
        logger: Logger instance
    
    Returns:
        Configuration dictionary
    
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid YAML
    """
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded config from {config_path}")
        return config
    except yaml.YAMLError as e:
        logger.error(f"Invalid YAML in {config_path}: {e}")
        raise


def resolve_paths(config: Dict[str, Any], repo_root: Path, logger: logging.Logger) -> Dict[str, Any]:
    """
    Convert relative paths in config to absolute paths.
    
    Args:
        config: Configuration dictionary
        repo_root: Repository root directory
        logger: Logger instance
    
    Returns:
        Config with resolved paths
    """
    path_keys = [
        ("data", "dataset_dir"),
        ("data", "energyplus", "meter_dir"),
        ("data", "energyplus", "epw_dir"),
        ("output", "base_dir"),
        ("logging", "log_dir"),
    ]
    
    for keys in path_keys:
        current = config
        for key in keys[:-1]:
            if key not in current:
                break
            current = current[key]
        else:
            last_key = keys[-1]
            if last_key in current and current[last_key]:
                path = Path(current[last_key])
                if not path.is_absolute():
                    current[last_key] = str(repo_root / path)
                    logger.debug(f"Resolved {'.'.join(keys)} to {current[last_key]}")
    
    return config


def main():
    """Main CLI entry point."""
    # Find repo root (parent of pipeline directory)
    repo_root = Path(__file__).parent.parent
    
    # Define default config path
    default_config = repo_root / "CONFIG.yml"
    
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="ECOS2026 - Energy Consumption Forecasting and Change Point Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py
  python main.py --config custom_config.yml
  python main.py --dataset MyDatasetName
  python main.py --debug
  python main.py --slurm
  python main.py --dry-run
        """
    )
    
    parser.add_argument(
        "--config",
        type=Path,
        default=default_config,
        help=f"Path to CONFIG.yml (default: {default_config})"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Process specific dataset only (default: all datasets)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    parser.add_argument(
        "--slurm",
        action="store_true",
        help="Submit jobs to SLURM cluster"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config and exit (don't process data)"
    )
    
    args = parser.parse_args()
    
    # Determine log level
    log_level = "DEBUG" if args.debug else "INFO"
    
    # Setup logging BEFORE any imports
    logger = setup_logging(log_level=log_level)
    logger.info("=" * 70)
    logger.info("ECOS2026 - Energy Consumption Time Series Analysis Pipeline")
    logger.info("=" * 70)
    
    try:
        # Load configuration
        logger.info(f"Loading config from {args.config}")
        config = load_config(args.config, logger)
        
        # Resolve paths
        config = resolve_paths(config, repo_root, logger)
        
        # Validate critical paths
        dataset_dir = Path(config["data"]["dataset_dir"])
        if not dataset_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
        logger.info(f"Dataset directory: {dataset_dir}")
        
        # Dry run - just validate config and exit
        if args.dry_run:
            logger.info("Dry run - validating config only")
            logger.info("✓ Config valid")
            logger.info("✓ Dataset directory accessible")
            logger.info("✓ Ready to run pipeline")
            return 0
        
        # Import pipeline modules AFTER logging is set up
        from pipeline_executor import PipelineExecutor
        
        logger.info("Initializing pipeline executor...")
        executor = PipelineExecutor(config, repo_root)
        
        if args.slurm:
            logger.info("SLURM mode: submitting jobs to cluster")
            executor.submit_to_slurm()
        else:
            logger.info("Local mode: running pipeline locally")
            
            # Get datasets to process
            if args.dataset:
                datasets = [args.dataset]
                logger.info(f"Processing single dataset: {args.dataset}")
            else:
                datasets = None
                logger.info(f"Processing all datasets in {dataset_dir}")
            
            # Run pipeline
            executor.run_local(datasets=datasets)
        
        logger.info("=" * 70)
        logger.info("Pipeline completed successfully!")
        logger.info("=" * 70)
        return 0
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=args.debug)
        return 1


if __name__ == "__main__":
    # Import pandas here for timestamp in logging setup
    import pandas as pd
    sys.exit(main())
