
"""
CAPTCHA Solver Bot - Main Entry Point
Entry point for command-line usage and testing
"""

import asyncio
import argparse
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
from captcha_solver.core.orchestrator import CaptchaOrchestrator
from captcha_solver.utils.config_loader import ConfigLoader
from captcha_solver.utils.logger import setup_logging
from captcha_solver.browser.driver_manager import WebDriverManager
from captcha_solver.utils.validators import validate_url, validate_config



async def main():
    """Main entry point for CAPTCHA solver"""
    
    parser = argparse.ArgumentParser(description='CAPTCHA Solver Bot')
    parser.add_argument('--url', type=str, help='URL containing CAPTCHA to solve')
    parser.add_argument('--config', type=str, default='data/configs/default_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--headless', action='store_true', 
                       help='Run browser in headless mode')
    parser.add_argument('--max-attempts', type=int, default=3,
                       help='Maximum number of solving attempts')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug logging')
    parser.add_argument('--gui', action='store_true',
                       help='Launch GUI interface')
    parser.add_argument('--benchmark', action='store_true',
                       help='Run performance benchmarks')
    parser.add_argument('--train', type=str, nargs='?', const='all',
                       help='Train models (specify type or "all")')
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = 'DEBUG' if args.debug else 'INFO'
    logger = setup_logging(log_level)
    
    try:
        # Load configuration
        config_loader = ConfigLoader()
        config = config_loader.load_config(args.config)
        
        # Validate configuration
        if not validate_config(config):
            logger.error("Invalid configuration file")
            return 1
        
        # Handle different modes
        if args.gui:
            from gui import launch_gui
            return launch_gui(config)
        
        elif args.benchmark:
            from tests.benchmarks.speed_tests import run_benchmarks
            return await run_benchmarks(config)
        
        elif args.train:
            from scripts.training.train_models import train_models
            return train_models(args.train, config)
        
        elif args.url:
            # Standard CAPTCHA solving mode
            if not validate_url(args.url):
                logger.error("Invalid URL provided")
                return 1
            
            return await solve_captcha_on_url(args.url, config, args)
        
        else:
            # Interactive mode
            return await interactive_mode(config)
            
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        return 0
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return 1

async def solve_captcha_on_url(url: str, config: dict, args) -> int:
    """Solve CAPTCHA on a specific URL"""
    
    logger = setup_logging()
    logger.info(f"Starting CAPTCHA solver for: {url}")
    
    # Initialize components
    driver_manager = WebDriverManager(config)
    orchestrator = CaptchaSolverOrchestrator(config)
    
    driver = None
    try:
        # Create browser session
        driver = await driver_manager.create_driver(headless=args.headless)
        
        # Navigate to URL
        logger.info(f"Navigating to {url}")
        driver.get(url)
        
        # Solve CAPTCHA
        result = await orchestrator.solve_captcha(driver, max_attempts=args.max_attempts)
        
        # Report results
        if result.success:
            logger.info(f" CAPTCHA solved successfully!")
            logger.info(f"Solution: {result.solution}")
            logger.info(f"Confidence: {result.confidence:.2%}")
            logger.info(f"Processing time: {result.processing_time:.2f}s")
            logger.info(f"Method used: {result.method_used}")
            return 0
        else:
            logger.error(f" Failed to solve CAPTCHA")
            logger.error(f"Error: {result.error_message}")
            return 1
            
    except Exception as e:
        logger.error(f"Error during solving process: {str(e)}")
        return 1
    finally:
        if driver:
            await driver_manager.cleanup_driver(driver)

async def interactive_mode(config: dict) -> int:
    """Interactive mode for testing and experimentation"""
    
    logger = setup_logging()
    print("\n🤖 CAPTCHA Solver Bot - Interactive Mode")
    print("=" * 50)
    
    while True:
        print("\nAvailable options:")
        print("1. Solve CAPTCHA on URL")
        print("2. Test specific solver type")
        print("3. View performance statistics")
        print("4. Train models")
        print("5. Run benchmarks")
        print("6. Exit")
        
        choice = input("\nEnter your choice (1-6): ").strip()
        
        if choice == '1':
            url = input("Enter URL: ").strip()
            if validate_url(url):
                max_attempts = input("Max attempts (default 3): ").strip() or "3"
                headless = input("Headless mode? (y/n, default n): ").strip().lower() == 'y'
                
                class Args:
                    max_attempts = int(max_attempts)
                    headless = headless
                
                result_code = await solve_captcha_on_url(url, config, Args())
                
                if result_code == 0:
                    print("Success!")
                else:
                    print(" Failed!")
            else:
                print("Invalid URL format")
        
        elif choice == '2':
            await test_solver_type(config)
        
        elif choice == '3':
            view_performance_stats(config)
        
        elif choice == '4':
            train_models_interactive(config)
        
        elif choice == '5':
            from tests.benchmarks.speed_tests import run_benchmarks
            await run_benchmarks(config)
        
        elif choice == '6':
            print("👋 Goodbye!")
            break
        
        else:
            print("Invalid choice. Please try again.")
    
    return 0

async def test_solver_type(config: dict):
    """Test a specific solver type"""
    
    from python.core.captcha_detector import CaptchaType
    
    print("\nAvailable solver types:")
    for i, captcha_type in enumerate(CaptchaType, 1):
        print(f"{i}. {captcha_type.value}")
    
    try:
        choice = int(input("Select solver type: ")) - 1
        captcha_types = list(CaptchaType)
        
        if 0 <= choice < len(captcha_types):
            selected_type = captcha_types[choice]
            url = input(f"Enter URL with {selected_type.value} CAPTCHA: ").strip()
            
            if validate_url(url):
                # Implementation for testing specific solver type
                print(f"Testing {selected_type.value} solver on {url}")
                # Add actual testing logic here
            else:
                print("Invalid URL format")
        else:
            print("Invalid selection")
    except ValueError:
        print("Please enter a valid number")

def view_performance_stats(config: dict):
    """Display performance statistics"""
    
    try:
        from python.core.performance_tracker import PerformanceTracker
        
        tracker = PerformanceTracker()
        stats = tracker.get_all_statistics()
        
        print("\n Performance Statistics")
        print("=" * 30)
        
        if not stats:
            print("No performance data available yet.")
            return
        
        for key, data in stats.items():
            success_rate = data['successes'] / data['attempts'] if data['attempts'] > 0 else 0
            avg_time = data['total_time'] / data['attempts'] if data['attempts'] > 0 else 0
            
            print(f"\n{key}:")
            print(f"  Attempts: {data['attempts']}")
            print(f"  Success Rate: {success_rate:.2%}")
            print(f"  Average Time: {avg_time:.2f}s")
            print(f"  Average Confidence: {data['avg_confidence']:.2%}")
            
    except Exception as e:
        print(f"Error loading performance statistics: {str(e)}")

def train_models_interactive(config: dict):
    """Interactive model training"""
    
    print("\nAvailable training options:")
    print("1. Text CAPTCHA models")
    print("2. Image selection models") 
    print("3. Audio CAPTCHA models")
    print("4. All models")
    
    try:
        choice = input("Select training option (1-4): ").strip()
        
        model_types = {
            '1': 'text',
            '2': 'image',
            '3': 'audio',
            '4': 'all'
        }
        
        if choice in model_types:
            from scripts.training.train_models import train_models
            result = train_models(model_types[choice], config)
            
            if result == 0:
                print(" Training completed successfully!")
            else:
                print(" Training failed!")
        else:
            print("Invalid selection")
    except Exception as e:
        print(f"Training error: {str(e)}")

if __name__ == "__main__":
    # Check Python version
    if sys.version_info < (3, 8):
        print(" Python 3.8 or higher is required")
        sys.exit(1)
    
    # Check if required directories exist
    required_dirs = ['python', 'data', 'models']
    for dir_name in required_dirs:
        if not os.path.exists(dir_name):
            print(f"Required directory '{dir_name}' not found")
            print("Please run setup script first: python scripts/setup/install_dependencies.py")
            sys.exit(1)
    
    # Run main function
    exit_code = asyncio.run(main())
    sys.exit(exit_code)