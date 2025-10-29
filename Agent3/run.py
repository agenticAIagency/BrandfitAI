#!/usr/bin/env python3
"""
Agent 3 Setup and Run Script
Handles directory creation, environment setup, and execution
"""

import os
import sys
from pathlib import Path
import logging
import importlib.util  # Import the correct library

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_directory_structure():
    """Create all required directories"""
    directories = [
        "data/creators",
        "data/personas",
        "data/indices",
        "logs",
        "schemas"
    ]
    
    logger.info("Creating directory structure...")
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"  ✓ Created {directory}")


def check_environment():
    """Check if environment is properly configured"""
    logger.info("Checking environment configuration...")
    
    # Check for .env file
    if not Path(".env").exists():
        logger.warning("⚠️  No .env file found!")
        logger.info("Creating .env from .env.example...")
        
        if Path(".env.example").exists():
            import shutil
            shutil.copy(".env.example", ".env")
            logger.info("✓ Created .env file")
            logger.warning("⚠️  Please edit .env and add your GEMINI_API_KEY")
            return False
        else:
            logger.error("❌ .env.example not found!")
            return False
    
    # Check for Gemini API key
    from dotenv import load_dotenv
    load_dotenv()
    
    gemini_key = os.getenv("GEMINI_API_KEY")
    if not gemini_key or gemini_key == "your_gemini_api_key_here":
        logger.warning("⚠️  GEMINI_API_KEY not set or using placeholder")
        logger.info("The system will run with mock responses")
        logger.info("To use real Gemini API:")
        logger.info("  1. Get API key from: https://makersuite.google.com/app/apikey")
        logger.info("  2. Edit .env and set: GEMINI_API_KEY=your_actual_key")
        return True  # Can still run with mock
    
    logger.info("✓ Environment configured")
    return True


def check_dependencies():
    """Check if all required packages are installed"""
    logger.info("Checking dependencies...")
    
    # These are the package names from requirements.txt
    required_packages = {
        "fastapi": "fastapi",
        "uvicorn": "uvicorn",
        "pydantic": "pydantic",
        "numpy": "numpy",
        "google-generativeai": "google.generativeai", # Maps package name to import name
        "python-dotenv": "dotenv",
        "pyyaml": "yaml",
        "loguru": "loguru",
    }
    
    missing = []
    for package_name, import_name in required_packages.items():
        spec = importlib.util.find_spec(import_name)
        if spec is None:
            missing.append(package_name)
    
    if missing:
        logger.error(f"❌ Missing packages: {', '.join(missing)}")
        logger.info("Install missing packages with:")
        logger.info("  pip install -r requirements.txt")
        return False
    
    logger.info("✓ All dependencies installed")
    return True


def run_quickstart():
    """Run the quickstart demo"""
    logger.info("\n" + "="*70)
    logger.info("Running Agent 3 Quickstart Demo")
    logger.info("="*70 + "\n")
    
    try:
        import asyncio
        from quickstart import run_persona_builder_demo
        
        # This will fail until quickstart.py is created
        persona = asyncio.run(run_persona_builder_demo())
        
        logger.info("\n✅ Quickstart completed successfully!")
        return True
        
    except ImportError:
        logger.error("❌ quickstart.py not found. Cannot run demo.")
        return False
    except Exception as e:
        logger.error(f"❌ Quickstart failed: {e}", exc_info=True)
        return False


def run_tests():
    """Run the test suite"""
    logger.info("\n" + "="*70)
    logger.info("Running Test Suite")
    logger.info("="*70 + "\n")
    
    try:
        import pytest
        
        # Run tests with verbose output
        exit_code = pytest.main([
            "tests/",
            "-v",
            "--tb=short",
            "--color=yes"
        ])
        
        if exit_code == 0:
            logger.info("\n✅ All tests passed!")
            return True
        else:
            logger.warning(f"\n⚠️  Some tests failed (exit code: {exit_code})")
            return False
            
    except ImportError:
        logger.warning("⚠️  pytest not found. Skipping tests.")
        logger.info("Install dev dependencies with: pip install pytest")
        return False
    except Exception as e:
        logger.error(f"❌ Test execution failed: {e}", exc_info=True)
        return False


def run_api_server():
    """Start the FastAPI server"""
    logger.info("\n" + "="*750)
    logger.info("Starting FastAPI Server")
    logger.info("="*70 + "\n")
    
    try:
        import uvicorn
        from dotenv import load_dotenv
        
        load_dotenv()
        
        host = os.getenv("API_HOST", "0.0.0.0")
        port = int(os.getenv("API_PORT", 8003))
        
        logger.info(f"Server will start on http://{host}:{port}")
        logger.info("API Documentation: http://localhost:8003/docs")
        logger.info("Press Ctrl+C to stop\n")
        
        uvicorn.run(
            "main:app",
            host=host,
            port=port,
            reload=True,
            log_level="info"
        )
        
    except ImportError:
         logger.error("❌ main.py not found. Cannot start server.")
    except KeyboardInterrupt:
        logger.info("\n✓ Server stopped")
    except Exception as e:
        logger.error(f"❌ Server failed: {e}", exc_info=True)


def print_menu():
    """Print main menu"""
    print("\n" + "="*70)
    print("Agent 3: Creator Persona Builder")
    print("="*70)
    print("\nOptions:")
    print("  1. Run Quickstart Demo (build sample persona)")
    print("  2. Run Tests")
    print("  3. Start API Server")
    print("  4. Setup Only (create directories, check env)")
    print("  5. Exit")
    print("="*70)


def main():
    """Main execution function"""
    # Always create directories and check environment
    create_directory_structure()
    env_ok = check_environment()
    deps_ok = check_dependencies()
    
    if not deps_ok:
        logger.error("\n❌ Cannot proceed without dependencies")
        logger.info("Install with: pip install -r requirements.txt")
        sys.exit(1)
    
    if not env_ok:
        logger.warning("\n⚠️  Environment not fully configured")
        logger.info("System will run with mock responses")
    
    # Interactive menu
    while True:
        print_menu()
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == "1":
            run_quickstart()
            input("\nPress Enter to continue...")
            
        elif choice == "2":
            run_tests()
            input("\nPress Enter to continue...")
            
        elif choice == "3":
            run_api_server()
            
        elif choice == "4":
            logger.info("\n✓ Setup complete!")
            logger.info("Directories created and environment checked")
            input("\nPress Enter to continue...")
            
        elif choice == "5":
            logger.info("\nGoodbye!")
            sys.exit(0)
            
        else:
            logger.warning("Invalid choice. Please enter 1-5")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\n\n✓ Exited by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"\n❌ Fatal error: {e}", exc_info=True)
        sys.exit(1)

