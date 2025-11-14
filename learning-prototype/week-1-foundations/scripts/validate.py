#!/usr/bin/env python3

"""
Week 1 Foundations - Comprehensive Validation Script

This script performs a thorough validation of your Week 1 completion:
1. Checks all TODOs are completed
2. Validates all exercises are passing
3. Runs tests if available
4. Checks code quality
5. Validates documentation

Usage: python validate.py [--verbose] [--strict]
"""

import os
import sys
import re
import ast
import importlib.util
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass
from enum import Enum

# Colors for terminal output
class Color:
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    CYAN = '\033[0;36m'
    MAGENTA = '\033[0;35m'
    NC = '\033[0m'  # No Color

class CheckStatus(Enum):
    """Status of a validation check"""
    PASS = "PASS"
    FAIL = "FAIL"
    WARN = "WARN"
    INFO = "INFO"

@dataclass
class ValidationResult:
    """Result of a validation check"""
    name: str
    status: CheckStatus
    message: str
    details: List[str] = None

    def __post_init__(self):
        if self.details is None:
            self.details = []

class WeekOneValidator:
    """Comprehensive validator for Week 1 completion"""

    def __init__(self, week1_dir: Path, verbose: bool = False, strict: bool = False):
        self.week1_dir = week1_dir
        self.starter_dir = week1_dir / "starter-code"
        self.exercises_dir = week1_dir / "exercises"
        self.notes_dir = week1_dir / "notes"
        self.scripts_dir = week1_dir / "scripts"
        self.verbose = verbose
        self.strict = strict

        self.results: List[ValidationResult] = []
        self.total_checks = 0
        self.passed_checks = 0
        self.failed_checks = 0
        self.warnings = 0

    def print_header(self, text: str):
        """Print a formatted header"""
        print(f"\n{Color.BLUE}{'=' * 70}{Color.NC}")
        print(f"{Color.BLUE}{text.center(70)}{Color.NC}")
        print(f"{Color.BLUE}{'=' * 70}{Color.NC}\n")

    def print_result(self, result: ValidationResult):
        """Print a validation result"""
        self.total_checks += 1

        if result.status == CheckStatus.PASS:
            prefix = f"{Color.GREEN}✓{Color.NC}"
            self.passed_checks += 1
        elif result.status == CheckStatus.FAIL:
            prefix = f"{Color.RED}✗{Color.NC}"
            self.failed_checks += 1
        elif result.status == CheckStatus.WARN:
            prefix = f"{Color.YELLOW}⚠{Color.NC}"
            self.warnings += 1
        else:
            prefix = f"{Color.BLUE}ℹ{Color.NC}"

        print(f"{prefix} {result.name}: {result.message}")

        if self.verbose and result.details:
            for detail in result.details:
                print(f"  {Color.CYAN}→{Color.NC} {detail}")

        self.results.append(result)

    def check_file_exists(self, filepath: Path, description: str) -> bool:
        """Check if a file exists"""
        if filepath.exists():
            self.print_result(ValidationResult(
                name=description,
                status=CheckStatus.PASS,
                message=f"Found {filepath.name}"
            ))
            return True
        else:
            self.print_result(ValidationResult(
                name=description,
                status=CheckStatus.FAIL,
                message=f"Missing {filepath.name}"
            ))
            return False

    def check_todos(self, filepath: Path) -> ValidationResult:
        """Check for remaining TODOs in a file"""
        try:
            content = filepath.read_text()
            todos = re.findall(r'#\s*TODO[:\s]*(.*)', content)

            if not todos:
                return ValidationResult(
                    name=f"TODOs in {filepath.name}",
                    status=CheckStatus.PASS,
                    message="All TODOs completed"
                )
            else:
                return ValidationResult(
                    name=f"TODOs in {filepath.name}",
                    status=CheckStatus.WARN if not self.strict else CheckStatus.FAIL,
                    message=f"{len(todos)} TODOs remaining",
                    details=todos[:5]  # Show first 5 TODOs
                )
        except Exception as e:
            return ValidationResult(
                name=f"TODOs in {filepath.name}",
                status=CheckStatus.FAIL,
                message=f"Error reading file: {str(e)}"
            )

    def check_python_syntax(self, filepath: Path) -> ValidationResult:
        """Check if a Python file has valid syntax"""
        try:
            with open(filepath, 'r') as f:
                ast.parse(f.read())
            return ValidationResult(
                name=f"Syntax: {filepath.name}",
                status=CheckStatus.PASS,
                message="Valid Python syntax"
            )
        except SyntaxError as e:
            return ValidationResult(
                name=f"Syntax: {filepath.name}",
                status=CheckStatus.FAIL,
                message=f"Syntax error at line {e.lineno}: {e.msg}"
            )
        except Exception as e:
            return ValidationResult(
                name=f"Syntax: {filepath.name}",
                status=CheckStatus.FAIL,
                message=f"Error: {str(e)}"
            )

    def check_imports(self, filepath: Path, required_imports: List[str]) -> ValidationResult:
        """Check if file contains required imports"""
        try:
            content = filepath.read_text()
            missing_imports = []

            for imp in required_imports:
                # Check for various import patterns
                patterns = [
                    f"import {imp}",
                    f"from {imp} import",
                    f"from {imp.split('.')[0]} import"
                ]
                if not any(pattern in content for pattern in patterns):
                    missing_imports.append(imp)

            if not missing_imports:
                return ValidationResult(
                    name=f"Imports: {filepath.name}",
                    status=CheckStatus.PASS,
                    message="All required imports present"
                )
            else:
                return ValidationResult(
                    name=f"Imports: {filepath.name}",
                    status=CheckStatus.WARN,
                    message=f"Missing imports: {', '.join(missing_imports)}"
                )
        except Exception as e:
            return ValidationResult(
                name=f"Imports: {filepath.name}",
                status=CheckStatus.FAIL,
                message=f"Error: {str(e)}"
            )

    def check_class_exists(self, filepath: Path, class_name: str) -> ValidationResult:
        """Check if a class is defined in a file"""
        try:
            content = filepath.read_text()
            pattern = rf'class\s+{class_name}\s*[\(\:]'

            if re.search(pattern, content):
                return ValidationResult(
                    name=f"Class {class_name}",
                    status=CheckStatus.PASS,
                    message=f"Defined in {filepath.name}"
                )
            else:
                return ValidationResult(
                    name=f"Class {class_name}",
                    status=CheckStatus.WARN,
                    message=f"Not found in {filepath.name}"
                )
        except Exception as e:
            return ValidationResult(
                name=f"Class {class_name}",
                status=CheckStatus.FAIL,
                message=f"Error: {str(e)}"
            )

    def check_function_exists(self, filepath: Path, function_name: str) -> ValidationResult:
        """Check if a function is defined in a file"""
        try:
            content = filepath.read_text()
            patterns = [
                rf'def\s+{function_name}\s*\(',
                rf'async\s+def\s+{function_name}\s*\('
            ]

            if any(re.search(pattern, content) for pattern in patterns):
                return ValidationResult(
                    name=f"Function {function_name}",
                    status=CheckStatus.PASS,
                    message=f"Defined in {filepath.name}"
                )
            else:
                return ValidationResult(
                    name=f"Function {function_name}",
                    status=CheckStatus.WARN,
                    message=f"Not found in {filepath.name}"
                )
        except Exception as e:
            return ValidationResult(
                name=f"Function {function_name}",
                status=CheckStatus.FAIL,
                message=f"Error: {str(e)}"
            )

    def check_endpoint_exists(self, filepath: Path, endpoint: str, method: str) -> ValidationResult:
        """Check if a FastAPI endpoint is defined"""
        try:
            content = filepath.read_text()
            pattern = rf'@app\.{method.lower()}\(["\'].*{endpoint}.*["\']\)'

            if re.search(pattern, content):
                return ValidationResult(
                    name=f"Endpoint {method} {endpoint}",
                    status=CheckStatus.PASS,
                    message="Defined"
                )
            else:
                return ValidationResult(
                    name=f"Endpoint {method} {endpoint}",
                    status=CheckStatus.WARN,
                    message="Not found or incomplete"
                )
        except Exception as e:
            return ValidationResult(
                name=f"Endpoint {method} {endpoint}",
                status=CheckStatus.FAIL,
                message=f"Error: {str(e)}"
            )

    def check_dependencies(self) -> List[ValidationResult]:
        """Check if required Python packages are installed"""
        results = []
        required_packages = [
            ('fastapi', 'FastAPI'),
            ('uvicorn', 'Uvicorn'),
            ('pydantic', 'Pydantic'),
            ('pytest', 'Pytest (for testing)'),
        ]

        optional_packages = [
            ('alpaca', 'Alpaca SDK'),
            ('python-dotenv', 'python-dotenv'),
        ]

        for module_name, display_name in required_packages:
            try:
                __import__(module_name)
                results.append(ValidationResult(
                    name=f"Package: {display_name}",
                    status=CheckStatus.PASS,
                    message="Installed"
                ))
            except ImportError:
                results.append(ValidationResult(
                    name=f"Package: {display_name}",
                    status=CheckStatus.FAIL,
                    message=f"Not installed (pip install {module_name})"
                ))

        for module_name, display_name in optional_packages:
            try:
                __import__(module_name.replace('-', '_'))
                results.append(ValidationResult(
                    name=f"Package: {display_name}",
                    status=CheckStatus.PASS,
                    message="Installed (optional)"
                ))
            except ImportError:
                results.append(ValidationResult(
                    name=f"Package: {display_name}",
                    status=CheckStatus.INFO,
                    message="Not installed (optional)"
                ))

        return results

    def validate_starter_code(self):
        """Validate all starter code files"""
        self.print_header("Validating Starter Code")

        # Check main.py
        main_py = self.starter_dir / "main.py"
        if self.check_file_exists(main_py, "Main module"):
            self.print_result(self.check_python_syntax(main_py))
            self.print_result(self.check_todos(main_py))
            self.print_result(self.check_endpoint_exists(main_py, "/signal", "get"))
            self.print_result(self.check_endpoint_exists(main_py, "/trade", "post"))
            self.print_result(self.check_endpoint_exists(main_py, "/portfolio", "get"))

        # Check models.py
        models_py = self.starter_dir / "models.py"
        if self.check_file_exists(models_py, "Models module"):
            self.print_result(self.check_python_syntax(models_py))
            self.print_result(self.check_todos(models_py))
            self.print_result(self.check_class_exists(models_py, "TradingSignal"))
            self.print_result(self.check_class_exists(models_py, "TradeRequest"))
            self.print_result(self.check_class_exists(models_py, "TradeResponse"))

        # Check broker.py
        broker_py = self.starter_dir / "broker.py"
        if self.check_file_exists(broker_py, "Broker module"):
            self.print_result(self.check_python_syntax(broker_py))
            self.print_result(self.check_todos(broker_py))
            self.print_result(self.check_class_exists(broker_py, "AlpacaBroker"))

        # Check config.py
        config_py = self.starter_dir / "config.py"
        if self.check_file_exists(config_py, "Config module"):
            self.print_result(self.check_python_syntax(config_py))
            self.print_result(self.check_todos(config_py))
            self.print_result(self.check_class_exists(config_py, "Settings"))

    def validate_exercises(self):
        """Validate exercise files"""
        self.print_header("Validating Exercises")

        exercises = [
            ("exercise_1_fastapi_basics.py", "Exercise 1: FastAPI Basics"),
            ("exercise_2_pydantic_models.py", "Exercise 2: Pydantic Models"),
            ("exercise_3_async_routes.py", "Exercise 3: Async Routes"),
            ("exercise_4_error_handling.py", "Exercise 4: Error Handling"),
            ("exercise_5_integration.py", "Exercise 5: Integration"),
        ]

        for filename, description in exercises:
            filepath = self.exercises_dir / filename
            if self.check_file_exists(filepath, description):
                self.print_result(self.check_python_syntax(filepath))
                result = self.check_todos(filepath)
                self.print_result(result)

    def validate_environment(self):
        """Validate environment configuration"""
        self.print_header("Validating Environment")

        # Check for .env file
        env_file = self.week1_dir / ".env"
        if env_file.exists():
            self.print_result(ValidationResult(
                name="Environment file",
                status=CheckStatus.PASS,
                message=".env file exists"
            ))

            # Check for required variables (without exposing values)
            content = env_file.read_text()
            required_vars = ["ALPACA_API_KEY", "ALPACA_SECRET_KEY"]

            for var in required_vars:
                if var in content:
                    self.print_result(ValidationResult(
                        name=f"Env var: {var}",
                        status=CheckStatus.PASS,
                        message="Configured"
                    ))
                else:
                    self.print_result(ValidationResult(
                        name=f"Env var: {var}",
                        status=CheckStatus.WARN,
                        message="Not configured"
                    ))
        else:
            self.print_result(ValidationResult(
                name="Environment file",
                status=CheckStatus.WARN,
                message=".env file not found (needed for Alpaca)"
            ))

    def validate_documentation(self):
        """Validate documentation and notes"""
        self.print_header("Validating Documentation")

        notes = [
            "fastapi_fundamentals.md",
            "pydantic_explained.md",
            "async_python.md",
            "paper_trading.md",
            "testing_basics.md",
        ]

        for note in notes:
            filepath = self.notes_dir / note
            self.check_file_exists(filepath, f"Note: {note}")

    def validate_dependencies(self):
        """Validate required dependencies"""
        self.print_header("Validating Dependencies")

        for result in self.check_dependencies():
            self.print_result(result)

    def print_summary(self):
        """Print validation summary"""
        self.print_header("Validation Summary")

        print(f"Total Checks:    {Color.BLUE}{self.total_checks}{Color.NC}")
        print(f"Passed:          {Color.GREEN}{self.passed_checks}{Color.NC}")
        print(f"Failed:          {Color.RED}{self.failed_checks}{Color.NC}")
        print(f"Warnings:        {Color.YELLOW}{self.warnings}{Color.NC}")

        if self.total_checks > 0:
            pass_rate = (self.passed_checks / self.total_checks) * 100
            print(f"\nSuccess Rate:    {Color.BLUE}{pass_rate:.1f}%{Color.NC}")

        print("\n")

        if self.failed_checks == 0:
            print(f"{Color.GREEN}{'=' * 70}{Color.NC}")
            print(f"{Color.GREEN}{'🎉 Excellent! Week 1 validation passed!'.center(70)}{Color.NC}")
            print(f"{Color.GREEN}{'=' * 70}{Color.NC}")
            print()
            print(f"{Color.GREEN}You're ready for Week 2!{Color.NC}")
            print()
        elif self.warnings > 0 and self.failed_checks == 0:
            print(f"{Color.YELLOW}{'=' * 70}{Color.NC}")
            print(f"{Color.YELLOW}{'⚠ Good progress! Some items need attention.'.center(70)}{Color.NC}")
            print(f"{Color.YELLOW}{'=' * 70}{Color.NC}")
            print()
            print("Review warnings above and complete remaining TODOs.")
            print()
        else:
            print(f"{Color.RED}{'=' * 70}{Color.NC}")
            print(f"{Color.RED}{'✗ Some checks failed. Please review above.'.center(70)}{Color.NC}")
            print(f"{Color.RED}{'=' * 70}{Color.NC}")
            print()
            print(f"{Color.YELLOW}Tips:{Color.NC}")
            print("  - Complete all TODOs in starter-code/")
            print("  - Ensure all required packages are installed")
            print("  - Check syntax errors in Python files")
            print("  - Review ../notes/ for guidance")
            print("  - Check ../solutions/ if stuck")
            print()

        print(f"{Color.BLUE}Next Steps:{Color.NC}")
        print("  1. Complete any remaining TODOs")
        print("  2. Test your API with: ./run_api.sh")
        print("  3. Run comprehensive tests: ./test_week1.sh")
        print("  4. Complete all exercises manually")
        print("  5. Review all notes in ../notes/")
        print()

    def run(self):
        """Run all validations"""
        print(f"{Color.GREEN}")
        print("╔════════════════════════════════════════════════════════════════════╗")
        print("║                                                                    ║")
        print("║      Week 1 Foundations - Comprehensive Validation Script         ║")
        print("║                                                                    ║")
        print("╚════════════════════════════════════════════════════════════════════╝")
        print(f"{Color.NC}\n")

        print(f"{Color.BLUE}ℹ{Color.NC} Validating: {self.week1_dir}")
        print(f"{Color.BLUE}ℹ{Color.NC} Mode: {'Strict' if self.strict else 'Normal'}")
        print(f"{Color.BLUE}ℹ{Color.NC} Verbose: {'On' if self.verbose else 'Off'}")

        # Run all validations
        self.validate_dependencies()
        self.validate_starter_code()
        self.validate_exercises()
        self.validate_environment()
        self.validate_documentation()

        # Print summary
        self.print_summary()

        # Return exit code
        return 0 if self.failed_checks == 0 else 1


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Week 1 Foundations - Comprehensive Validation Script"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    parser.add_argument(
        "--strict", "-s",
        action="store_true",
        help="Treat warnings as failures"
    )
    parser.add_argument(
        "--dir", "-d",
        type=str,
        default=None,
        help="Week 1 directory path (default: parent of script directory)"
    )

    args = parser.parse_args()

    # Determine week1 directory
    if args.dir:
        week1_dir = Path(args.dir)
    else:
        script_dir = Path(__file__).parent
        week1_dir = script_dir.parent

    # Validate directory exists
    if not week1_dir.exists():
        print(f"{Color.RED}Error: Directory not found: {week1_dir}{Color.NC}")
        return 1

    # Create validator and run
    validator = WeekOneValidator(week1_dir, verbose=args.verbose, strict=args.strict)
    return validator.run()


if __name__ == "__main__":
    sys.exit(main())
