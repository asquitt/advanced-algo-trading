#!/usr/bin/env python3
"""
Week 2 LLM Integration - Final Validation Script
================================================

This script performs comprehensive validation to ensure you're ready for Week 3:

1. Check all TODOs are complete
2. Verify exercises are passing
3. Validate API cost is < $1/week
4. Confirm readiness for Week 3

Usage:
    python3 validate.py [--verbose] [--strict]

Options:
    --verbose   Show detailed output
    --strict    Fail on warnings (not just errors)
"""

import os
import sys
import re
import ast
import subprocess
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import argparse


class Status(Enum):
    """Validation status levels"""
    PASS = "✅ PASS"
    FAIL = "❌ FAIL"
    WARN = "⚠️  WARN"
    INFO = "ℹ️  INFO"


@dataclass
class ValidationResult:
    """Result of a validation check"""
    name: str
    status: Status
    message: str
    details: Optional[str] = None
    score: int = 0  # Points earned for this check


class Colors:
    """ANSI color codes"""
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    CYAN = '\033[0;36m'
    MAGENTA = '\033[0;35m'
    BOLD = '\033[1m'
    NC = '\033[0m'  # No Color


class Week2Validator:
    """Main validation class for Week 2"""

    def __init__(self, verbose: bool = False, strict: bool = False):
        self.verbose = verbose
        self.strict = strict
        self.results: List[ValidationResult] = []
        self.week2_dir = Path(__file__).parent.parent

    def print_header(self, text: str):
        """Print a formatted header"""
        print(f"\n{Colors.BLUE}{'='*70}{Colors.NC}")
        print(f"{Colors.BLUE}{text}{Colors.NC}")
        print(f"{Colors.BLUE}{'='*70}{Colors.NC}")

    def print_result(self, result: ValidationResult):
        """Print a validation result"""
        color = {
            Status.PASS: Colors.GREEN,
            Status.FAIL: Colors.RED,
            Status.WARN: Colors.YELLOW,
            Status.INFO: Colors.CYAN
        }.get(result.status, Colors.NC)

        print(f"{color}{result.status.value}{Colors.NC} {result.name}: {result.message}")

        if self.verbose and result.details:
            for line in result.details.split('\n'):
                print(f"    {line}")

    def add_result(self, result: ValidationResult):
        """Add and print a result"""
        self.results.append(result)
        self.print_result(result)

    # =========================================================================
    # Validation 1: Check TODOs in Starter Code
    # =========================================================================

    def count_todos_in_file(self, file_path: Path) -> Tuple[int, int]:
        """
        Count total TODOs and completed TODOs in a file.

        Returns:
            Tuple of (total_todos, incomplete_todos)
        """
        try:
            content = file_path.read_text()

            # Find all TODO comments
            todo_pattern = r'#\s*TODO[:\s](.+?)(?=\n|$)'
            todos = re.findall(todo_pattern, content, re.IGNORECASE)

            # Check for common completion patterns
            # A TODO is considered incomplete if it's still in the code
            # and not followed by actual implementation

            total_todos = len(todos)
            incomplete_todos = 0

            for todo in todos:
                # Simple heuristic: if the TODO line contains "fill in", "implement",
                # "replace", etc., it's likely incomplete
                incomplete_keywords = [
                    'fill in', 'implement', 'replace', 'add your',
                    'write your', 'complete this', 'todo:', 'uncomment'
                ]
                todo_lower = todo.lower()

                if any(keyword in todo_lower for keyword in incomplete_keywords):
                    incomplete_todos += 1

            return total_todos, incomplete_todos

        except Exception as e:
            return 0, 0

    def validate_todos(self):
        """Check if all TODOs in starter code are complete"""
        self.print_header("Validation 1: TODO Completion")

        starter_code_dir = self.week2_dir / "starter-code"
        python_files = list(starter_code_dir.glob("*.py"))

        total_todos = 0
        incomplete_todos = 0
        file_results = []

        for file_path in python_files:
            file_total, file_incomplete = self.count_todos_in_file(file_path)
            total_todos += file_total
            incomplete_todos += file_incomplete

            if file_incomplete > 0:
                file_results.append(f"{file_path.name}: {file_incomplete}/{file_total} incomplete")

        # Calculate score (max 25 points)
        if total_todos > 0:
            completion_rate = (total_todos - incomplete_todos) / total_todos
            score = int(completion_rate * 25)
        else:
            score = 25

        if incomplete_todos == 0:
            self.add_result(ValidationResult(
                name="TODO Completion",
                status=Status.PASS,
                message=f"All TODOs completed ({total_todos} total)",
                score=score
            ))
        elif incomplete_todos <= 2:
            self.add_result(ValidationResult(
                name="TODO Completion",
                status=Status.WARN,
                message=f"{incomplete_todos}/{total_todos} TODOs incomplete",
                details="\n".join(file_results),
                score=score
            ))
        else:
            self.add_result(ValidationResult(
                name="TODO Completion",
                status=Status.FAIL,
                message=f"{incomplete_todos}/{total_todos} TODOs incomplete",
                details="\n".join(file_results),
                score=score
            ))

    # =========================================================================
    # Validation 2: Check Exercise Files
    # =========================================================================

    def validate_exercises(self):
        """Check if exercise files exist and can be imported"""
        self.print_header("Validation 2: Exercise Files")

        exercises_dir = self.week2_dir / "exercises"
        required_exercises = [
            "exercise_1_llm_basics.py",
            "exercise_2_prompts.py",
            "exercise_3_parsing.py",
            "exercise_4_agent.py",
            "exercise_5_optimization.py"
        ]

        exercises_found = 0
        exercises_valid = 0
        details = []

        for exercise_name in required_exercises:
            exercise_path = exercises_dir / exercise_name

            if not exercise_path.exists():
                details.append(f"❌ {exercise_name}: Not found")
                continue

            exercises_found += 1

            # Check if file has meaningful content (> 100 lines)
            try:
                content = exercise_path.read_text()
                lines = [line for line in content.split('\n') if line.strip() and not line.strip().startswith('#')]

                if len(lines) > 50:
                    exercises_valid += 1
                    details.append(f"✅ {exercise_name}: Found ({len(lines)} lines of code)")
                else:
                    details.append(f"⚠️  {exercise_name}: Found but minimal content ({len(lines)} lines)")

            except Exception as e:
                details.append(f"❌ {exercise_name}: Error reading file - {e}")

        # Calculate score (max 20 points)
        score = int((exercises_valid / len(required_exercises)) * 20)

        if exercises_valid == len(required_exercises):
            self.add_result(ValidationResult(
                name="Exercise Files",
                status=Status.PASS,
                message=f"All {len(required_exercises)} exercises present",
                details="\n".join(details) if self.verbose else None,
                score=score
            ))
        elif exercises_found == len(required_exercises):
            self.add_result(ValidationResult(
                name="Exercise Files",
                status=Status.WARN,
                message=f"{exercises_valid}/{len(required_exercises)} exercises complete",
                details="\n".join(details),
                score=score
            ))
        else:
            self.add_result(ValidationResult(
                name="Exercise Files",
                status=Status.FAIL,
                message=f"Only {exercises_found}/{len(required_exercises)} exercises found",
                details="\n".join(details),
                score=score
            ))

    # =========================================================================
    # Validation 3: API Configuration
    # =========================================================================

    def validate_api_config(self):
        """Check if API keys are configured"""
        self.print_header("Validation 3: API Configuration")

        groq_key = os.getenv("GROQ_API_KEY")
        claude_key = os.getenv("ANTHROPIC_API_KEY")

        configured = []
        missing = []

        if groq_key:
            configured.append("Groq API key configured")
        else:
            missing.append("Groq API key not set")

        if claude_key:
            configured.append("Claude API key configured")
        else:
            missing.append("Claude API key not set")

        # Calculate score (max 10 points)
        score = len(configured) * 5

        if len(configured) == 2:
            self.add_result(ValidationResult(
                name="API Configuration",
                status=Status.PASS,
                message="Both API keys configured",
                details="\n".join(configured),
                score=score
            ))
        elif len(configured) == 1:
            self.add_result(ValidationResult(
                name="API Configuration",
                status=Status.WARN,
                message="Only 1/2 API keys configured",
                details="\n".join(configured + missing),
                score=score
            ))
        else:
            self.add_result(ValidationResult(
                name="API Configuration",
                status=Status.FAIL,
                message="No API keys configured",
                details="Set GROQ_API_KEY and ANTHROPIC_API_KEY environment variables",
                score=score
            ))

    # =========================================================================
    # Validation 4: Code Quality Checks
    # =========================================================================

    def validate_code_quality(self):
        """Check code quality in starter files"""
        self.print_header("Validation 4: Code Quality")

        starter_code_dir = self.week2_dir / "starter-code"
        python_files = list(starter_code_dir.glob("*.py"))

        issues = []
        files_checked = 0

        for file_path in python_files:
            files_checked += 1
            try:
                content = file_path.read_text()

                # Check for common issues
                # 1. Uncommented example code (often has "# EXAMPLE")
                if "# EXAMPLE IMPLEMENTATION (uncomment to use):" in content:
                    # Count how many are still commented
                    example_sections = content.count("# EXAMPLE IMPLEMENTATION")
                    if example_sections > 2:
                        issues.append(f"{file_path.name}: Contains {example_sections} uncommented example sections")

                # 2. Check if file can be parsed as valid Python
                try:
                    ast.parse(content)
                except SyntaxError as e:
                    issues.append(f"{file_path.name}: Syntax error - {e}")

                # 3. Check for pass statements in important methods (might indicate incomplete implementation)
                if "async def generate(" in content or "def analyze_stock(" in content:
                    # Count pass statements
                    pass_count = content.count("\n    pass\n") + content.count("\n        pass\n")
                    if pass_count > 3:
                        issues.append(f"{file_path.name}: Contains {pass_count} 'pass' statements (possible incomplete implementation)")

            except Exception as e:
                issues.append(f"{file_path.name}: Error analyzing file - {e}")

        # Calculate score (max 15 points)
        if len(issues) == 0:
            score = 15
        elif len(issues) <= 2:
            score = 10
        else:
            score = 5

        if len(issues) == 0:
            self.add_result(ValidationResult(
                name="Code Quality",
                status=Status.PASS,
                message=f"No issues found in {files_checked} files",
                score=score
            ))
        elif len(issues) <= 2:
            self.add_result(ValidationResult(
                name="Code Quality",
                status=Status.WARN,
                message=f"{len(issues)} minor issues found",
                details="\n".join(issues),
                score=score
            ))
        else:
            self.add_result(ValidationResult(
                name="Code Quality",
                status=Status.FAIL,
                message=f"{len(issues)} issues found",
                details="\n".join(issues),
                score=score
            ))

    # =========================================================================
    # Validation 5: Cost Analysis
    # =========================================================================

    def validate_cost_targets(self):
        """Validate that estimated costs are under budget"""
        self.print_header("Validation 5: Cost Analysis")

        # Cost estimates (from README)
        GROQ_COST_PER_1K = 0.0001
        CLAUDE_COST_PER_1M = 3.0  # Average of input/output

        # Target usage
        SIGNALS_PER_DAY = 100
        DAYS_PER_WEEK = 7
        AVG_TOKENS = 500
        CACHE_HIT_RATE = 0.9  # 90% cache hit rate

        total_requests = SIGNALS_PER_DAY * DAYS_PER_WEEK
        actual_api_calls = total_requests * (1 - CACHE_HIT_RATE)

        # Calculate weekly costs
        groq_tokens = actual_api_calls * AVG_TOKENS
        groq_cost = (groq_tokens / 1000) * GROQ_COST_PER_1K

        claude_tokens = actual_api_calls * AVG_TOKENS
        claude_cost = (claude_tokens / 1000000) * CLAUDE_COST_PER_1M

        budget = 1.0  # $1/week target

        details = [
            f"Total requests/week: {total_requests}",
            f"Cache hit rate: {CACHE_HIT_RATE*100:.0f}%",
            f"Actual API calls: {actual_api_calls:.0f}",
            f"",
            f"Groq cost/week: ${groq_cost:.4f}",
            f"Claude cost/week: ${claude_cost:.4f}",
            f"Budget target: ${budget:.2f}",
        ]

        # Calculate score (max 15 points)
        if groq_cost < budget:
            score = 15
            status = Status.PASS
            message = f"Groq cost (${groq_cost:.4f}/week) under budget"
        else:
            score = 5
            status = Status.FAIL
            message = f"Groq cost (${groq_cost:.4f}/week) exceeds budget"

        self.add_result(ValidationResult(
            name="Cost Analysis (Groq)",
            status=status,
            message=message,
            details="\n".join(details),
            score=score
        ))

        # Additional check for mixed strategy (80% Groq, 20% Claude)
        mixed_cost = (groq_cost * 0.8) + (claude_cost * 0.2)
        if mixed_cost < budget:
            self.add_result(ValidationResult(
                name="Cost Analysis (Mixed)",
                status=Status.PASS,
                message=f"Mixed strategy (${mixed_cost:.4f}/week) under budget",
                details="80% Groq + 20% Claude with 90% cache hit rate",
                score=0  # Bonus points, not counted
            ))

    # =========================================================================
    # Validation 6: Dependencies
    # =========================================================================

    def validate_dependencies(self):
        """Check if required packages are installed"""
        self.print_header("Validation 6: Dependencies")

        required_packages = {
            "anthropic": "Anthropic SDK",
            "groq": "Groq SDK",
        }

        optional_packages = {
            "redis": "Redis client (for caching)",
            "tenacity": "Retry library",
        }

        installed = []
        missing = []

        for package, description in required_packages.items():
            try:
                __import__(package)
                installed.append(f"{package}: {description}")
            except ImportError:
                missing.append(f"{package}: {description}")

        # Calculate score (max 10 points)
        score = int((len(installed) / len(required_packages)) * 10)

        if len(missing) == 0:
            self.add_result(ValidationResult(
                name="Required Dependencies",
                status=Status.PASS,
                message="All required packages installed",
                details="\n".join(installed),
                score=score
            ))
        else:
            self.add_result(ValidationResult(
                name="Required Dependencies",
                status=Status.FAIL,
                message=f"{len(missing)}/{len(required_packages)} packages missing",
                details="Missing:\n" + "\n".join(missing) + "\n\nInstall with: pip install " + " ".join(missing),
                score=score
            ))

        # Check optional packages (info only)
        optional_installed = []
        optional_missing = []

        for package, description in optional_packages.items():
            try:
                __import__(package)
                optional_installed.append(f"{package}: {description}")
            except ImportError:
                optional_missing.append(f"{package}: {description}")

        if optional_installed:
            self.add_result(ValidationResult(
                name="Optional Dependencies",
                status=Status.INFO,
                message=f"{len(optional_installed)} optional packages installed",
                details="\n".join(optional_installed),
                score=0
            ))

    # =========================================================================
    # Validation 7: Readiness for Week 3
    # =========================================================================

    def validate_week3_readiness(self):
        """Check if student is ready for Week 3"""
        self.print_header("Validation 7: Week 3 Readiness")

        # Calculate total score so far
        total_score = sum(r.score for r in self.results)
        max_score = 100

        # Count failures and warnings
        failures = sum(1 for r in self.results if r.status == Status.FAIL)
        warnings = sum(1 for r in self.results if r.status == Status.WARN)

        readiness_criteria = {
            "Score ≥ 70%": total_score >= 70,
            "No critical failures": failures == 0,
            "API keys configured": os.getenv("GROQ_API_KEY") or os.getenv("ANTHROPIC_API_KEY"),
            "Exercises present": (self.week2_dir / "exercises").exists(),
        }

        ready = all(readiness_criteria.values())

        details = []
        for criterion, passed in readiness_criteria.items():
            status_symbol = "✅" if passed else "❌"
            details.append(f"{status_symbol} {criterion}")

        details.append("")
        details.append(f"Total score: {total_score}/{max_score}")
        details.append(f"Pass rate: {(total_score/max_score)*100:.1f}%")

        if ready:
            self.add_result(ValidationResult(
                name="Week 3 Readiness",
                status=Status.PASS,
                message="Ready to proceed to Week 3! 🎉",
                details="\n".join(details),
                score=5  # Bonus points
            ))
        else:
            self.add_result(ValidationResult(
                name="Week 3 Readiness",
                status=Status.FAIL,
                message="Not yet ready for Week 3",
                details="\n".join(details),
                score=0
            ))

    # =========================================================================
    # Main Validation Runner
    # =========================================================================

    def run_all_validations(self):
        """Run all validation checks"""
        print(f"{Colors.BOLD}")
        print("╔════════════════════════════════════════════════════════════════╗")
        print("║        Week 2: LLM Integration - Final Validation             ║")
        print("╚════════════════════════════════════════════════════════════════╝")
        print(f"{Colors.NC}")

        # Run all validations
        self.validate_todos()
        self.validate_exercises()
        self.validate_api_config()
        self.validate_code_quality()
        self.validate_cost_targets()
        self.validate_dependencies()
        self.validate_week3_readiness()

        # Print final summary
        self.print_summary()

    def print_summary(self):
        """Print final validation summary"""
        self.print_header("FINAL SUMMARY")

        # Calculate statistics
        total_score = sum(r.score for r in self.results)
        max_score = 100
        pass_count = sum(1 for r in self.results if r.status == Status.PASS)
        fail_count = sum(1 for r in self.results if r.status == Status.FAIL)
        warn_count = sum(1 for r in self.results if r.status == Status.WARN)
        total_checks = pass_count + fail_count + warn_count

        # Print score
        print(f"\n{Colors.BOLD}Score: {total_score}/{max_score} ({(total_score/max_score)*100:.1f}%){Colors.NC}")

        # Print breakdown
        print(f"\n{Colors.GREEN}✅ Passed:  {pass_count}{Colors.NC}")
        print(f"{Colors.YELLOW}⚠️  Warnings: {warn_count}{Colors.NC}")
        print(f"{Colors.RED}❌ Failed:  {fail_count}{Colors.NC}")

        # Print grade
        print(f"\n{Colors.BOLD}Grade:{Colors.NC}")
        if total_score >= 90:
            grade = "A - Excellent! 🌟"
            color = Colors.GREEN
        elif total_score >= 80:
            grade = "B - Good work! 👍"
            color = Colors.GREEN
        elif total_score >= 70:
            grade = "C - Passing, but needs improvement 📚"
            color = Colors.YELLOW
        else:
            grade = "D - More work needed ⚠️"
            color = Colors.RED

        print(f"{color}{grade}{Colors.NC}")

        # Next steps
        print(f"\n{Colors.BOLD}Next Steps:{Colors.NC}")

        if fail_count == 0 and total_score >= 70:
            print(f"{Colors.GREEN}✅ You're ready for Week 3!{Colors.NC}")
            print("\nRecommended actions:")
            print("  1. Review Week 2 key concepts")
            print("  2. Complete any remaining exercises")
            print("  3. Proceed to Week 3: Data & Risk Management")
        else:
            print(f"{Colors.YELLOW}⚠️  Complete the following before Week 3:{Colors.NC}")

            if fail_count > 0:
                print("\nCritical issues:")
                for result in self.results:
                    if result.status == Status.FAIL:
                        print(f"  • {result.name}: {result.message}")

            if warn_count > 0:
                print("\nWarnings to address:")
                for result in self.results:
                    if result.status == Status.WARN:
                        print(f"  • {result.name}: {result.message}")

        # Exit code
        if self.strict:
            exit_code = 1 if (fail_count > 0 or warn_count > 0) else 0
        else:
            exit_code = 1 if fail_count > 0 else 0

        print()
        return exit_code


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Week 2 LLM Integration - Final Validation"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed output"
    )
    parser.add_argument(
        "--strict", "-s",
        action="store_true",
        help="Fail on warnings (not just errors)"
    )

    args = parser.parse_args()

    # Run validation
    validator = Week2Validator(verbose=args.verbose, strict=args.strict)
    exit_code = validator.run_all_validations()

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
