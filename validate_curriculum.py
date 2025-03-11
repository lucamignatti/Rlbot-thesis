#!/usr/bin/env python3
"""
Curriculum Validation Script

This script validates a curriculum to ensure all components work correctly
before starting a long training session.

Usage:
  python validate_curriculum.py [--curriculum_module module_name] [--curriculum_func function_name]
  
Example:
  python validate_curriculum.py --curriculum_module curriculum.curriculum --curriculum_func create_lucy_skg_curriculum
"""

import argparse
import importlib
import traceback
import sys
from typing import Any, Optional, Tuple
from curriculum.base import CurriculumManager

def import_module_function(module_path: str, function_name: str) -> Tuple[Any, Optional[Exception]]:
    """
    Import a function from a module path.
    
    Args:
        module_path: The path to the module (e.g. 'curriculum.curriculum')
        function_name: The name of the function (e.g. 'create_lucy_skg_curriculum')
    
    Returns:
        Tuple containing:
        - The imported function, or None if import failed
        - Any exception that occurred during import, or None on success
    """
    try:
        module = importlib.import_module(module_path)
        function = getattr(module, function_name)
        return function, None
    except ImportError as e:
        return None, e
    except AttributeError as e:
        return None, e
    except Exception as e:
        return None, e

def validate_curriculum(curriculum_module: str = "curriculum.curriculum", 
                       curriculum_func: str = "create_lucy_skg_curriculum",
                       debug: bool = True):
    """
    Validate a curriculum by creating it and running the validation function.
    
    Args:
        curriculum_module: Module path containing the curriculum creation function
        curriculum_func: Name of the function that creates the curriculum
        debug: Whether to run with debug output enabled
    
    Returns:
        bool: True if validation passed, False otherwise
    """
    print(f"\n=== CURRICULUM VALIDATION ===")
    print(f"Loading curriculum from {curriculum_module}.{curriculum_func}...")
    
    # Import the curriculum creation function
    creator_func, import_error = import_module_function(curriculum_module, curriculum_func)
    if import_error:
        print(f"ERROR: Failed to import curriculum from {curriculum_module}.{curriculum_func}")
        print(f"       {import_error}")
        return False
    
    # Create the curriculum
    try:
        print(f"Creating curriculum...")
        curriculum_manager = creator_func(debug=debug, use_wandb=False)
        if not isinstance(curriculum_manager, CurriculumManager):
            print(f"ERROR: The function did not return a CurriculumManager object")
            print(f"       Got: {type(curriculum_manager)}")
            return False
    except Exception as e:
        print(f"ERROR: Failed to create curriculum")
        print(f"       {e}")
        traceback.print_exc(file=sys.stdout)
        return False
    
    # Run validation on the curriculum
    try:
        print(f"Running validation on curriculum with {len(curriculum_manager.stages)} stages...\n")
        validation_result = curriculum_manager.validate_all_stages()
        return validation_result
    except Exception as e:
        print(f"ERROR: Exception during curriculum validation")
        print(f"       {e}")
        traceback.print_exc(file=sys.stdout)
        return False

def main():
    parser = argparse.ArgumentParser(description="Validate a curriculum before training")
    parser.add_argument("--curriculum_module", default="curriculum.curriculum", 
                        help="Module path containing curriculum (default: curriculum.curriculum)")
    parser.add_argument("--curriculum_func", default="create_lucy_skg_curriculum", 
                        help="Function name that creates curriculum (default: create_lucy_skg_curriculum)")
    parser.add_argument("--no_debug", action="store_true", 
                        help="Disable debug output from curriculum")
    args = parser.parse_args()
    
    success = validate_curriculum(
        curriculum_module=args.curriculum_module,
        curriculum_func=args.curriculum_func,
        debug=not args.no_debug
    )
    
    if success:
        print("\n✅ Curriculum validation PASSED - Your curriculum is ready for training!")
        sys.exit(0)
    else:
        print("\n❌ Curriculum validation FAILED - Please fix the issues before starting training")
        sys.exit(1)

if __name__ == "__main__":
    main()