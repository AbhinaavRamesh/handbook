#!/usr/bin/env python3
"""
Master script to generate all Neural Networks visualizations.

Usage:
    python generate_all_nn_visuals.py           # Generate all
    python generate_all_nn_visuals.py --module perceptron  # Generate one module
"""

import os
import subprocess
import argparse
import sys

MODULES = [
    'perceptron',
    'activation',
    'backprop',
    'init',
    'norm',
    'cnn',
    'rnn',
    'autoencoder',
    'gan',
    'optimizer',
    'regularization',
    'gradient'
]

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def generate_module(module_name):
    """Generate visualizations for a single module."""
    script = f'generate_{module_name}_visuals.py'
    script_path = os.path.join(SCRIPT_DIR, script)

    if os.path.exists(script_path):
        print(f"\n{'='*60}")
        print(f"Generating {module_name} visualizations...")
        print('='*60)
        try:
            subprocess.run([sys.executable, script_path], check=True, cwd=SCRIPT_DIR)
            print(f"Completed {module_name}")
        except subprocess.CalledProcessError as e:
            print(f"Error generating {module_name}: {e}")
            return False
    else:
        print(f"Warning: {script} not found, skipping...")
        return False

    return True


def main():
    parser = argparse.ArgumentParser(description='Generate Neural Networks visualizations')
    parser.add_argument('--module', choices=MODULES, help='Generate specific module only')
    args = parser.parse_args()

    print("="*60)
    print("Neural Networks Visualization Generator")
    print("="*60)
    print(f"Script directory: {SCRIPT_DIR}")
    print(f"Available modules: {', '.join(MODULES)}")

    success_count = 0
    fail_count = 0

    if args.module:
        if generate_module(args.module):
            success_count += 1
        else:
            fail_count += 1
    else:
        for module in MODULES:
            if generate_module(module):
                success_count += 1
            else:
                fail_count += 1

    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    print(f"Successful: {success_count}")
    print(f"Failed/Skipped: {fail_count}")

    if fail_count == 0:
        print("\nAll visualizations generated successfully!")
    else:
        print(f"\nSome modules were skipped or failed. Check logs above.")


if __name__ == '__main__':
    main()
