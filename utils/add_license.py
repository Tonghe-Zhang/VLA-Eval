"""
Script to add MIT License headers to Python files.

Usage:
    python add_license.py /path/to/directory [--author "Your Name"] [--year 2025] [--dry-run]
    
Examples:
    # Add license to all .py files in a directory
    python add_license.py /home/user/project
    
    # Specify custom author and year
    python add_license.py /home/user/project --author "John Doe" --year 2024
    
    # Dry run (show what would be changed without modifying files)
    python add_license.py /home/user/project --dry-run
    
    # Process multiple directories
    python add_license.py /path/to/dir1 /path/to/dir2 /path/to/dir3
"""

import os
import sys
import argparse
from pathlib import Path


def get_license_text(author: str = "Tonghe Zhang", year: int = 2025) -> str:
    """Generate MIT License text with specified author and year."""
    return f"""# MIT License

# Copyright (c) {year} {author}

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""


def has_license(content: str) -> bool:
    """
    Check if file already contains a license header.
    Detects common license indicators.
    """
    license_indicators = [
        "Copyright (c)",
        "MIT License",
        "Apache License",
        "GNU General Public License",
        "BSD License",
        "Licensed under",
        "SPDX-License-Identifier"
    ]
    
    # Check first 50 lines for license
    lines = content.split('\n')[:50]
    first_lines = '\n'.join(lines)
    
    return any(indicator in first_lines for indicator in license_indicators)


def add_license_to_file(filepath: Path, license_text: str, dry_run: bool = False) -> tuple[bool, str]:
    """
    Add MIT License to a Python file if it doesn't already have one.
    
    Returns:
        (modified, message): Whether file was modified and a status message
    """
    try:
        # Read existing content
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check if license already exists
        if has_license(content):
            return False, f"⊘ License already present: {filepath}"
        
        # Handle shebang lines (#!/usr/bin/env python, etc.)
        lines = content.split('\n')
        shebang = ""
        start_idx = 0
        
        if lines and lines[0].startswith('#!'):
            shebang = lines[0] + '\n\n'
            start_idx = 1
            remaining_content = '\n'.join(lines[start_idx:])
        else:
            remaining_content = content
        
        # Create new content with license
        new_content = shebang + license_text + remaining_content.lstrip()
        
        if dry_run:
            return True, f"✓ Would add license to: {filepath}"
        
        # Write the new content
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        return True, f"✓ Added license to: {filepath}"
    
    except Exception as e:
        return False, f"✗ Error processing {filepath}: {e}"


def process_directory(directory: Path, license_text: str, dry_run: bool = False) -> tuple[int, int, int]:
    """
    Process all Python files in a directory recursively.
    
    Returns:
        (added, skipped, errors): Counts of files processed
    """
    if not directory.exists():
        print(f"✗ Directory does not exist: {directory}")
        return 0, 0, 0
    
    if not directory.is_dir():
        print(f"✗ Not a directory: {directory}")
        return 0, 0, 0
    
    # Find all Python files recursively
    python_files = list(directory.rglob("*.py"))
    
    if not python_files:
        print(f"⊘ No Python files found in: {directory}")
        return 0, 0, 0
    
    print(f"\n{'='*70}")
    print(f"Processing directory: {directory}")
    print(f"Found {len(python_files)} Python file(s)")
    print(f"{'='*70}\n")
    
    added = 0
    skipped = 0
    errors = 0
    
    for filepath in sorted(python_files):
        modified, message = add_license_to_file(filepath, license_text, dry_run)
        print(message)
        
        if "Error" in message:
            errors += 1
        elif modified:
            added += 1
        else:
            skipped += 1
    
    return added, skipped, errors


def main():
    parser = argparse.ArgumentParser(
        description="Add MIT License headers to Python files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        'directories',
        nargs='+',
        type=str,
        help='Directory or directories to process'
    )
    
    parser.add_argument(
        '--author',
        type=str,
        default='Tonghe Zhang',
        help='Copyright holder name (default: Tonghe Zhang)'
    )
    
    parser.add_argument(
        '--year',
        type=int,
        default=2025,
        help='Copyright year (default: 2025)'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be changed without modifying files'
    )
    
    args = parser.parse_args()
    
    # Generate license text
    license_text = get_license_text(args.author, args.year)
    
    if args.dry_run:
        print("\n" + "="*70)
        print("DRY RUN MODE - No files will be modified")
        print("="*70)
    
    # Process all directories
    total_added = 0
    total_skipped = 0
    total_errors = 0
    
    for dir_path in args.directories:
        directory = Path(dir_path).resolve()
        added, skipped, errors = process_directory(directory, license_text, args.dry_run)
        total_added += added
        total_skipped += skipped
        total_errors += errors
    
    # Print summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"Directories processed: {len(args.directories)}")
    print(f"Files with license added: {total_added}")
    print(f"Files skipped (already have license): {total_skipped}")
    print(f"Errors: {total_errors}")
    
    if args.dry_run:
        print(f"\n⚠ DRY RUN MODE - Run without --dry-run to apply changes")
    
    print(f"{'='*70}\n")
    
    # Exit with error code if there were errors
    sys.exit(1 if total_errors > 0 else 0)


if __name__ == "__main__":
    main()