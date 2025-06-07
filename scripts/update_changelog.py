#!/usr/bin/env python3
import sys
import datetime
from pathlib import Path

def update_markdown_changelog(change_type, description):
    """Update the Markdown changelog."""
    changelog_path = Path("CHANGELOG.md")
    
    if not changelog_path.exists():
        print("Error: CHANGELOG.md not found!")
        sys.exit(1)
    
    with open(changelog_path, 'r') as f:
        content = f.read()
    
    # Find the Unreleased section
    unreleased_section = "## [Unreleased]"
    if unreleased_section not in content:
        print("Error: Could not find [Unreleased] section in CHANGELOG.md")
        sys.exit(1)
    
    # Add the new entry
    new_entry = f"\n### {change_type}\n- {description}"
    content = content.replace(unreleased_section, unreleased_section + new_entry)
    
    with open(changelog_path, 'w') as f:
        f.write(content)

def update_rst_changelog(change_type, description):
    """Update the RST changelog."""
    changelog_path = Path("docs/source/changelog.rst")
    
    if not changelog_path.exists():
        print("Error: changelog.rst not found!")
        sys.exit(1)
    
    with open(changelog_path, 'r') as f:
        content = f.read()
    
    # Find the Unreleased section
    unreleased_section = "[Unreleased]"
    if unreleased_section not in content:
        print("Error: Could not find [Unreleased] section in changelog.rst")
        sys.exit(1)
    
    # Add the new entry
    new_entry = f"\n{change_type}\n~~~~~\n- {description}"
    content = content.replace(unreleased_section, unreleased_section + new_entry)
    
    with open(changelog_path, 'w') as f:
        f.write(content)

def update_changelog(change_type, description):
    """
    Add a new entry to both changelog formats.
    
    Args:
        change_type (str): Type of change (Added, Changed, Deprecated, Removed, Fixed, Security)
        description (str): Description of the change
    """
    update_markdown_changelog(change_type, description)
    update_rst_changelog(change_type, description)
    print(f"Successfully added changelog entry: {change_type} - {description}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python update_changelog.py <change_type> <description>")
        print("Example: python update_changelog.py Added 'New feature for data processing'")
        sys.exit(1)
    
    change_type = sys.argv[1]
    description = sys.argv[2]
    
    valid_types = ["Added", "Changed", "Deprecated", "Removed", "Fixed", "Security"]
    if change_type not in valid_types:
        print(f"Error: Invalid change type. Must be one of: {', '.join(valid_types)}")
        sys.exit(1)
    
    update_changelog(change_type, description) 