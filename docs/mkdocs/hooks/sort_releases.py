"""Hook to sort release notes in descending order."""

import re
from pathlib import Path


def on_nav(nav, config, files):
    """Sort release notes in descending version order."""
    # Find the Release Notes section
    for item in nav:
        if hasattr(item, "title") and item.title == "Release Notes":
            if hasattr(item, "children") and item.children is not None:
                # Sort children by version number in descending order
                def version_key(nav_item):
                    if hasattr(nav_item, "file") and nav_item.file:
                        # Extract version from filename (e.g., v1.2.md -> (1, 2))
                        filename = Path(nav_item.file.src_path).stem
                        match = re.match(r"v?(\d+)\.(\d+)", filename)
                        if match:
                            major, minor = match.groups()
                            return (int(major), int(minor))
                    return (0, 0)

                # Sort in descending order (highest version first)
                item.children.sort(key=version_key, reverse=True)
            break

    return nav
