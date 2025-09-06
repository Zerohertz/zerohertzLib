"""Generate API reference pages for MkDocs."""

import mkdocs_gen_files

# Define the modules to document
modules = ["algorithm", "api", "mlops", "monitoring", "plot", "quant", "util", "vision"]

# Generate main module page
with mkdocs_gen_files.open("api/zerohertzLib.md", "w") as f:
    f.write(
        """# zerohertzLib

::: zerohertzLib
    options:
        show_root_heading: true
        show_root_toc_entry: false
        heading_level: 2
        members_order: source
        show_source: false
        show_bases: true
        filters:
          - "!^_"
          - "!MIT License"
"""
    )

# Generate individual module pages
for module in modules:
    module_path = f"api/{module}.md"

    with mkdocs_gen_files.open(module_path, "w") as f:
        f.write(
            f"""# {module.title()}

::: zerohertzLib.{module}
    options:
        show_root_heading: true
        show_root_toc_entry: false
        heading_level: 2
        members_order: source
        show_source: false
        show_bases: true  
        filters:
          - "!^_"
          - "!MIT License"
"""
        )

# Generate navigation
nav_content = """# API Reference

## Modules

* [Overview](zerohertzLib.md)
"""

for module in modules:
    nav_content += f"* [{module.title()}]({module}.md)\n"

with mkdocs_gen_files.open("api/SUMMARY.md", "w") as nav_file:
    nav_file.write(nav_content)
