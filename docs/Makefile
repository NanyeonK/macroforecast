# Minimal Makefile for Sphinx documentation
# Targets match the standard Sphinx pattern used by sphinx-quickstart.

SPHINXBUILD   = sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build

.PHONY: help html clean cleanhtml

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

html:
	$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS) $(O)

clean:
	rm -rf "$(BUILDDIR)"

# cleanhtml: full clean rebuild.
# Run this whenever the package version changes (pyproject.toml /
# macroforecast/__init__.py bumped). Sphinx incremental builds do not
# re-render unchanged pages when only __version__ changes, so the version
# metadata in the HTML footer and docsearch meta tags will be stale on pages
# whose source .md file has not been modified.
cleanhtml:
	rm -rf "$(BUILDDIR)/html"
	$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS) $(O)
