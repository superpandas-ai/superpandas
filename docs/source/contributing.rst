Contributing to SuperPandas
============================

We love your input! We want to make contributing to SuperPandas as easy and transparent as possible, whether it's:

- Reporting a bug
- Discussing the current state of the code
- Submitting a fix
- Proposing new features
- Becoming a maintainer

Development Process
-------------------

We use GitHub to host code, to track issues and feature requests, as well as accept pull requests.

1. Fork the repo and create your branch from `main`
2. If you've added code that should be tested, add tests
3. If you've changed APIs, update the documentation
4. Ensure the test suite passes
5. Make sure your code lints
6. Issue that pull request!

Pull Request Process
--------------------

1. Update the README.md with details of changes to the interface, if applicable
2. Update the docs/ with any necessary documentation changes
3. The PR will be merged once you have the sign-off of at least one other developer

Any contributions you make will be under the MIT Software License
------------------------------------------------------------------

In short, when you submit code changes, your submissions are understood to be under the same `MIT License <http://choosealicense.com/licenses/mit/>`_ that covers the project. Feel free to contact the maintainers if that's a concern.

Report bugs using GitHub's issue tracker
----------------------------------------

We use GitHub issues to track public bugs. Report a bug by `opening a new issue <https://github.com/superpandas-ai/superpandas/issues/new>`_; it's that easy!

Write bug reports with detail, background, and sample code
----------------------------------------------------------

**Great Bug Reports** tend to have:

- A quick summary and/or background
- Steps to reproduce
  - Be specific!
  - Give sample code if you can
- What you expected would happen
- What actually happens
- Notes (possibly including why you think this might be happening, or stuff you tried that didn't work)

Development Setup
-----------------

1. Clone the repository:

   .. code-block:: bash

      git clone https://github.com/superpandas-ai/superpandas.git
      cd superpandas

2. Create a virtual environment:

   .. code-block:: bash

      python -m venv venv
      source venv/bin/activate  # On Windows: venv\Scripts\activate

3. Install development dependencies:

   .. code-block:: bash

      pip install -e ".[dev]"
      pip install -r requirements-docs.txt

4. Run tests:

   .. code-block:: bash

      pytest

5. Build documentation:

   .. code-block:: bash

      cd docs
      make html

Code Style
-----------

We use `black <https://github.com/psf/black>`_ for code formatting and `flake8 <https://flake8.pycqa.org/>`_ for linting. Please ensure your code follows these style guidelines.

Development Guidelines
----------------------

1. **Code Style**:
   - Use black for code formatting
   - Use flake8 for linting
   - Follow PEP 8 guidelines
   - Use type hints for function arguments and return values

2. **Testing**:
   - Write unit tests for new features
   - Ensure all tests pass before submitting PR
   - Maintain or improve test coverage
   - Use pytest for testing

3. **Documentation**:
   - Add docstrings to all new functions and classes
   - Follow NumPy docstring style
   - Update relevant documentation files
   - Add examples for new features

4. **Git Workflow**:
   - Create feature branches from main
   - Use descriptive commit messages
   - Keep commits focused and atomic
   - Rebase on main before submitting PR

5. **Pull Requests**:
   - Reference related issues
   - Include tests and documentation
   - Update changelog if needed
   - Request review from maintainers

Documentation
-------------

We use Sphinx for documentation. When adding new features or changing existing ones, please update the documentation accordingly. The documentation is built automatically on ReadTheDocs.

License
--------

By contributing, you agree that your contributions will be licensed under its MIT License. 