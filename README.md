## About the Project

A comprehensive implementation of matrix decomposition techniques, which form a foundation of scientific and engineering applications. 

The project combines mathematical intuition with visual exploration through Marimo (reactive notebooks), allowing users to gain deeper insights into each implementation.

> This is the first draft of the project. More implementations—such as **Householder Reflection**, **Bidiagonalization**, **LU Decomposition**, and others—will be added soon.


## 🧪 Testing

To test the export process, run `.github/scripts/build.py` from the root directory.

```bash
uv run .github/scripts/build.py
```

This will export all notebooks in a folder called `_site/` in the root directory. Then to serve the site, run:

```bash
python -m http.server -d _site
```

This will serve the site at `http://localhost:8000`.
