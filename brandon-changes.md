# Documentation for MM-TSFLIB Evaluator Testing

This guide explains how to use `uv`, a fast Python package installer and resolver, to manage your project's dependencies listed in an `environment.txt` file within a virtual environment (`venv`).

## **Prerequisites:**

*   **`uv` installed:** You need `uv` available on your system. If you don't have it, install it (visit the [official `uv` documentation](https://github.com/astral-sh/uv) for the latest methods, often via `curl` or `pipx`):
    ```bash
    # Example using curl (check official docs for the recommended way)
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Or using pipx
    pipx install uv
    ```
*   **Project Directory:** You should be working within your project's main directory in your terminal.

## **Scripts:**

**Library Installations**

``` bash
cd venv

uv pip install -r old_environment.txt 
# Old Environment File With Library Versions

uv pip install -r environment.txt
# Successful Installations of Most Libraries in old_environment.txt

uv pip install -r error_lib.txt
# Libraries From old_environment.txt but Without versions that did not has successful installations.
```
**Running Evaluators**
```bash
cd src

uv run bash run.sh
# Evaluator Script
```

## **Errors Faced**

**Library Errors**

1. Path errors with `uv pip install -r old_environment.txt`.

2. Builds failed for various libraries in the original nonfunctioning environment.txt (old_environment.txt without versions). These libraries are in `error_lib.txt`.

I attempted to patch these errors by removing versions in `old_environment.txt` thus circumventing the path errors, but some libraries still didn't work. These libraries are noted in `error_lib.txt`.

**Script Errors**

Once a functioning run.sh was created I began to run into errors relating to the actual code specifically surrounding the BERT LLM code in `src/exp/exp_long_term_forcast.py`. Attempts were made to circumvent these errors, but were reverted to maintain the majority of the directory. Furthermore the leading belief is that these errors are caused by the missing libraries mentioned earlier.