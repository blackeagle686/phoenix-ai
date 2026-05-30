# 🚀 Publishing to PyPI

Follow these steps to make your SDK available via `pip install Phoenix AI-sdk`.

---

## 1. Prerequisites
You will need two tools installed:
```bash
pip install build twine
```

## 2. Prepare the Distribution
Run the following command in the root of your project:
```bash
python -m build
```
This will create a `dist/` folder containing two files:
- `phoenix-1.0.0-py3-none-any.whl` (The compiled wheel)
- `phoenix-1.0.0.tar.gz` (The source distribution)

## 3. Account Setup
1. Create an account on [PyPI (Production)](https://pypi.org/account/register/).
2. **Recommended**: Create an account on [TestPyPI](https://test.pypi.org/account/register/) first to verify your package looks correct.
3. Generate an **API Token** in your Account Settings. Use this instead of your password for security.

## 4. Upload to TestPyPI (Verification)
Upload your package to the test repository first:
```bash
python -m twine upload --repository testpypi dist/*
```
When prompted:
- **Username**: `__token__`
- **Password**: (Your API Token starting with `pypi-`)

Test the installation:
```bash
pip install --index-url https://test.pypi.org/simple/ --no-deps Phoenix AI-sdk
```

## 5. Upload to PyPI (Final)
Once you are happy with the test, upload to the real PyPI:
```bash
python -m twine upload dist/*
```
Again, use `__token__` as the username and your production API Token as the password.

---

## 🔧 Troubleshooting
- **Name Already Taken**: If `Phoenix AI-sdk` is already taken on PyPI, you will need to change the `name` field in `pyproject.toml` to something unique (e.g., `Phoenix AI-sdk-yourname`).
- **Version Mismatch**: You cannot upload the same version twice. Increment the `version` in `pyproject.toml` for every update.
- **Missing Files**: Check `MANIFEST.in` if your `templates/` or `static/` folders are missing after installation.

