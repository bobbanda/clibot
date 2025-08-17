# Fix for Ultralytics Installation Hash Mismatch Error

The error you're encountering is a hash mismatch for the scipy package. This typically happens due to:
1. Corrupted download
2. Network interruption
3. Proxy/firewall issues

## Solution Steps:

### 1. Clear pip cache and retry
```bash
# In your activated virtual environment
pip cache purge
pip install --no-cache-dir ultralytics
```

### 2. If that fails, try with increased timeout
```bash
pip install --no-cache-dir --timeout 300 ultralytics
```

### 3. If still failing, install scipy separately first
```bash
# Install scipy with retry
pip install --no-cache-dir --timeout 300 "scipy>=1.4.1"

# Then install ultralytics
pip install --no-cache-dir ultralytics
```

### 4. Alternative: Use different index URL
```bash
pip install --no-cache-dir --index-url https://pypi.org/simple/ ultralytics
```

### 5. If hash mismatch persists, force reinstall without hash checking (use with caution)
```bash
pip install --no-cache-dir --force-reinstall --no-deps scipy
pip install --no-cache-dir ultralytics
```

### 6. Nuclear option: Download wheels manually
```bash
# Download scipy wheel manually
wget https://files.pythonhosted.org/packages/db/0a/92b1de4a7adc7a15dcf5bddc6e191f6f29ee663b30511ce20467ef9b82e4/scipy-1.15.3-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl

# Verify the file (optional)
sha256sum scipy-1.15.3-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl

# Install from local file
pip install scipy-1.15.3-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl

# Then install ultralytics
pip install ultralytics
```

## Notes:
- The hash mismatch suggests the downloaded file is corrupted
- This often happens on slow or unstable connections
- Using `--no-cache-dir` ensures fresh downloads
- The `--timeout` option helps with slow connections