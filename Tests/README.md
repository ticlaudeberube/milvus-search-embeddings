# Test Documentation

## Installation

Install required testing packages:
```bash
pip install pytest coverage
```

## Running Tests

### Single Test File
```bash
# Run specific test file
python -m pytest Tests/test_MilvusUtils.py

# Run with verbose output
python -m pytest Tests/test_MilvusUtils.py -v
```

### All Tests
```bash
# Run all tests in Tests directory
python -m pytest Tests/

# Run with verbose output
python -m pytest Tests/ -v
```

## Coverage Reports

### Generate Coverage
```bash
# Run tests with coverage
python -m coverage run -m pytest Tests/test_MilvusUtils.py

# Generate coverage report
python -m coverage report -m

# Generate HTML coverage report
python -m coverage html
```

### View Coverage
- Terminal report: `python -m coverage report -m`
- HTML report: Open `htmlcov/index.html` in browser