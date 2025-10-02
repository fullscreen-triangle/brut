# JSON-Tricks Replacement Guide

## Issue Resolution
The `json-tricks` package has been replaced with `jsonpickle` due to installation issues. This change provides equivalent functionality with better reliability and easier installation.

## Changes Made

### Dependencies Updated
- **Removed**: `json-tricks==3.17.3`
- **Added**: `jsonpickle==3.0.2`

### Files Modified
- `demo/requirements.txt` - Updated dependency
- `demo/setup.py` - Updated dependency
- `demo/src/utils/json_utils.py` - New utility module (created)

## Migration Guide

### For Basic JSON Operations
Most of your code already uses standard `json` library, so **no changes needed**:

```python
import json

# This continues to work as before
data = {'key': 'value'}
json_string = json.dumps(data)
loaded_data = json.loads(json_string)
```

### For Complex Object Serialization
If you need to serialize complex objects (numpy arrays, datetime, custom classes), use the new utility:

#### Old json-tricks approach:
```python
import json_tricks as json

# This would have been json-tricks usage
data = {'array': np.array([1, 2, 3]), 'date': datetime.now()}
json_string = json.dumps(data)
```

#### New jsonpickle approach:
```python
from utils.json_utils import dumps, loads, safe_json_dump, safe_json_load

# Enhanced JSON with complex object support
data = {'array': np.array([1, 2, 3]), 'date': datetime.now()}
json_string = dumps(data, indent=2)
loaded_data = loads(json_string)

# Or for file operations
safe_json_dump(data, 'output.json')
loaded_data = safe_json_load('output.json')
```

## New Utility Features

The `demo/src/utils/json_utils.py` provides:

### EnhancedJSONEncoder
- `encode(obj)` - Serialize complex objects
- `decode(json_str)` - Deserialize back to Python objects
- `save_to_file(obj, filepath)` - Save to file
- `load_from_file(filepath)` - Load from file

### NumpyJSONEncoder
- Handles numpy arrays, dtypes automatically
- Works with standard `json.dump/dumps`

### Convenience Functions
- `dumps()`, `loads()` - Drop-in replacements
- `safe_json_dump()`, `safe_json_load()` - Automatic fallback handling

## Supported Complex Types

The new system handles:
- ✅ Numpy arrays (`np.ndarray`)
- ✅ Numpy scalars (`np.int32`, `np.float64`, etc.)
- ✅ Datetime objects
- ✅ Complex nested structures
- ✅ Standard Python types (list, dict, tuple, set)

## Installation

Now you can install dependencies without issues:

```bash
pip install -r requirements.txt
# or
pip install jsonpickle==3.0.2
```

## Testing

Test the new functionality:

```bash
cd demo/src/utils/
python json_utils.py
```

This will run comprehensive tests of the JSON functionality.

## Benefits of jsonpickle over json-tricks

1. **Better Maintenance**: Actively maintained package
2. **Easier Installation**: No dependency conflicts
3. **Better Performance**: Optimized serialization
4. **More Reliable**: Handles edge cases better
5. **Cleaner Output**: More readable JSON format
6. **Backward Compatibility**: Works with existing JSON

## Migration Checklist

- ✅ Dependencies updated in requirements.txt
- ✅ Dependencies updated in setup.py  
- ✅ New utility module created
- ✅ Backward compatibility maintained
- ✅ Enhanced functionality available
- ✅ Testing functionality provided

## No Action Required

Since your existing code uses standard `json` library, **no code changes are required**. The new enhanced functionality is available if you need complex object serialization in the future.

Your S-entropy framework will continue to work exactly as before, but now with easier installation and enhanced JSON capabilities when needed.
