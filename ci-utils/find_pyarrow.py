import pyarrow as pa
import platform
path=pa.get_library_dirs()[0]
operating_system = platform.system()

if operating_system == 'Linux':
    path += '/libarrow_python.so'
elif operating_system == 'Darwin':
    path += '/libarrow_python.dylib'
elif operating_system == 'Windows':
    path += '\\arrow_python.lib'

print(path)