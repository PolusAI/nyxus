import os
import sys

def link_arrow_lib():
    
    if not arrow_headers_found():
        raise ImportError("The pyarrow library was not found. Pyarrow must be installed ot link the Arrow library.")
        
    import pyarrow as pa
    
    if os.sys.platform == "win32":
        for lib_dir in pa.get_library_dirs():
            if sys.version_info[0]==3 and sys.version_info[1]>=8: 
                # since add_dll_dir is added in Python3.8
                os.add_dll_directory(lib_dir)
            else:
                os.environ['PATH'] = lib_dir + os.pathsep + os.environ['PATH']

def arrow_headers_found():
    
    try:
        import pyarrow as pa

        return True
        
    except:
        
        return False
    
    