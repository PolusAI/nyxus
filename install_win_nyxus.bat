echo off
set PATH=C:/Program Files (x86)/tiff/bin;C:/Program Files (x86)/zlib/bin;%PATH%
set TIFF_INCLUDE_DIR=C:/Program Files (x86)/tiff/include
set TIFF_LIBRARY_RELEASE=C:/Program Files (x86)/tiff/lib/tiff.lib
python -m pip install delvewheel
python setup.py bdist_wheel -d dist
pushd dist
for %%j in (*.whl) do (
  delvewheel repair %%j
  pip install wheelhouse\%%j 
  ) 
popd