echo on
set PATH=%LIB_PATH%;%PATH%
echo %LIB_PATH%
echo %PATH%
python -m pip install delvewheel
python setup.py bdist_wheel -d dist
pushd dist
for %%j in (*.whl) do (
  delvewheel repair %%j
  pip install wheelhouse\%%j 
  ) 
popd