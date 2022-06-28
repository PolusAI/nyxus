echo on
set PATH=D:/a/nyxus/nyxus/build/temp.win-amd64-3.10/Release/local_install/bin;%PATH%
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