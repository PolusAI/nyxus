PYTHON_VERSIONS=("cp36-cp36m" "cp37-cp37m" "cp38-cp38" "cp39-cp39")

for PYTHON_VERSION in ${PYTHON_VERSIONS[@]}; do
    env CMAKE_ARGS="-DPython_INCLUDE_DIR=/opt/python/${PYTHON_VERSION}/include/python3.6m -DPython_LIBRARY=/opt/python/${PYTHON_VERSION}/lib/" /opt/python/${PYTHON_VERSION}/bin/python setup.py bdist_wheel -d dist

done

for whl in ./dist/*.whl; do
    auditwheel repair --plat manylinux2010_x86_64 $whl -w ./dist
    rm $whl
done