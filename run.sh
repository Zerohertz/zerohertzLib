# sphinx-apidoc -f -o sphinx/source zerohertzLib
# sed -i '/.. automodule::/a\   :private-members:' sphinx/source/*.rst

pip uninstall zerohertzLib -y
pip uninstall zerohertzLib -y
rm -rf build
rm -rf dist
rm -rf *.egg-info
python setup.py sdist bdist_wheel
pip install dist/*.whl
cd sphinx
rm -rf build
make html
mv build/html ../docs
cd ../docs
python -m http.server