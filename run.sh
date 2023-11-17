cd ..
rm -rf tmp
cp -r zerohertzLib tmp
cd tmp
pip uninstall zerohertzLib -y
pip uninstall zerohertzLib -y
rm -rf build
rm -rf dist
rm -rf *.egg-info
rm -rf docs
python setup.py sdist bdist_wheel
pip install dist/*.whl
# sphinx-apidoc -f -o sphinx/source zerohertzLib --implicit-namespaces
# sed -i '/.. automodule::/a\   :private-members:' sphinx/source/*.rst
python sphinx/release_note.py
cd sphinx
rm -rf build
make html
mv build/html ../docs
cd ../docs
python -m http.server 1547