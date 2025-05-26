cd cpp_im2ht
rm -rf build
python setup.py clean
python setup.py build
python setup.py install --user
cd ..

