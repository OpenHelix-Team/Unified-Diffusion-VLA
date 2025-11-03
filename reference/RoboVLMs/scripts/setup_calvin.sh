apt-get update -yqq

# Install dependency for calvin
apt-get -yqq install libegl1 libgl1

# Install EGL mesa
apt-get install -yqq libegl1-mesa libegl1-mesa-dev
apt-get install -yqq mesa-utils libosmesa6-dev llvm
apt-get -yqq install meson
apt-get -yqq build-dep mesa

conda install -c conda-forge gcc=12.1.0 gxx_linux-64 -y

git clone --recurse-submodules https://github.com/mees/calvin.git

CALVIN_ROOT=$(pwd)/calvin
cd ${CALVIN_ROOT}
sed -i '11d' calvin_models/requirements.txt
sed -i '12d' calvin_models/requirements.txt
sed -i '13d' calvin_models/requirements.txt
sed -i '14d' calvin_models/requirements.txt

sh install.sh

# CALVIN spesicifcally requires the following version of numpy
pip install numpy==1.21.0

# Download dataset
cd ${CALVIN_ROOT}/dataset
# sh download_data.sh debug
# sh download_data.sh D
# sh download_data.sh ABC
sh download_data.sh ABCD