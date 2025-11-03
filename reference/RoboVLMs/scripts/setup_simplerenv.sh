apt-get update -yqq

apt-get -yqq install libegl1-mesa libegl1
apt-get -yqq install libgl1
apt-get -yqq install libosmesa6-dev

apt-get install -yqq --no-install-recommends libvulkan-dev vulkan-tools
apt-get install -yqq ffmpeg

conda install -c conda-forge gcc=12.1.0 gxx_linux-64 -y

pip install mediapy decord

# Install numpy<2.0 
pip install numpy==1.24.4

git clone https://github.com/simpler-env/SimplerEnv --recurse-submodules

SIMPLER_ROOT=$(pwd)/SimplerEnv

cd ${SIMPLER_ROOT}/ManiSkill2_real2sim
pip install -e .
cd ..
pip install -e .
