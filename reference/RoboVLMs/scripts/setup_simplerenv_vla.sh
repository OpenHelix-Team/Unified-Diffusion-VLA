apt-get update -yqq

apt-get -yqq install libegl1-mesa libegl1
apt-get -yqq install libgl1
apt-get -yqq install libosmesa6-dev

apt-get install -yqq --no-install-recommends libvulkan-dev vulkan-tools
apt-get install -yqq ffmpeg

