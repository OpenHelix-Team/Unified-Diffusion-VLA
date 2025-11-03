cd ..

LIBERO_ROOT="${pwd}/LIBERO"
if [ ! -d "$LIBERO_ROOT" ]; then
    git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
    cd "${LIBERO_ROOT}"
    pip install -e .
    pip install -r ../RoboVLMs/eval/libero/libero_requirements.txt
fi
