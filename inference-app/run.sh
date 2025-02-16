mkdir -p build
cd build
export PICO_SDK_PATH=/home/kelu/projets/pico-sdk/
cmake .. -DPICO_BOARD=${pico}
make -j
cp pico_inference_app.uf2 ../
