#nsys profile --stats=true -t cuda jsrun -n1 -a1 -c1 -g1 ./main3d.gnu.TPROF.CUDA.runOnGpu.ex
jsrun -n1 -a1 -c1 -g1 ./main3d.gnu.TPROF.CUDA.ex
#nvprof --print-gpu-summary jsrun -n1 -a1 -c1 -g1 ./main3d.gnu.TPROF.CUDA.runOnGpu.ex
