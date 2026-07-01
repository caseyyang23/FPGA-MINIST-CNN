open_project mnist_cnn_hls
set_top cnn_accel

add_files cnn_accel.cpp
add_files cnn_accel.h
add_files data/weights.h
add_files data/test_images.h
add_files data/test_labels.h
add_files data/golden_layers.h
add_files -tb tb_cnn.cpp

open_solution "solution1" -flow_target vivado
create_clock -period 10 -name default

csim_design

exit
