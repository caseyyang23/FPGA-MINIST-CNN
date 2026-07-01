############################################################
## This file is generated automatically by Vitis HLS.
## Please DO NOT edit it.
## Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
############################################################
open_project hls_minst_cnn
set_top cnn_accel
add_files mnist_fpga_cnn/hls/cnn_accel.cpp
add_files mnist_fpga_cnn/hls/cnn_accel.h
add_files mnist_fpga_cnn/hls/data/weights.h
add_files -tb mnist_fpga_cnn/hls/data/test_labels.h -cflags "-Wno-unknown-pragmas" -csimflags "-Wno-unknown-pragmas"
add_files -tb mnist_fpga_cnn/hls/data/test_images.h -cflags "-Wno-unknown-pragmas" -csimflags "-Wno-unknown-pragmas"
add_files -tb mnist_fpga_cnn/hls/tb_cnn.cpp -cflags "-Wno-unknown-pragmas" -csimflags "-Wno-unknown-pragmas"
add_files -tb mnist_fpga_cnn/hls/data/golden_layers.h -cflags "-Wno-unknown-pragmas" -csimflags "-Wno-unknown-pragmas"
open_solution "solution3" -flow_target vivado
set_part {xc7z020-clg400-1}
create_clock -period 10 -name default
source "./hls_minst_cnn/solution3/directives.tcl"
csim_design
csynth_design
cosim_design
export_design -format ip_catalog
