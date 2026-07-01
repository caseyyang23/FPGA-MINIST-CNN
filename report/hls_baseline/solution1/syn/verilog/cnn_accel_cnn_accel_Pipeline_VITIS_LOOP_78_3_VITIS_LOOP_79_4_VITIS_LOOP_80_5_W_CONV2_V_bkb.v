// ==============================================================
// Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2022.2 (64-bit)
// Version: 2022.2
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// ==============================================================
`timescale 1 ns / 1 ps
module cnn_accel_cnn_accel_Pipeline_VITIS_LOOP_78_3_VITIS_LOOP_79_4_VITIS_LOOP_80_5_W_CONV2_V_bkb (
    address0, ce0, q0, 
    reset, clk);

parameter DataWidth = 8;
parameter AddressWidth = 11;
parameter AddressRange = 1152;
 
input[AddressWidth-1:0] address0;
input ce0;
output reg[DataWidth-1:0] q0;

input reset;
input clk;

 
reg [DataWidth-1:0] rom0[0:AddressRange-1];


initial begin
     
    $readmemh("./cnn_accel_cnn_accel_Pipeline_VITIS_LOOP_78_3_VITIS_LOOP_79_4_VITIS_LOOP_80_5_W_CONV2_V_bkb.dat", rom0);
end

  
always @(posedge clk) 
begin 
    if (ce0) 
    begin
        q0 <= rom0[address0];
    end
end


endmodule

