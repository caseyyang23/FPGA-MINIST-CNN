// ==============================================================
// Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2022.2 (64-bit)
// Version: 2022.2
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// ==============================================================
`timescale 1 ns / 1 ps
module cnn_accel_cnn_accel_Pipeline_VITIS_LOOP_128_1_VITIS_LOOP_129_2_VITIS_LOOP_130_3_W_FC1_Vcud (
    address0, ce0, q0, 
    reset, clk);

parameter DataWidth = 8;
parameter AddressWidth = 15;
parameter AddressRange = 25088;
 
input[AddressWidth-1:0] address0;
input ce0;
output reg[DataWidth-1:0] q0;

input reset;
input clk;

 
reg [DataWidth-1:0] rom0[0:AddressRange-1];


initial begin
     
    $readmemh("./cnn_accel_cnn_accel_Pipeline_VITIS_LOOP_128_1_VITIS_LOOP_129_2_VITIS_LOOP_130_3_W_FC1_Vcud.dat", rom0);
end

  
always @(posedge clk) 
begin 
    if (ce0) 
    begin
        q0 <= rom0[address0];
    end
end


endmodule

