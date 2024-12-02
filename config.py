
# Control factors
DEBUG: bool = False
C_CODE_FILE: str = "./data/gemm-p.c"
CONFIG_FILE: str = "./data/gemm-p_ds_config.json"
WORK_DIR: str = "./work"

# DSE
MAX_ITER = 5

# Prompts

# Constants:
KERNEL_NAME: str = "gemm-p"
MAKEFILE_STR = f"""
# Copyright (C) 2019 Falcon Computing Solutions, Inc. - All rights reserved.
#
# Choose target FPGA platform & vendor
VENDOR=XILINX
#DEVICE=xilinx_aws-vu9p-f1-04261818_dynamic_5_0
#DEVICE=xilinx_u250_xdma_201830_2

#DEVICE=xilinx_vcu1525_xdma_201830_1
# Host Code Compilation settings
#HOST_SRC_FILES=./src/digitrec_host.cpp ./src/util.cpp

# Executable names and arguments
EXE=test
ACC_EXE=test_acc
# Testing mode
EXE_ARGS= data

CXX=g++
CXX_INC_DIRS=-I ./ -I $(MACH_COMMON_DIR)
CXX_FLAGS+= $(CXX_INC_DIRS) -Wall -O3 -std=c++11
ifeq ($(VENDOR),XILINX)
CXX_FLAGS +=-lstdc++ -L$(XILINX_SDX)/lib/lnx64.o
endif

CFLAGS=-I $(XILINX_HLS)/include

# Accelerated Kernel settings
KERNEL_NAME={KERNEL_NAME}
KERNEL_SRC_FILES=./src/{KERNEL_NAME}.c
KERNEL_INC_DIR=$(CXX_INC_DIRS)

# MerlinCC Options
CMP_OPT=-d11 --attribute burst_total_size_threshold=36700160 --attribute burst_single_size_threshold=36700160 -funsafe-math-optimizations
LNK_OPT=-d11

ROOT_DIR:=$(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

MCC_COMMON_DIR=$(ROOT_DIR)/../mcc_common
include $(MCC_COMMON_DIR)/mcc_common.mk
"""