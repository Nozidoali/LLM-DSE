import os
import dotenv
import argparse

from datetime import datetime
DATE_STR = datetime.now().strftime("%Y%m%d_%H%M%S")
# DATE_STR = "20241210_184345"

# Load environment variables
dotenv.load_dotenv()

# parse arguments
parser = argparse.ArgumentParser(description='DSE for HLS')
parser.add_argument('--benchmark', type=str, default="bradybench_17", help='benchmark name')
parser.add_argument('--folder', type=str, default="./data", help='folder name')
args = parser.parse_args()

# Control factors
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEVICE = os.getenv("DEVICE")
DEBUG: bool = False
BENCHMARK: str = args.benchmark
WORK_DIR: str = os.getenv("WORK_DIR") + f"/work_{BENCHMARK}_{DATE_STR}"
OPENAI_LOGFILE = f"{WORK_DIR}/openai.log"
C_CODE_FILE: str = os.path.join(args.folder, f"{BENCHMARK}.c")
CONFIG_FILE: str = os.path.join(args.folder, f"{BENCHMARK}.json")
PICKLE_FILE: str = os.path.join(args.folder, f"{BENCHMARK}.pickle")
COMPILE_TIMEOUT_MINUTES: int = 40
COMPILE_TIMEOUT: int = COMPILE_TIMEOUT_MINUTES * 60
ENABLE_CODE_ANAL_AGENT: bool = False

if not os.path.exists(WORK_DIR): os.makedirs(WORK_DIR)

# DSE
MAX_ITER = 100
NUM_WORKERS = 8

def print_config():
	print(f"-"*80)
	print(f"OPENAI_LOGFILE: {OPENAI_LOGFILE}")
	print(f"DEVICE: {DEVICE}")
	print(f"DEBUG: {DEBUG}")
	print(f"BENCHMARK: {BENCHMARK}")
	print(f"WORK_DIR: {WORK_DIR}")
	print(f"C_CODE_FILE: {C_CODE_FILE}")
	print(f"CONFIG_FILE: {CONFIG_FILE}")
	print(f"PICKLE_FILE: {PICKLE_FILE}")
	print(f"COMPILE_TIMEOUT: {COMPILE_TIMEOUT}")
	print(f"MAX_ITER: {MAX_ITER}")
	print(f"-"*80)

# Prompts

# Constants:
KERNEL_NAME: str = "top"
MAKEFILE_STR = f"""
# Copyright (C) 2019 Falcon Computing Solutions, Inc. - All rights reserved.
#
# Choose target FPGA platform & vendor
VENDOR=XILINX
#DEVICE=xilinx_aws-vu9p-f1-04261818_dynamic_5_0
#DEVICE=xilinx_u250_xdma_201830_2
DEVICE={DEVICE}

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

MCC_COMMON_STR = f"""
# Copyright (C) 2017 Falcon Computing Solutions, Inc. - All rights reserved.
#####################################################################

# prevent make from printing directory related messages
MAKEFLAGS += --no-print-directory

#####################################################################
# User Settings
#####################################################################


# Modify target platform to your choosing
ifeq ($(VENDOR),XILINX)
    ifeq ($(DEVICE),)
        #DEVICE=xilinx:adm-pcie-ku3:2ddr-xpr:4.0###
        #DEVICE=xilinx_vcu1525_dynamic_5_1
				#DEVICE=xilinx_u250_xdma_201830_1
        #DEVICE=xilinx_vcu1525_xdma_201830_1
        DEVICE=xilinx_u200_gen3x16_xdma_2_202110_1
    endif
    #PLATFORM=sdaccel::$(DEVICE)
    PLATFORM=vitis::$(DEVICE)
    PATH_D=/opt/xilinx/platforms/$(DEVICE)/$(DEVICE).xpfm
#-p=/opt/xilinx/platforms/xilinx_vcu1525_xdma_201830_1/xilinx_vcu1525_xdma_201830_1.xpfm
    BIN_EXT=xclbin
    SIM_ENV=XCL_EMULATION_MODE
    SIM_ENV_VAL=1
	REMOTE?=merlin1
	PROFILE_SUMMARY=profile_summary_sdx
else ifeq ($(VENDOR),INTEL)
    ifeq ($(DEVICE),)
        DEVICE=a10gx
    endif
    PLATFORM=aocl::$(DEVICE)
    BIN_EXT=aocx
    SIM_ENV=CL_CONTEXT_EMULATOR_DEVICE_ALTERA
    SIM_ENV_VAL=altera
	REMOTE?=merlin2
	PROFILE_SUMMARY=profile_summary_aocl
else
    $(error VENDOR variable not set correctly.)
endif

# Additional CXX flags used when compiling accelerated executable
CXX_ACC_FLAGS+= -D MCC_ACC -L. -Wl,-rpath=./ -fPIC
ifeq ($(VENDOR),XILINX)
    ifeq ($(XILINX_SDX),)
        ifeq ($(XILINX_VITIS),)
		$(error XILINX_SDX or XILINX_VITIS must be set when targeting Xilinx FPGA cards.)
	endif
    endif	
    CXX_ACC_FLAGS+= -L$(XILINX_SDX)/runtime/lib/x86_64 -lxilinxopencl
endif
CXX_ACC_FLAGS+= -Wl,-rpath=./ -fPIC 

CXX_ACC_SIM_FLAGS=$(CXX_ACC_FLAGS) -D MCC_SIM -D MCC_ACC_H_FILE=\"__merlin$(KERNEL_NAME)_sim.h\" \
                  -l$(KERNEL_NAME)_sim  
CXX_ACC_HW_FLAGS=$(CXX_ACC_FLAGS) -D MCC_ACC_H_FILE=\"__merlin$(KERNEL_NAME).h\" \
                  -l$(KERNEL_NAME) 
ACC_SIM_EXE=$(ACC_EXE)_sim

#####################################################################
# Merlin Compiler related settings 
#####################################################################
EXEC := merlincc
EST_OUTPUT := merlin.rpt
ACCGEN_OUTPUT=$(KERNEL_NAME)
SIMGEN_OUTPUT=$(KERNEL_NAME)_sim
ACCHW_BUILD_TIME=$(shell date +%Y%m%d_%H%M%S)
BITGEN_OUTPUT=$(KERNEL_NAME)_hw.$(BIN_EXT)
BITGEN_OUTPUT_TIMESTAMPED=$(KERNEL_NAME)_$(ACCHW_BUILD_TIME).$(BIN_EXT)
AFIGEN_OUTPUT=$(USER)_$(KERNEL_NAME)_hw
AFIGEN_OUTPUT_TIMESTAMPED=$(USER)_$(KERNEL_NAME)_hw_$(ACCHW_BUILD_TIME).awsxclbin
ACC_PKG=$(KERNEL_NAME)_pkg
ACC_PKG_FILE=$(ACC_PKG).tar.gz

#####################################################################
# DO NOT MODIFY BELOW THIS LINE 
#####################################################################

.PHONY: default clean

default:
	@echo "**** Merlin Compiler Makefile"
	@echo "**** Copyright (C) 2017 Falcon Computing Solutions, Inc. - All rights reserved."
	@echo ""
	@echo "$$ make <target>"
	@echo "Available targets:"
	@echo "           run - Compile and Run TB Executable on CPU (without Acceleration)"
	@echo "       mcc_acc - Use Merlin Compiler to compile the Acceleration Kernel code"
	@echo "    mcc_runsim - Generate kernel binary for simulation on CPU and run it"
	@echo "  mcc_estimate - Get resource and performance estimates"
	@echo "    mcc_bitgen - Generate kernel binary for Acceleration Platform"
	@echo "    mcc_afigen - Generate AFI for AWS F1 FPGA"
	@echo "     mcc_runhw - Run TB Executable on Host CPU and accelerated kernel on platform HW"
	@echo "       mcc_pkg - Package all necessary files for running accelerated application"
	@echo "         clean - Remove all output products (except for bitgen outputs)"

all: run runsim estimate bitgen 

mcc_runhw: $(BITGEN_OUTPUT) $(ACC_EXE) 
	./$(ACC_EXE) $(EXE_ARGS) $(BITGEN_OUTPUT) 

mcc_bitgen : $(BITGEN_OUTPUT)

mcc_afigen : $(AFIGEN_OUTPUT).awsxclbin


$(BITGEN_OUTPUT) : $(ACCGEN_OUTPUT).mco
	$(EXEC) $^ -o $(BITGEN_OUTPUT) $(LNK_OPT) --platform=$(PLATFORM)
	cp $(BITGEN_OUTPUT) $(BITGEN_OUTPUT_TIMESTAMPED)

mcc_estimate : $(EST_OUTPUT)

$(EST_OUTPUT) : $(ACCGEN_OUTPUT).mco
	$(EXEC) $^ --report=estimate $(LNK_OPT) -p=$(PATH_D) --kernel_frequency 250  #--platform=$(PLATFORM)

mcc_runsim: mcc_simgen $(ACC_EXE)
	$(SIM_ENV)=$(SIM_ENV_VAL) ./$(ACC_EXE) $(EXE_ARGS) $(SIMGEN_OUTPUT).$(BIN_EXT)

mcc_simgen : $(SIMGEN_OUTPUT).$(BIN_EXT)

$(SIMGEN_OUTPUT).$(BIN_EXT) : $(ACCGEN_OUTPUT).mco
	$(EXEC) $^ -march=sw_emu -D MCC_SIM -o $(SIMGEN_OUTPUT) $(LNK_OPT) --platform=$(PLATFORM)

mcc_accexe: $(ACC_EXE)

$(ACC_EXE): $(HOST_SRC_FILES) $(KERNEL_SRC_FILES) 
	$(CXX) $(CXX_FLAGS) $(CXX_ACC_HW_FLAGS) $^ -o $(ACC_EXE)

mcc_accsimexe $(ACC_SIM_EXE): $(HOST_SRC_FILES) $(KERNEL_SRC_FILES) 
	$(CXX) $(CXX_FLAGS) $(CXX_ACC_SIM_FLAGS) $^ -o $(ACC_SIM_EXE)

mcc_acc : $(ACCGEN_OUTPUT).mco

mcc_pkg : $(ACC_PKG_FILE)

$(ACC_PKG_FILE) : $(BITGEN_OUTPUT) $(ACC_EXE)
	mkdir -p $(ACC_PKG)
	cp $(ACC_EXE) $(ACC_PKG)
	cp $(BITGEN_OUTPUT) $(ACC_PKG)
	cp $(KERNEL_NAME)_hw.gbs $(ACC_PKG) 2>/dev/null || :
	cp $(AFIGEN_OUTPUT).awsxclbin $(ACC_PKG) 2>/dev/null || :
	cp lib$(KERNEL_NAME).so $(ACC_PKG)
	tar cf $(ACC_PKG).tar $(ACC_PKG)
	gzip $(ACC_PKG).tar
	echo "Package $(ACC_PKG_FILE) is ready"

$(ACCGEN_OUTPUT).mco : $(KERNEL_SRC_FILES)
	$(EXEC) -c $^ -D $(VENDOR) -o $(ACCGEN_OUTPUT) $(CMP_OPT) -p=$(PATH_D) $(KERNEL_INC_DIR) #--platform=$(PLATFORM)

$(SIMGEN_OUTPUT).mco : $(KERNEL_SRC_FILES)
	$(EXEC) -D MCC_SIM -c $^ -D $(VENDOR) -o $(SIMGEN_OUTPUT) $(CMP_OPT) -p=$(PATH_D) $(KERNEL_INC_DIR) #--platform=$(PLATFORM)

run: $(EXE)
	./$(EXE) $(EXE_ARGS)

exe $(EXE): $(HOST_SRC_FILES) $(KERNEL_SRC_FILES) 
	$(CXX) $(CXX_FLAGS) -D MCC_CPU $^ -o $(EXE)

$(HOST_SRC_FILES) :
	echo ""

$(KERNEL_SRC_FILES) :
	echo ""

$(AFIGEN_OUTPUT).awsxclbin : $(BITGEN_OUTPUT) 
	$(SDK_DIR)/../SDAccel/tools/create_sdaccel_afi.sh -xclbin=$(BITGEN_OUTPUT) \
	             -o=$(AFIGEN_OUTPUT) -s3_bucket=fcs-fpga -s3_dcp_key=dcp -s3_logs_key=log
	cp $(AFIGEN_OUTPUT).awsxclbin $(AFIGEN_OUTPUT_TIMESTAMPED) 

clean:
	rm -rfv merlin.log *.o $(SRC_DIR)/data_int*.txt $(EXE)
	rm -rf $(ACCGEN_OUTPUT).mco $(SIMGEN_OUTPUT).mco $(ACC_EXE) $(ACC_SIM_EXE) $(SIMGEN_OUTPUT).$(BIN_EXT) emconfig.json $(EST_OUTPUT) lib$(KERNEL_NAME)_sim.so lib$(KERNEL_NAME).so __merlin$(KERNEL_NAME).h


####################################################################
AOC_SIMGEN_OUTPUT=$(KERNEL_NAME)_sim_aoc.$(BIN_EXT)
AOC_EST_OUTPUT=$(KERNEL_NAME)_aoc_$(shell date +%Y%m%d_%H%M%S).aoco
AOC_BITGEN_OUTPUT=$(KERNEL_NAME)_aoc_$(shell date +%Y%m%d_%H%M%S).$(BIN_EXT)

aoc_runsim_alt: aoc_simgen $(EXE)
	$(SIM_ENV)=$(SIM_ENV_VAL) ./$(EXE) $(EXE_ARGS) $(AOC_SIMGEN_OUTPUT)

aoc_runsim: mcc_simgen aoc_simgen $(ACC_EXE)
	$(SIM_ENV)=$(SIM_ENV_VAL) ./$(ACC_EXE) $(EXE_ARGS) $(AOC_SIMGEN_OUTPUT)

aoc_simgen : $(AOC_SIMGEN_OUTPUT) 

$(AOC_SIMGEN_OUTPUT) : $(AOC_KERNEL_SRC_FILES)
	aoc -march=emulator $^ $(KERNEL_INC_DIR) -o $(AOC_SIMGEN_OUTPUT) -v

aoc_est: $(AOC_EST_OUTPUT)

$(AOC_EST_OUTPUT) : $(AOC_KERNEL_SRC_FILES)
	aoc -c $^ $(AOC_OPT) $(KERNEL_INC_DIR) -o $@ -v -v -v --report

aoc_bitgen : $(AOC_BITGEN_OUTPUT)

$(AOC_BITGEN_OUTPUT) : $(AOC_KERNEL_SRC_FILES)
	aoc $^ $(AOC_OPT) $(KERNEL_INC_DIR) -o $@ -v -v -v --report

####################################################################
XOC_SIMGEN_OUTPUT=$(KERNEL_NAME)_sim_xoc.$(BIN_EXT)
XOC_BITGEN_OUTPUT=$(KERNEL_NAME)_xoc_$(shell date +%Y%m%d_%H%M%S).$(BIN_EXT)

xoc_runsim: mcc_simgen xoc_simgen $(ACC_EXE)
	$(SIM_ENV)=$(SIM_ENV_VAL) ./$(ACC_EXE) $(EXE_ARGS) $(XOC_SIMGEN_OUTPUT)

xoc_simgen : $(XOC_SIMGEN_OUTPUT) 

$(XOC_SIMGEN_OUTPUT) : $(XOC_KERNEL_SRC_FILES)
	xocc -t sw_emu --platform $(XDEVICE) $^ $(KERNEL_INC_DIR) -o $(XOC_SIMGEN_OUTPUT)

xoc_bitgen : $(XOC_BITGEN_OUTPUT)

$(XOC_BITGEN_OUTPUT) : $(XOC_KERNEL_SRC_FILES)
	xocc -t hw --platform $(XDEVICE) $^ $(XOC_OPT) $(KERNEL_INC_DIR) -o $@


###################################
# auxilary targets							
test_remote_host: 
	ssh $(REMOTE) -t "hostname"

test_remote_fail:  test_remote_host
	@echo "test passed ! "
	@echo ""

remote_sw_copied.o : $(ACC_EXE) lib$(KERNEL_NAME).so
	ssh $(REMOTE) -t "mkdir -p `pwd`"
	scp $(ACC_EXE) lib$(KERNEL_NAME).so $(REMOTE):`pwd`
	touch remote_sw_copied.o

remote_hw_copied.o : $(BITGEN_OUTPUT)
	ssh $(REMOTE) -t "mkdir -p `pwd`"
	make -n mcc_runhw > runhw.cmd
	echo "touch finished.o" >> runhw.cmd
	scp $(BITGEN_OUTPUT) runhw.cmd $(REMOTE):`pwd`
	-scp -r ../data $(REMOTE):`pwd`/../data
	touch remote_hw_copied.o
###################################


mcc_runhw_clean: 
	rm -rf remote_hw_copied.o remote_sw_copied.o
	rm -rf runhw.log profile.mon

mcc_bitgen_remote: test_remote_host $(BITGEN_OUTPUT) mcc_accexe remote_sw_copied.o remote_hw_copied.o

mcc_runhw_remote: mcc_bitgen_remote
	-ssh $(REMOTE) -t "cd `pwd`; bash -c 'source ~/vendor.sh; chmod 777 ./runhw.cmd; rm -rf finished.o; timeout 2m ./runhw.cmd' |& tee runhw.log; echo " "; ls -al finished.o |& tee -a runhw.log"
	scp $(REMOTE):`pwd`/runhw.log .
	scp $(REMOTE):`pwd`/profile.mon .
	make $(PROFILE_SUMMARY)


profile_summary_aocl:
	@echo +--------------------------------------------------+---------------+-------------+-----------------+
	@perl $$MERLIN_COMPILER_HOME/mars-gen/scripts/aocl/analyze_profile.pl | grep 'EXECUTE TIME' 
	@echo +--------------------------------------------------+---------------+-------------+-----------------+
	@perl $$MERLIN_COMPILER_HOME/mars-gen/scripts/aocl/analyze_profile.pl | grep TOTAL_KERNEL_TIME
	@perl $$MERLIN_COMPILER_HOME/mars-gen/scripts/aocl/analyze_profile.pl | grep TOTAL_PCIE_TIME
	@echo +--------------------------------------------------+---------------+-------------+-----------------+

profile_summary_sdx:
	@echo +--------------------------------------------------+---------------+-------------+-----------------+
	@python $$MERLIN_COMPILER_HOME/mars-gen/scripts/sdaccel/analyze_profile.py sdaccel_profile_summary.csv | grep 'Compute Unit Utilization'
	@python $$MERLIN_COMPILER_HOME/mars-gen/scripts/sdaccel/analyze_profile.py sdaccel_profile_summary.csv | grep 'Data Transfer' 
	@echo +--------------------------------------------------+---------------+-------------+-----------------+


test_profile:
	echo $$MERLIN_COMPILER_HOME
	-$(PROFILE_CMD) 

mcc_runhw_watch: 
	ssh $(REMOTE) -t "watch 'echo PID user cpu mem time cmd; ps a o pid= -o user= -o %cpu -o %mem -o time -o cmd | grep -e $(ACC_EXE) | grep -v grep' "
"""
