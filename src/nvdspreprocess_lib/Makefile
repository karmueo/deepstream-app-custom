################################################################################
# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
################################################################################

CUDA_VER=12.6
ifeq ($(CUDA_VER),)
  $(error "CUDA_VER is not set")
endif

TARGET_DEVICE = $(shell gcc -dumpmachine | cut -f1 -d -)
REPOSITORY_ROOT := $(abspath ../..)

CXX:= g++
NVCC:=/usr/local/cuda-$(CUDA_VER)/bin/nvcc

NVDS_PREPROCESS_SOURCE_DIR?=/opt/nvidia/deepstream/deepstream/sources/gst-plugins/gst-nvdspreprocess/nvdspreprocess_lib

LIB:=libcustom2d_preprocess.so

CFLAGS+= -fPIC -DHAVE_CONFIG_H -std=c++17 -Wall -Werror \
	 -ffile-prefix-map=$(REPOSITORY_ROOT)=. \
	 -I /usr/local/cuda-$(CUDA_VER)/include \
	 -I $(NVDS_PREPROCESS_SOURCE_DIR) \
	 -I /opt/nvidia/deepstream/deepstream/sources/gst-plugins/include \
	 -I /opt/nvidia/deepstream/deepstream/sources/gst-plugins/gst-nvdspreprocess/include \
	 -I /opt/nvidia/deepstream/deepstream/sources/includes

CFLAGS += -g
CXXFLAGS += -g

GST_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream/lib/gst-plugins/
LIB_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream/lib/

LIBS := -shared -Wl,-no-undefined \
	-L/usr/local/cuda-$(CUDA_VER)/lib64/ -lcudart -ldl \
	-L$(LIB_INSTALL_DIR) -lnvdsgst_helper -lnvdsgst_meta -lnvds_meta -lnvbufsurface -lnvbufsurftransform\
	-lcuda -Wl,-rpath,$(LIB_INSTALL_DIR)

OBJS:= nvdspreprocess_2d_lib.o nvdspreprocess_impl.o nvdspreprocess_conversion.o

PKGS:= gstreamer-1.0 gstreamer-base-1.0 gstreamer-video-1.0

CFLAGS+=$(shell pkg-config --cflags $(PKGS))
LIBS+=$(shell pkg-config --libs $(PKGS))

all: $(LIB)

nvdspreprocess_2d_lib.o: $(NVDS_PREPROCESS_SOURCE_DIR)/nvdspreprocess_lib.cpp Makefile
	@echo $(CFLAGS)
	$(CXX) -c -o $@ $(CFLAGS) $<

nvdspreprocess_impl.o: $(NVDS_PREPROCESS_SOURCE_DIR)/nvdspreprocess_impl.cpp Makefile
	@echo $(CFLAGS)
	$(CXX) -c -o $@ $(CFLAGS) $<

nvdspreprocess_conversion.o: $(NVDS_PREPROCESS_SOURCE_DIR)/nvdspreprocess_conversion.cu Makefile
	@echo $(CFLAGS)
	$(NVCC) -c -o $@ --compiler-options '-fPIC -ffile-prefix-map=$(REPOSITORY_ROOT)=.' \
		-I /usr/local/cuda-$(CUDA_VER)/include \
		-I $(NVDS_PREPROCESS_SOURCE_DIR) $<

$(LIB): $(OBJS) $(DEP) Makefile
	@echo $(CFLAGS)
	$(CXX) -o $@ $(OBJS) $(LIBS)

install: $(LIB)
	cp -rv $(LIB) $(GST_INSTALL_DIR)

clean:
	rm -rf $(OBJS) $(LIB)
