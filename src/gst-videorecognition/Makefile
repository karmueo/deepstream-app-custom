CUDA_VER?=12.6
ifeq ($(CUDA_VER),)
  $(error "CUDA_VER is not set")
endif
TARGET_DEVICE = $(shell gcc -dumpmachine | cut -f1 -d -)
CXX:= g++


SRCS:= gstvideorecognition.cpp

INCS:= $(wildcard *.h)
LIB:=libgst_videorecognition.so

NVDS_VERSION:=7.1

CFLAGS+= -fPIC -DDS_VERSION=\"7.1.0\" \
	 -I /usr/local/cuda-$(CUDA_VER)/include \
	 -I /opt/nvidia/deepstream/deepstream/sources/includes \
	 -I /usr/include/gstreamer-1.0

GST_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib/gst-plugins/
LIB_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib/

LIBS := -shared -Wl,-no-undefined \
	-L/usr/local/cuda-$(CUDA_VER)/lib64/ -lcudart -ldl \
	-lnppc -lnppig -lnpps -lnppicc -lnppidei \
	-L$(LIB_INSTALL_DIR) -lnvdsgst_helper -lnvdsgst_meta -lnvds_meta -lnvbufsurface -lnvbufsurftransform\
	-Wl,-rpath,$(LIB_INSTALL_DIR)

OBJS:= $(SRCS:.cpp=.o)

PKGS:= gstreamer-1.0 gstreamer-base-1.0 gstreamer-video-1.0

CFLAGS+=$(shell pkg-config --cflags $(PKGS))
LIBS+=$(shell pkg-config --libs $(PKGS))

CFLAGS += -g
CXXFLAGS += -g

all: $(LIB)

%.o: %.cpp $(INCS) Makefile
	@echo $(CFLAGS)
	$(CXX) -c -o $@ $(CFLAGS) $<

$(LIB): $(OBJS) $(DEP) Makefile
	@echo $(CFLAGS)
	$(CXX) -o $@ $(OBJS) $(LIBS)

install: $(LIB)
	cp -rv $(LIB) $(GST_INSTALL_DIR)

clean:
	rm -rf $(OBJS) $(LIB)
