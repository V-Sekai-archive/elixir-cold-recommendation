# RecGPT NVTX NIF - optional markers for nsys/Nsight Systems profiling
# Uses dlopen; no link-time dependency on libnvToolsExt
# NIFs are loaded by the VM; only need -ldl for our dlopen

UNAME_SYS := $(shell uname -s)
PREFIX = $(MIX_APP_PATH)
PRIV = $(PREFIX)/priv
BUILD = build

ERL_CFLAGS ?= -I$(ERTS_INCLUDE_DIR) -I$(ERL_EI_INCLUDE_DIR)
CFLAGS += -fPIC -O2 -Wall -Wextra
LDFLAGS += -shared -ldl

.PHONY: all clean

all: $(PRIV)/recgpt_nvtx.so

$(PRIV)/recgpt_nvtx.so: $(BUILD)/recgpt_nvtx.o
	@mkdir -p $(PRIV)
	$(CC) $(BUILD)/recgpt_nvtx.o $(LDFLAGS) -o $(PRIV)/recgpt_nvtx.so

$(BUILD)/recgpt_nvtx.o: recgpt_nvtx.c
	@mkdir -p $(BUILD)
	$(CC) $(CFLAGS) $(ERL_CFLAGS) -c recgpt_nvtx.c -o $(BUILD)/recgpt_nvtx.o

clean:
	rm -rf $(BUILD) $(PRIV)/recgpt_nvtx.so
