SRCDIR=src

VPATH=$(SRCDIR)

SRC=$(wildcard $(SRCDIR)/*.c)
HDR=$(wildcard $(SRCDIR)/*.h)
OBJ=$(patsubst %.c,%.o,$(SRC))
TGT=$(patsubst %.c,%,$(SRC))

LDLIBS=-lOpenCL

CFLAGS=-std=c99 -g -Wall

all: $(TGT)

$(OBJ): %.o: %.c $(HDR)

clean:
	$(RM) $(OBJ) $(TGT)
