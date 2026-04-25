CC      = gcc
CFLAGS  = -Wall -Wextra -O2 -std=c11
TARGET  = kvstore
SRCS    = main.c store.c parser.c
OBJS    = $(SRCS:.c=.o)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

clean:
	rm -f $(OBJS) $(TARGET)

.PHONY: clean
