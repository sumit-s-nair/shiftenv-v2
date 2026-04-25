#include <stdio.h>
#include "protocol.h"

void protocol_init() {
    printf("Protocol init\n");
}

void protocol_encode(const char *msg) {
    printf("Encoded: %s\n", msg);
}
