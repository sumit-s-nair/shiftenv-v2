#include "network.h"
#include "protocol.h"
#include "../common/log.h"

void network_init() {
    protocol_init();
    log_info("Network init");
}

void network_send(const char *msg) {
    protocol_encode(msg);
    log_info("Network send");
}

void network_shutdown() {
    log_info("Network shutdown");
}
