#include "processor.h"
#include "parser.h"
#include "../common/log.h"

void processor_init() {
    log_info("Processor init");
}

void processor_process() {
    const char *data = "123";
    int value = parse_int(data);
    log_info("Processing complete");
}

void processor_shutdown() {
    log_info("Processor shutdown");
}
