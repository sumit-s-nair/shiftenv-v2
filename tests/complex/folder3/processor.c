#include <stdlib.h>
#include "processor.h"

struct Processor {
    int multiplier;
};

void* create_processor(int multiplier) {
    struct Processor* p = (struct Processor*)malloc(sizeof(struct Processor));
    if (p) p->multiplier = multiplier;
    return p;
}

int process_data(void* proc_handle, int raw_data) {
    return raw_data * ((struct Processor*)proc_handle)->multiplier;
}

void free_processor(void* proc_handle) {
    free(proc_handle);
}
