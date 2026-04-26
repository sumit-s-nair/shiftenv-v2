#include <stdio.h>
#include "storage.h"
#include "reporter.h"

void generate_report(void* storage_handle) {
    /* Interdependent: Reporter needs to query Storage via opaque handle */
    int total = get_total_sum(storage_handle);
    printf("Final System Report. Total Sum Stored: %d\n", total);
}
