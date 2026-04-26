#include <stdio.h>
#include "counter.h"
#include "ops.h"

int main() {
    void* my_counter = create_counter(10);

    increment_counter(my_counter, 5);
    printf("Counter value: %d\n", get_value(my_counter));

    free_counter(my_counter);
    return 0;
}
