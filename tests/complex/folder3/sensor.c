#include <stdlib.h>
#include "sensor.h"

struct Sensor {
    int current_val;
};

void* create_sensor(int seed) {
    struct Sensor* s = (struct Sensor*)malloc(sizeof(struct Sensor));
    if (s) s->current_val = seed;
    return s;
}

int poll_sensor(void* sensor_handle) {
    struct Sensor* s = (struct Sensor*)sensor_handle;
    s->current_val += 5; /* Simulate changing data */
    return s->current_val;
}

void free_sensor(void* sensor_handle) {
    free(sensor_handle);
}
