#ifndef SENSOR_H
#define SENSOR_H

void* create_sensor(int seed);
int poll_sensor(void* sensor_handle);
void free_sensor(void* sensor_handle);

#endif
