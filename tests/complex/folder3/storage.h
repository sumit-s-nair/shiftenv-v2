#ifndef STORAGE_H
#define STORAGE_H

void* create_storage(void);
void save_value(void* storage_handle, int value);
int get_total_sum(void* storage_handle);
void free_storage(void* storage_handle);

#endif
