#ifndef PROCESSOR_H
#define PROCESSOR_H

void* create_processor(int multiplier);
int process_data(void* proc_handle, int raw_data);
void free_processor(void* proc_handle);

#endif
