#include "sensor.h"
#include "processor.h"
#include "storage.h"
#include "reporter.h"

int main() {
    void* my_sensor = create_sensor(10);
    void* my_proc = create_processor(2);
    void* my_storage = create_storage();

    for (int i = 0; i < 3; i++) {
        int raw = poll_sensor(my_sensor);
        int processed = process_data(my_proc, raw);
        save_value(my_storage, processed);
    }

    generate_report(my_storage);

    free_sensor(my_sensor);
    free_processor(my_proc);
    free_storage(my_storage);
    return 0;
}
