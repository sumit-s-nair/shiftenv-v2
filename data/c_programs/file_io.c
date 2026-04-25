#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TMPFILE "/tmp/c2rust_test_file_io.txt"

void write_lines(const char *path, const char **lines, int n) {
    FILE *f = fopen(path, "w");
    if (!f) { perror("fopen"); exit(1); }
    for (int i = 0; i < n; i++) {
        fprintf(f, "%s\n", lines[i]);
    }
    fclose(f);
}

int count_lines(const char *path) {
    FILE *f = fopen(path, "r");
    if (!f) { perror("fopen"); exit(1); }
    int count = 0;
    char buf[256];
    while (fgets(buf, sizeof(buf), f)) count++;
    fclose(f);
    return count;
}

void print_file(const char *path) {
    FILE *f = fopen(path, "r");
    if (!f) { perror("fopen"); exit(1); }
    char buf[256];
    while (fgets(buf, sizeof(buf), f)) {
        buf[strcspn(buf, "\n")] = '\0';
        printf("%s\n", buf);
    }
    fclose(f);
}

int main(void) {
    const char *lines[] = {"alpha", "beta", "gamma"};
    write_lines(TMPFILE, lines, 3);
    printf("Lines: %d\n", count_lines(TMPFILE));
    print_file(TMPFILE);
    remove(TMPFILE);
    return 0;
}
