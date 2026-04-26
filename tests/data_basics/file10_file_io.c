#include <stdio.h>
#include <stdlib.h>

int main() {
    FILE *fp = fopen("output.txt", "w");
    if (fp == NULL) {
        printf("Error opening file\n");
        return 1;
    }
    
    fprintf(fp, "Hello, world!\n");
    fclose(fp);
    
    return 0;
}
