// tool for formatting text for use in latex code

#include <stdio.h>
#include <string.h>

int main() {
    FILE* in_ptr = fopen("input.txt", "r");
    FILE* out_ptr = fopen("output.txt", "w");
    char in_row[1000], out_row[1000];
    char* b = in_row;
    b = fgets(in_row, 1000, in_ptr);
    // debugging
    // printf("%s", in_row);
    while (b != NULL){
        if (strlen(b) == 1){
            out_row[0] = '\n';
            out_row[1] = '\0';
            fwrite(out_row, 1, 1, out_ptr);
        }
        else {
            out_row[0] = '\\';
            out_row[1] = 'i';
            out_row[2] = 't';
            out_row[3] = 'e';
            out_row[4] = 'm';
            out_row[5] = ' ';
            for (int i=2 ; i<1000 ; i++){
                out_row[i+4] = in_row[i];
                if (in_row[i] == '\0'){
                    break;
                    out_row[i+4] = '\n';
                    out_row[i+5] = '\0';
                }
            }
            fwrite(out_row, strlen(out_row), 1, out_ptr);
        }
        b = fgets(in_row, 1000, in_ptr);
    }

    fclose(out_ptr);
    fclose(in_ptr);
}
