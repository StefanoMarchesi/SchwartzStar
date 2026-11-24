#include <libproc.h>
#include <inttypes.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>

int main(int argc, char **argv) {
    const bool current = argc == 3 && strcmp(argv[2], "--current") == 0;
    if (argc != 2 && !current) {
        fprintf(stderr, "Usage: %s PID [--current]\n", argv[0]);
        return 2;
    }
    struct rusage_info_v4 info = {0};
    if (proc_pid_rusage(atoi(argv[1]), RUSAGE_INFO_V4, (rusage_info_t *) &info) != 0) {
        return 1;
    }
    const uint64_t peak = info.ri_lifetime_max_phys_footprint;
    const uint64_t footprint = !current && peak > info.ri_phys_footprint ? peak : info.ri_phys_footprint;
    printf("%" PRIu64 "\n", footprint);
    return 0;
}
