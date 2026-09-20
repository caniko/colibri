/* Strict GPU residency accounting (COLI_STRICT_RESIDENCY=1): the tier must
 * report exactly which experts are not VRAM-resident, and the miss
 * predicate must fire on any routed expert the mask leaves out. The
 * engine turns a nonzero verification into a fatal refusal instead of
 * silently serving hybrid inference; these unit-test the accounting the
 * fatal path stands on, using the same fake CUDA backend as the other
 * tier tests (no GPU). Non-strict behavior is unchanged by construction
 * (G.strict defaults to 0) and stays covered by the existing suite. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../compat.h"   /* setenv/unsetenv: MinGW has neither */

#include "qwen36_fake_cuda.h"

#include "../qwen36_tier.c"

static int fails;
static void check(int ok, const char *what) {
    if (!ok) { printf("  FAIL: %s\n", what); fails++; }
}

int main(void) {
    enum { NL = 1, NE = 64, D = 64, IH = 32, TOPK = 8 };
    setenv("COLI_CUDA", "1", 1);
    setenv("COLI_STRICT_RESIDENCY", "1", 1);
    setenv("QT_NO_WARMSTART", "1", 1);
    fake_ndev = 1;

    if (!qt_init(NL, NE, D, IH, NE, TOPK, 0 /* per-row */, 1 /* int4 */)) {
        printf("  FAIL: strict tier would not start\n");
        return 1;
    }
    check(qt_strict_on() == 1, "strict flag not latched from env");

    static unsigned char g4[NE][D * IH / 2], u4[NE][D * IH / 2], d4[NE][D * IH / 2];
    static float sc[NE][2 * IH + D];
    for (int eid = 0; eid < NE; eid++) {
        memset(g4[eid], (unsigned char)(eid + 1), sizeof g4[eid]);
        memset(u4[eid], (unsigned char)(eid + 2), sizeof u4[eid]);
        memset(d4[eid], (unsigned char)(eid + 3), sizeof d4[eid]);
        for (int i = 0; i < 2 * IH + D; i++) sc[eid][i] = 1.0f;
    }

    /* Half placed: verification must report exactly the missing half. */
    for (int eid = 0; eid < NE; eid += 2)
        qt_note_block(0, eid, g4[eid], u4[eid], d4[eid], sc[eid], sc[eid] + IH, sc[eid] + 2 * IH);
    qt_fill_wait();
    check(qt_verify_full_placement() == NE / 2, "partial placement miscounted");

    /* Rest placed: full placement verifies clean. */
    for (int eid = 1; eid < NE; eid += 2)
        qt_note_block(0, eid, g4[eid], u4[eid], d4[eid], sc[eid], sc[eid] + IH, sc[eid] + 2 * IH);
    qt_fill_wait();
    check(qt_verify_full_placement() == 0, "full placement not verified");

    /* Miss predicate: any clear bit among the K routed experts fires. */
    check(qt_strict_miss(0xFFu, 8) == 0, "full mask misreported as miss");
    check(qt_strict_miss(0xFEu, 8) == 1, "single miss not detected");
    check(qt_strict_miss(0xFFFFFFFFu, 32) == 0, "32-wide full mask misreported");
    check(qt_strict_miss(0x7FFFFFFFu, 32) == 1, "32-wide miss not detected");
    check(qt_strict_miss(0u, 0) == 0, "empty routing misreported");

    if (fails == 0) printf("strict residency accounting ok\n");
    return fails != 0;
}
