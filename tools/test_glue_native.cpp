/* Native validation of the WASM flat-array glue contract.
   Compiles the exact same sources that Emscripten builds, so if this passes,
   the JS marshaling (js/simapi.js) matches the C ABI.

   Build & run:
     clang++ -std=c++17 -O2 -o /tmp/test_glue tools/test_glue_native.cpp cpp/sim.cpp cpp/l2.cpp cpp/wasm_glue.cpp
     /tmp/test_glue
*/

#include <cstdio>
#include <cmath>

extern "C" {
int gotham_wasm_simulate(const double* m, const double* g, const double* c, double* out);
int gotham_wasm_l2_simulate(const double* m, const double* g, const double* c,
                            double* out, char* names, int names_cap);
void gotham_wasm_memory(const double* m, const double* c, double* out);
}

static int failures = 0;

static void check(double actual, double expected, double tol, const char* label) {
  if (std::fabs(actual - expected) / expected > tol) {
    std::printf("FAIL %s: got %.4f expected %.4f\n", label, actual, expected);
    failures++;
  } else {
    std::printf("ok   %s = %.4f\n", label, actual);
  }
}

int main() {
  /* LLaMA-3 8B */
  const double model[11] = {8.03, 0.0, 32, 4096, 32, 0, 128256, 14336, 0, 0, 0};
  /* H100 */
  const double gpu1[5] = {989.5, 1979.0, 67.0, 3350.0, 80.0};
  const double gpu2[17] = {989.5, 1979.0, 67.0, 3350.0, 80.0,
                           132, 1.83, 228 * 1024.0, 256 * 1024.0, 256 * 1024.0,
                           50 * 1024.0 * 1024.0, 7000.0, 0.0, 128.0, 2048, 64, 32};
  const double cfg1[8] = {2, 1, 2048, 1, 1, 0, 1.0, 1.0};
  double out[35];
  if (!gotham_wasm_simulate(model, gpu1, cfg1, out)) { std::printf("L1 failed\n"); return 1; }
  check(out[5], 2032.08, 0.02, "prefill intensity");
  check(out[6], 989.5e12, 0.01, "prefill achieved");
  check(out[18], 1.0, 0.05, "decode intensity");
  check(out[22], 196.0, 0.06, "decode tok/s");
  check(out[33], 295.37, 0.01, "ridge");

  double mem[4];
  gotham_wasm_memory(model, cfg1, mem);
  check(mem[0], 16.06e9, 0.01, "weights bytes");

  const double cfg2[17] = {0, 1, 2048, 1, 1, 0, 1.0, 1.0, 1, 1, 1, 64, 64, 0.8, 4, 512, 64};
  double out2[313];
  char names[24 * 32];
  if (!gotham_wasm_l2_simulate(model, gpu2, cfg2, out2, names, 24 * 32)) { std::printf("L2 failed\n"); return 1; }
  check(out2[295], 27751.0, 0.05, "L2 prefill tok/s");
  check(out2[312], 1.0, 0.0, "L2 valid");

  const double cfg2d[17] = {1, 1, 2048, 1, 1, 0, 1.0, 1.0, 1, 1, 1, 64, 64, 0.8, 4, 512, 64};
  if (!gotham_wasm_l2_simulate(model, gpu2, cfg2d, out2, names, 24 * 32)) { std::printf("L2 decode failed\n"); return 1; }
  check(out2[295], 194.0, 0.10, "L2 decode tok/s");
  std::printf("kernels: %s | %s | %s\n", names, names + 32, names + 64);

  if (failures) { std::printf("%d FAILURES\n", failures); return 1; }
  std::printf("All native glue checks passed.\n");
  return 0;
}
