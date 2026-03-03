/*
 * NVTX NIF for RecGPT: annotate CPU phases for nsys/Nsight Systems profiling.
 * Uses dlopen to load libnvToolsExt at runtime; no-ops when unavailable.
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdlib.h>
#include <string.h>
#include "erl_nif.h"

static ERL_NIF_TERM nif_range_push(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]);
static ERL_NIF_TERM nif_range_pop(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]);

static void *nvtx_handle = NULL;
static int (*nvtx_range_push_a)(const char *) = NULL;
static int (*nvtx_range_pop)(void) = NULL;

static int load_nvtx(void) {
  if (nvtx_handle != NULL)
    return (nvtx_range_push_a != NULL && nvtx_range_pop != NULL);

  const char *libs[] = {
    "libnvToolsExt.so.1",
    "libnvToolsExt.so",
    NULL
  };
  for (int i = 0; libs[i] != NULL; i++) {
    nvtx_handle = dlopen(libs[i], RTLD_NOW | RTLD_LOCAL);
    if (nvtx_handle) {
      nvtx_range_push_a = (int (*)(const char *))dlsym(nvtx_handle, "nvtxRangePushA");
      nvtx_range_pop = (int (*)(void))dlsym(nvtx_handle, "nvtxRangePop");
      if (nvtx_range_push_a && nvtx_range_pop)
        return 1;
      dlclose(nvtx_handle);
      nvtx_handle = NULL;
    }
  }
  return 0;
}

static ERL_NIF_TERM nif_range_push(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
  char name[256];
  if (argc != 1)
    return enif_make_badarg(env);
  if (enif_get_atom(env, argv[0], name, sizeof(name), ERL_NIF_LATIN1)) {
    /* ok */
  } else if (enif_is_list(env, argv[0])) {
    unsigned len;
    if (!enif_get_list_length(env, argv[0], &len) || len >= sizeof(name) - 1)
      return enif_make_badarg(env);
    if (!enif_get_string(env, argv[0], name, sizeof(name), ERL_NIF_LATIN1))
      return enif_make_badarg(env);
  } else if (enif_is_binary(env, argv[0])) {
    ErlNifBinary bin;
    if (!enif_inspect_binary(env, argv[0], &bin) || bin.size >= sizeof(name) - 1)
      return enif_make_badarg(env);
    memcpy(name, bin.data, bin.size);
    name[bin.size] = '\0';
  } else {
    return enif_make_badarg(env);
  }
  if (load_nvtx()) {
    nvtx_range_push_a(name);
  }
  return enif_make_atom(env, "ok");
}

static ERL_NIF_TERM nif_range_pop(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
  (void)argc;
  (void)argv;
  if (load_nvtx()) {
    nvtx_range_pop();
  }
  return enif_make_atom(env, "ok");
}

static ErlNifFunc nif_funcs[] = {
  {"range_push", 1, nif_range_push, 0},
  {"range_pop", 0, nif_range_pop, 0}
};

ERL_NIF_INIT(Elixir.RecGPT.NVTX, nif_funcs, NULL, NULL, NULL, NULL);
