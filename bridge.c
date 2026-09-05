#include "bridge.h"
#include <stdio.h>

#ifdef _WIN32
#include <windows.h>
#define strdup _strdup

static wchar_t *wide_path(const char *path) {
  int n = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, path, -1, NULL, 0);
  if (!n) return NULL;
  wchar_t *result = malloc((size_t)n * sizeof(wchar_t));
  if (result && !MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, path, -1, result, n)) {
    free(result);
    return NULL;
  }
  return result;
}
#else
#include <dlfcn.h>
#endif

void oc_close(oc_runtime *r) {
  if (!r) return;
  if (r->memory) r->api->ReleaseMemoryInfo(r->memory);
  if (r->env) r->api->ReleaseEnv(r->env);
  if (r->library) {
#ifdef _WIN32
    FreeLibrary((HMODULE)r->library);
#else
    dlclose(r->library);
#endif
  }
  free(r);
}

int oc_open(const char *path, oc_runtime **out, char **error) {
  *out = NULL;
  *error = NULL;
  oc_runtime *r = calloc(1, sizeof(*r));
  if (!r) {
    *error = strdup("allocate runtime: out of memory");
    return -1;
  }
  const OrtApiBase *(ORT_API_CALL *get_api)(void) = NULL;
#ifdef _WIN32
  wchar_t *wide = wide_path(path);
  if (!wide) {
    *error = strdup("invalid UTF-8 library path or out of memory");
    goto fail;
  }
  r->library = LoadLibraryExW(wide, NULL, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
  free(wide);
  if (r->library) get_api = (const OrtApiBase *(ORT_API_CALL *)(void))GetProcAddress((HMODULE)r->library, "OrtGetApiBase");
  if (!get_api) {
    char message[96];
    snprintf(message, sizeof(message), "load ONNX Runtime: Windows error %lu", (unsigned long)GetLastError());
    *error = strdup(message);
    goto fail;
  }
#else
  r->library = dlopen(path, RTLD_NOW | RTLD_LOCAL);
  if (!r->library) {
    *error = strdup(dlerror());
    goto fail;
  }
  get_api = (const OrtApiBase *(*)(void))dlsym(r->library, "OrtGetApiBase");
  if (!get_api) {
    *error = strdup("shared library does not export OrtGetApiBase");
    goto fail;
  }
#endif
  const OrtApiBase *base = get_api();
  r->api = base ? base->GetApi(ORT_API_VERSION) : NULL;
  if (!r->api) {
    *error = strdup("ONNX Runtime must support C API version 29 (1.29 or newer)");
    goto fail;
  }
  r->version = base->GetVersionString();
  OrtStatus *status = r->api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "onnxcraft", &r->env);
  if (!status) status = r->api->DisableTelemetryEvents(r->env);
  if (!status) status = r->api->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &r->memory);
  if (!status) status = r->api->GetAllocatorWithDefaultOptions(&r->allocator);
  if (status) {
    int code = r->api->GetErrorCode(status);
    *error = strdup(r->api->GetErrorMessage(status));
    r->api->ReleaseStatus(status);
    oc_close(r);
    return code;
  }
  *out = r;
  return 0;
fail:
  oc_close(r);
  return -1;
}

// Each fallible initialization has one cleanup path. Status ownership always
// passes to Go; no C function prints, aborts, or calls back into Go.
#define CHECK(call) do { status = (call); if (status) goto fail; } while (0)

OrtStatus *oc_options_new(oc_runtime *r, const oc_options *config, OrtSessionOptions **out) {
  const OrtApi *a = r->api;
  OrtSessionOptions *o = NULL;
  OrtStatus *status = NULL;
  CHECK(a->CreateSessionOptions(&o));
  CHECK(a->SetIntraOpNumThreads(o, config->intra_threads));
  CHECK(a->SetInterOpNumThreads(o, config->inter_threads));
  CHECK(a->SetSessionExecutionMode(o, config->parallel ? ORT_PARALLEL : ORT_SEQUENTIAL));
  const GraphOptimizationLevel levels[] = {ORT_ENABLE_ALL, ORT_DISABLE_ALL, ORT_ENABLE_BASIC, ORT_ENABLE_EXTENDED};
  CHECK(a->SetSessionGraphOptimizationLevel(o, levels[config->optimization]));
  *out = o;
  return NULL;
fail:
  if (o) a->ReleaseSessionOptions(o);
  return status;
}

OrtStatus *oc_config(oc_runtime *r, OrtSessionOptions *o, const char *key, const char *value) {
  return r->api->AddSessionConfigEntry(o, key, value);
}

OrtStatus *oc_provider(oc_runtime *r, OrtSessionOptions *o, const char *name, const char **keys, const char **values, size_t count) {
  const OrtApi *a = r->api;
  OrtStatus *status;
  if (strcmp(name, "CUDA") == 0) {
    OrtCUDAProviderOptionsV2 *p = NULL;
    status = a->CreateCUDAProviderOptions(&p);
    if (!status) status = a->UpdateCUDAProviderOptions(p, keys, values, count);
    if (!status) status = a->SessionOptionsAppendExecutionProvider_CUDA_V2(o, p);
    if (p) a->ReleaseCUDAProviderOptions(p);
    return status;
  }
  if (strcmp(name, "TensorRT") == 0) {
    OrtTensorRTProviderOptionsV2 *p = NULL;
    status = a->CreateTensorRTProviderOptions(&p);
    if (!status) status = a->UpdateTensorRTProviderOptions(p, keys, values, count);
    if (!status) status = a->SessionOptionsAppendExecutionProvider_TensorRT_V2(o, p);
    if (p) a->ReleaseTensorRTProviderOptions(p);
    return status;
  }
  if (strcmp(name, "OpenVINO") == 0)
    return a->SessionOptionsAppendExecutionProvider_OpenVINO_V2(o, keys, values, count);
  return a->SessionOptionsAppendExecutionProvider(o, name, keys, values, count);
}

void oc_options_free(oc_runtime *r, OrtSessionOptions *o) { r->api->ReleaseSessionOptions(o); }

void oc_session_free(oc_session *s) {
  if (!s) return;
  const OrtApi *a = s->runtime->api;
  if (s->handle) a->ReleaseSession(s->handle);
  if (s->ports) {
    for (size_t i = 0; i < s->inputs + s->outputs; i++) {
      if (s->ports[i].name) s->runtime->allocator->Free(s->runtime->allocator, s->ports[i].name);
      free(s->ports[i].shape);
    }
  }
  free(s->ports);
  free(s->names);
  free(s);
}

OrtStatus *oc_load(oc_runtime *r, const char *path, const void *data, size_t length, const OrtSessionOptions *options, oc_session **out) {
  const OrtApi *a = r->api;
  oc_session *s = calloc(1, sizeof(*s));
  if (!s) return a->CreateStatus(ORT_FAIL, "allocate session: out of memory");
  s->runtime = r;
  OrtStatus *status = NULL;
  OrtTypeInfo *info = NULL;
  if (path) {
#ifdef _WIN32
    wchar_t *wide = wide_path(path);
    if (!wide) {
      status = a->CreateStatus(ORT_FAIL, "invalid UTF-8 model path or out of memory");
      goto fail;
    }
    status = a->CreateSession(r->env, wide, options, &s->handle);
    free(wide);
    if (status) goto fail;
#else
    CHECK(a->CreateSession(r->env, path, options, &s->handle));
#endif
  } else {
    CHECK(a->CreateSessionFromArray(r->env, data, length, options, &s->handle));
  }
  CHECK(a->SessionGetInputCount(s->handle, &s->inputs));
  CHECK(a->SessionGetOutputCount(s->handle, &s->outputs));
  size_t n = s->inputs + s->outputs;
  if (n < s->inputs || !s->outputs || n > SIZE_MAX / sizeof(oc_port)) {
    status = a->CreateStatus(ORT_FAIL, "invalid model input/output count");
    goto fail;
  }
  s->ports = calloc(n, sizeof(*s->ports));
  s->names = calloc(n, sizeof(*s->names));
  if (!s->ports || !s->names) {
    status = a->CreateStatus(ORT_FAIL, "allocate model metadata: out of memory");
    goto fail;
  }
  for (size_t i = 0; i < n; i++) {
    oc_port *p = &s->ports[i];
    if (i < s->inputs) {
      CHECK(a->SessionGetInputName(s->handle, i, r->allocator, &p->name));
      CHECK(a->SessionGetInputTypeInfo(s->handle, i, &info));
    } else {
      CHECK(a->SessionGetOutputName(s->handle, i - s->inputs, r->allocator, &p->name));
      CHECK(a->SessionGetOutputTypeInfo(s->handle, i - s->inputs, &info));
    }
    s->names[i] = p->name;
    ONNXType kind;
    CHECK(a->GetOnnxTypeFromTypeInfo(info, &kind));
    if (kind != ONNX_TYPE_TENSOR) {
      status = a->CreateStatus(ORT_NOT_IMPLEMENTED, "model inputs and outputs must be dense tensors");
      goto fail;
    }
    const OrtTensorTypeAndShapeInfo *tensor = NULL;
    CHECK(a->CastTypeInfoToTensorInfo(info, &tensor));
    CHECK(a->GetTensorElementType(tensor, &p->type));
    CHECK(a->GetDimensionsCount(tensor, &p->rank));
    if (p->rank) {
      if (p->rank > SIZE_MAX / sizeof(int64_t) || !(p->shape = malloc(p->rank * sizeof(int64_t)))) {
        status = a->CreateStatus(ORT_FAIL, "allocate tensor dimensions: out of memory");
        goto fail;
      }
      CHECK(a->GetDimensions(tensor, p->shape, p->rank));
    }
    a->ReleaseTypeInfo(info);
    info = NULL;
  }
  *out = s;
  return NULL;
fail:
  if (info) a->ReleaseTypeInfo(info);
  oc_session_free(s);
  return status;
}

void oc_outputs_free(oc_session *s, oc_tensor *outputs) {
  for (size_t i = 0; i < s->outputs; i++) {
    if (outputs[i].value) s->runtime->api->ReleaseValue(outputs[i].value);
    memset(&outputs[i], 0, sizeof(outputs[i]));
  }
}

static OrtStatus *run(oc_session *s, OrtRunOptions *options, const oc_tensor *inputs, const oc_tensor *preallocated, oc_tensor *outputs) {
  const OrtApi *a = s->runtime->api;
  size_t n = s->inputs + s->outputs;
  OrtValue *local[16] = {0};
  OrtValue **values = n <= 16 ? local : calloc(n, sizeof(*values));
  if (!values) return a->CreateStatus(ORT_FAIL, "allocate run values: out of memory");
  OrtStatus *status = NULL;
  // ORT requires a non-NULL data address even for a zero-element tensor.
  static uint64_t empty;
  for (size_t i = 0; i < n; i++) {
    if (i >= s->inputs && !preallocated) break;
    const oc_tensor *t = i < s->inputs ? &inputs[i] : &preallocated[i - s->inputs];
    CHECK(a->CreateTensorWithDataAsOrtValue(s->runtime->memory, t->bytes ? t->data : &empty,
      t->bytes, t->shape, t->rank, t->type, &values[i]));
  }
  CHECK(a->Run(s->handle, options, (const char *const *)s->names, (const OrtValue *const *)values,
    s->inputs, (const char *const *)(s->names + s->inputs), s->outputs, values + s->inputs));
  if (!preallocated) {
    for (size_t i = 0; i < s->outputs; i++) {
      oc_tensor *t = &outputs[i];
      t->value = values[s->inputs + i];
      values[s->inputs + i] = NULL;
      const OrtMemoryInfo *memory = NULL;
      CHECK(a->GetTensorMemoryInfo(t->value, &memory));
      OrtMemoryInfoDeviceType device;
      a->MemoryInfoGetDeviceType(memory, &device);
      if (device != OrtMemoryInfoDeviceType_CPU) {
        status = a->CreateStatus(ORT_NOT_IMPLEMENTED, "output tensor is not in CPU memory");
        goto fail;
      }
      CHECK(a->GetTensorElementTypeAndShapeDataReference(t->value, &t->type, &t->shape, &t->rank));
      CHECK(a->GetTensorSizeInBytes(t->value, &t->bytes));
      CHECK(a->GetTensorMutableData(t->value, &t->data));
    }
  }
fail:
  for (size_t i = 0; i < n; i++) if (values[i]) a->ReleaseValue(values[i]);
  if (values != local) free(values);
  if (status && outputs) oc_outputs_free(s, outputs);
  return status;
}

OrtStatus *oc_run(oc_session *s, OrtRunOptions *options, const oc_tensor *inputs, oc_tensor *outputs) {
  return run(s, options, inputs, NULL, outputs);
}

// This entry point retains no Go pointers after returning, including on error.
OrtStatus *oc_run_into(oc_session *s, OrtRunOptions *options, const oc_tensor *inputs, const oc_tensor *outputs) {
  return run(s, options, inputs, outputs, NULL);
}

OrtStatus *oc_run_options(oc_runtime *r, OrtRunOptions **out) { return r->api->CreateRunOptions(out); }
void oc_terminate(oc_runtime *r, OrtRunOptions *options) {
  OrtStatus *status = r->api->RunOptionsSetTerminate(options);
  if (status) r->api->ReleaseStatus(status);
}
void oc_run_options_free(oc_runtime *r, OrtRunOptions *options) { r->api->ReleaseRunOptions(options); }
int oc_error_code(oc_runtime *r, OrtStatus *status) { return r->api->GetErrorCode(status); }
const char *oc_error_message(oc_runtime *r, OrtStatus *status) { return r->api->GetErrorMessage(status); }
void oc_error_free(oc_runtime *r, OrtStatus *status) { r->api->ReleaseStatus(status); }
