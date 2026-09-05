#pragma once
#include "internal/ort/onnxruntime_c_api.h"

typedef struct {
  void *library;
  const OrtApi *api;
  OrtEnv *env;
  OrtMemoryInfo *memory;
  OrtAllocator *allocator;
  const char *version;
} oc_runtime;

typedef struct {
  char *name;
  ONNXTensorElementDataType type;
  int64_t *shape;
  size_t rank;
} oc_port;

typedef struct {
  oc_runtime *runtime;
  OrtSession *handle;
  size_t inputs, outputs;
  oc_port *ports;
  char **names;
} oc_session;

typedef struct {
  int intra_threads, inter_threads, parallel, optimization;
} oc_options;

typedef struct {
  void *data;
  const int64_t *shape;
  size_t bytes, rank;
  ONNXTensorElementDataType type;
  OrtValue *value;
} oc_tensor;

int oc_open(const char *path, oc_runtime **out, char **error);
void oc_close(oc_runtime *r);
OrtStatus *oc_options_new(oc_runtime *r, const oc_options *config, OrtSessionOptions **out);
OrtStatus *oc_config(oc_runtime *r, OrtSessionOptions *o, const char *key, const char *value);
OrtStatus *oc_provider(oc_runtime *r, OrtSessionOptions *o, const char *name, const char **keys, const char **values, size_t count);
void oc_options_free(oc_runtime *r, OrtSessionOptions *o);
OrtStatus *oc_load(oc_runtime *r, const char *path, const void *data, size_t length, const OrtSessionOptions *options, oc_session **out);
void oc_session_free(oc_session *s);
OrtStatus *oc_run(oc_session *s, OrtRunOptions *options, const oc_tensor *inputs, oc_tensor *outputs);
OrtStatus *oc_run_into(oc_session *s, OrtRunOptions *options, const oc_tensor *inputs, const oc_tensor *outputs);
void oc_outputs_free(oc_session *s, oc_tensor *outputs);
OrtStatus *oc_run_options(oc_runtime *r, OrtRunOptions **out);
void oc_terminate(oc_runtime *r, OrtRunOptions *options);
void oc_run_options_free(oc_runtime *r, OrtRunOptions *options);
int oc_error_code(oc_runtime *r, OrtStatus *status);
const char *oc_error_message(oc_runtime *r, OrtStatus *status);
void oc_error_free(oc_runtime *r, OrtStatus *status);
