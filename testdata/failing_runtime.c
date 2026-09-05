// A deliberately failing implementation of the subset of ORT used by the
// bridge. Every native allocation is logged so tests can verify rollback.
#include "internal/ort/onnxruntime_c_api.h"
#include <stdio.h>

struct OrtStatus { int unused; };
struct OrtEnv { int unused; };
struct OrtMemoryInfo { int unused; };
struct OrtSessionOptions { int unused; };
struct OrtSession { int unused; };
struct OrtTypeInfo { int unused; };
struct OrtTensorTypeAndShapeInfo { int unused; };
struct OrtValue { void *data; int owned; int64_t shape; };

static void event(const char *name) {
  FILE *f = fopen(getenv("OC_TEST_LOG"), "a");
  if (f) { fprintf(f, "%s\n", name); fclose(f); }
}
static int fails(const char *stage) { return strcmp(getenv("OC_TEST_FAIL"), stage) == 0; }
static OrtStatus *failure(void) { event("status+"); return calloc(1, sizeof(OrtStatus)); }
static OrtErrorCode ORT_API_CALL code(const OrtStatus *s) { (void)s; return ORT_FAIL; }
static const char *ORT_API_CALL message(const OrtStatus *s) { (void)s; return "injected failure"; }
static OrtStatus *ORT_API_CALL create_status(OrtErrorCode c, const char *s) { (void)c; (void)s; return failure(); }
static void ORT_API_CALL release_status(OrtStatus *s) { event("status-"); free(s); }

static OrtStatus *ORT_API_CALL create_env(OrtLoggingLevel l, const char *id, OrtEnv **out) {
  (void)l; (void)id;
  if (fails("env")) return failure();
  event("env+"); *out = calloc(1, sizeof(OrtEnv)); return NULL;
}
static void ORT_API_CALL release_env(OrtEnv *e) { event("env-"); free(e); }
static OrtStatus *ORT_API_CALL telemetry(const OrtEnv *e) { (void)e; return NULL; }
static OrtStatus *ORT_API_CALL memory(OrtAllocatorType a, OrtMemType m, OrtMemoryInfo **out) {
  (void)a; (void)m;
  if (fails("memory")) return failure();
  event("memory+"); *out = calloc(1, sizeof(OrtMemoryInfo)); return NULL;
}
static void ORT_API_CALL release_memory(OrtMemoryInfo *m) { event("memory-"); free(m); }
static void ORT_API_CALL allocator_free(OrtAllocator *a, void *p) { (void)a; event("name-"); free(p); }
static OrtAllocator allocator = {.version = ORT_API_VERSION, .Free = allocator_free};
static OrtStatus *ORT_API_CALL get_allocator(OrtAllocator **out) { *out = &allocator; return NULL; }

static OrtStatus *ORT_API_CALL options(OrtSessionOptions **out) {
  event("options+"); *out = calloc(1, sizeof(OrtSessionOptions)); return NULL;
}
static void ORT_API_CALL release_options(OrtSessionOptions *o) { event("options-"); free(o); }
static OrtStatus *ORT_API_CALL threads(OrtSessionOptions *o, int n) { (void)o; (void)n; return fails("options") ? failure() : NULL; }
static OrtStatus *ORT_API_CALL execution(OrtSessionOptions *o, ExecutionMode m) { (void)o; (void)m; return NULL; }
static OrtStatus *ORT_API_CALL optimization(OrtSessionOptions *o, GraphOptimizationLevel l) { (void)o; (void)l; return NULL; }
static OrtStatus *ORT_API_CALL create_session(const OrtEnv *e, const ORTCHAR_T *p, const OrtSessionOptions *o, OrtSession **out) {
  (void)e; (void)p; (void)o;
  if (fails("session")) return failure();
  event("session+"); *out = calloc(1, sizeof(OrtSession)); return NULL;
}
static void ORT_API_CALL release_session(OrtSession *s) { event("session-"); free(s); }
static OrtStatus *ORT_API_CALL input_count(const OrtSession *s, size_t *n) { (void)s; *n = 1; return NULL; }
static OrtStatus *ORT_API_CALL output_count(const OrtSession *s, size_t *n) { (void)s; *n = 2; return NULL; }
static OrtStatus *ORT_API_CALL name(const OrtSession *s, size_t i, OrtAllocator *a, char **out) {
  (void)s; (void)i; (void)a;
  event("name+"); *out = malloc(2); memcpy(*out, "x", 2); return NULL;
}
static OrtStatus *ORT_API_CALL type_info(const OrtSession *s, size_t i, OrtTypeInfo **out) {
  (void)s; (void)i;
  event("type+"); *out = calloc(1, sizeof(OrtTypeInfo)); return NULL;
}
static void ORT_API_CALL release_type(OrtTypeInfo *p) { event("type-"); free(p); }
static OrtStatus *ORT_API_CALL kind(const OrtTypeInfo *p, ONNXType *out) { (void)p; *out = ONNX_TYPE_TENSOR; return NULL; }
static OrtStatus *ORT_API_CALL cast(const OrtTypeInfo *p, const OrtTensorTypeAndShapeInfo **out) {
  *out = (const OrtTensorTypeAndShapeInfo *)p; return NULL;
}
static OrtStatus *ORT_API_CALL element(const OrtTensorTypeAndShapeInfo *p, ONNXTensorElementDataType *out) {
  (void)p; *out = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; return NULL;
}
static OrtStatus *ORT_API_CALL rank(const OrtTensorTypeAndShapeInfo *p, size_t *out) {
  (void)p; *out = 1; return fails("metadata") ? failure() : NULL;
}
static OrtStatus *ORT_API_CALL dimensions(const OrtTensorTypeAndShapeInfo *p, int64_t *out, size_t n) { (void)p; (void)n; *out = 1; return NULL; }

static int tensor_count;
static OrtStatus *ORT_API_CALL tensor(const OrtMemoryInfo *m, void *data, size_t size, const int64_t *shape, size_t rank, ONNXTensorElementDataType type, OrtValue **out) {
  (void)m; (void)size; (void)shape; (void)rank; (void)type;
  if (++tensor_count == 2 && fails("tensor")) return failure();
  event("value+"); *out = calloc(1, sizeof(OrtValue)); (*out)->data = data; (*out)->shape = 1; return NULL;
}
static void ORT_API_CALL release_value(OrtValue *v) { event("value-"); if (v->owned) free(v->data); free(v); }
static OrtStatus *ORT_API_CALL run(OrtSession *s, const OrtRunOptions *o, const char *const *names, const OrtValue *const *inputs, size_t ni, const char *const *on, size_t no, OrtValue **outputs) {
  (void)s; (void)o; (void)names; (void)inputs; (void)ni; (void)on;
  for (size_t i = 0; i < no; i++) {
    if (!outputs[i]) {
      event("value+"); outputs[i] = calloc(1, sizeof(OrtValue));
      outputs[i]->owned = 1; outputs[i]->data = calloc(1, sizeof(float)); outputs[i]->shape = 1;
    }
    if (fails("run")) return failure();
  }
  return NULL;
}
static OrtMemoryInfo cpu;
static OrtStatus *ORT_API_CALL tensor_memory(const OrtValue *v, const OrtMemoryInfo **out) { (void)v; *out = &cpu; return NULL; }
static void ORT_API_CALL device_type(const OrtMemoryInfo *m, OrtMemoryInfoDeviceType *out) { (void)m; *out = OrtMemoryInfoDeviceType_CPU; }
static OrtStatus *ORT_API_CALL tensor_shape(const OrtValue *v, ONNXTensorElementDataType *type, const int64_t **shape, size_t *rank) {
  if (fails("output")) return failure();
  *type = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; *shape = &v->shape; *rank = 1; return NULL;
}
static OrtStatus *ORT_API_CALL tensor_bytes(const OrtValue *v, size_t *out) { (void)v; *out = sizeof(float); return NULL; }
static OrtStatus *ORT_API_CALL tensor_data(OrtValue *v, void **out) { *out = v->data; return NULL; }

static const OrtApi api = {
  .CreateStatus = create_status, .GetErrorCode = code, .GetErrorMessage = message, .ReleaseStatus = release_status,
  .CreateEnv = create_env, .ReleaseEnv = release_env, .DisableTelemetryEvents = telemetry,
  .CreateCpuMemoryInfo = memory, .ReleaseMemoryInfo = release_memory, .GetAllocatorWithDefaultOptions = get_allocator,
  .CreateSessionOptions = options, .ReleaseSessionOptions = release_options,
  .SetIntraOpNumThreads = threads, .SetInterOpNumThreads = threads, .SetSessionExecutionMode = execution,
  .SetSessionGraphOptimizationLevel = optimization,
  .CreateSession = create_session, .ReleaseSession = release_session,
  .SessionGetInputCount = input_count, .SessionGetOutputCount = output_count,
  .SessionGetInputName = name, .SessionGetOutputName = name,
  .SessionGetInputTypeInfo = type_info, .SessionGetOutputTypeInfo = type_info,
  .ReleaseTypeInfo = release_type, .GetOnnxTypeFromTypeInfo = kind, .CastTypeInfoToTensorInfo = cast,
  .GetTensorElementType = element, .GetDimensionsCount = rank, .GetDimensions = dimensions,
  .CreateTensorWithDataAsOrtValue = tensor, .ReleaseValue = release_value, .Run = run,
  .GetTensorMemoryInfo = tensor_memory, .MemoryInfoGetDeviceType = device_type,
  .GetTensorElementTypeAndShapeDataReference = tensor_shape, .GetTensorSizeInBytes = tensor_bytes, .GetTensorMutableData = tensor_data,
};
static const OrtApi *ORT_API_CALL get_api(uint32_t version) { return version == ORT_API_VERSION ? &api : NULL; }
static const char *ORT_API_CALL version(void) { return "failure-fixture"; }
static const OrtApiBase base = {.GetApi = get_api, .GetVersionString = version};
const OrtApiBase *ORT_API_CALL OrtGetApiBase(void) { return &base; }
