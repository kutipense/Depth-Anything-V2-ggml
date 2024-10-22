#include <algorithm>
#include <cmath>
#include <unordered_map>
#include <vector>

#define SAFETENSORS_CPP_IMPLEMENTATION
#include "ggml-backend.h"
#include "ggml.h"
#include "safetensors.hh"
#include "stb_image.h"
#include "tinycolormap.hpp"

template <typename T>
struct image {
  long int nx, ny;
  std::vector<T> data;
};

image<uint8_t> load_image(const std::string &fname) {
  int nx, ny, nc;
  auto data = stbi_load(fname.c_str(), &nx, &ny, &nc, 3);
  assert(data != nullptr);
  image<uint8_t> img = {.nx = nx, .ny = ny, .data = std::vector<uint8_t>(data, data + nx * ny * 3)};
  stbi_image_free(data);
  return img;
}

safetensors::safetensors_t load_safetensor(const std::string &filename) {
  std::string warn, err;
  safetensors::safetensors_t st;
  bool ret = safetensors::load_from_file(filename, &st, &warn, &err);
  assert(ret && safetensors::validate_data_offsets(st, err));
  return st;
}

image<float> preprocess_image(const image<uint8_t> &img) {
  const int nx = img.nx, ny = img.ny;
  image<float> result = {.nx = nx, .ny = ny, .data = std::vector<float>(img.data.size(), 0)};

  const float mean[3] = {0.485, 0.456, 0.406};
  const float stdv[3] = {0.229, 0.224, 0.225};

  for (int y = 0; y < ny; y++) {
    for (int x = 0; x < nx; x++) {
#pragma unroll
      for (int c = 0; c < 3; c++) {
        auto ind           = 3 * (y * nx + x) + c;
        auto t_ind         = (c * ny + y) * nx + x;
        result.data[t_ind] = (img.data[ind] / 255.0f - mean[c]) / stdv[c];
      }
    }
  }

  return result;
}

image<uint8_t> postprocess_image(const image<float> &img) {
  auto _min_it = std::min_element(img.data.begin(), img.data.end());
  auto _max_it = std::max_element(img.data.begin(), img.data.end());
  auto _max = *_max_it, _min = *_min_it;
  float range = _max - _min;

  image<uint8_t> result_image = {.nx = img.nx, .ny = img.ny};
  result_image.data.reserve(img.nx * img.ny * 3);
  auto size = img.nx * img.ny;
  for (int i = 0, j = 0; i < size; i++, j += 3) {
    float value                     = (img.data[i] - _min) / range;
    const tinycolormap::Color color = tinycolormap::GetColor(value);
    result_image.data[j]            = color.ri();
    result_image.data[j + 1]        = color.gi();
    result_image.data[j + 2]        = color.bi();
  }

  return result_image;
}

inline ggml_tensor *conv2d(ggml_context *ctx, ggml_tensor *x, ggml_tensor *w, ggml_tensor *b, int s, int p) {
  x = ggml_conv_2d(ctx, w, x, s, s, p, p, 1, 1);
  return b == nullptr ? x : ggml_add(ctx, x, ggml_repeat(ctx, b, x));
}

inline ggml_tensor *linear(ggml_context *ctx, ggml_tensor *x, ggml_tensor *w, ggml_tensor *b) {
  x = ggml_mul_mat(ctx, w, x);
  return b == nullptr ? x : ggml_add(ctx, x, ggml_repeat(ctx, b, x));
}

struct dptv2_config {
  int img_size, patch_size, in_channels, embed_dim, depth, mlp_ratio, num_heads, features;
  std::vector<int> out_channels, intermediate_layers;
};

void load_tensor(safetensors::safetensors_t *st, const std::string &key, ggml_tensor *tensor) {
  safetensors::tensor_t dst;
  bool success = st->tensors.at(key, &dst);
  assert(success);
  size_t start = dst.data_offsets[0], end = dst.data_offsets[1];
  assert(end - start == ggml_nbytes(tensor));
  ggml_backend_tensor_set(tensor, st->storage.data() + start, 0, ggml_nbytes(tensor));
}

struct model_base {
  dptv2_config *config;

  model_base() : config(nullptr) {}
  model_base(dptv2_config *_config) : config(_config) {}

  void virtual init(ggml_context *_ctx) = 0;
  size_t virtual ctx_size()             = 0;
};

struct patch_embeddings : model_base {
  struct ggml_tensor *conv2d_w, *conv2d_b;

  patch_embeddings(dptv2_config *_config) : model_base(_config) {}

  size_t ctx_size() override { return 2 * ggml_tensor_overhead(); }

  void init(ggml_context *_ctx) {
    conv2d_w = ggml_new_tensor_4d(_ctx, GGML_TYPE_F32, config->patch_size, config->patch_size, 3, config->embed_dim);
    conv2d_b = ggml_new_tensor_3d(_ctx, GGML_TYPE_F32, 1, 1, config->embed_dim);
  }

  void load(safetensors::safetensors_t *st, const std::string &key) {
    load_tensor(st, key + ".patch_embeddings.projection.weight", conv2d_w);
    load_tensor(st, key + ".patch_embeddings.projection.bias", conv2d_b);
  }

  ggml_tensor *forward(ggml_context *_ctx, ggml_tensor *x) {
    x = conv2d(_ctx, x, conv2d_w, conv2d_b, conv2d_w->ne[0], 0);
    return ggml_transpose(_ctx, ggml_reshape_3d(_ctx, x, x->ne[0] * x->ne[1], x->ne[2], x->ne[3]));
  }
};

struct embeddings : model_base {
  patch_embeddings patch_emb;
  ggml_tensor *cls_tk, *mask_tk, *position_emb;

  embeddings(dptv2_config *_config) : model_base(_config), patch_emb(_config) {}

  size_t ctx_size() override { return patch_emb.ctx_size() + 3 * ggml_tensor_overhead(); }

  void init(ggml_context *_ctx) {
    const int num_tokens = 1;

    patch_emb.init(_ctx);

    int num_patches = pow(config->img_size / config->patch_size, 2);
    cls_tk          = ggml_new_tensor_3d(_ctx, GGML_TYPE_F32, config->embed_dim, 1, 1);
    mask_tk         = ggml_new_tensor_2d(_ctx, GGML_TYPE_F32, config->embed_dim, 1);
    position_emb    = ggml_new_tensor_3d(_ctx, GGML_TYPE_F32, config->embed_dim, num_patches + num_tokens, 1);
  }

  void load(safetensors::safetensors_t *st, const std::string &key) {
    patch_emb.load(st, key + ".embeddings");
    load_tensor(st, key + ".embeddings.position_embeddings", position_emb);
    load_tensor(st, key + ".embeddings.cls_token", cls_tk);
    load_tensor(st, key + ".embeddings.mask_token", mask_tk);
  }

  ggml_tensor *forward(ggml_context *_ctx, ggml_tensor *x) {
    x = patch_emb.forward(_ctx, x);

    struct ggml_tensor repeat_tensor {
      .ne = { config->embed_dim, 1, x->ne[2], 1 }
    };

    x = ggml_concat(_ctx, ggml_repeat(_ctx, cls_tk, &repeat_tensor), x, 1);
    x = ggml_add(_ctx, x, position_emb);

    return x;
  }
};

struct attention : model_base {
  float scale;

  ggml_tensor *query_w, *key_w, *value_w;
  ggml_tensor *query_b, *key_b, *value_b;

  attention(dptv2_config *_config) : model_base(_config) {
    scale = 1.0f / sqrt(_config->embed_dim / config->num_heads);
  }

  size_t ctx_size() override { return 6 * ggml_tensor_overhead(); }

  void init(ggml_context *_ctx) {
    query_w = ggml_new_tensor_2d(_ctx, GGML_TYPE_F32, config->embed_dim, config->embed_dim);
    key_w   = ggml_new_tensor_2d(_ctx, GGML_TYPE_F32, config->embed_dim, config->embed_dim);
    value_w = ggml_new_tensor_2d(_ctx, GGML_TYPE_F32, config->embed_dim, config->embed_dim);
    query_b = ggml_new_tensor_2d(_ctx, GGML_TYPE_F32, config->embed_dim, 1);
    key_b   = ggml_new_tensor_2d(_ctx, GGML_TYPE_F32, config->embed_dim, 1);
    value_b = ggml_new_tensor_2d(_ctx, GGML_TYPE_F32, config->embed_dim, 1);
  }

  void load(safetensors::safetensors_t *st, const std::string &key) {
    auto _key = key + ".attention.attention";
    load_tensor(st, _key + ".query.weight", query_w), load_tensor(st, _key + ".query.bias", query_b);
    load_tensor(st, _key + ".key.weight", key_w), load_tensor(st, _key + ".key.bias", key_b);
    load_tensor(st, _key + ".value.weight", value_w), load_tensor(st, _key + ".value.bias", value_b);
  }

  ggml_tensor *forward(ggml_context *_ctx, ggml_tensor *x) {
    int C = x->ne[0], N = x->ne[1], B = x->ne[2];
    int ch = C / config->num_heads;

    auto q = linear(_ctx, x, query_w, query_b);
    auto k = linear(_ctx, x, key_w, key_b);
    auto v = linear(_ctx, x, value_w, value_b);

    q = ggml_permute(_ctx, ggml_reshape_4d(_ctx, q, ch, config->num_heads, N, B), 0, 2, 1, 3);
    k = ggml_permute(_ctx, ggml_reshape_4d(_ctx, k, ch, config->num_heads, N, B), 0, 2, 1, 3);
    v = ggml_permute(_ctx, ggml_reshape_4d(_ctx, v, ch, config->num_heads, N, B), 0, 2, 1, 3);

    auto qkT  = ggml_soft_max(_ctx, ggml_scale(_ctx, ggml_mul_mat(_ctx, k, q), scale));
    auto attn = ggml_mul_mat(_ctx, ggml_cont(_ctx, ggml_transpose(_ctx, v)), qkT);
    attn      = ggml_cont(_ctx, ggml_permute(_ctx, attn, 0, 2, 1, 3));
    attn      = ggml_reshape_3d(_ctx, attn, attn->ne[0] * attn->ne[1], attn->ne[2], attn->ne[3]);

    return attn;
  }
};

struct mlp : model_base {
  ggml_tensor *ln_0_w, *ln_0_b, *ln_1_w, *ln_1_b;  // linear

  mlp(dptv2_config *_config) : model_base(_config) {}

  size_t ctx_size() override { return 4 * ggml_tensor_overhead(); }

  void init(ggml_context *_ctx) {
    auto hidden_features = config->embed_dim * config->mlp_ratio;
    ln_0_w               = ggml_new_tensor_2d(_ctx, GGML_TYPE_F32, config->embed_dim, hidden_features);
    ln_1_w               = ggml_new_tensor_2d(_ctx, GGML_TYPE_F32, hidden_features, config->embed_dim);
    ln_0_b               = ggml_new_tensor_2d(_ctx, GGML_TYPE_F32, hidden_features, 1);
    ln_1_b               = ggml_new_tensor_2d(_ctx, GGML_TYPE_F32, config->embed_dim, 1);
  }

  void load(safetensors::safetensors_t *st, const std::string &key) {
    load_tensor(st, key + ".mlp.fc1.weight", ln_0_w), load_tensor(st, key + ".mlp.fc1.bias", ln_0_b);
    load_tensor(st, key + ".mlp.fc2.weight", ln_1_w), load_tensor(st, key + ".mlp.fc2.bias", ln_1_b);
  }

  ggml_tensor *forward(ggml_context *_ctx, ggml_tensor *x) {
    return linear(_ctx, ggml_gelu(_ctx, linear(_ctx, x, ln_0_w, ln_0_b)), ln_1_w, ln_1_b);
  }
};

struct layer : model_base {
  float norm_eps = 1e-6;
  attention attn;
  mlp _mlp;

  ggml_tensor *ln_0_w, *ln_0_b;                // linear
  ggml_tensor *ls_0_w, *ls_1_w;                // layer scale
  ggml_tensor *n_0_w, *n_0_b, *n_1_w, *n_1_b;  // layer norm

  layer(dptv2_config *_config) : model_base(_config), attn(_config), _mlp(_config) {}

  size_t ctx_size() override { return attn.ctx_size() + _mlp.ctx_size() + 8 * ggml_tensor_overhead(); }

  void init(ggml_context *_ctx) {
    attn.init(_ctx), _mlp.init(_ctx);

    ln_0_w = ggml_new_tensor_2d(_ctx, GGML_TYPE_F32, config->embed_dim, config->embed_dim);
    ln_0_b = ggml_new_tensor_2d(_ctx, GGML_TYPE_F32, config->embed_dim, 1);
    ls_0_w = ggml_new_tensor_2d(_ctx, GGML_TYPE_F32, config->embed_dim, 1);
    ls_1_w = ggml_new_tensor_2d(_ctx, GGML_TYPE_F32, config->embed_dim, 1);
    n_0_w  = ggml_new_tensor_2d(_ctx, GGML_TYPE_F32, config->embed_dim, 1);
    n_1_w  = ggml_new_tensor_2d(_ctx, GGML_TYPE_F32, config->embed_dim, 1);
    n_0_b  = ggml_new_tensor_2d(_ctx, GGML_TYPE_F32, config->embed_dim, 1);
    n_1_b  = ggml_new_tensor_2d(_ctx, GGML_TYPE_F32, config->embed_dim, 1);
  }

  void load(safetensors::safetensors_t *st, const std::string &key) {
    attn.load(st, key), _mlp.load(st, key);

    load_tensor(st, key + ".layer_scale1.lambda1", ls_0_w), load_tensor(st, key + ".layer_scale2.lambda1", ls_1_w);
    load_tensor(st, key + ".norm1.weight", n_0_w), load_tensor(st, key + ".norm1.bias", n_0_b);
    load_tensor(st, key + ".norm2.weight", n_1_w), load_tensor(st, key + ".norm2.bias", n_1_b);
    load_tensor(st, key + ".attention.output.dense.weight", ln_0_w);
    load_tensor(st, key + ".attention.output.dense.bias", ln_0_b);
  }

  ggml_tensor *forward(ggml_context *_ctx, ggml_tensor *x) {
    ggml_tensor *out;
    out = ggml_add(_ctx, ggml_mul(_ctx, ggml_norm(_ctx, x, norm_eps), n_0_w), n_0_b);
    out = ggml_mul(_ctx, ggml_add(_ctx, ggml_mul_mat(_ctx, ln_0_w, attn.forward(_ctx, out)), ln_0_b), ls_0_w);
    x   = ggml_add(_ctx, x, out);
    out = ggml_add(_ctx, ggml_mul(_ctx, ggml_norm(_ctx, x, norm_eps), n_1_w), n_1_b);
    x   = ggml_add(_ctx, x, ggml_mul(_ctx, _mlp.forward(_ctx, out), ls_1_w));
    return x;
  }
};

struct encoder : model_base {
  std::vector<layer> layers;

  encoder(dptv2_config *_config) : model_base(_config) { layers = std::vector<layer>(_config->depth, layer{_config}); }

  size_t ctx_size() override { return layers.front().ctx_size() * config->depth; }

  void init(ggml_context *_ctx) {
    for (auto &l : layers) l.init(_ctx);
  }

  void load(safetensors::safetensors_t *st, const std::string &key) {
    for (size_t i = 0; i < layers.size(); i++) layers[i].load(st, key + ".encoder.layer." + std::to_string(i));
  }

  std::vector<ggml_tensor *> forward(ggml_context *_ctx, ggml_tensor *x) {
    std::vector<ggml_tensor *> outputs;
    for (auto &l : layers) {
      x = l.forward(_ctx, x);
      outputs.push_back(x);
    }
    return outputs;
  }
};

struct backbone : model_base {
  float norm_eps = 1e-6;
  embeddings embd;
  encoder enc;

  ggml_tensor *n_0_w, *n_0_b;

  backbone(dptv2_config *_config) : model_base(_config), embd(_config), enc(_config) {}

  size_t ctx_size() override { return embd.ctx_size() + enc.ctx_size() + 2 * ggml_tensor_overhead(); }

  void init(ggml_context *_ctx) {
    embd.init(_ctx), enc.init(_ctx);

    n_0_w = ggml_new_tensor_2d(_ctx, GGML_TYPE_F32, config->embed_dim, 1);
    n_0_b = ggml_new_tensor_2d(_ctx, GGML_TYPE_F32, config->embed_dim, 1);
  }

  void load(safetensors::safetensors_t *st, const std::string &key) {
    embd.load(st, key), enc.load(st, key);
    load_tensor(st, key + ".layernorm.weight", n_0_w), load_tensor(st, key + ".layernorm.bias", n_0_b);
  }

  std::vector<ggml_tensor *> forward(ggml_context *_ctx, ggml_tensor *x) {
    auto outputs = enc.forward(_ctx, embd.forward(_ctx, x));

    std::vector<ggml_tensor *> intermediate_outputs;
    for (auto i : config->intermediate_layers) {
      auto out = outputs[i];
      out      = ggml_add(_ctx, ggml_mul(_ctx, ggml_norm(_ctx, out, norm_eps), n_0_w), n_0_b);
      intermediate_outputs.push_back(out);
    }
    return intermediate_outputs;
  }
};

struct head : model_base {
  int in_features, out_features;
  int patch_h, patch_w;

  struct ggml_tensor *conv2d_0_w, *conv2d_0_b;
  struct ggml_tensor *conv2d_1_w, *conv2d_1_b;
  struct ggml_tensor *conv2d_2_w, *conv2d_2_b;

  head(dptv2_config *_config) : model_base(_config) {
    in_features  = config->features;
    out_features = config->features / 2;
    patch_h      = config->img_size / config->patch_size * config->patch_size;
    patch_w      = patch_h;
  }

  size_t ctx_size() override { return 6 * ggml_tensor_overhead(); }

  void init(ggml_context *_ctx) {
    conv2d_0_w = ggml_new_tensor_4d(_ctx, GGML_TYPE_F32, 3, 3, in_features, out_features);
    conv2d_1_w = ggml_new_tensor_4d(_ctx, GGML_TYPE_F32, 3, 3, out_features, 32);
    conv2d_2_w = ggml_new_tensor_4d(_ctx, GGML_TYPE_F32, 1, 1, 32, 1);
    conv2d_0_b = ggml_new_tensor_3d(_ctx, GGML_TYPE_F32, 1, 1, out_features);
    conv2d_1_b = ggml_new_tensor_3d(_ctx, GGML_TYPE_F32, 1, 1, 32);
    conv2d_2_b = ggml_new_tensor_3d(_ctx, GGML_TYPE_F32, 1, 1, 1);
  }

  void load(safetensors::safetensors_t *st, const std::string &key) {
    load_tensor(st, key + ".conv1.weight", conv2d_0_w), load_tensor(st, key + ".conv1.bias", conv2d_0_b);
    load_tensor(st, key + ".conv2.weight", conv2d_1_w), load_tensor(st, key + ".conv2.bias", conv2d_1_b);
    load_tensor(st, key + ".conv3.weight", conv2d_2_w), load_tensor(st, key + ".conv3.bias", conv2d_2_b);
  }

  ggml_tensor *forward(ggml_context *_ctx, ggml_tensor *x) {
    x = conv2d(_ctx, x, conv2d_0_w, conv2d_0_b, 1, 1);
    x = ggml_upscale_ext(_ctx, x, patch_h, patch_w, x->ne[2], x->ne[3]);
    x = ggml_relu(_ctx, conv2d(_ctx, x, conv2d_1_w, conv2d_1_b, 1, 1));
    x = ggml_relu(_ctx, conv2d(_ctx, x, conv2d_2_w, conv2d_2_b, 1, 0));
    return x;
  }
};

struct residual_layer : model_base {
  int in_features;

  struct ggml_tensor *conv2d_0_w, *conv2d_0_b;
  struct ggml_tensor *conv2d_1_w, *conv2d_1_b;

  residual_layer(dptv2_config *_config) : model_base(_config) { in_features = config->features; }

  size_t ctx_size() override { return 4 * ggml_tensor_overhead(); }

  void init(ggml_context *_ctx) {
    conv2d_0_w = ggml_new_tensor_4d(_ctx, GGML_TYPE_F32, 3, 3, in_features, in_features);
    conv2d_1_w = ggml_new_tensor_4d(_ctx, GGML_TYPE_F32, 3, 3, in_features, in_features);
    conv2d_0_b = ggml_new_tensor_3d(_ctx, GGML_TYPE_F32, 1, 1, in_features);
    conv2d_1_b = ggml_new_tensor_3d(_ctx, GGML_TYPE_F32, 1, 1, in_features);
  }

  void load(safetensors::safetensors_t *st, const std::string &key) {
    load_tensor(st, key + ".convolution1.weight", conv2d_0_w), load_tensor(st, key + ".convolution1.bias", conv2d_0_b);
    load_tensor(st, key + ".convolution2.weight", conv2d_1_w), load_tensor(st, key + ".convolution2.bias", conv2d_1_b);
  }

  ggml_tensor *forward(ggml_context *_ctx, ggml_tensor *x) {
    auto out = ggml_relu(_ctx, conv2d(_ctx, ggml_relu(_ctx, x), conv2d_0_w, conv2d_0_b, 1, 1));
    out      = ggml_add(_ctx, conv2d(_ctx, out, conv2d_1_w, conv2d_1_b, 1, 1), x);
    return out;
  }
};

struct fusion_stage : model_base {
  residual_layer res0, res1;
  struct ggml_tensor *conv2d_0_w, *conv2d_0_b;

  fusion_stage(dptv2_config *_config) : model_base(_config), res0(_config), res1(_config) {}

  size_t ctx_size() override { return res1.ctx_size() * 2 + 2 * ggml_tensor_overhead(); }

  void init(ggml_context *_ctx) {
    res0.init(_ctx), res1.init(_ctx);

    conv2d_0_w = ggml_new_tensor_4d(_ctx, GGML_TYPE_F32, 1, 1, config->features, config->features);
    conv2d_0_b = ggml_new_tensor_3d(_ctx, GGML_TYPE_F32, 1, 1, config->features);
  }

  void load(safetensors::safetensors_t *st, const std::string &key) {
    res0.load(st, key + ".residual_layer1"), res1.load(st, key + ".residual_layer2");
    load_tensor(st, key + ".projection.weight", conv2d_0_w), load_tensor(st, key + ".projection.bias", conv2d_0_b);
  }

  ggml_tensor *forward(ggml_context *_ctx, ggml_tensor *layer0, ggml_tensor *layer1, int size0, int size1) {
    return this->forward(_ctx, ggml_add(_ctx, layer0, res0.forward(_ctx, layer1)), size0, size1);
  }

  ggml_tensor *forward(ggml_context *_ctx, ggml_tensor *layer0, int size0, int size1) {
    layer0 = res1.forward(_ctx, layer0);
    layer0 = ggml_upscale_ext(_ctx, layer0, size0, size1, layer0->ne[2], layer0->ne[3]);
    layer0 = conv2d(_ctx, layer0, conv2d_0_w, conv2d_0_b, 1, 0);
    return layer0;
  }
};

struct reassemble_stage : model_base {
  int patch_h, patch_w;
  std::vector<int> out_channels;

  std::array<ggml_tensor *, 4> conv2d_w, conv2d_b;

  struct ggml_tensor *convt2d_0_w, *convt2d_0_b;
  struct ggml_tensor *convt2d_1_w, *convt2d_1_b;
  struct ggml_tensor *conv2d_4_w, *conv2d_4_b;

  reassemble_stage(dptv2_config *_config) : model_base(_config) {
    out_channels = _config->out_channels;
    patch_h      = config->img_size / config->patch_size;
    patch_w      = patch_h;
  }

  size_t ctx_size() override { return 14 * ggml_tensor_overhead(); }

  void init(ggml_context *_ctx) {
#pragma unroll
    for (size_t i = 0; i < 4; i++) {
      conv2d_w[i] = ggml_new_tensor_4d(_ctx, GGML_TYPE_F32, 1, 1, config->embed_dim, out_channels[i]);
      conv2d_b[i] = ggml_new_tensor_3d(_ctx, GGML_TYPE_F32, 1, 1, out_channels[i]);
    }

    convt2d_0_w = ggml_new_tensor_4d(_ctx, GGML_TYPE_F32, 4, 4, out_channels[0], out_channels[0]);
    convt2d_1_w = ggml_new_tensor_4d(_ctx, GGML_TYPE_F32, 2, 2, out_channels[1], out_channels[1]);
    conv2d_4_w  = ggml_new_tensor_4d(_ctx, GGML_TYPE_F32, 3, 3, out_channels[3], out_channels[3]);
    convt2d_0_b = ggml_new_tensor_3d(_ctx, GGML_TYPE_F32, 1, 1, out_channels[0]);
    convt2d_1_b = ggml_new_tensor_3d(_ctx, GGML_TYPE_F32, 1, 1, out_channels[1]);
    conv2d_4_b  = ggml_new_tensor_3d(_ctx, GGML_TYPE_F32, 1, 1, out_channels[3]);
  }

  void load(safetensors::safetensors_t *st, const std::string &key) {
    auto _key = key + ".layers.";
#pragma unroll
    for (size_t i = 0; i < 4; i++) {
      load_tensor(st, _key + std::to_string(i) + ".projection.weight", conv2d_w[i]);
      load_tensor(st, _key + std::to_string(i) + ".projection.bias", conv2d_b[i]);
    }

    load_tensor(st, _key + "0.resize.weight", convt2d_0_w), load_tensor(st, _key + "0.resize.bias", convt2d_0_b);
    load_tensor(st, _key + "1.resize.weight", convt2d_1_w), load_tensor(st, _key + "1.resize.bias", convt2d_1_b);
    load_tensor(st, _key + "3.resize.weight", conv2d_4_w), load_tensor(st, _key + "3.resize.bias", conv2d_4_b);
  }

  std::vector<ggml_tensor *> forward(ggml_context *_ctx, std::vector<ggml_tensor *> inputs) {
    auto out0 = inputs[0], out1 = inputs[1], out2 = inputs[2], out3 = inputs[3];

    out0 = ggml_view_2d(_ctx, out0, out0->ne[0], out0->ne[1] - 1, out0->nb[1], out0->nb[1]);
    out0 = ggml_cont(_ctx, ggml_permute(_ctx, out0, 1, 0, 2, 3));
    out0 = ggml_reshape_4d(_ctx, out0, patch_w, patch_h, out0->ne[1], out0->ne[2]);
    out0 = conv2d(_ctx, out0, conv2d_w[0], conv2d_b[0], 1, 0);
    out0 = ggml_conv_transpose_2d_p0(_ctx, ggml_cast(_ctx, convt2d_0_w, GGML_TYPE_F16), out0, 4);
    out0 = ggml_add(_ctx, out0, ggml_repeat(_ctx, convt2d_0_b, out0));

    out1 = ggml_view_2d(_ctx, out1, out1->ne[0], out1->ne[1] - 1, out1->nb[1], out1->nb[1]);
    out1 = ggml_cont(_ctx, ggml_permute(_ctx, out1, 1, 0, 2, 3));
    out1 = ggml_reshape_4d(_ctx, out1, patch_w, patch_h, out1->ne[1], out1->ne[2]);
    out1 = conv2d(_ctx, out1, conv2d_w[1], conv2d_b[1], 1, 0);
    out1 = ggml_conv_transpose_2d_p0(_ctx, ggml_cast(_ctx, convt2d_1_w, GGML_TYPE_F16), out1, 2);
    out1 = ggml_add(_ctx, out1, ggml_repeat(_ctx, convt2d_1_b, out1));

    out2 = ggml_view_2d(_ctx, out2, out2->ne[0], out2->ne[1] - 1, out2->nb[1], out2->nb[1]);
    out2 = ggml_cont(_ctx, ggml_permute(_ctx, out2, 1, 0, 2, 3));
    out2 = ggml_reshape_4d(_ctx, out2, patch_w, patch_h, out2->ne[1], out2->ne[2]);
    out2 = conv2d(_ctx, out2, conv2d_w[2], conv2d_b[2], 1, 0);

    out3 = ggml_view_2d(_ctx, out3, out3->ne[0], out3->ne[1] - 1, out3->nb[1], out3->nb[1]);
    out3 = ggml_cont(_ctx, ggml_permute(_ctx, out3, 1, 0, 2, 3));
    out3 = ggml_reshape_4d(_ctx, out3, patch_w, patch_h, out3->ne[1], out3->ne[2]);
    out3 = conv2d(_ctx, out3, conv2d_w[3], conv2d_b[3], 1, 0);
    out3 = conv2d(_ctx, out3, conv2d_4_w, conv2d_4_b, 2, 1);

    return {out0, out1, out2, out3};
  }
};

struct neck : model_base {
  int patch_h, patch_w;
  std::vector<int> out_channels;
  fusion_stage fusion0, fusion1, fusion2, fusion3;
  reassemble_stage reassemble;

  std::array<ggml_tensor *, 4> conv2d_w;

  neck(dptv2_config *_config)
      : model_base(_config),
        reassemble(_config),
        fusion0(_config),
        fusion1(_config),
        fusion2(_config),
        fusion3(_config) {
    out_channels = _config->out_channels;
    patch_h      = config->img_size / config->patch_size;
    patch_w      = patch_h;
  }

  size_t ctx_size() override { return fusion0.ctx_size() * 4 + reassemble.ctx_size() + 8 * ggml_tensor_overhead(); }

  void init(ggml_context *_ctx) {
    reassemble.init(_ctx), fusion0.init(_ctx), fusion1.init(_ctx), fusion2.init(_ctx), fusion3.init(_ctx);

#pragma unroll
    for (size_t i = 0; i < 4; i++)
      conv2d_w[i] = ggml_new_tensor_4d(_ctx, GGML_TYPE_F32, 3, 3, out_channels[i], config->features);
  }

  void load(safetensors::safetensors_t *st, const std::string &key) {
    reassemble.load(st, key + ".reassemble_stage");
    fusion0.load(st, key + ".fusion_stage.layers.0"), fusion1.load(st, key + ".fusion_stage.layers.1");
    fusion2.load(st, key + ".fusion_stage.layers.2"), fusion3.load(st, key + ".fusion_stage.layers.3");

#pragma unroll
    for (size_t i = 0; i < 4; i++) load_tensor(st, key + ".convs." + std::to_string(i) + ".weight", conv2d_w[i]);
  }

  ggml_tensor *forward(ggml_context *_ctx, std::vector<ggml_tensor *> inputs) {
    auto ra_outs = reassemble.forward(_ctx, inputs);

    std::array<ggml_tensor *, 4> outs;
#pragma unroll
    for (size_t i = 0; i < 4; i++) outs[i] = conv2d(_ctx, ra_outs[i], conv2d_w[i], nullptr, 1, 1);

    auto path3 = fusion0.forward(_ctx, outs[3], /*  */ outs[2]->ne[0], outs[2]->ne[1]);
    auto path2 = fusion1.forward(_ctx, path3, outs[2], outs[1]->ne[0], outs[1]->ne[1]);
    auto path1 = fusion2.forward(_ctx, path2, outs[1], outs[0]->ne[0], outs[0]->ne[1]);
    auto path0 = fusion3.forward(_ctx, path1, outs[0], outs[0]->ne[0] * 2, outs[0]->ne[1] * 2);

    return path0;
  }
};

struct dptv2 : model_base {
  backbone bck;
  neck nck;
  head hd;

  ggml_context *ctx;
  ggml_backend *backend;
  ggml_cgraph *gf              = nullptr;
  ggml_context *ctx_cgraph     = nullptr;
  ggml_gallocr_t allocr        = nullptr;
  ggml_backend_buffer_t buffer = nullptr;

  dptv2(dptv2_config *_config, ggml_backend *_backend)
      : model_base(_config), backend(_backend), bck(_config), nck(_config), hd(_config) {
    struct ggml_init_params params {
      ctx_size(), NULL, true,
    };

    ctx = ggml_init(params);

    struct ggml_tensor *input = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 518, 518, 3, 1);
    ggml_set_name(input, "input");
    ggml_set_input(input);

    init(ctx);

    buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
  }

  ~dptv2() {
    ggml_free(ctx);
    ggml_backend_buffer_free(buffer);
    ggml_backend_free(backend);
  }

  image<float> compute_graph(struct ggml_tensor *input) {
    ggml_tensor *output;

    {
      // create graph
      struct ggml_init_params params0 = {
          ggml_tensor_overhead() * GGML_DEFAULT_GRAPH_SIZE + ggml_graph_overhead(),
          NULL,
          true,
      };

      ctx_cgraph = ggml_init(params0);
      gf         = ggml_new_graph(ctx_cgraph);

      auto bck_out = bck.forward(ctx_cgraph, input);
      auto nck_out = nck.forward(ctx_cgraph, bck_out);
      auto hd_out  = hd.forward(ctx_cgraph, nck_out);

      output = hd_out;

      ggml_build_forward_expand(gf, hd_out);
    }

    allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    ggml_gallocr_alloc_graph(allocr, gf);

    int n_threads = 8;
    ggml_backend_cpu_set_n_threads(backend, n_threads);
    ggml_backend_graph_compute(backend, gf);

    image<float> output_image = {
        .nx   = output->ne[0],
        .ny   = output->ne[1],
        .data = std::vector<float>(output->ne[0] * output->ne[1]),
    };

    ggml_backend_tensor_get(output, output_image.data.data(), 0, ggml_nbytes(output));

    ggml_free(ctx_cgraph);
    ggml_gallocr_free(allocr);

    return output_image;
  }

  size_t ctx_size() { return ggml_tensor_overhead() + bck.ctx_size() + nck.ctx_size() + hd.ctx_size(); }

  void init(ggml_context *_ctx = nullptr) override {
    if (_ctx == nullptr) _ctx = ctx;
    bck.init(_ctx), nck.init(_ctx), hd.init(_ctx);
  }

  void load(safetensors::safetensors_t *st) { bck.load(st, "backbone"), nck.load(st, "neck"), hd.load(st, "head"); }
};

std::unordered_map<std::string, dptv2_config> configs = {
    {
        "s",
        {.img_size            = 518,
         .patch_size          = 14,
         .in_channels         = 3,
         .embed_dim           = 384,
         .depth               = 12,
         .mlp_ratio           = 4,
         .num_heads           = 6,
         .features            = 64,
         .out_channels        = {48, 96, 192, 384},
         .intermediate_layers = {2, 5, 8, 11}},
    },
    {
        "b",
        {.img_size            = 518,
         .patch_size          = 14,
         .in_channels         = 3,
         .embed_dim           = 768,
         .depth               = 12,
         .mlp_ratio           = 4,
         .num_heads           = 12,
         .features            = 128,
         .out_channels        = {96, 192, 384, 768},
         .intermediate_layers = {2, 5, 8, 11}},
    },
    {
        "l",
        {.img_size            = 518,
         .patch_size          = 14,
         .in_channels         = 3,
         .embed_dim           = 1024,
         .depth               = 24,
         .mlp_ratio           = 4,
         .num_heads           = 16,
         .features            = 256,
         .out_channels        = {256, 512, 1024, 1024},
         .intermediate_layers = {4, 11, 17, 23}},
    },
};

int main(int argc, char **argv) {
  assert(argc == 4 || argc == 5 && "usage: dptv2 [s|b|l] weights input_file output_file");
  std::string vit_size = argv[1], weights = argv[2], input_filename = argv[3],
              output_filename = (argc == 4 ? "output.jpg" : argv[4]);
  assert(vit_size == "s" || vit_size == "b" || vit_size == "l" && "use [s|b|l]");

  dptv2_config *config  = &configs[vit_size];
  ggml_backend *backend = ggml_backend_cpu_init();

  auto model = dptv2(config, backend);
  {
    auto safetensor = load_safetensor(weights);
    model.load(&safetensor);
  }

  image<uint8_t> img        = load_image(input_filename);
  image<uint8_t> img_scaled = {
      .nx   = config->img_size,
      .ny   = config->img_size,
      .data = std::vector<uint8_t>(config->img_size * config->img_size * 3),
  };

  stbir_resize_uint8(img.data.data(), img.nx, img.ny, 0, img_scaled.data.data(), img_scaled.nx, img_scaled.ny, 0, 3);
  image<float> input_image = preprocess_image(img_scaled);
  auto input               = ggml_get_tensor(model.ctx, "input");

  ggml_backend_tensor_set(input, input_image.data.data(), 0, ggml_nbytes(input));

  auto depth_image                 = model.compute_graph(input);
  image<uint8_t> depth_image_bytes = postprocess_image(depth_image);
  image<uint8_t> output_image      = {.nx = img.nx, .ny = img.ny, .data = std::vector<uint8_t>(img.nx * img.ny * 3)};
  stbir_resize_uint8(depth_image_bytes.data.data(), depth_image_bytes.nx, depth_image_bytes.ny, 0,
                     output_image.data.data(), output_image.nx, output_image.ny, 0, 3);

  stbi_write_jpg(output_filename.c_str(), output_image.nx, output_image.ny, 3, output_image.data.data(), 90);
  return 0;
}
