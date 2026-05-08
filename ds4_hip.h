#ifndef DS4_HIP_H
#define DS4_HIP_H

#include <stdbool.h>
#include <stdint.h>

#ifdef DS4_HIP

/* =========================================================================
 * HIP Tensor and Command Lifetime.
 * =========================================================================
 *
 * Opaque device tensor used by the DS4-specific HIP executor.
 * Analogous to ds4_metal.h but using AMD HIP API.
 */

typedef struct ds4_hip_tensor ds4_hip_tensor;

int ds4_hip_init(void);
void ds4_hip_cleanup(void);

ds4_hip_tensor *ds4_hip_tensor_alloc(uint64_t bytes);
ds4_hip_tensor *ds4_hip_tensor_view(const ds4_hip_tensor *base, uint64_t offset, uint64_t bytes);
void ds4_hip_tensor_free(ds4_hip_tensor *tensor);
uint64_t ds4_hip_tensor_bytes(const ds4_hip_tensor *tensor);
void *ds4_hip_tensor_contents(ds4_hip_tensor *tensor);
int ds4_hip_tensor_write(ds4_hip_tensor *tensor, uint64_t offset, const void *data, uint64_t bytes);
int ds4_hip_tensor_read(const ds4_hip_tensor *tensor, uint64_t offset, void *data, uint64_t bytes);
int ds4_hip_tensor_copy(ds4_hip_tensor *dst, uint64_t dst_offset,
                        const ds4_hip_tensor *src, uint64_t src_offset,
                        uint64_t bytes);

int ds4_hip_begin_commands(void);
int ds4_hip_flush_commands(void);
int ds4_hip_end_commands(void);
int ds4_hip_synchronize(void);

int ds4_hip_set_model_map(const void *model_map, uint64_t model_size);
int ds4_hip_set_model_map_range(const void *model_map, uint64_t model_size, uint64_t map_offset, uint64_t map_size);
void ds4_hip_set_quality(bool quality);
void ds4_hip_print_memory_report(const char *label);

/* =========================================================================
 * Embeddings and Indexer Helpers.
 * ========================================================================= */

int ds4_hip_embed_token_hc_tensor(
        ds4_hip_tensor *out_hc,
        const void       *model_map,
        uint64_t          model_size,
        uint64_t          weight_offset,
        uint32_t          n_vocab,
        uint32_t          token,
        uint32_t          n_embd,
        uint32_t          n_hc);

int ds4_hip_embed_tokens_hc_tensor(
        ds4_hip_tensor       *out_hc,
        const ds4_hip_tensor *tokens,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint32_t                n_vocab,
        uint32_t                n_tokens,
        uint32_t                n_embd,
        uint32_t                n_hc);

int ds4_hip_indexer_score_one_tensor(
        ds4_hip_tensor       *scores,
        const ds4_hip_tensor *q,
        const ds4_hip_tensor *weights,
        const ds4_hip_tensor *index_comp,
        uint32_t                n_comp,
        uint32_t                n_head,
        uint32_t                head_dim,
        float                   scale);

/* =========================================================================
 * Dense Layer (Matmul).
 * ========================================================================= */

int ds4_hip_dense_tensor(
        ds4_hip_tensor       *out,
        const ds4_hip_tensor *in,
        const ds4_hip_tensor *weights,
        uint32_t                n_rows,
        uint32_t                n_cols,
        uint32_t                n_embd,
        bool                    bias);

/* =========================================================================
 * Attention.
 * ========================================================================= */

int ds4_hip_flash_attn_tensor(
        ds4_hip_tensor       *out,
        const ds4_hip_tensor *q,
        const ds4_hip_tensor *k,
        const ds4_hip_tensor *v,
        uint32_t                n_tokens,
        uint32_t                n_heads,
        uint32_t                head_dim,
        float                   scale);

int ds4_hip_attn_score_one_tensor(
        ds4_hip_tensor       *scores,
        const ds4_hip_tensor *q,
        const ds4_hip_tensor *k,
        const ds4_hip_tensor *mask,
        uint32_t                n_tokens,
        uint32_t                n_heads,
        uint32_t                head_dim,
        float                   scale);

int ds4_hip_attn_softmax_tensor(
        ds4_hip_tensor       *out,
        const ds4_hip_tensor *scores,
        uint32_t                n_tokens,
        uint32_t                n_heads,
        uint32_t                seq_len);

int ds4_hip_attn_combine_tensor(
        ds4_hip_tensor       *out,
        const ds4_hip_tensor *v,
        const ds4_hip_tensor *attn,
        uint32_t                n_tokens,
        uint32_t                n_heads,
        uint32_t                head_dim);

/* =========================================================================
 * MoE (Mixture of Experts).
 * ========================================================================= */

int ds4_hip_moe_router_tensor(
        ds4_hip_tensor       *selected,
        ds4_hip_tensor       *weights,
        const ds4_hip_tensor *logits,
        uint32_t                n_tokens,
        uint32_t                n_experts,
        uint32_t                top_k);

int ds4_hip_moe_expert_tensor(
        ds4_hip_tensor       *out,
        const ds4_hip_tensor *in,
        const ds4_hip_tensor *gate_up,
        const ds4_hip_tensor *down,
        uint32_t                n_tokens,
        uint32_t                hidden_dim,
        uint32_t                ffn_dim);

int ds4_hip_moe_combine_tensor(
        ds4_hip_tensor       *out,
        const ds4_hip_tensor *expert_outs,
        const ds4_hip_tensor *selected,
        uint32_t                n_tokens,
        uint32_t                n_experts,
        uint32_t                ffn_dim);

/* =========================================================================
 * Normalization and Activation.
 * ========================================================================= */

int ds4_hip_rms_norm_tensor(
        ds4_hip_tensor       *out,
        const ds4_hip_tensor *in,
        const ds4_hip_tensor *weight,
        uint32_t                n_tokens,
        uint32_t                dim,
        float                   eps);

int ds4_hip_swiglu_tensor(
        ds4_hip_tensor       *out,
        const ds4_hip_tensor *gate,
        const ds4_hip_tensor *up,
        uint32_t                n_tokens,
        uint32_t                dim);

int ds4_hip_silu_tensor(
        ds4_hip_tensor       *out,
        const ds4_hip_tensor *in,
        uint32_t                n_elements);

int ds4_hip_gelu_tensor(
        ds4_hip_tensor       *out,
        const ds4_hip_tensor *in,
        uint32_t                n_elements);

/* =========================================================================
 * RoPE (Rotary Positional Embedding).
 * ========================================================================= */

int ds4_hip_rope_tensor(
        ds4_hip_tensor       *out,
        const ds4_hip_tensor *in,
        const float          *freqs,
        uint32_t                n_tokens,
        uint32_t                n_heads,
        uint32_t                head_dim,
        uint32_t                offset);

int ds4_hip_rope_cos_sin_tensor(
        float                 *cos_out,
        float                 *sin_out,
        const float          *freqs,
        uint32_t                n_tokens,
        uint32_t                head_dim);

/* =========================================================================
 * KV Cache Management.
 * ========================================================================= */

int ds4_hip_kv_cache_update_tensor(
        ds4_hip_tensor       *k_cache,
        ds4_hip_tensor       *v_cache,
        const ds4_hip_tensor *k_new,
        const ds4_hip_tensor *v_new,
        uint32_t                layer,
        uint32_t                pos,
        uint32_t                n_tokens,
        uint32_t                n_heads,
        uint32_t                head_dim);

int ds4_hip_kv_cache_slice_tensor(
        ds4_hip_tensor       *out_k,
        ds4_hip_tensor       *out_v,
        const ds4_hip_tensor *k_cache,
        const ds4_hip_tensor *v_cache,
        uint32_t                layer,
        uint32_t                start_pos,
        uint32_t                n_tokens,
        uint32_t                n_heads,
        uint32_t                head_dim);

/* =========================================================================
 * Indexer (Compressed Attention).
 * ========================================================================= */

int ds4_hip_indexer_topk_tensor(
        ds4_hip_tensor       *topk_ids,
        const ds4_hip_tensor *scores,
        uint32_t                n_rows,
        uint32_t                k);

int ds4_hip_indexer_gather_tensor(
        ds4_hip_tensor       *out,
        const ds4_hip_tensor *src,
        const ds4_hip_tensor *indices,
        uint32_t                n_rows,
        uint32_t                k,
        uint32_t                elem_size);

/* =========================================================================
 * Utility Operations.
 * ========================================================================= */

int ds4_hip_add_tensor(
        ds4_hip_tensor       *out,
        const ds4_hip_tensor *a,
        const ds4_hip_tensor *b,
        uint32_t                n_elements);

int ds4_hip_copy_tensor(
        ds4_hip_tensor       *dst,
        const ds4_hip_tensor *src,
        uint64_t               bytes);

int ds4_hip_fill_tensor(
        ds4_hip_tensor       *tensor,
        float                 value,
        uint32_t                n_elements);

int ds4_hip_softmax_tensor(
        ds4_hip_tensor       *out,
        const ds4_hip_tensor *in,
        uint32_t                n_rows,
        uint32_t                n_cols);

/* =========================================================================
 * Model Weights Management.
 * ========================================================================= */

int ds4_hip_copy_weights_to_device(const void *weights_ptr, uint64_t weights_size);
void ds4_hip_free_weights_device(void);

/* Layer weight management for hybrid mode */
int ds4_hip_copy_layer_weights_to_device(int layer_idx, const void *weights_ptr, uint64_t weights_size);
void ds4_hip_free_layer_weights(int layer_idx);

/* Tensor decompression and GPU copy (experimental) */
typedef struct {
    void *device_ptr;
    void *host_ptr;
    uint64_t size;
    int layer_idx;
    int ref_count;
} ds4_hip_tensor_cache;

int ds4_hip_decompress_tensor_to_gpu(
    void **device_out,
    const void *compressed_data,
    uint64_t n_elements,
    int tensor_type);

void ds4_hip_free_tensor_cache(ds4_hip_tensor_cache *cache);

/* GPU-accelerated softmax and argmax for logits */
int ds4_hip_softmax_argmax(int *token, const float *logits, uint32_t n_vocab);

int ds4_hip_generate(
        const void *model,
        const void *weights,
        const int *prompt_tokens,
        int prompt_len,
        int n_predict,
        int ctx_size,
        void (*emit_token)(void *, int),
        void *emit_ud);

/* =========================================================================
 * Graph Operations.
 * ========================================================================= */

typedef struct ds4_hip_graph ds4_hip_graph;

int ds4_hip_graph_alloc(ds4_hip_graph **out, uint32_t max_nodes);
void ds4_hip_graph_free(ds4_hip_graph *g);
int ds4_hip_graph_add_node(ds4_hip_graph *g, void *kernel, void **args, uint32_t n_args);
int ds4_hip_graph_finalize(ds4_hip_graph *g);
int ds4_hip_graph_execute(ds4_hip_graph *g);

#endif /* DS4_HIP */

#endif /* DS4_HIP_H */