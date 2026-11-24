#include "Commands/Commands.h"
#include "IO/Gguf.h"
#include "Core/LocalFunction.h"
#include "Core/Device.h"
#include "Inference/LLM/GenerationMemory.h"
#include "Models/LLM/DecoderFactory.h"
#include "Distributed/ProcessGroup.h"
#include "Text/HuggingFaceTokenizer.h"
#include "Text/TokenStream.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace elman {
namespace {
using Clock = std::chrono::steady_clock;
void require(bool condition, const std::string& message) {
    if (!condition) throw std::invalid_argument("Generation: " + message);
}
double elapsed(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}
Json integers(const std::vector<int32_t>& values) {
    Json::Array result;
    for (int value : values) result.emplace_back(value);
    return result;
}
int integer(const std::string& value) {
    size_t used = 0;
    int result = std::stoi(value, &used);
    require(used == value.size(), "invalid integer: " + value);
    return result;
}
double number(const std::string& value) {
    size_t used = 0;
    double result = std::stod(value, &used);
    require(used == value.size() && std::isfinite(result), "invalid number: " + value);
    return result;
}
uint64_t seed_number(const std::string& value) {
    require(!value.empty() && std::all_of(value.begin(), value.end(), [](char c) { return c >= '0' && c <= '9'; }),
            "seed must be an unsigned decimal integer");
    size_t used = 0;
    unsigned long long result = std::stoull(value, &used);
    require(used == value.size(), "invalid seed");
    return result;
}
size_t memory_mib(const std::string& value) {
    require(!value.empty() && std::all_of(value.begin(), value.end(), [](char c) { return c >= '0' && c <= '9'; }),
            "memory MiB must be an unsigned decimal integer");
    unsigned long long result = std::stoull(value);
    require(result <= std::numeric_limits<size_t>::max(), "memory value exceeds size_t");
    return checked_memory_mul(size_t(result), 1024 * 1024);
}
Json memory_json(const MemoryReport& report) {
    return Json::Object{{"budget_bytes", double(report.budget_bytes)},
        {"estimated_peak_bytes", double(report.estimated_peak_bytes)}, {"measured_peak_bytes", double(report.measured_peak_bytes)},
        {"host_budget_bytes", double(report.host_budget_bytes)}, {"host_estimated_peak_bytes", double(report.host_estimated_peak_bytes)},
        {"host_measured_peak_bytes", double(report.host_measured_peak_bytes)}};
}
Json device_memory_json(const std::vector<DeviceMemoryReport>& reports) {
    Json::Array result;
    for (const DeviceMemoryReport& item : reports) result.emplace_back(Json::Object{{"ordinal", item.ordinal}, {"memory", memory_json(item.memory)}});
    return result;
}
Json allocation_json(const AllocationPoolStats& stats) {
    return Json::Object{{"requests", double(stats.requests)}, {"allocations", double(stats.allocations)},
        {"reuses", double(stats.reuses)}, {"evictions", double(stats.evictions)},
        {"allocated_bytes", double(stats.allocated_bytes)}, {"peak_reserved_bytes", double(stats.peak_reserved_bytes)},
        {"peak_cached_bytes", double(stats.peak_cached_bytes)}};
}
Json staging_json(const HostStagingStats& stats) {
    return Json::Object{{"allocated_bytes", double(stats.allocated_bytes)}, {"pending_bytes", double(stats.pending_bytes)},
        {"cached_bytes", double(stats.cached_bytes)}, {"peak_bytes", double(stats.peak_bytes)},
        {"metadata_bytes", double(stats.metadata_bytes)}, {"buffers", double(stats.buffers)},
        {"allocations", double(stats.allocations)}, {"reuses", double(stats.reuses)}, {"waits", double(stats.waits)},
        {"fallbacks", double(stats.fallbacks)}, {"uploads", double(stats.uploads)}, {"downloads", double(stats.downloads)},
        {"uploaded_bytes", double(stats.uploaded_bytes)}, {"downloaded_bytes", double(stats.downloaded_bytes)}};
}
}

int generate_command(int argc, char** argv) {
    require(argc >= 3, "generate requires a model directory, safetensors checkpoint or GGUF file");
    std::string model_path = argv[2], input_path, output_path, backend_name_, dtype_name_;
    GenerationOptions options;
    bool stream = false, fused_pointwise = true, fused_normalization = true, fused_attention_softmax = true, tiled_attention = true, packed_projections = true;
    bool direct_paged_attention = false, absorbed_attention = true;
    bool temperature_set = false, top_k_set = false, top_p_set = false, penalty_set = false, sampling_set = false;
    std::vector<std::string> stops;
    std::vector<int> device_ordinals;
    int simulated_devices = 0;
    bool tensor_parallel = false;
    for (int i = 3; i < argc; ++i) {
        std::string key = argv[i];
        if (key == "--stream") { stream = true; continue; }
        if (key == "--cuda-graph") { options.cuda_graph = true; continue; }
        if (key == "--tensor-parallel") { tensor_parallel = true; continue; }
        if (key == "--mla-expanded") { absorbed_attention = false; continue; }
        if (key == "--sample") { options.do_sample = true; continue; }
        if (key == "--no-memory-reuse") { options.reuse_memory = false; continue; }
        if (key == "--synchronous") { options.asynchronous = false; continue; }
        if (key == "--no-pinned-staging") { options.pinned_staging = false; continue; }
        if (key == "--paged-cache") { options.paged_cache = true; continue; }
        if (key == "--direct-paged-attention") { direct_paged_attention = options.paged_cache = true; continue; }
        if (key == "--full-logits") { options.full_logits = true; continue; }
        if (key == "--no-fusion") { fused_pointwise = fused_normalization = fused_attention_softmax = tiled_attention = packed_projections = false; continue; }
        if (key == "--no-normalization-fusion") { fused_normalization = false; continue; }
        if (key == "--no-attention-fusion") { fused_attention_softmax = tiled_attention = false; continue; }
        if (key == "--no-tiled-attention") { tiled_attention = false; continue; }
        if (key == "--no-packed-projections") { packed_projections = false; continue; }
        require(i + 1 < argc, "missing value for " + key);
        std::string value = argv[++i];
        if (key == "--input") input_path = value;
        else if (key == "--output") output_path = value;
        else if (key == "--backend") backend_name_ = value;
        else if (key == "--dtype") dtype_name_ = value;
        else if (key == "--simulate-devices") simulated_devices = integer(value);
        else if (key == "--devices") {
            require(device_ordinals.empty() && !value.empty(), "--devices requires one nonempty comma-separated list");
            size_t begin = 0;
            do {
                const size_t end = value.find(',', begin);
                const std::string token = value.substr(begin, end == std::string::npos ? end : end - begin);
                require(!token.empty() && std::all_of(token.begin(), token.end(), [](char c) { return c >= '0' && c <= '9'; }),
                    "device ordinals must be nonnegative decimal integers");
                device_ordinals.push_back(integer(token));
                if (end == std::string::npos) break;
                begin = end + 1;
            } while (true);
        }
        else if (key == "--max-new-tokens") options.max_new_tokens = integer(value);
        else if (key == "--max-context") options.max_context = integer(value);
        else if (key == "--prefill-chunk-size") options.prefill_chunk_size = integer(value);
        else if (key == "--memory-limit-mib") options.memory.limit_bytes = memory_mib(value);
        else if (key == "--memory-reserve-mib") options.memory.reserve_bytes = memory_mib(value);
        else if (key == "--stop") stops.push_back(value);
        else if (key == "--temperature") { options.sampling.temperature = number(value); temperature_set = sampling_set = true; }
        else if (key == "--top-k") { options.sampling.top_k = integer(value); top_k_set = sampling_set = true; }
        else if (key == "--top-p") { options.sampling.top_p = number(value); top_p_set = sampling_set = true; }
        else if (key == "--repetition-penalty") { options.sampling.repetition_penalty = number(value); penalty_set = sampling_set = true; }
        else if (key == "--seed") { options.seed = seed_number(value); sampling_set = true; }
        else throw std::invalid_argument("Unknown generation option: " + key);
    }
    require(!input_path.empty(), "generate requires --input JSON");
    const std::filesystem::path supplied_path = std::filesystem::absolute(std::filesystem::u8path(model_path)).lexically_normal();
    require(std::filesystem::exists(supplied_path), "Model path does not exist: " + model_path);
    const auto gguf = find_gguf_model(supplied_path);
    if (!gguf.empty()) model_path = gguf.u8string();
    else if (std::filesystem::is_regular_file(supplied_path)) {
        require(supplied_path.filename() == "model.safetensors" || supplied_path.filename() == "model.safetensors.index.json",
            "Use the model directory, model.safetensors or model.safetensors.index.json for a complete checkpoint");
        model_path = supplied_path.parent_path().u8string();
    }
    require(!gguf.empty() || std::filesystem::is_directory(std::filesystem::u8path(model_path)), "Model path must be a directory or supported checkpoint file");
    require(!gguf.empty() || std::filesystem::exists(std::filesystem::u8path(model_path) / "config.json"),
        "The model directory must contain config.json alongside its weights and tokenizer files");
    if (backend_name_.empty()) backend_name_ = "cpu";
    require(options.max_new_tokens >= 0 && options.max_context > 0, "invalid output or context limit");
    require(options.prefill_chunk_size >= 0, "prefill_chunk_size must be nonnegative");
    require(options.do_sample || !sampling_set, "sampling options require --sample");
    Backend backend = parse_backend(backend_name_);
    require(backend_available(backend), "backend is unavailable");
    require(device_ordinals.empty() || backend == Backend::CUDA, "--devices requires CUDA");
    require(simulated_devices == 0 || (simulated_devices >= 2 && backend == Backend::CUDA && device_ordinals.size() <= 1),
        "--simulate-devices requires at least two stages on a single CUDA device");
    require(!simulated_devices || (!options.paged_cache && !options.cuda_graph), "simulated device stages require eager contiguous execution");
    require(device_ordinals.size() <= 1 || !options.paged_cache, "layer partitioning requires contiguous caches");
    int ordinal = device_ordinals.empty() ? current_device(backend) : device_ordinals.front();
    std::shared_ptr<ProcessGroup> parallel;
    if (tensor_parallel) {
        require(device_ordinals.empty() && !simulated_devices && !options.paged_cache && !options.cuda_graph,
                "Tensor parallel generation requires eager contiguous caches with one device per rank");
        const ProcessGroupOptions ranks = process_group_environment();
        require(ranks.size > 1, "Launch tensor parallel generation with two or more ranks");
        parallel = std::make_shared<ProcessGroup>(ranks);
        if (backend == Backend::CUDA) ordinal = ranks.local_rank;
    }
    const bool primary = !parallel || parallel->rank() == 0;
    DeviceScope input_device(backend, ordinal);
    DType dtype = parse_dtype(dtype_name_.empty() ? (!gguf.empty() && backend != Backend::CPU ? "float16" :
        backend == Backend::CUDA ? "bfloat16" : "float32") : dtype_name_);
    require(dtype == DType::Float32 || dtype == DType::Float16 || dtype == DType::BFloat16, "dtype must be float32, float16 or bfloat16");
    Json requests = Json::read(input_path);
    Json::Array items = requests.is_array() ? requests.array() : Json::Array{requests};
    require(!items.empty(), "request array is empty");
    std::unique_ptr<HuggingFaceTokenizer> tokenizer;
    if (std::any_of(items.begin(), items.end(), [](const Json& item) { return !item.contains("input_ids"); }))
        tokenizer = std::make_unique<HuggingFaceTokenizer>(model_path);
    std::filesystem::path generation_path = std::filesystem::path(model_path) / "generation_config.json";
    if (std::filesystem::exists(generation_path)) {
        Json config = Json::read(generation_path.string());
        if (options.do_sample) {
            if (!temperature_set) options.sampling.temperature = config.number_or("temperature", 1);
            if (!top_k_set) options.sampling.top_k = config.int_or("top_k", 0);
            if (!top_p_set) options.sampling.top_p = config.number_or("top_p", 1);
            if (!penalty_set) options.sampling.repetition_penalty = config.number_or("repetition_penalty", 1);
        }
        if (config.contains("eos_token_id") && !config.at("eos_token_id").is_null()) {
            const Json& eos = config.at("eos_token_id");
            if (eos.is_array()) for (const Json& value : eos.array()) options.eos_ids.push_back(value.integer());
            else options.eos_ids.push_back(eos.integer());
        }
    } else if (tokenizer) options.eos_ids = tokenizer->stop_ids();
    if (options.do_sample) validate_sampling(options.sampling);
    std::vector<std::vector<int32_t>> inputs;
    std::vector<std::vector<std::string>> request_stops;
    for (const Json& item : items) {
        require(int(item.contains("prompt")) + int(item.contains("messages")) + int(item.contains("input_ids")) == 1,
            "each request needs exactly one of prompt, messages or input_ids");
        if (item.contains("input_ids")) {
            std::vector<int32_t> ids;
            for (const Json& token : item.at("input_ids").array()) ids.push_back(token.integer());
            inputs.push_back(std::move(ids));
            require(!stream && stops.empty() && !item.contains("stop"), "input_ids requests return IDs and do not support text streaming/stops");
        } else inputs.push_back(item.contains("prompt") ? tokenizer->encode(item.at("prompt").string()) :
                                tokenizer->encode_chat(item.at("messages").array()));
        std::vector<std::string> strings = stops;
        if (item.contains("stop")) {
            const Json& value = item.at("stop");
            if (value.is_array()) for (const Json& stop : value.array()) strings.push_back(stop.string());
            else strings.push_back(value.string());
        }
        TextStopFilter validate_stops(strings);
        request_stops.push_back(std::move(strings));
    }
    DecoderConfig model_config = read_decoder_config(model_path);
    require(simulated_devices <= model_config.layers, "simulated stages exceed the number of layers");
    require(!model_config.latent_cache_rank || !options.paged_cache, "DeepSeek generation uses its compressed contiguous cache");
    require(!model_config.recurrent_layers || !options.paged_cache, "Qwen3.5 generation uses its hybrid recurrent cache");
    if (model_config.latent_cache_rank || model_config.recurrent_layers) packed_projections = false;
    options.max_context = std::min(options.max_context, model_config.max_positions);
    require(!options.cuda_graph || (backend == Backend::CUDA && !model_config.latent_cache_rank && !model_config.recurrent_layers &&
        device_ordinals.size() <= 1 && tiled_attention && options.max_context >= 2),
        "--cuda-graph requires single-device Qwen, tiled attention and context >= 2");
    std::vector<DeviceLoadPlan> load_plans = parallel
        ? std::vector<DeviceLoadPlan>{plan_qwen_tensor_parallel_load(model_path, backend, dtype, packed_projections, parallel->rank(), parallel->size())}
        : plan_decoder_load(model_path, backend, dtype, packed_projections, device_ordinals);
    const DecoderMemoryPlan& load_plan = load_plans.front().memory;
    // Reject invalid or unaffordable requests before allocating model weights.
    for (const std::vector<int>& input : inputs) {
        int count = validate_generation_request(model_config, input, options);
        DecoderMemoryPlan plan = plan_decoder_generation(model_config, backend, dtype, input.size(), size_t(count), options.do_sample, tiled_attention, packed_projections,
            options.paged_cache, options.prefill_chunk_size, options.full_logits, load_plan.weight_workspace_bytes);
        detail::add_graph_memory(plan, model_config, dtype, options, load_plan.weight_workspace_bytes);
        plan.host_additional_bytes = checked_memory_add(plan.host_additional_bytes,
            (backend == Backend::CUDA && options.asynchronous && options.pinned_staging ? host_staging_memory_bound() : 0));
        std::vector<DeviceLoadPlan> preflight_plans = load_plans;
        for (DeviceLoadPlan& local : preflight_plans) {
            local.memory.parameter_bytes = checked_memory_add(local.memory.parameter_bytes,
                checked_memory_mul(plan.additional_bytes, size_t(std::max(1, simulated_devices))));
            const size_t stages = simulated_devices ? size_t(simulated_devices) : device_ordinals.size();
            local.memory.host_additional_bytes = checked_memory_add(plan.host_additional_bytes,
                stages > 1 ? checked_memory_mul(stages, host_staging_memory_bound()) : 0);
        }
        DecoderMemoryAdmission preflight(backend, options.memory, preflight_plans, false);
    }
    std::chrono::steady_clock::time_point start = Clock::now();
    MemoryReport load_memory;
    std::vector<DeviceMemoryReport> load_device_memory;
    std::unique_ptr<DecoderModel> owned_model = [&] {
        DecoderMemoryAdmission admission(backend, options.memory, load_plans, true);
        DecoderLoadOptions settings;
        settings.fused_pointwise = fused_pointwise;
        settings.fused_normalization = fused_normalization;
        settings.fused_attention_softmax = fused_attention_softmax;
        settings.tiled_attention = tiled_attention;
        settings.packed_projections = packed_projections;
        settings.direct_paged_attention = direct_paged_attention;
        settings.absorbed_attention = absorbed_attention;
        settings.devices = device_ordinals;
        settings.simulated_devices = simulated_devices;
        settings.tensor_parallel = parallel;
        std::unique_ptr<DecoderModel> loaded = load_decoder(model_path, backend, dtype, settings);
        load_memory = admission.report();
        load_device_memory = admission.device_reports();
        return loaded;
    }();
    const DecoderModel& model = *owned_model;
    if (parallel) parallel->barrier();
    double load_ms = elapsed(start);
    Json::Array results;
    LocalFunction record_result{[&](const std::string& name, const std::vector<int32_t>& input, const GenerationResult& result) {
        if (!primary || output_path.empty()) return;
        results.emplace_back(Json::Object{{"name", name},
            {"input_ids", integers(input)}, {"generated_ids", integers(result.ids)}, {"text", result.text},
            {"matched_stop", result.matched_stop},
            {"finish_reason", result.finish_reason}, {"prompt_tokens", static_cast<int>(input.size())},
            {"generated_tokens", static_cast<int>(result.ids.size())},
            {"prefill_ms", result.prefill_ms}, {"decode_ms", result.decode_ms}, {"memory", memory_json(result.memory)},
            {"device_memory", device_memory_json(result.device_memory)},
            {"cuda_graph", Json::Object{{"commands", double(result.graph_commands)}, {"nodes", double(result.graph_nodes)},
                {"replays", double(result.graph_replays)}, {"capture_ms", result.graph_capture_ms},
                {"retained_bytes", double(result.graph_retained_bytes)}}},
            {"prefill_tokens", result.prefill_tokens}, {"prefill_chunks", result.prefill_chunks},
            {"allocations", allocation_json(result.allocations)}});
    }};
    for (size_t i = 0; i < items.size(); ++i) {
        std::function<void(const std::string&)> on_text;
        if (stream && primary) on_text = [](const std::string& piece) { std::cout << piece << std::flush; };
        const bool token_input = items[i].contains("input_ids");
        GenerationResult result = token_input ? generate(model, inputs[i], options) :
            generate_text(model, *tokenizer, inputs[i], options, request_stops[i], on_text);
        record_result(items[i].string_or("name", std::to_string(i)), inputs[i], result);
        if (primary) {
            if (!stream) std::cout << (token_input ? integers(result.ids).dump(0) : result.text);
            std::cout << '\n' << std::flush;
        }
    }
    if (primary && !output_path.empty()) Json(Json::Object{{"backend", backend_name(backend)}, {"dtype", dtype_name(dtype)},
        {"strategy", options.do_sample ? "sample" : "greedy"}, {"load_ms", load_ms}, {"max_context", options.max_context},
        {"load_memory", memory_json(load_memory)},
        {"load_device_memory", device_memory_json(load_device_memory)},
        {"execution_stages", model.execution_stages()}, {"simulated_device_placement", model.simulated_device_placement()},
        {"tensor_parallel_size", model.tensor_parallel_size()},
        {"activation_transfers", double(model.activation_transfers())},
        {"memory_reuse", options.reuse_memory},
        {"paged_cache", options.paged_cache},
        {"cuda_graph", options.cuda_graph},
        {"full_logits", options.full_logits}, {"prefill_chunk_size", options.prefill_chunk_size},
        {"asynchronous", backend == Backend::CUDA && options.asynchronous},
        {"pinned_staging", backend == Backend::CUDA && options.asynchronous && options.pinned_staging},
        // These counters cover the reused worker stream across all CLI requests.
        {"stream_host_staging", staging_json(generation_host_staging_stats(backend))},
        {"fused_pointwise", model.fused_pointwise()},
        {"fused_normalization", model.fused_normalization()},
        {"fused_attention_softmax", model.fused_attention_softmax()},
        {"tiled_attention", model.tiled_attention_enabled()},
        {"packed_projections", model.packed_projections()},
        {"direct_paged_attention", model.direct_paged_attention()},
        {"attention_tiling", model.attention_tiling_name()},
        {"sampling", Json::Object{{"temperature", options.sampling.temperature}, {"top_k", options.sampling.top_k},
            {"top_p", options.sampling.top_p}, {"repetition_penalty", options.sampling.repetition_penalty},
            {"seed", std::to_string(options.seed)}, {"rng", "splitmix64-high24"}, {"sorting", "stable"}}},
        {"results", results}}).write(output_path);
    return 0;
}
} // namespace elman
