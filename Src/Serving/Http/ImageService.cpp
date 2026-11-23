#include "ImageService.h"
#include "Core/LocalFunction.h"

#include "ChatService.h"
#include "Serving/ServerLog.h"
#include "Core/Device.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>
#include <string_view>

namespace elman {
namespace {

DiffusionInput image_input(const Json& body, bool flux2, const char* sampler) {
    try {
        for (const Json::Object::value_type& entry : body.object()) {
            const std::string& key = entry.first;
            if (key != "prompt" && key != "negative_prompt" && key != "width" && key != "height" &&
                key != "steps" && key != "cfg_scale" && key != "seed" && key != "sampler_name" &&
                key != "batch_size" && key != "n_iter" && key != "send_images" && key != "save_images" &&
                key != "output_format" && key != "force_task_id" && key != "live_preview")
                throw std::invalid_argument("Unsupported image parameter: " + key);
        }
        if (body.int_or("batch_size", 1) != 1 || body.int_or("n_iter", 1) != 1)
            throw std::invalid_argument("Request one image at a time");
        if (body.string_or("sampler_name", sampler) != sampler || body.bool_or("save_images", false))
            throw std::invalid_argument(std::string("Use ") + sampler + " and download the returned image");
        DiffusionInput input;
        input.prompt = body.at("prompt").string();
        input.negative_prompt = body.string_or("negative_prompt", "");
        input.width = body.int_or("width", flux2 ? 512 : 256);
        input.height = body.int_or("height", flux2 ? 512 : 256);
        input.steps = body.int_or("steps", flux2 ? 4 : 12);
        input.guidance = body.number_or("cfg_scale", flux2 ? 1.0 : input.guidance);
        double seed = body.number_or("seed", -1);
        if (input.prompt.find_first_not_of(" \t\r\n") == std::string::npos ||
            input.prompt.size() > 4096 || input.negative_prompt.size() > 4096)
            throw std::invalid_argument("A prompt is required and each prompt must be at most 4096 UTF-8 bytes");
        const int alignment = flux2 ? 16 : 8;
        LocalFunction supported_size{[alignment](int value) { return value >= 64 && value <= 2048 && value % alignment == 0; }};
        if (!supported_size(input.width) || !supported_size(input.height))
            throw std::invalid_argument("Width and height must be multiples of " + std::to_string(alignment) + " from 64 to 2048");
        if (input.steps < 1 || input.steps > 50) throw std::invalid_argument("Steps must be from 1 to 50");
        if (!std::isfinite(input.guidance) || input.guidance < 0 || input.guidance > 30)
            throw std::invalid_argument("CFG scale must be from 0 to 30");
        if (flux2 && (input.guidance != 1.0 || !input.negative_prompt.empty()))
            throw std::invalid_argument("FLUX.2 Klein requires CFG scale 1 and an empty negative prompt");
        if (!std::isfinite(seed) || seed < -1 || seed > std::numeric_limits<uint32_t>::max() || std::floor(seed) != seed)
            throw std::invalid_argument("Seed must be -1 or an integer from 0 to 4294967295");
        if (seed == -1) {
            std::random_device random;
            input.seed = std::uniform_int_distribution<uint32_t>{}(random);
        } else input.seed = static_cast<uint32_t>(seed);
        return input;
    } catch (const std::exception& error) {
        throw ChatError(400, "invalid_image_request", error.what());
    }
}

std::string base64(std::string_view bytes) {
    constexpr std::string_view alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string encoded;
    encoded.reserve((bytes.size() + 2) / 3 * 4);
    for (size_t offset = 0; offset < bytes.size(); offset += 3) {
        const size_t count = std::min<size_t>(3, bytes.size() - offset);
        uint32_t value = 0;
        for (size_t i = 0; i < count; ++i)
            value |= uint32_t(static_cast<unsigned char>(bytes[offset + i])) << (16 - 8 * i);
        for (size_t i = 0; i < 4; ++i)
            encoded.push_back(i <= count ? alphabet[(value >> (18 - 6 * i)) & 63] : '=');
    }
    return encoded;
}

std::string encode_bitmap(const Tensor& image) {
    const Shape& shape = image.shape();
    if (shape.size() != 4 || shape[0] != 1 || shape[1] != 3 || shape[2] <= 0 || shape[3] <= 0)
        throw std::invalid_argument("Invalid RGB image shape");
    const uint64_t height = shape[2], width = shape[3];
    const uint64_t row_size = (width * 3 + 3) & ~uint64_t{3};
    const uint64_t pixel_bytes = row_size * height;
    constexpr uint32_t header_bytes = 54;
    if (pixel_bytes > std::numeric_limits<uint32_t>::max() - header_bytes)
        throw std::overflow_error("Image exceeds BMP size limit");

    std::string bmp = "BM";
    LocalFunction write{[&](uint32_t value, int bytes = 4) {
        for (int i = 0; i < bytes; ++i) bmp.push_back(static_cast<char>((value >> (i * 8)) & 255));
    }};
    write(header_bytes + static_cast<uint32_t>(pixel_bytes));
    write(0);
    write(header_bytes);
    write(40);
    write(static_cast<uint32_t>(width));
    write(static_cast<uint32_t>(-static_cast<int32_t>(height)));  // Top-down rows.
    write(1, 2);
    write(24, 2);
    write(0);
    write(static_cast<uint32_t>(pixel_bytes));
    bmp.resize(header_bytes + static_cast<size_t>(pixel_bytes), '\0');

    const std::vector<float> pixels = image.data();
    for (size_t y = 0; y < height; ++y) {
        for (size_t x = 0; x < width; ++x) {
            for (size_t c = 0; c < 3; ++c) {
                const float value = pixels[((2 - c) * height + y) * width + x];
                if (!std::isfinite(value)) throw std::runtime_error("Non-finite image pixel");
                bmp[header_bytes + y * row_size + x * 3 + c] = static_cast<char>(std::lround(std::clamp(value, 0.0f, 1.0f) * 255.0f));
            }
        }
    }
    return bmp;
}

} // namespace

ImageService::ImageService(const std::filesystem::path& directory, std::string name, Backend backend, DType dtype, bool flux2)
    : name_(std::move(name)), backend_(backend),
      session_(flux2 ? decltype(session_)(std::in_place_type<Flux2Session>, directory, backend, current_device(backend), dtype)
                     : decltype(session_)(DiffusionSession::load(directory, backend, current_device(backend), dtype))) {}

const char* ImageService::sampler() const {
    return std::holds_alternative<Flux2Session>(session_) ? "FlowMatchEuler" : "DDIM";
}

Json ImageService::generate(const Json& body, const std::function<void()>& check,
                            const std::function<void(const Json&)>& emit) {
    const bool flux2 = std::holds_alternative<Flux2Session>(session_);
    const DiffusionInput input = image_input(body, flux2, sampler());
    bool send_images, live_preview;
    std::string id;
    try {
        send_images = body.bool_or("send_images", true);
        live_preview = body.bool_or("live_preview", false) && send_images;
        if (body.string_or("output_format", "bmp") != "bmp") throw std::invalid_argument("output_format must be bmp");
        id = body.string_or("force_task_id", "");
        if (id.size() > 64 || !std::all_of(id.begin(), id.end(), [](char c) {
                return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '-' || c == '_';
            })) throw std::invalid_argument("Invalid force_task_id");
    } catch (const std::exception& error) { throw ChatError(400, "invalid_image_request", error.what()); }

    {
        std::lock_guard<std::mutex> lock(jobs_mutex_);
        if (id.empty()) {
            do { id = "job-" + std::to_string(next_job_++); } while (jobs_.count(id));
        }
        if (!jobs_.emplace(id, Job{"Queued", 0, input.steps}).second)
            throw ChatError(409, "task_already_running", "This task ID is already queued or running");
    }
    const std::string scope = std::string(flux2 ? "flux2 " : "sd ") + id;
    LocalFunction update{[&](const std::string& status, int completed, int steps) {
        {
            std::lock_guard<std::mutex> lock(jobs_mutex_);
            ImageService::Job& job = jobs_.at(id);
            job.status = status;
            job.completed_steps = completed;
            job.steps = steps;
        }
        server_status(scope, status);
        if (emit) emit(Json::Object{{"type", "progress"}, {"status", status}, {"step", completed}, {"steps", steps}});
    }};
    LocalFunction finish{[&] {
        std::lock_guard<std::mutex> lock(jobs_mutex_);
        jobs_.erase(id);
        if (active_ == id) active_.clear();
    }};
    std::unique_lock<std::timed_mutex> lock(mutex_, std::defer_lock);
    Json response;
    try {
        server_status(scope, "Queued (" + std::to_string(input.width) + "x" + std::to_string(input.height) +
            ", " + std::to_string(input.steps) + " steps, seed " + std::to_string(input.seed) + ")");
        if (emit) emit(Json::Object{{"type", "progress"}, {"status", "Queued"}, {"step", 0}, {"steps", input.steps}});
        while (!lock.try_lock_for(std::chrono::milliseconds(25))) check();
        check();
        {
            std::lock_guard<std::mutex> progress_lock(jobs_mutex_);
            active_ = id;
        }
        const DiffusionSession* stable = std::get_if<DiffusionSession>(&session_);
        const Flux2Session* flux = std::get_if<Flux2Session>(&session_);
        if (!stable && !flux) throw std::bad_variant_access();
        DiffusionStepCallback after_step;
        if (live_preview) after_step = [&](int step, int steps, const Tensor& predicted_original) {
            check();
            update("Decoding preview " + std::to_string(step) + "/" + std::to_string(steps), step, steps);
            std::string preview;
            try {
                const Tensor decoded = stable ? stable->decode_preview(predicted_original, 128)
                                              : flux->decode_preview(predicted_original, 128);
                preview = base64(encode_bitmap(decoded));
            } catch (const std::exception& error) {
                if (emit) throw std::runtime_error("Could not decode preview " + std::to_string(step) + ": " + error.what());
                server_status(scope, "Preview " + std::to_string(step) + " skipped: " + error.what());
                return;
            }
            check();
            {
                std::lock_guard<std::mutex> progress_lock(jobs_mutex_);
                ImageService::Job& job = jobs_.at(id);
                job.preview_step = step;
                job.preview = preview;
            }
            if (emit) emit(Json::Object{{"type", "preview"}, {"step", step}, {"steps", steps}, {"image", std::move(preview)}});
        };
        const DiffusionTrace trace = [&](const std::string&, const Tensor&) { check(); };
        const DiffusionProgressCallback progress = [&](const DiffusionProgress& value) {
            check();
            const int completed = value.stage == DiffusionProgress::Stage::Denoising ? value.step - 1 : value.step;
            update(diffusion_progress_text(value), completed, value.total_steps);
        };
        StableDiffusionOutput result = stable ? stable->generate(input, trace, progress, after_step)
                                              : flux->generate(input, trace, progress, after_step);
        check();
        Json::Array images;
        if (send_images) {
            update("Encoding BMP", input.steps, input.steps);
            images.emplace_back(base64(encode_bitmap(result.image)));
        }
        check();
        Json info = Json::Object{{"seed", double(input.seed)}, {"width", input.width}, {"height", input.height},
            {"steps", input.steps}, {"cfg_scale", input.guidance}, {"sampler_name", sampler()}, {"sd_model_name", name_},
            {"output_format", "bmp"}};
        response = Json::Object{{"images", std::move(images)}, {"parameters", body}, {"info", info.dump(0)}};
    } catch (const std::exception& error) {
        finish();
        server_status(scope, std::string("Stopped: ") + error.what());
        throw;
    }
    finish();
    server_status(scope, "Completed");
    return response;
}

bool ImageService::has_active_job() const {
    std::lock_guard<std::mutex> lock(jobs_mutex_);
    return !active_.empty();
}

Json ImageService::progress(std::string_view id, int after_preview_step, bool skip_image) const {
    std::lock_guard<std::mutex> lock(jobs_mutex_);
    const auto found = jobs_.find(id.empty() ? std::string_view(active_) : id);
    const Job idle;
    const Job& status = found == jobs_.end() ? idle : found->second;
    const double fraction = status.steps ? double(status.completed_steps) / status.steps : 0;
    const bool send_preview = !skip_image && status.preview_step > after_preview_step;
    return Json::Object{{"progress", fraction}, {"eta_relative", Json{}},
        {"current_image", send_preview ? Json(status.preview) : Json{}}, {"preview_step", status.preview_step},
        {"textinfo", status.status.empty() ? Json{} : Json(status.status)},
        {"state", Json::Object{{"job", status.status}, {"sampling_step", status.completed_steps}, {"sampling_steps", status.steps}}}};
}

} // namespace elman
