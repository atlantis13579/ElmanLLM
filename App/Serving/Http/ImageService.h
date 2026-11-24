#pragma once

#include "Inference/Diffusion/Flux2Session.h"

#include <map>
#include <mutex>
#include <string_view>
#include <variant>

namespace elman {

class ImageService {
public:
    ImageService(const std::filesystem::path& directory, std::string name, Backend backend, DType dtype, bool flux2);
    const std::string& name() const { return name_; }
    Backend backend() const { return backend_; }
    const char* sampler() const;
    bool has_active_job() const;
    Json generate(const Json& body, const std::function<void()>& check,
                  const std::function<void(const Json&)>& emit = {});
    Json progress(std::string_view id, int after_preview_step, bool skip_image) const;

private:
    struct Job {
        std::string status;
        int completed_steps = 0;
        int steps = 0;
        int preview_step = 0;
        std::string preview;
    };

    std::string name_;
    Backend backend_;
    std::variant<DiffusionSession, Flux2Session> session_;
    std::timed_mutex mutex_;
    mutable std::mutex jobs_mutex_;
    std::map<std::string, Job, std::less<>> jobs_;
    std::string active_;
    uint64_t next_job_ = 1;
};

} // namespace elman
