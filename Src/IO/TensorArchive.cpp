#include "IO/TensorArchive.h"

#include <algorithm>
#include <cctype>
#include <stdexcept>

#include "IO/Gguf.h"
#include "IO/NumPy.h"
#include "IO/Safetensors.h"
#include "IO/TorchWeights.h"

namespace elman {

std::string model_file_extension(const std::filesystem::path& path) {
    auto extension = path.extension().u8string();
    std::transform(extension.begin(), extension.end(), extension.begin(),
                   [](unsigned char value) { return char(std::tolower(value)); });
    return extension;
}

bool is_tensor_archive(const std::filesystem::path& path) {
    const auto extension = model_file_extension(path);
    return extension == ".safetensors" || extension == ".npy" || extension == ".npz" ||
           extension == ".pt" || extension == ".pth" || extension == ".bin" || extension == ".gguf";
}

TensorArchive load_tensor_archive(const std::filesystem::path& path, Backend backend, const std::string& single_name) {
    const auto extension = model_file_extension(path);
    if (extension == ".npy" || extension == ".npz") return io::numpy_archive(path, backend, single_name);
    if (extension == ".pt" || extension == ".pth" || extension == ".bin") return io::torch_weights(path, backend);
    TensorArchive tensors;
    if (extension == ".safetensors") {
        const SafetensorsFile file(path);
        for (const auto& pair : file.tensors()) tensors.emplace(pair.first, file.load(pair.first, backend));
    } else if (extension == ".gguf") {
        const GgufFile file(path);
        for (const auto& pair : file.tensors()) tensors.emplace(pair.first, file.load(pair.first, backend));
    } else throw std::invalid_argument("Unsupported tensor archive extension " + extension);
    return tensors;
}

}  // namespace elman
