#include <algorithm>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
#include <vector>

#include "Arguments.h"

#ifndef _WIN32
#include <cerrno>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>
#ifdef __linux__
#include <sched.h>
#endif
extern char** environ;
#endif

namespace {

using Arguments = std::vector<std::string>;
using Environment = std::map<std::string, std::string>;
using elman::tools::integer_argument;

struct Options {
    int processes = 0, port = 29500, timeout = 60, cpu_threads = 0;
    bool explicit_threads = false;
    std::string address = "127.0.0.1";
    std::vector<int> numa_nodes;
    Arguments command;
};

void usage() {
    std::cout << "Usage: elman_launch_distributed --processes N [--master-address IPv4]\n"
                 "       [--master-port PORT] [--timeout SECONDS] [--cpu-threads N]\n"
                 "       [--numa-nodes NODE,...] -- COMMAND [ARG ...]\n";
}

std::vector<int> parse_list(std::string_view input, char separator, std::string_view option) {
    std::vector<int> values;
    size_t next = 0;
    do {
        next = input.find(separator);
        values.push_back(integer_argument(input.substr(0, next), option));
        if (next != std::string_view::npos) input.remove_prefix(next + 1);
    } while (next != std::string_view::npos);
    return values;
}

int available_cpus() {
#ifdef _WIN32
    DWORD_PTR process = 0, system = 0;
    if (GetProcessAffinityMask(GetCurrentProcess(), &process, &system) && process) {
        int count = 0;
        for (; process; process &= process - 1) ++count;
        return count;
    }
#elif defined(__linux__)
    cpu_set_t affinity;
    CPU_ZERO(&affinity);
    if (!sched_getaffinity(0, sizeof(affinity), &affinity) && CPU_COUNT(&affinity)) return CPU_COUNT(&affinity);
#endif
    return static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
}

Options parse(const Arguments& arguments) {
    Options options;
    for (size_t i = 0; i < arguments.size(); ++i) {
        const std::string_view argument = arguments[i];
        if (argument == "--" || argument.rfind("--", 0) != 0) {
            const size_t first = i + (argument == "--" ? 1 : 0);
            options.command.assign(arguments.begin() + first, arguments.end());
            break;
        }
        const size_t equals = argument.find('=');
        const std::string_view key = argument.substr(0, equals);
        if (equals == std::string_view::npos && ++i == arguments.size())
            throw std::invalid_argument("Missing value for " + std::string(key));
        const std::string_view value = equals == std::string_view::npos ? std::string_view(arguments[i]) : argument.substr(equals + 1);
        if (key == "--processes") options.processes = integer_argument(value, key);
        else if (key == "--master-address") options.address = value;
        else if (key == "--master-port") options.port = integer_argument(value, key);
        else if (key == "--timeout") options.timeout = integer_argument(value, key);
        else if (key == "--cpu-threads") {
            options.cpu_threads = integer_argument(value, key);
            options.explicit_threads = true;
        } else if (key == "--numa-nodes") options.numa_nodes = parse_list(value, ',', key);
        else throw std::invalid_argument("Unknown option " + std::string(key));
    }
    if (options.processes <= 0 || options.command.empty() || options.command.front().empty() || options.port <= 0 ||
        options.port > 65535 || options.timeout <= 0 || options.timeout > 3600)
        throw std::invalid_argument("Provide a positive process count, executable, port in 1-65535 and timeout in 1-3600 seconds");
    const std::vector<int> octets = parse_list(options.address, '.', "--master-address");
    if (octets.size() != 4 || std::any_of(octets.begin(), octets.end(), [](int n) { return n < 0 || n > 255; }))
        throw std::invalid_argument("Master address must be IPv4");
    const std::string address = std::to_string(octets[0]) + '.' + std::to_string(octets[1]) + '.' +
                         std::to_string(octets[2]) + '.' + std::to_string(octets[3]);
    if (options.address != address) throw std::invalid_argument("Master address must use canonical dotted IPv4 notation");
    if (!options.explicit_threads) options.cpu_threads = std::clamp(available_cpus() / options.processes, 1, 256);
    if (options.cpu_threads < 1 || options.cpu_threads > 256) throw std::invalid_argument("CPU threads must be in 1-256");
    if (!options.numa_nodes.empty()) {
#ifndef __linux__
        throw std::invalid_argument("NUMA placement requires Linux and numactl");
#endif
        if (options.numa_nodes.size() != size_t(options.processes) ||
            std::any_of(options.numa_nodes.begin(), options.numa_nodes.end(), [](int n) { return n < 0; }))
            throw std::invalid_argument("Provide one nonnegative NUMA node per local worker");
    }
    return options;
}

#ifdef _WIN32
volatile LONG interrupted = 0;
BOOL WINAPI handle_control(DWORD kind) {
    if (kind != CTRL_C_EVENT && kind != CTRL_BREAK_EVENT) return FALSE;
    InterlockedExchange(&interrupted, 1);
    return TRUE;
}
bool was_interrupted() { return InterlockedCompareExchange(&interrupted, 0, 0) != 0; }

std::wstring widen(const std::string& text) {
    if (text.empty()) return {};
    const int length = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, text.data(), int(text.size()), nullptr, 0);
    if (!length) throw std::invalid_argument("Invalid UTF-8 argument");
    std::wstring wide(static_cast<size_t>(length), L'\0');
    MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, text.data(), int(text.size()), wide.data(), length);
    return wide;
}

std::wstring quote(const std::wstring& value) {
    std::wstring result = L"\"";
    size_t slashes = 0;
    for (const wchar_t ch : value) {
        if (ch == L'\\') {
            ++slashes;
            continue;
        }
        result.append(ch == L'"' ? slashes * 2 + 1 : slashes, L'\\');
        result += ch;
        slashes = 0;
    }
    result.append(slashes * 2, L'\\');
    return result + L'"';
}

struct EnvironmentOrder {
    bool operator()(const std::wstring& left, const std::wstring& right) const {
        return CompareStringOrdinal(left.c_str(), -1, right.c_str(), -1, TRUE) == CSTR_LESS_THAN;
    }
};

std::wstring environment_block(const Environment& updates, bool explicit_threads) {
    struct EnvironmentStrings {
        wchar_t* data = GetEnvironmentStringsW();
        ~EnvironmentStrings() { if (data) FreeEnvironmentStringsW(data); }
    } original;
    if (!original.data) throw std::runtime_error("Cannot read Windows environment");
    std::map<std::wstring, std::wstring, EnvironmentOrder> values;
    for (const wchar_t* entry = original.data; *entry;) {
        const std::wstring text(entry);
        const size_t equals = text.find(L'=', 1);
        if (equals != std::wstring::npos) values[text.substr(0, equals)] = text.substr(equals + 1);
        entry += text.size() + 1;
    }
    for (const std::pair<const std::string, std::string>& entry : updates) {
        const std::string& key = entry.first;
        const std::string& value = entry.second;
        const std::wstring name = widen(key);
        if (key != "ELMAN_CPU_THREADS" || explicit_threads || !values.count(name)) values[name] = widen(value);
    }
    std::wstring block;
    for (const std::pair<const std::wstring, std::wstring>& entry : values) {
        const std::wstring& name = entry.first;
        const std::wstring& value = entry.second;
        block += name + L'=' + value;
        block += L'\0';
    }
    block += L'\0';
    return block;
}

using Process = HANDLE;
constexpr Process invalid_process = nullptr;
Process spawn(const Arguments& arguments, const Environment& updates, bool explicit_threads) {
    std::wstring command;
    for (const std::string& argument : arguments) {
        if (!command.empty()) command += L' ';
        command += quote(widen(argument));
    }
    std::wstring environment = environment_block(updates, explicit_threads);
    STARTUPINFOW startup{};
    startup.cb = sizeof(startup);
    startup.dwFlags = STARTF_USESTDHANDLES;
    startup.hStdInput = GetStdHandle(STD_INPUT_HANDLE);
    startup.hStdOutput = GetStdHandle(STD_OUTPUT_HANDLE);
    startup.hStdError = GetStdHandle(STD_ERROR_HANDLE);
    PROCESS_INFORMATION info{};
    if (!CreateProcessW(nullptr, command.data(), nullptr, nullptr, TRUE,
                        CREATE_UNICODE_ENVIRONMENT, environment.data(), nullptr, &startup, &info))
        throw std::system_error(int(GetLastError()), std::system_category(), "Cannot start " + arguments.front());
    CloseHandle(info.hThread);
    return info.hProcess;
}

bool poll(Process& process, int& status) {
    const DWORD result = WaitForSingleObject(process, 0);
    if (result == WAIT_TIMEOUT) return false;
    DWORD code = 0;
    if (result != WAIT_OBJECT_0 || !GetExitCodeProcess(process, &code))
        throw std::system_error(int(GetLastError()), std::system_category(), "Cannot wait for worker");
    CloseHandle(process);
    process = invalid_process;
    status = static_cast<int>(code);
    return true;
}

void stop(std::vector<Process>& workers) noexcept {
    for (Process process : workers) if (process) TerminateProcess(process, 1);
    for (Process process : workers) if (process) {
        WaitForSingleObject(process, 3000);
        CloseHandle(process);
    }
}
#else
volatile std::sig_atomic_t interrupted = 0;
void handle_signal(int) { interrupted = 1; }
bool was_interrupted() { return interrupted != 0; }

std::vector<char*> pointers(Arguments& items) {
    std::vector<char*> result;
    result.reserve(items.size() + 1);
    for (std::string& item : items) result.push_back(item.data());
    result.push_back(nullptr);
    return result;
}

using Process = pid_t;
constexpr Process invalid_process = 0;
Process spawn(Arguments arguments, const Environment& updates, bool explicit_threads) {
    Environment values;
    for (char** entry = environ; *entry; ++entry) {
        const std::string text(*entry);
        const size_t equals = text.find('=');
        if (equals != std::string::npos) values[text.substr(0, equals)] = text.substr(equals + 1);
    }
    for (const std::pair<const std::string, std::string>& entry : updates) {
        const std::string& key = entry.first;
        const std::string& value = entry.second;
        if (key != "ELMAN_CPU_THREADS" || explicit_threads || !values.count(key)) values[key] = value;
    }
    Arguments environment;
    for (const std::pair<const std::string, std::string>& entry : values) {
        const std::string& key = entry.first;
        const std::string& value = entry.second;
        environment.push_back(key + '=' + value);
    }
    std::vector<char*> argv = pointers(arguments), envp = pointers(environment);
    Process child;
    const int error = posix_spawnp(&child, argv[0], nullptr, nullptr, argv.data(), envp.data());
    if (error) throw std::system_error(error, std::generic_category(), "Cannot start " + arguments.front());
    return child;
}

bool poll(Process& process, int& status) {
    int raw = 0;
    const pid_t child = waitpid(process, &raw, WNOHANG);
    if (child == 0 || (child < 0 && errno == EINTR)) return false;
    if (child < 0) throw std::system_error(errno, std::generic_category(), "Cannot wait for worker");
    process = invalid_process;
    status = WIFEXITED(raw) ? WEXITSTATUS(raw) : 128 + WTERMSIG(raw);
    return true;
}

void stop(std::vector<Process>& workers) noexcept {
    for (Process process : workers) if (process) kill(process, SIGTERM);
    const std::chrono::steady_clock::time_point deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
    while (std::chrono::steady_clock::now() < deadline) {
        bool pending = false;
        for (Process& process : workers) if (process) {
            const pid_t result = waitpid(process, nullptr, WNOHANG);
            if (result > 0 || (result < 0 && errno == ECHILD)) process = invalid_process;
            else pending = true;
        }
        if (!pending) return;
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    for (Process process : workers) if (process) kill(process, SIGKILL);
    for (Process process : workers) if (process) while (waitpid(process, nullptr, 0) < 0 && errno == EINTR) {}
}
#endif

int launch(const Options& options) {
#ifdef _WIN32
    if (!SetConsoleCtrlHandler(handle_control, TRUE))
        throw std::system_error(int(GetLastError()), std::system_category(), "Cannot register console handler");
#else
    struct sigaction action{};
    action.sa_handler = handle_signal;
    sigemptyset(&action.sa_mask);
    if (sigaction(SIGINT, &action, nullptr) != 0 || sigaction(SIGTERM, &action, nullptr) != 0)
        throw std::system_error(errno, std::generic_category(), "Cannot register signal handler");
#endif
    struct Workers {
        std::vector<Process> processes;
        ~Workers() { stop(processes); }
    } workers;
    workers.processes.reserve(static_cast<size_t>(options.processes));
    Environment environment{{"ELMAN_WORLD_SIZE", std::to_string(options.processes)}, {"ELMAN_MASTER_ADDR", options.address},
        {"ELMAN_MASTER_PORT", std::to_string(options.port)}, {"ELMAN_DISTRIBUTED_TIMEOUT", std::to_string(options.timeout)},
        {"ELMAN_CPU_THREADS", std::to_string(options.cpu_threads)}};
    for (int rank = 0; rank < options.processes; ++rank) {
        if (was_interrupted()) return 130;
        environment["ELMAN_RANK"] = environment["ELMAN_LOCAL_RANK"] = std::to_string(rank);
        Arguments command;
        if (!options.numa_nodes.empty()) {
            const std::string node = std::to_string(options.numa_nodes[rank]);
            command = {"numactl", "--cpunodebind=" + node, "--membind=" + node};
        }
        command.insert(command.end(), options.command.begin(), options.command.end());
        workers.processes.push_back(spawn(command, environment, options.explicit_threads));
    }
    int remaining = options.processes;
    while (remaining) {
        if (was_interrupted()) return 130;
        for (size_t rank = 0; rank < workers.processes.size(); ++rank) {
            Process& worker = workers.processes[rank];
            int status = 0;
            if (worker == invalid_process || !poll(worker, status)) continue;
            --remaining;
            if (status) {
                std::cerr << "Elman worker " << rank << " exited with status " << status << '\n';
                return status;
            }
        }
        if (remaining) std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    return 0;
}

}  // namespace

#ifdef _WIN32
int wmain(int argc, wchar_t** argv) {
#else
int main(int argc, char** argv) {
#endif
    try {
        Arguments arguments;
        for (int i = 1; i < argc; ++i) arguments.push_back(elman::tools::utf8_argument(argv[i]));
        if (arguments.size() == 1 && (arguments.front() == "--help" || arguments.front() == "-h")) {
            usage();
            return 0;
        }
        return launch(parse(arguments));
    } catch (const std::exception& error) {
        std::cerr << "Elman launcher: " << error.what() << '\n';
        return 1;
    }
}
