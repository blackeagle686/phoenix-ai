#pragma once

#include <map>
#include <memory>
#include <string>
#include "types.h"
#include "backend.h"

namespace phoenix {

class Dispatcher {
public:
    static Dispatcher& instance();

    void register_backend(Device device, std::unique_ptr<Backend> backend);
    Backend* get_backend(Device device);

private:
    Dispatcher() = default;
    std::map<Device, std::unique_ptr<Backend>> backends_;
};

} // namespace phoenix
