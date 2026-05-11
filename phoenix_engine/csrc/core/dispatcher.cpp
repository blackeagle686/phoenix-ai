#include "dispatcher.h"
#include <stdexcept>

namespace phoenix {

Dispatcher& Dispatcher::instance() {
    static Dispatcher instance;
    return instance;
}

void Dispatcher::register_backend(Device device, std::unique_ptr<Backend> backend) {
    backends_[device] = std::move(backend);
}

Backend* Dispatcher::get_backend(Device device) {
    auto it = backends_.find(device);
    if (it == backends_.end()) {
        throw std::runtime_error("No backend registered for device: " + device_to_string(device));
    }
    return it->second.get();
}

} // namespace phoenix
