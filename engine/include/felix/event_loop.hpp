#pragma once

#include "datastream.hpp"
#include <memory>
#include <functional>

namespace felix {

    class StrategyHandle; // We will need a way to call into Python/Strategy

    class EventLoop {
    public:
        EventLoop();
        ~EventLoop();

        void run(DataStream& stream, std::function<void(const TickRecord&)> on_tick);

    private:
        // Internal state
    };

}
