#include "felix/event_loop.hpp"
#include "felix/tick_record.hpp"
#include <iostream>

namespace felix {

    EventLoop::EventLoop() {
    }

    EventLoop::~EventLoop() {
    }

    void EventLoop::run(DataStream& stream, std::function<void(const TickRecord&)> on_tick) {
        std::cout << "Starting EventLoop run..." << std::endl;
        TickRecord tick;
        while(stream.next(tick)) {
            on_tick(tick);
        }
        std::cout << "EventLoop finished." << std::endl;
    }

}
