#pragma once

#include <cstdint>
#include <string>

namespace felix {

    enum class Side {
        BUY,
        SELL
    };

    struct Order {
        uint64_t id;
        uint64_t symbol_id;
        Side side;
        double price;  // 0.0 for Market Orders
        double volume;
        bool active;
    };

    struct Fill {
        uint64_t order_id;
        uint64_t symbol_id;
        double price;
        double volume;
        Side side;
    };

}
