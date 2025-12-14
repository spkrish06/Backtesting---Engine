#pragma once

#include "execution.hpp"
#include <vector>
#include <map>

namespace felix {

    class OrderBook {
    public:
        void add_order(const Order& order);
        void cancel_order(uint64_t order_id);

        // Check if any resting orders match the new market price
        // Returns a list of fills generated
        std::vector<Fill> check_fills(double market_price);

        size_t active_orders_count() const { return orders_.size(); }

    private:
        // Simple storage: Map ID -> Order
        std::map<uint64_t, Order> orders_;
    };

}
