#include "felix/order_book.hpp"

namespace felix {

    void OrderBook::add_order(const Order& order) {
        orders_[order.id] = order;
    }

    void OrderBook::cancel_order(uint64_t order_id) {
        auto it = orders_.find(order_id);
        if (it != orders_.end()) {
            it->second.active = false; 
            orders_.erase(it);
        }
    }

    std::vector<Fill> OrderBook::check_fills(double market_price) {
        std::vector<Fill> fills;
        
        for (auto it = orders_.begin(); it != orders_.end(); ) {
            const auto& order = it->second;
            bool filled = false;

            if (order.side == Side::BUY) {
                if (market_price <= order.price) {
                    filled = true;
                }
            } else { // SELL
                if (market_price >= order.price) {
                    filled = true;
                }
            }

            if (filled) {
                // Generate Fill
                fills.push_back({
                    order.id,
                    order.symbol_id,
                    order.price, // Limit Price Fill (Optimistic)
                    order.volume,
                    order.side
                });
                
                // Remove filled order
                it = orders_.erase(it);
            } else {
                ++it;
            }
        }

        return fills;
    }

}
