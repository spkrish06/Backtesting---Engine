#include "felix/matching.hpp"
#include <algorithm>

namespace felix {

    MatchingEngine::MatchingEngine() : slippage_{} {}
    
    MatchingEngine::MatchingEngine(const SlippageConfig& config) : slippage_(config) {}

    void MatchingEngine::set_fill_callback(FillCallback cb) {
        on_fill_ = cb;
    }
    
    void MatchingEngine::set_slippage_config(const SlippageConfig& config) {
        slippage_ = config;
    }

    double MatchingEngine::apply_slippage(double price, Side side) const {
        // Apply slippage: buyer pays more, seller receives less
        double direction = (side == Side::BUY) ? 1.0 : -1.0;
        double slippage_pct = slippage_.fixed_bps / 10000.0;
        return price * (1.0 + direction * slippage_pct);
    }
    
    void MatchingEngine::activate_pending_orders(uint64_t current_time) {
        // Activate orders where t >= t_active (§8.1 latency model)
        auto it = pending_orders_.begin();
        while (it != pending_orders_.end()) {
            if (current_time >= it->t_active) {
                // Order is now active, add to order book
                books_[it->order.symbol_id].add_order(it->order);
                it = pending_orders_.erase(it);
            } else {
                ++it;
            }
        }
    }

    void MatchingEngine::process_tick(const TickRecord& tick) {
        // First, we activate any pending orders that have reached t_active
        activate_pending_orders(tick.timestamp);
        
        // Then check for fills
        auto it = books_.find(tick.symbol_id);
        if (it == books_.end()) {
            return;
        }

        std::vector<Fill> fills = it->second.check_fills(tick.price);

        if (on_fill_) {
            for (auto& fill : fills) {
                // Apply slippage to fill price
                fill.price = apply_slippage(fill.price, fill.side);
                on_fill_(fill);
            }
        }
    }

    uint64_t MatchingEngine::submit_order(const Order& order) {
        // Immediate activation (no latency)
        Order o = order;
        if (o.id == 0) {
            o.id = next_order_id_++;
        }
        o.active = true;

        if (o.price > 0) {
            books_[o.symbol_id].add_order(o);
        }
        
        return o.id;
    }
    
    uint64_t MatchingEngine::submit_order_with_timestamp(const Order& order, uint64_t t_signal) {
        // Order with latency model: t_active = t_signal + latency (§8.1)
        Order o = order;
        if (o.id == 0) {
            o.id = next_order_id_++;
        }
        o.active = true;
        
        // Convert latency from microseconds to nanoseconds (timestamp is in ns)
        uint64_t latency_ns = static_cast<uint64_t>(slippage_.latency_us * 1000.0);
        
        PendingOrder pending;
        pending.order = o;
        pending.t_signal = t_signal;
        pending.t_active = t_signal + latency_ns;
        
        pending_orders_.push_back(pending);
        
        return o.id;
    }

    void MatchingEngine::cancel_order(uint64_t order_id) {
        // Cancel from order books
        for (auto& [sym_id, book] : books_) {
            book.cancel_order(order_id);
        }
        
        // Also remove from pending orders
        pending_orders_.erase(
            std::remove_if(pending_orders_.begin(), pending_orders_.end(),
                [order_id](const PendingOrder& po) { return po.order.id == order_id; }),
            pending_orders_.end()
        );
    }

}


