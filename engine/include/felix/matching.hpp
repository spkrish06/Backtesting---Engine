#pragma once

#include "tick_record.hpp"
#include "execution.hpp"
#include "order_book.hpp"
#include <map>
#include <queue>
#include <functional>
#include <vector>

namespace felix {

    // Slippage & Latency configuration per our assignment spec §8
    struct SlippageConfig {
        double fixed_bps = 5.0;          // Fixed slippage in basis points (0.05%)
        double latency_us = 100.0;       // Order activation latency in microseconds
    };
    
    // Pending order with activation time (§8.1)
    struct PendingOrder {
        Order order;
        uint64_t t_signal;      // When order was submitted
        uint64_t t_active;      // When order becomes eligible: t_signal + latency
    };

    class MatchingEngine {
    public:
        using FillCallback = std::function<void(const Fill&)>;

        MatchingEngine();
        explicit MatchingEngine(const SlippageConfig& config);

        void set_fill_callback(FillCallback cb);
        void set_slippage_config(const SlippageConfig& config);

        // Main entry point for market data
        void process_tick(const TickRecord& tick);

        // Order entry with latency model
        uint64_t submit_order(const Order& order);
        uint64_t submit_order_with_timestamp(const Order& order, uint64_t t_signal);
        void cancel_order(uint64_t order_id);

        // Getters
        const SlippageConfig& slippage_config() const { return slippage_; }
        size_t pending_order_count() const { return pending_orders_.size(); }

    private:
        FillCallback on_fill_;
        std::map<uint64_t, OrderBook> books_;
        uint64_t next_order_id_ = 1;
        SlippageConfig slippage_;
        
        // Pending orders queue (orders waiting for activation)
        std::vector<PendingOrder> pending_orders_;
        
        // Activate pending orders that have reached t_active
        void activate_pending_orders(uint64_t current_time);
        
        // Apply slippage to fill price
        double apply_slippage(double price, Side side) const;
    };

}


