#pragma once

#include "portfolio.hpp"
#include "execution.hpp"

namespace felix {

    struct RiskLimits {
        double max_drawdown = 0.20;          // 20% max drawdown
        double max_position_size = 10000.0;  // Max notional per position
        double max_order_size = 1000.0;      // Max shares per order
    };

    class RiskEngine {
    public:
        RiskEngine(const RiskLimits& limits = {});

        // Check if an order passes risk checks
        bool check_order(const Order& order, const Portfolio& portfolio, double current_price);
        
        // Check portfolio health
        bool check_drawdown(const Portfolio& portfolio, double peak_equity);
        
        // Getters
        const RiskLimits& limits() const { return limits_; }
        
    private:
        RiskLimits limits_;
    };

}
