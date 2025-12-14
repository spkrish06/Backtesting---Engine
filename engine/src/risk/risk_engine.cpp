#include "felix/risk.hpp"
#include <cmath>

namespace felix {

    RiskEngine::RiskEngine(const RiskLimits& limits) : limits_(limits) {}

    bool RiskEngine::check_order(const Order& order, const Portfolio& portfolio, double current_price) {
        // Check order size
        if (order.volume > limits_.max_order_size) {
            return false;
        }

        // Check position size after fill
        const Position& pos = portfolio.get_position(order.symbol_id);
        double new_qty = pos.quantity;
        if (order.side == Side::BUY) {
            new_qty += order.volume;
        } else {
            new_qty -= order.volume;
        }

        double notional = std::abs(new_qty) * current_price;
        if (notional > limits_.max_position_size) {
            return false;
        }

        return true;
    }

    bool RiskEngine::check_drawdown(const Portfolio& portfolio, double peak_equity) {
        double current = portfolio.equity();
        if (peak_equity <= 0) return true;
        
        double drawdown = (peak_equity - current) / peak_equity;
        return drawdown < limits_.max_drawdown;
    }

}
