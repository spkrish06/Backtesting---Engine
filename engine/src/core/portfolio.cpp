#include "felix/portfolio.hpp"
#include <cmath>

namespace felix {

    Portfolio::Portfolio(double initial_cash) : cash_(initial_cash) {}

    void Portfolio::on_fill(const Fill& fill) {
        Position& pos = positions_[fill.symbol_id];
        
        double trade_value = fill.price * fill.volume;
        double direction = (fill.side == Side::BUY) ? 1.0 : -1.0;
        double signed_qty = fill.volume * direction;

        // Update average price
        if (std::abs(pos.quantity) < 1e-9) {
            // New position
            pos.avg_price = fill.price;
        } else if ((pos.quantity > 0 && fill.side == Side::BUY) ||
                   (pos.quantity < 0 && fill.side == Side::SELL)) {
            // Adding to position
            double total_value = pos.avg_price * std::abs(pos.quantity) + trade_value;
            pos.avg_price = total_value / (std::abs(pos.quantity) + fill.volume);
        } else {
            // Reducing/flipping position - realize P&L
            double closed_qty = std::min(std::abs(pos.quantity), fill.volume);
            double pnl = closed_qty * (fill.price - pos.avg_price) * (pos.quantity > 0 ? 1 : -1);
            pos.realized_pnl += pnl;
            cash_ += pnl;
        }

        // Update quantity
        pos.quantity += signed_qty;
        
        // Adjust cash for trade
        if (fill.side == Side::BUY) {
            cash_ -= trade_value;
        } else {
            cash_ += trade_value;
        }

        // Store last price
        last_prices_[fill.symbol_id] = fill.price;
    }

    double Portfolio::unrealized_pnl(uint64_t symbol_id, double current_price) const {
        auto it = positions_.find(symbol_id);
        if (it == positions_.end()) return 0.0;
        
        const Position& pos = it->second;
        return pos.quantity * (current_price - pos.avg_price);
    }

    double Portfolio::total_unrealized_pnl(double current_price) const {
        double total = 0.0;
        for (const auto& [sym, pos] : positions_) {
            auto price_it = last_prices_.find(sym);
            double price = (price_it != last_prices_.end()) ? price_it->second : current_price;
            total += pos.quantity * (price - pos.avg_price);
        }
        return total;
    }

    double Portfolio::equity() const {
        double eq = cash_;
        for (const auto& [sym, pos] : positions_) {
            auto price_it = last_prices_.find(sym);
            if (price_it != last_prices_.end()) {
                eq += pos.quantity * (price_it->second - pos.avg_price);
            }
        }
        return eq;
    }

    const Position& Portfolio::get_position(uint64_t symbol_id) const {
        static Position empty = {0, 0, 0};
        auto it = positions_.find(symbol_id);
        return (it != positions_.end()) ? it->second : empty;
    }

    void Portfolio::update_prices(uint64_t symbol_id, double price) {
        last_prices_[symbol_id] = price;
    }

}
