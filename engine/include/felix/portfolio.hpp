#pragma once

#include "execution.hpp"
#include <map>
#include <cstdint>

namespace felix {

    struct Position {
        double quantity;     // Positive = Long, Negative = Short
        double avg_price;
        double realized_pnl;
    };

    class Portfolio {
    public:
        Portfolio(double initial_cash = 100000.0);

        // Process a fill and update positions
        void on_fill(const Fill& fill);

        // Getters
        double cash() const { return cash_; }
        double equity() const;  // Cash + Unrealized P&L
        double unrealized_pnl(uint64_t symbol_id, double current_price) const;
        double total_unrealized_pnl(double current_price) const; // Simplified: assumes one symbol
        
        const Position& get_position(uint64_t symbol_id) const;
        
        // Mark-to-market update
        void update_prices(uint64_t symbol_id, double price);

    private:
        double cash_;
        std::map<uint64_t, Position> positions_;
        std::map<uint64_t, double> last_prices_;
    };

}
