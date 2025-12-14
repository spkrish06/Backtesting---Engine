#include <iostream>
#include <cassert>
#include "felix/matching.hpp"

using namespace felix;

int main() {
    std::cout << "Testing MatchingEngine..." << std::endl;

    MatchingEngine engine;
    
    bool filled = false;
    engine.set_fill_callback([&](const Fill& fill) {
        std::cout << "Filled: Order " << fill.order_id 
                  << " Price " << fill.price 
                  << " Qty " << fill.volume << std::endl;
        filled = true;
    });

    // 1. Submit Limit BUY @ 100
    Order o;
    o.symbol_id = 1;
    o.side = Side::BUY;
    o.price = 100.0;
    o.volume = 10;
    
    uint64_t oid = engine.submit_order(o);
    assert(oid > 0);

    // 2. Tick @ 101 (Should NOT fill)
    TickRecord t1;
    t1.symbol_id = 1;
    t1.price = 101.0;
    engine.process_tick(t1);
    assert(!filled);

    // 3. Tick @ 99 (Should FILL)
    TickRecord t2;
    t2.symbol_id = 1;
    t2.price = 99.0;
    engine.process_tick(t2);
    assert(filled);

    std::cout << "Example Limit Order Test Passed!" << std::endl;
    return 0;
}
