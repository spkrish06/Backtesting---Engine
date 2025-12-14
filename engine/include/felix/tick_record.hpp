#pragma once

#include <cstdint>

namespace felix {

    #pragma pack(push, 1)
    struct TickRecord {
        uint64_t timestamp; // Nanoseconds
        uint64_t symbol_id; 
        double price;
        double volume;
        uint8_t flags;      // 1 = Trade, 0 = Quote
        uint8_t _pad[7];    // Align to 32 bytes
    };
    #pragma pack(pop)

    static_assert(sizeof(TickRecord) == 40, "TickRecord must be 40 bytes");

}
