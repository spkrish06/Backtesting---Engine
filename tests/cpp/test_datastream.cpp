#include "felix/datastream.hpp"
#include "felix/tick_record.hpp"
#include <iostream>
#include <vector>
#include <fstream>
#include <cassert>
#include <filesystem>

int main() {
    std::string filename = "test_data.bin";
    {
        std::ofstream out(filename, std::ios::binary);
        std::vector<felix::TickRecord> ticks(10);
        for(int i=0; i<10; ++i) {
            ticks[i].timestamp = 1000 + i;
            ticks[i].symbol_id = 1;
            ticks[i].price = 100.0 + i;
            ticks[i].volume = 1.0;
            ticks[i].flags = (i % 2); 
        }
        out.write(reinterpret_cast<const char*>(ticks.data()), ticks.size() * sizeof(felix::TickRecord));
    }

    felix::DataStream stream;
    stream.load(filename);

    felix::TickRecord tick;
    int count = 0;
    while(stream.next(tick)) {
        assert(tick.timestamp == 1000 + count);
        assert(tick.price == 100.0 + count);
        count++;
    }

    assert(count == 10);
    std::cout << "Test passed! Read " << count << " ticks." << std::endl;

    std::filesystem::remove(filename);
    return 0;
}
