#include "felix/datastream.hpp"
#include "felix/tick_record.hpp"
#include <iostream>

namespace felix {

    struct DataStream::Impl {
        std::vector<TickRecord> data;
        size_t current_idx = 0;
    };

    DataStream::DataStream() : pImpl(std::make_unique<Impl>()) {
    }

    DataStream::~DataStream() = default;

    void DataStream::load(const std::string& file_path) {
        std::cout << "Loading " << file_path << std::endl;
        FILE* f = fopen(file_path.c_str(), "rb");
        if (!f) {
            std::cerr << "Failed to open " << file_path << std::endl;
            return;
        }

        fseek(f, 0, SEEK_END);
        size_t size = ftell(f);
        fseek(f, 0, SEEK_SET);

        size_t count = size / sizeof(TickRecord);
        pImpl->data.resize(count);
        
        size_t read = fread(pImpl->data.data(), sizeof(TickRecord), count, f);
        if (read != count) {
            std::cerr << "Read incomplete: " << read << " / " << count << std::endl;
        }
        
        fclose(f);
        std::cout << "Loaded " << count << " ticks." << std::endl;
        pImpl->current_idx = 0;
    }

    bool DataStream::next(TickRecord& out_tick) {
        if (pImpl->current_idx < pImpl->data.size()) {
            out_tick = pImpl->data[pImpl->current_idx++];
            return true;
        }
        return false;
    }

    void DataStream::reset() {
        pImpl->current_idx = 0;
    }

}
