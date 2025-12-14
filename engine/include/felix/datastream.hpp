#pragma once

#include <string>
#include <vector>
#include <memory>
#include <span>

namespace felix {

    struct TickRecord; // Forward declaration

    class DataStream {
    public:
        DataStream();
        ~DataStream();

        void load(const std::string& file_path);
        bool next(TickRecord& out_tick);
        void reset();

    private:
        struct Impl;
        std::unique_ptr<Impl> pImpl;
    };

}
