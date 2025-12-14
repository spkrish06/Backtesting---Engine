#include <pybind11/pybind11.h>
#include "felix/event_loop.hpp"
#include "felix/datastream.hpp"
#include "felix/tick_record.hpp"
#include "felix/matching.hpp"
#include "felix/execution.hpp"
#include "felix/portfolio.hpp"
#include "felix/risk.hpp"

namespace py = pybind11;

PYBIND11_MODULE(felix_engine, m) {
    m.doc() = "Felix HFT Backtester Engine";

    py::class_<felix::TickRecord>(m, "TickRecord")
        .def_readonly("timestamp", &felix::TickRecord::timestamp)
        .def_readonly("symbol_id", &felix::TickRecord::symbol_id)
        .def_readonly("price", &felix::TickRecord::price)
        .def_readonly("volume", &felix::TickRecord::volume)
        .def_readonly("flags", &felix::TickRecord::flags);

    py::class_<felix::DataStream>(m, "DataStream")
        .def(py::init<>())
        .def("load", &felix::DataStream::load);

    py::enum_<felix::Side>(m, "Side")
        .value("BUY", felix::Side::BUY)
        .value("SELL", felix::Side::SELL);

    py::class_<felix::Order>(m, "Order")
        .def(py::init<>())
        .def_readwrite("id", &felix::Order::id)
        .def_readwrite("symbol_id", &felix::Order::symbol_id)
        .def_readwrite("side", &felix::Order::side)
        .def_readwrite("price", &felix::Order::price)
        .def_readwrite("volume", &felix::Order::volume);

    py::class_<felix::Fill>(m, "Fill")
        .def(py::init<>())
        .def_readwrite("order_id", &felix::Fill::order_id)
        .def_readwrite("symbol_id", &felix::Fill::symbol_id)
        .def_readwrite("price", &felix::Fill::price)
        .def_readwrite("volume", &felix::Fill::volume)
        .def_readwrite("side", &felix::Fill::side);

    py::class_<felix::SlippageConfig>(m, "SlippageConfig")
        .def(py::init<>())
        .def_readwrite("fixed_bps", &felix::SlippageConfig::fixed_bps)
        .def_readwrite("latency_us", &felix::SlippageConfig::latency_us);

    py::class_<felix::MatchingEngine>(m, "MatchingEngine")
        .def(py::init<>())
        .def(py::init<const felix::SlippageConfig&>())
        .def("process_tick", &felix::MatchingEngine::process_tick)
        .def("submit_order", &felix::MatchingEngine::submit_order)
        .def("submit_order_with_timestamp", &felix::MatchingEngine::submit_order_with_timestamp)
        .def("cancel_order", &felix::MatchingEngine::cancel_order)
        .def("set_slippage_config", &felix::MatchingEngine::set_slippage_config)
        .def("pending_order_count", &felix::MatchingEngine::pending_order_count);

    py::class_<felix::Position>(m, "Position")
        .def_readonly("quantity", &felix::Position::quantity)
        .def_readonly("avg_price", &felix::Position::avg_price)
        .def_readonly("realized_pnl", &felix::Position::realized_pnl);

    py::class_<felix::Portfolio>(m, "Portfolio")
        .def(py::init<double>(), py::arg("initial_cash") = 100000.0)
        .def("on_fill", &felix::Portfolio::on_fill)
        .def("cash", &felix::Portfolio::cash)
        .def("equity", &felix::Portfolio::equity)
        .def("get_position", &felix::Portfolio::get_position)
        .def("update_prices", &felix::Portfolio::update_prices);

    py::class_<felix::RiskLimits>(m, "RiskLimits")
        .def(py::init<>())
        .def_readwrite("max_drawdown", &felix::RiskLimits::max_drawdown)
        .def_readwrite("max_position_size", &felix::RiskLimits::max_position_size)
        .def_readwrite("max_order_size", &felix::RiskLimits::max_order_size);

    py::class_<felix::RiskEngine>(m, "RiskEngine")
        .def(py::init<const felix::RiskLimits&>(), py::arg("limits") = felix::RiskLimits{})
        .def("check_order", &felix::RiskEngine::check_order)
        .def("check_drawdown", &felix::RiskEngine::check_drawdown);

    py::class_<felix::EventLoop>(m, "EventLoop")
        .def(py::init<>())
        .def("run", [](felix::EventLoop& self, felix::DataStream& stream, py::object strategy) {
            // Trampoline: call strategy.on_tick for each tick
            self.run(stream, [&](const felix::TickRecord& tick) {
                strategy.attr("on_tick")(tick);
            });
        })
        .def("run_with_matching", [](felix::EventLoop& self, felix::DataStream& stream, 
                                     felix::MatchingEngine& engine, py::object strategy) {
            // Set up fill callback to notify Python strategy
            engine.set_fill_callback([&](const felix::Fill& fill) {
                try {
                    strategy.attr("on_fill")(fill);
                } catch (py::error_already_set& e) {
                    // Isolate Python failures per spec Section 11
                }
            });
            
            // Run event loop, process each tick through matching engine
            self.run(stream, [&](const felix::TickRecord& tick) {
                strategy.attr("on_tick")(tick);
                engine.process_tick(tick);
            });
        });
}
