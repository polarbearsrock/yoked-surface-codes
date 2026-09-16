#include "stim/circuit/circuit.h"
#include "stim/simulators/frame_simulator.h"
#include "stim/util_bot/probability_util.h"
#include <cstdio>
#include <fstream>
#include <iostream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// Observe multiple rows without introducing any random draw or simulated gate.
int main(int argc, char **argv) {
    if (argc != 6) throw std::invalid_argument("circuit prefix seed batch_size comma_separated_rows");
    FILE *input = fopen(argv[1], "r");
    if (!input) throw std::runtime_error("cannot open circuit");
    auto circuit = stim::Circuit::from_file(input);
    fclose(input);
    std::string prefix = argv[2];
    uint64_t seed = std::stoull(argv[3]);
    size_t shots = std::stoull(argv[4]);
    std::vector<size_t> selected;
    std::stringstream row_stream(argv[5]);
    std::string token;
    while (std::getline(row_stream, token, ',')) {
        size_t row = std::stoull(token);
        if (row >= shots) throw std::invalid_argument("row outside batch");
        selected.push_back(row);
    }
    if (std::set<size_t>(selected.begin(), selected.end()).size() != selected.size())
        throw std::invalid_argument("duplicate row");
    auto stats = circuit.compute_stats();
    stim::FrameSimulator<128> sim(stats, stim::FrameSimulatorMode::STORE_DETECTIONS_TO_MEMORY, 0,
        std::mt19937_64(seed ^ stim::INTENTIONAL_VERSION_SEED_INCOMPATIBILITY));
    sim.configure_for(stats, stim::FrameSimulatorMode::STORE_DETECTIONS_TO_MEMORY, shots);
    sim.reset_all();
    std::vector<std::ofstream> logs;
    std::vector<size_t> faults(selected.size(), 0);
    for (auto row : selected) {
        logs.emplace_back(prefix + "_shot_" + std::to_string(row) + "_faults.csv");
        logs.back() << "instruction_index,tick,gate,target_index,q0,pauli0,q1,pauli1,measurement_index\n";
    }
    size_t instruction = 0, ticks = 0, measurements = 0;
    circuit.for_each_operation([&](const stim::CircuitInstruction &op) {
        bool channel = op.gate_type == stim::GateType::DEPOLARIZE1 || op.gate_type == stim::GateType::DEPOLARIZE2 ||
            op.gate_type == stim::GateType::X_ERROR || op.gate_type == stim::GateType::Y_ERROR || op.gate_type == stim::GateType::Z_ERROR;
        bool measure = op.gate_type == stim::GateType::M;
        std::vector<std::vector<int>> before(selected.size());
        if (channel || measure) {
            std::set<uint32_t> unique;
            for (auto t : op.targets) {
                auto q = t.qubit_value();
                if (!unique.insert(q).second) throw std::runtime_error("repeated noise/measurement target");
                for (size_t j = 0; j < selected.size(); j++) {
                    auto row = selected[j];
                    before[j].push_back((bool)sim.x_table[q][row] + 2 * (bool)sim.z_table[q][row]);
                }
            }
        }
        sim.do_gate(op);
        if (channel) {
            size_t stride = op.gate_type == stim::GateType::DEPOLARIZE2 ? 2 : 1;
            for (size_t j = 0; j < selected.size(); j++) {
                auto row = selected[j];
                for (size_t k = 0; k < op.targets.size(); k += stride) {
                    auto q0 = op.targets[k].qubit_value();
                    int p0 = before[j][k] ^ ((bool)sim.x_table[q0][row] + 2 * (bool)sim.z_table[q0][row]);
                    int q1 = -1, p1 = 0;
                    if (stride == 2) {
                        q1 = op.targets[k+1].qubit_value();
                        p1 = before[j][k+1] ^ ((bool)sim.x_table[q1][row] + 2 * (bool)sim.z_table[q1][row]);
                    }
                    if (p0 || p1) {
                        logs[j] << instruction << ',' << ticks << ',' << stim::GATE_DATA[op.gate_type].name << ',' << k << ','
                            << q0 << ',' << "IXZY"[p0] << ',' << q1 << ',' << "IXZY"[p1] << ",-1\n";
                        faults[j]++;
                    }
                }
            }
        } else if (measure) {
            for (size_t j = 0; j < selected.size(); j++) {
                auto row = selected[j];
                for (size_t k = 0; k < op.targets.size(); k++) {
                    bool flip = sim.m_record.lookback(op.targets.size()-k)[row] ^ (before[j][k]&1);
                    if (flip) {
                        logs[j] << instruction << ',' << ticks << ",M," << k << ',' << op.targets[k].qubit_value()
                            << ",readout,-1,I," << measurements+k << '\n';
                        faults[j]++;
                    }
                }
            }
        } else if (op.gate_type == stim::GateType::TICK) {
            ticks++;
        } else if ((stim::GATE_DATA[op.gate_type].flags & stim::GATE_IS_NOISY) && op.args.size() && op.args[0] != 0) {
            throw std::runtime_error("unhandled noisy instruction: " + op.str());
        }
        measurements += op.compute_stats(nullptr).num_measurements;
        instruction++;
    });
    auto save = [&](const stim::simd_bit_table<128> &table, size_t bits, const std::string &name) {
        auto transposed = table.transposed();
        std::ofstream output(prefix + name, std::ios::binary);
        for (size_t row = 0; row < shots; row++) {
            if (bits % 8) transposed[row].u8[bits/8] &= (1 << (bits%8))-1;
            output.write(reinterpret_cast<char *>(transposed[row].u8), (bits+7)/8);
        }
    };
    save(sim.det_record.storage, stats.num_detectors, "_detectors.b8");
    save(sim.obs_record, stats.num_observables, "_observables.b8");
    for (size_t j=0; j<selected.size(); j++)
        std::cout << "shot=" << selected[j] << " faults=" << faults[j] << '\n';
}
