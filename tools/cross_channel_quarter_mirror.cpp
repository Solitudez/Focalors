#include "ns/cross_channel_quarter_mirror.h"
#include "io/para_reader.h"

#include <iostream>
#include <stdexcept>
#include <string>

int main(int argc, char* argv[])
{
    const auto para_map = IO::paras_to_map(argc, argv);

    CrossChannelQuarterMirror::Options options;
    std::string                        quarter_root;
    std::string                        output_root;
    if (!IO::read_string(para_map, "quarter_root", quarter_root))
        throw std::runtime_error("quarter_root is required.");
    if (!IO::read_string(para_map, "output_root", output_root))
        output_root = quarter_root + "_mirrored_full";

    options.quarter_root = quarter_root;
    options.output_root  = output_root;
    IO::read_number(para_map, "quarter_step", options.quarter_step);
    IO::read_number(para_map, "assignment_tolerance", options.assignment_tolerance);

    const auto result = CrossChannelQuarterMirror::reconstruct(options);
    std::cout << "Cross-channel quarter mirror completed.\n"
              << "  source: " << result.source_final_dir << " (step " << result.quarter_step << ")\n"
              << "  output: " << result.output_final_dir << "\n"
              << "  fields: " << result.reconstructed_variables.size() << " variables, " << result.written_file_count
              << " files\n"
              << "  max assignment conflict: " << result.max_assignment_conflict << std::endl;
    return 0;
}
