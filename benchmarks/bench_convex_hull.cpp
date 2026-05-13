#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "../src/nyx/features/convex_hull.h"

namespace
{
using Clock = std::chrono::steady_clock;

struct Workload
{
	std::string case_name;
	int side = 0;
};

struct Options
{
	int cycles = 3;
	std::vector<Workload> workloads;
};

std::vector<Pixel2> make_x_major_sorted(int side)
{
	std::vector<Pixel2> pixels;
	pixels.reserve(static_cast<size_t>(side) * static_cast<size_t>(side));

	for (int x = 0; x < side; ++x)
		for (int y = 0; y < side; ++y)
			pixels.emplace_back(x, y, 1u);

	return pixels;
}

std::vector<Pixel2> make_y_major_raster(int side)
{
	std::vector<Pixel2> pixels;
	pixels.reserve(static_cast<size_t>(side) * static_cast<size_t>(side));

	for (int y = 0; y < side; ++y)
		for (int x = 0; x < side; ++x)
			pixels.emplace_back(x, y, 1u);

	return pixels;
}

std::vector<Pixel2> make_pixels(const std::string& case_name, int side)
{
	if (case_name == "x_major_sorted")
		return make_x_major_sorted(side);

	if (case_name == "y_major_raster")
		return make_y_major_raster(side);

	auto pixels = make_x_major_sorted(side);
	if (case_name == "reversed")
	{
		std::reverse(pixels.begin(), pixels.end());
		return pixels;
	}

	if (case_name == "shuffled")
	{
		std::mt19937 rng(1234567u + static_cast<unsigned int>(side));
		std::shuffle(pixels.begin(), pixels.end(), rng);
		return pixels;
	}

	throw std::runtime_error("unknown benchmark case: " + case_name);
}

int reps_for_side(int side)
{
	if (side <= 64)
		return 160;
	if (side <= 128)
		return 80;
	if (side <= 256)
		return 30;
	if (side <= 512)
		return 10;
	return 4;
}

double median(std::vector<double> values)
{
	std::sort(values.begin(), values.end());
	const size_t n = values.size();
	if (n % 2 == 1)
		return values[n / 2];
	return 0.5 * (values[n / 2 - 1] + values[n / 2]);
}

void initialize_convex_roi(LR& roi, const std::vector<Pixel2>& pixels, int side)
{
	roi.raw_pixels = pixels;
	roi.convHull_CH.clear();
	roi.fvals.assign(static_cast<size_t>(Nyxus::FeatureIMQ::_COUNT_), std::vector<double>(1, 0.0));

	// ConvexHullFeature depends on PERIMETER only for circularity; this keeps
	// the benchmark focused on convex hull construction, not contour tracing.
	roi.fvals[static_cast<int>(Nyxus::Feature2D::PERIMETER)][0] = 4.0 * static_cast<double>(side - 1);
}

void calculate_convex_hull(LR& roi, ConvexHullFeature& feature, const Fsettings& settings)
{
	feature.calculate(roi, settings);
	feature.save_value(roi.fvals);
}

void run_workload(int cycle, const Workload& workload)
{
	const auto pixels = make_pixels(workload.case_name, workload.side);
	const int reps = reps_for_side(workload.side);

	LR roi;
	initialize_convex_roi(roi, pixels, workload.side);

	Fsettings settings;
	ConvexHullFeature feature;

	uint64_t checksum = 0;
	for (int i = 0; i < 2; ++i)
	{
		calculate_convex_hull(roi, feature, settings);
		checksum += static_cast<uint64_t>(roi.convHull_CH.size());
		checksum += static_cast<uint64_t>(roi.fvals[static_cast<int>(Nyxus::Feature2D::CONVEX_HULL_AREA)][0]);
	}

	std::vector<double> elapsed_ms;
	elapsed_ms.reserve(static_cast<size_t>(reps));

	for (int i = 0; i < reps; ++i)
	{
		const auto start = Clock::now();
		calculate_convex_hull(roi, feature, settings);
		const auto stop = Clock::now();

		checksum += static_cast<uint64_t>(roi.convHull_CH.size());
		checksum += static_cast<uint64_t>(roi.fvals[static_cast<int>(Nyxus::Feature2D::CONVEX_HULL_AREA)][0]);
		elapsed_ms.push_back(std::chrono::duration<double, std::milli>(stop - start).count());
	}

	const double sum = std::accumulate(elapsed_ms.begin(), elapsed_ms.end(), 0.0);
	const auto [min_it, max_it] = std::minmax_element(elapsed_ms.begin(), elapsed_ms.end());
	const double hull_area = roi.fvals[static_cast<int>(Nyxus::Feature2D::CONVEX_HULL_AREA)][0];
	const double solidity = roi.fvals[static_cast<int>(Nyxus::Feature2D::SOLIDITY)][0];

	std::cout
		<< cycle << ','
		<< workload.case_name << ','
		<< workload.side << ','
		<< pixels.size() << ','
		<< reps << ','
		<< std::fixed << std::setprecision(6)
		<< median(elapsed_ms) << ','
		<< (sum / static_cast<double>(elapsed_ms.size())) << ','
		<< *min_it << ','
		<< *max_it << ','
		<< roi.convHull_CH.size() << ','
		<< hull_area << ','
		<< solidity << ','
		<< checksum
		<< '\n';
}

void add_default_workloads(std::vector<Workload>& workloads)
{
	for (int side : {128, 256, 512, 1024})
		workloads.push_back({"x_major_sorted", side});

	for (int side : {64, 128, 256, 512, 1024})
		workloads.push_back({"y_major_raster", side});

	// Keep pathological unsorted cases capped so this can also be run on older
	// revisions where shuffled input can create a huge invalid hull.
	for (int side : {64, 128, 256})
		workloads.push_back({"reversed", side});

	for (int side : {64, 128})
		workloads.push_back({"shuffled", side});
}

void print_usage(const char* program)
{
	std::cerr
		<< "Usage: " << program << " [--cycles N] [--case NAME --side N ...]\n"
		<< "\n"
		<< "Cases: x_major_sorted, y_major_raster, reversed, shuffled\n"
		<< "If no --case/--side pairs are provided, a default workload is used.\n";
}

int parse_int(std::string_view value, std::string_view flag)
{
	const std::string text(value);
	char* end = nullptr;
	const long parsed = std::strtol(text.c_str(), &end, 10);
	if (end == nullptr || *end != '\0' || parsed <= 0)
		throw std::runtime_error("invalid positive integer for " + std::string(flag));
	return static_cast<int>(parsed);
}

Options parse_options(int argc, char** argv)
{
	Options options;
	std::string pending_case;

	for (int i = 1; i < argc; ++i)
	{
		const std::string_view arg(argv[i]);
		if (arg == "--help" || arg == "-h")
		{
			print_usage(argv[0]);
			std::exit(0);
		}

		if (arg == "--cycles")
		{
			if (++i >= argc)
				throw std::runtime_error("--cycles requires a value");
			options.cycles = parse_int(argv[i], "--cycles");
			continue;
		}

		if (arg == "--case")
		{
			if (++i >= argc)
				throw std::runtime_error("--case requires a value");
			pending_case = argv[i];
			continue;
		}

		if (arg == "--side")
		{
			if (pending_case.empty())
				throw std::runtime_error("--side must follow --case");
			if (++i >= argc)
				throw std::runtime_error("--side requires a value");
			options.workloads.push_back({pending_case, parse_int(argv[i], "--side")});
			pending_case.clear();
			continue;
		}

		throw std::runtime_error("unknown argument: " + std::string(arg));
	}

	if (!pending_case.empty())
		throw std::runtime_error("--case must be paired with --side");

	if (options.workloads.empty())
		add_default_workloads(options.workloads);

	return options;
}
}

int main(int argc, char** argv)
{
	try
	{
		const Options options = parse_options(argc, argv);
		std::cout << "cycle,case,side,points,reps,median_ms,mean_ms,min_ms,max_ms,hull_size,hull_area,solidity,checksum\n";

		for (int cycle = 1; cycle <= options.cycles; ++cycle)
			for (const Workload& workload : options.workloads)
				run_workload(cycle, workload);
	}
	catch (const std::exception& e)
	{
		std::cerr << "bench_convex_hull: " << e.what() << "\n";
		print_usage(argv[0]);
		return 1;
	}

	return 0;
}
