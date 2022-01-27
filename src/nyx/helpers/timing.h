#pragma once

#include <chrono>
#include <iostream>
#include <string>
#include <map>

/// @brief A timer providing categorized accumulatable time measurements
class Stopwatch
{
public:
	using Unit = std::micro;
	static constexpr const char* UnitString = "micro-second";

	Stopwatch(const std::string& header_, const std::string& tail_);
	~Stopwatch();
	static void add_measurement_once(const std::string& measurement_name, double value) 
	{ 
		totals[measurement_name] = value; 
	}
	static void print_stats();
	static void save_stats(const std::string & fpath);

protected:
	std::string header, tail;
	std::chrono::time_point<std::chrono::system_clock> start, end;
	static std::map <std::string, double> totals;
};

#ifdef CHECKTIMING
	#define STOPWATCH(H,T) Stopwatch stopWatch(H,T);
#else
	#define STOPWATCH(H,T)
#endif