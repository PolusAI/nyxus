#pragma once

#include <chrono>
#include <iostream>
#include <string>
#include <map>

class Stopwatch
{
public:
	Stopwatch (const std::string& header_, const std::string & tail_)
	{
		header = header_;
		tail = tail_;

		if (totals.find(header) == totals.end())
			totals[header] = 0.0;

		start = std::chrono::system_clock::now();
		if (header.length() > 0)
			std::cout << header << "\n";
	}
	~Stopwatch()
	{
		end = std::chrono::system_clock::now();
		std::chrono::duration<double, std::micro> elap = end - start;
		std::cout << tail << " " << elap.count() << " us\n";
		totals[header] = totals[header] + elap.count();
	}
	static void print_totals()
	{
		std::cout << "Totals\n" << "--------\n";
		for (auto& t : totals)
			std::cout << t.first << " : " << t.second << "\n";
		std::cout << "--------\n";
	}
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