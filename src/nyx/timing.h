#pragma once

#include <chrono>
#include <iostream>
#include <string>

class Stopwatch
{
public:
	Stopwatch (const std::string & clockName)
	{
		clock_name = clockName;
		start = std::chrono::system_clock::now();
	}
	~Stopwatch()
	{
		end = std::chrono::system_clock::now();
		std::chrono::duration<double, std::micro> elap = end - start;
		std::cout << clock_name << " " << elap.count() << " us\n";
	}
protected:
	std::string clock_name;
	std::chrono::time_point<std::chrono::system_clock> start, end;
};

#ifdef CHECKTIMING
	#define STOPWATCH(X) Stopwatch stopWatch(X);
#else
	#define STOPWATCH(X)
#endif