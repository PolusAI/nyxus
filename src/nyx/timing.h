#pragma once

#include <chrono>
#include <iostream>
#include <string>

class Stopwatch
{
public:
	Stopwatch (const std::string& header, const std::string & tail)
	{
		summary_text = tail;
		start = std::chrono::system_clock::now();
		if (header.length() > 0)
			std::cout << header << "\n";
	}
	~Stopwatch()
	{
		end = std::chrono::system_clock::now();
		std::chrono::duration<double, std::micro> elap = end - start;
		std::cout << summary_text << " " << elap.count() << " us\n";
	}
protected:
	std::string summary_text;
	std::chrono::time_point<std::chrono::system_clock> start, end;
};

#ifdef CHECKTIMING
	#define STOPWATCH(H,T) Stopwatch stopWatch(H,T);
#else
	#define STOPWATCH(H,T)
#endif