#include "counttime.h"
#include <iostream>
void Times::start()
{
	t_start = std::chrono::steady_clock::now();
}
void Times::end()
{
	std::chrono::steady_clock::time_point t_end = std::chrono::steady_clock::now();
	std::chrono::steady_clock::duration d = t_end - t_start;

	if (out_type == MILLISECOND_TYPE)
		std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(d).count()
		<< "ms" << std::endl;
	else if (out_type == SECOND_TYPE)
		std::cout << std::chrono::duration_cast<std::chrono::seconds>(d).count()
		<< "s" << std::endl;
	else if (out_type == MICROSECOND_TYPE)
		std::cout << std::chrono::duration_cast<std::chrono::microseconds>(d).count()
		<< "us" << std::endl;
	else 
		std::cout << std::chrono::duration_cast<std::chrono::nanoseconds>(d).count()
		<< "ns" << std::endl;
}
void Times::delay_ms(int ms)
{
	std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
	double dd;
	do
	{
		std::chrono::steady_clock::time_point end= std::chrono::steady_clock::now();
		std::chrono::steady_clock::duration d = end - start;
		dd = std::chrono::duration_cast<std::chrono::milliseconds>(d).count();
	} while (dd < ms);
	
}