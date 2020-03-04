#include <chrono>
/*
**chrono is a precision-neutral library for time and date
**  Clock:
**  std::chrono::system_clock : current time according to the system - is not steady
**  std::chrono::steady_clock : goes at a uniform
**  std::chrono::high_resolution : provides smallest possible tick period
*/
#define SECOND_TYPE       1  //output unit is second
#define MILLISECOND_TYPE  0  //output unit is millisecond
#define MICROSECOND_TYPE  2  //output unit is microsecond
#define NANOSECOND_TYPE   3  //output unit is nanosecond

class Times {
public:
	Times(int type = 0) :out_type(type) {}; //default output unit is millisecond
	void start();
	void end();
	void delay_ms(int ms);
private:
	std::chrono::steady_clock::time_point t_start;
	int out_type = 0;
};